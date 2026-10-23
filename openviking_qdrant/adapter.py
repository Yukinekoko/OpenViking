# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""OpenViking ``CollectionAdapter`` backed by Qdrant.

The adapter intentionally keeps all Qdrant specifics below the OpenViking
interface.  Compared with the historical fork it does not require changes to
``factory.py``, ``vectordb_config.py``, ``base.py`` or ``collection.py``:

* dotted backend class paths are discovered by the upstream factory;
* filter-only ``query()`` calls go to Qdrant scroll instead of synthesizing a
  random dense vector, so strict-mode ``max_query_limit`` is not involved;
* event time decay is implemented locally with the same curve and candidate
  amplification as the native engine;
* filter-based deletion uses Qdrant's native delete endpoint.
"""

from __future__ import annotations

from datetime import datetime, timezone
from threading import RLock
from typing import Any

from openviking.storage.expr import FilterExpr
from openviking.storage.vectordb.collection.collection import Collection
from openviking.storage.vectordb_adapters.base import CollectionAdapter
from openviking.utils.time_decay import build_time_decay_fusion_spec
from openviking.utils.time_utils import parse_iso_datetime

from .collection import QdrantCollection
from .config import QdrantSettings
from .filter import compile_qdrant_filter, filter_has_contains
from .meta import META_SUFFIX
from .rest import QdrantRestClient
from .utils import finite_score


class QdrantCollectionAdapter(CollectionAdapter):
    """Adapter that maps the OpenViking collection contract to Qdrant REST."""

    mode = "qdrant"
    USE_CONTENT_FIELD = False

    def __init__(
        self,
        settings: QdrantSettings,
        *,
        client: QdrantRestClient | None = None,
    ) -> None:
        super().__init__(
            collection_name=settings.collection_name,
            index_name=settings.index_name,
        )
        self._settings = settings
        self._client = client or QdrantRestClient(
            settings.url,
            api_key=settings.api_key,
            timeout_seconds=settings.timeout_seconds,
            max_retries=settings.max_retries,
            backoff_base_seconds=settings.backoff_base_seconds,
            backoff_max_seconds=settings.backoff_max_seconds,
        )
        self._project_name = settings.project_name or "default"
        self._distance_metric = settings.distance_metric
        self._sparse_weight = settings.sparse_weight
        self._sparse_enabled = settings.sparse_weight > 0.0
        self._dense_vector_name = settings.dense_vector_name
        self._sparse_vector_name = settings.sparse_vector_name
        self._metadata_collection_name_override = settings.metadata_collection_name
        self._dimension = settings.dimension
        self._DATA_BATCH_SIZE = settings.batch_size
        self._lock = RLock()
        self._qdrant: QdrantCollection | None = None
        self._physical_collection_name = self._physical_name(self._collection_name)
        self._metadata_collection_name = self._metadata_name()

    # ------------------------------------------------------------------
    # construction / loading
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: Any) -> "QdrantCollectionAdapter":
        from .compat import check_core_compatibility

        check_core_compatibility()
        return cls(QdrantSettings.from_config(config))

    # ------------------------------------------------------------------
    # operations / diagnostics
    # ------------------------------------------------------------------

    def health_check(self) -> dict[str, Any]:
        """Best-effort health/reuse report for operations and smoke tests.

        Uses the same transport (and therefore the same retries/credentials)
        as the adapter itself; never raises for network failures.
        """
        from .diagnose import run_diagnostics

        return run_diagnostics(
            settings=self._settings,
            client=self._client,
        )

    def reuse_assessment(self) -> dict[str, Any]:
        """Return whether the existing Qdrant collection can serve 0.5.1.1 reads."""
        return self.health_check().get("reuse", {})

    def _physical_name(self, collection_name: str) -> str:
        return f"{self._project_name}__{collection_name}"

    def _metadata_name(self) -> str:
        return self._metadata_collection_name_override or (
            f"{self._physical_collection_name}{META_SUFFIX}"
        )

    def _new_collection(self) -> QdrantCollection:
        self._physical_collection_name = self._physical_name(self._collection_name)
        self._metadata_collection_name = self._metadata_name()
        return QdrantCollection(
            client=self._client,
            collection_name=self._physical_collection_name,
            metadata_collection_name=self._metadata_collection_name,
            dense_vector_name=self._dense_vector_name,
            sparse_vector_name=self._sparse_vector_name,
            vector_dim=self._dimension,
            distance=self._distance_metric,
            sparse_enabled=self._sparse_enabled,
            sparse_weight=self._sparse_weight,
            query_candidate_cap=self._settings.query_candidate_cap,
        )

    def _load_existing_collection_if_needed(self) -> None:
        if self._collection is not None:
            return
        with self._lock:
            if self._collection is not None:
                return
            candidate = self._new_collection()
            if not candidate.collection_exists():
                candidate.close()
                return
            if not candidate.has_openviking_metadata():
                candidate.close()
                raise RuntimeError(
                    "Qdrant collection exists but OpenViking metadata is missing: "
                    f"{self._physical_collection_name}. "
                    "Use a different project/name, restore metadata, or drop the stale "
                    "collection explicitly."
                )
            try:
                candidate.get_meta_data(raise_on_error=True)
            except Exception:
                candidate.close()
                raise
            self._qdrant = candidate
            self._collection = Collection(candidate)

    def _create_backend_collection(self, meta: dict[str, Any]) -> Collection:
        candidate = self._new_collection()
        candidate.create_remote_collection(meta)
        self._qdrant = candidate
        return Collection(candidate)

    # ------------------------------------------------------------------
    # index metadata hooks
    # ------------------------------------------------------------------

    def _sanitize_scalar_index_fields(
        self,
        scalar_index_fields: list[str],
        fields_meta: list[dict[str, Any]],
    ) -> list[str]:
        del fields_meta
        return list(dict.fromkeys([*scalar_index_fields, "uri_depth", "scope_roots"]))

    def _build_default_index_meta(
        self,
        *,
        index_name: str,
        distance: str,
        use_sparse: bool,
        sparse_weight: float,
        scalar_index_fields: list[str],
    ) -> dict[str, Any]:
        return {
            "IndexName": index_name,
            "VectorIndex": {
                "IndexType": "hnsw_hybrid" if use_sparse else "hnsw",
                "Distance": distance,
            },
            "ScalarIndex": scalar_index_fields,
            "SparseWeight": sparse_weight,
        }

    def _compile_filter(self, expr: FilterExpr | dict[str, Any] | None) -> dict[str, Any]:
        return compile_qdrant_filter(expr)

    # ------------------------------------------------------------------
    # query
    # ------------------------------------------------------------------

    def query(
        self,
        *,
        query_vector: list[float] | None = None,
        sparse_query_vector: dict[str, float] | None = None,
        filter: dict[str, Any] | FilterExpr | None = None,
        limit: int = 10,
        offset: int = 0,
        output_fields: list[str] | None = None,
        order_by: str | None = None,
        order_desc: bool = False,
        advance: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        decay_request = _time_decay_request(advance)
        has_contains = filter_has_contains(filter)
        if has_contains and (query_vector or sparse_query_vector):
            raise NotImplementedError(
                "Contains is not supported for Qdrant vector search; only "
                "filter-only enumeration has a client-side fallback"
            )
        if has_contains and order_by:
            raise NotImplementedError(
                "Contains is not supported for Qdrant scalar-ordered search"
            )
        if decay_request is not None and not (query_vector or sparse_query_vector):
            raise ValueError("time-decay ranking requires a query vector")

        coll = self.get_collection()
        compiled = {} if has_contains else self._compile_filter(filter)

        if query_vector or sparse_query_vector:
            compiled_advance = self._compile_advance(decay_request)
            result = coll.search_by_vector(
                index_name=self._index_name,
                dense_vector=query_vector,
                sparse_vector=sparse_query_vector,
                limit=limit,
                offset=offset,
                filters=compiled,
                output_fields=output_fields,
                advance=compiled_advance,
                return_detail_info=decay_request is not None,
            )
        elif order_by:
            result = coll.search_by_scalar(
                index_name=self._index_name,
                field=order_by,
                order="desc" if order_desc else "asc",
                limit=limit,
                offset=offset,
                filters=compiled,
                output_fields=output_fields,
            )
        elif has_contains:
            assert self._qdrant is not None
            result = self._qdrant.search_by_random(
                index_name=self._index_name,
                limit=limit,
                offset=offset,
                filters={},
                output_fields=output_fields,
                post_filter=filter,
            )
        else:
            result = coll.search_by_random(
                index_name=self._index_name,
                limit=limit,
                offset=offset,
                filters=compiled,
                output_fields=output_fields,
            )
        return self._records_from_result(result)

    def _compile_advance(
        self,
        decay_request: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if decay_request is None:
            return None
        if "origin_ms" in decay_request:
            return {"time_decay": dict(decay_request)}
        protection = decay_request.get("protection")
        if not isinstance(protection, str):
            raise ValueError("time_decay protection must be a duration string")
        origin_raw = decay_request.get("origin")
        if isinstance(origin_raw, str):
            origin = parse_iso_datetime(origin_raw)
        elif isinstance(origin_raw, datetime):
            origin = origin_raw
        elif origin_raw is None:
            origin = datetime.now(timezone.utc)
        else:
            raise ValueError("time_decay origin must be an ISO 8601 string")
        spec = build_time_decay_fusion_spec(protection=protection, origin=origin)
        return {
            "time_decay": {
                "field": spec.field,
                "origin_ms": float(spec.origin_ms),
                "offset_ms": int(spec.offset_ms),
                "scale_ms": int(spec.scale_ms),
                "decay": float(spec.decay),
            }
        }

    def _records_from_result(self, result: Any) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for item in result.data:
            record = dict(item.fields) if item.fields else {}
            record["id"] = item.id
            record["_score"] = finite_score(item.score)
            if item.origin_score is not None:
                record["_origin_score"] = finite_score(item.origin_score)
            if item.addition_score is not None:
                record["_time_score"] = finite_score(item.addition_score)
            records.append(self._normalize_record_for_read(record))
        return records

    # ------------------------------------------------------------------
    # enumeration / mutation overrides
    # ------------------------------------------------------------------

    def search_by_random(
        self,
        *,
        filter: dict[str, Any] | FilterExpr | None = None,
        limit: int = 10,
        offset: int = 0,
        output_fields: list[str] | None = None,
        advance: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if not filter_has_contains(filter):
            return super().search_by_random(
                filter=filter,
                limit=limit,
                offset=offset,
                output_fields=output_fields,
                advance=advance,
            )
        self.get_collection()
        assert self._qdrant is not None
        result = self._qdrant.search_by_random(
            index_name=self._index_name,
            limit=limit,
            offset=offset,
            filters={},
            output_fields=output_fields,
            advance=advance,
            post_filter=filter,
        )
        return self._records_from_result(result)

    def count(self, filter: dict[str, Any] | FilterExpr | None = None) -> int:
        if filter_has_contains(filter):
            self.get_collection()
            assert self._qdrant is not None
            return self._qdrant.count_with_post_filter(filter)
        return super().count(filter)

    def strict_count(self, filter: dict[str, Any] | FilterExpr | None = None) -> int:
        if filter_has_contains(filter):
            self.get_collection()
            assert self._qdrant is not None
            return self._qdrant.count_with_post_filter(filter)
        return super().strict_count(filter)

    def delete(
        self,
        *,
        ids: list[str] | None = None,
        filter: dict[str, Any] | FilterExpr | None = None,
    ) -> int:
        if ids is not None:
            return super().delete(ids=ids)
        if filter is None:
            return 0
        coll = self.get_collection()
        assert self._qdrant is not None
        if filter_has_contains(filter):
            matched = self._qdrant.find_matching_ids(filter)
            batch_size = self._DATA_BATCH_SIZE or 100
            for start in range(0, len(matched), batch_size):
                coll.delete_data(matched[start : start + batch_size])
            return len(matched)

        compiled = self._compile_filter(filter)
        total = self._qdrant.aggregate_data(
            index_name=self._index_name,
            op="count",
            filters=compiled,
        ).agg.get("_total", 0)
        if total:
            self._qdrant.delete_by_filter(compiled)
        return int(total)

    def update_data(self, data_list: list[dict[str, Any]]):
        normalized = [self._normalize_record_for_write(record) for record in data_list]
        result = self.get_collection().update_data(normalized)
        return [str(item) for item in result or []]

    def search_by_keywords(
        self,
        keywords: list[str] | None = None,
        query: str | None = None,
        limit: int = 10,
        offset: int = 0,
        filter: dict[str, Any] | FilterExpr | None = None,
        output_fields: list[str] | None = None,
        mode: str | None = None,
        fields: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        del (
            keywords,
            query,
            limit,
            offset,
            filter,
            output_fields,
            mode,
            fields,
        )
        raise NotImplementedError(
            "Qdrant does not store the OpenViking content field; keyword/full-text "
            "search is unavailable and grep uses the filesystem fallback"
        )


def _time_decay_request(advance: dict[str, Any] | None) -> dict[str, Any] | None:
    if advance is None:
        return None
    if not isinstance(advance, dict):
        raise TypeError("advance must be a mapping")
    unknown = sorted(set(advance) - {"time_decay"})
    if unknown:
        raise NotImplementedError(
            f"Qdrant advanced ranking does not support: {unknown}"
        )
    request = advance.get("time_decay")
    if request is None:
        return None
    if not isinstance(request, dict):
        raise ValueError("time_decay advance must be a mapping")
    return request


__all__ = ["QdrantCollectionAdapter"]
