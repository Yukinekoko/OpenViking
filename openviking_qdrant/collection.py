# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Qdrant implementation of OpenViking's ``ICollection`` contract."""

from __future__ import annotations

import copy
import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import RLock
from typing import Any, Iterator
from urllib.parse import quote

from openviking.storage.vectordb.collection.collection import ICollection
from openviking.storage.vectordb.collection.result import (
    AggregateResult,
    DataItem,
    FetchDataInCollectionResult,
    SearchItemResult,
    SearchResult,
)
from openviking.utils.time_decay import build_time_decay_fusion_spec
from openviking.utils.time_utils import parse_iso_datetime

from .filter import filter_matches_record
from .meta import (
    META_MARKER_ID,
    META_SUFFIX,
    META_VECTOR_NAME,
    build_meta_view,
    index_scalar_fields,
    is_valid_metadata,
    metadata_payload,
)
from .rest import QdrantError, QdrantRestClient, collection_path
from .sparse import SparseTermDictionary
from .utils import (
    build_qdrant_payload,
    finite_score,
    payload_selector,
    payload_to_record,
    project_record,
    to_qdrant_point_id,
)

logger = logging.getLogger(__name__)

_SCROLL_PAGE_SIZE = 256
_UPSERT_CHUNK_SIZE = 256
_INTERNAL_SCALAR_FIELDS = ("uri_depth", "scope_roots")
_DISTANCE_MAP = {
    "cosine": "Cosine",
    "ip": "Dot",
    "dot": "Dot",
    "l2": "Euclid",
    "euclid": "Euclid",
    "l1": "Manhattan",
    "manhattan": "Manhattan",
}


@dataclass
class _Hit:
    point_id: str
    record: dict[str, Any]
    score: float
    origin_score: float | None = None
    addition_score: float | None = None


class QdrantCollection(ICollection):
    """REST-backed Qdrant collection with OpenViking payload semantics."""

    def __init__(
        self,
        *,
        client: QdrantRestClient,
        collection_name: str,
        metadata_collection_name: str,
        dense_vector_name: str,
        sparse_vector_name: str,
        vector_dim: int,
        distance: str,
        sparse_enabled: bool,
        sparse_weight: float,
        query_candidate_cap: int = 100_000,
    ) -> None:
        self._client = client
        self._collection_name = collection_name
        self._metadata_collection_name = metadata_collection_name
        self._dense_vector_name = dense_vector_name
        self._sparse_vector_name = sparse_vector_name
        self._vector_dim = int(vector_dim)
        self._distance = self._normalize_distance(distance)
        self._sparse_enabled = bool(sparse_enabled)
        self._sparse_weight = float(sparse_weight)
        self._query_candidate_cap = int(query_candidate_cap)
        self._schema: dict[str, Any] = {}
        self._indexes: dict[str, dict[str, Any]] = {}
        self._loaded = False
        self._sparse_dictionary: SparseTermDictionary | None = None
        self._sparse_lock = RLock()
        self._load_lock = RLock()

    # ------------------------------------------------------------------
    # lifecycle / metadata
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_distance(distance: str) -> str:
        value = str(distance or "cosine").strip().lower()
        if value not in _DISTANCE_MAP:
            raise ValueError(f"Unsupported Qdrant distance metric: {distance!r}")
        return _DISTANCE_MAP[value]

    @staticmethod
    def _result(response: dict[str, Any]) -> Any:
        return response.get("result", response)

    def _exists(self, name: str) -> bool:
        try:
            self._client.request("GET", collection_path(name))
        except QdrantError as exc:
            if exc.status == 404:
                return False
            raise
        return True

    def collection_exists(self) -> bool:
        return self._exists(self._collection_name)

    def create_remote_collection(self, metadata: dict[str, Any]) -> None:
        """Create the data + metadata collections and write the marker."""
        self._schema = copy.deepcopy(metadata)
        self._schema.setdefault("CollectionName", self._collection_name)
        if self._vector_dim <= 0:
            for field in self._schema.get("Fields", []):
                if (
                    str(field.get("FieldType", "")).lower() == "vector"
                    and int(field.get("Dim") or 0) > 0
                ):
                    self._vector_dim = int(field["Dim"])
                    break
        if self._vector_dim <= 0:
            raise ValueError("Qdrant backend requires a positive dense vector dimension")
        self._create_data_collection()
        self._create_metadata_collection()
        self._write_metadata_marker()
        self._loaded = True

    def _create_data_collection(self) -> None:
        if self._exists(self._collection_name):
            return
        body: dict[str, Any] = {
            "vectors": {
                self._dense_vector_name: {
                    "size": self._vector_dim,
                    "distance": self._distance,
                }
            }
        }
        if self._sparse_enabled:
            body["sparse_vectors"] = {self._sparse_vector_name: {}}
        try:
            self._client.request(
                "PUT",
                collection_path(self._collection_name),
                body,
                params={"wait": "true"},
            )
        except QdrantError as exc:
            if exc.status != 409:
                raise

    def _create_metadata_collection(self) -> None:
        if self._exists(self._metadata_collection_name):
            return
        body = {"vectors": {META_VECTOR_NAME: {"size": 1, "distance": "Dot"}}}
        try:
            self._client.request(
                "PUT",
                collection_path(self._metadata_collection_name),
                body,
                params={"wait": "true"},
            )
        except QdrantError as exc:
            if exc.status != 409:
                raise

    def has_openviking_metadata(self) -> bool:
        payload = self._load_metadata_marker()
        return is_valid_metadata(payload, self._collection_name)

    def _metadata_payload(self) -> dict[str, Any]:
        return metadata_payload(
            collection_name=self._collection_name,
            schema=self._schema,
            dense_vector_name=self._dense_vector_name,
            sparse_vector_name=self._sparse_vector_name,
            vector_dim=self._vector_dim,
            distance=self._distance,
            sparse_enabled=self._sparse_enabled,
            sparse_weight=self._sparse_weight,
            indexes=self._indexes,
        )

    def _write_metadata_marker(self) -> None:
        self._upsert_points(
            self._metadata_collection_name,
            [
                {
                    "id": META_MARKER_ID,
                    "vector": {META_VECTOR_NAME: [0.0]},
                    "payload": self._metadata_payload(),
                }
            ],
        )

    def _load_metadata_marker(self) -> dict[str, Any] | None:
        if not self._exists(self._metadata_collection_name):
            return None
        points = self._retrieve_points(
            self._metadata_collection_name,
            [META_MARKER_ID],
            with_vectors=False,
        )
        if not points:
            return None
        payload = points[0].get("payload")
        return payload if isinstance(payload, dict) else None

    def _ensure_loaded(self, *, raise_on_error: bool = False) -> bool:
        if self._loaded:
            return True
        with self._load_lock:
            if self._loaded:
                return True
            try:
                marker = self._load_metadata_marker()
            except QdrantError as exc:
                # ``raise_on_error=False`` is documented as "best effort": a
                # transient/unreachable Qdrant must not make callers like the
                # upstream ``_filter_known_fields`` probe crash.  Genuine
                # programming errors (TypeError etc.) are still propagated.
                message = (
                    f"Could not read OpenViking metadata for Qdrant collection "
                    f"{self._collection_name!r}: {exc}"
                )
                if raise_on_error:
                    raise
                logger.warning(message)
                return False
            if not is_valid_metadata(marker, self._collection_name):
                message = (
                    f"Qdrant collection {self._collection_name!r} is missing valid "
                    "OpenViking metadata"
                )
                if raise_on_error:
                    raise RuntimeError(message)
                logger.warning(message)
                return False
            assert marker is not None
            self._schema = copy.deepcopy(marker["schema"])
            self._vector_dim = int(marker.get("vector_dim") or self._vector_dim)
            self._dense_vector_name = str(
                marker.get("dense_vector_name") or self._dense_vector_name
            )
            self._sparse_vector_name = str(
                marker.get("sparse_vector_name") or self._sparse_vector_name
            )
            self._distance = str(marker.get("distance") or self._distance)
            self._sparse_enabled = bool(
                marker.get("sparse_enabled", self._sparse_enabled)
            )
            if "sparse_weight" in marker:
                self._sparse_weight = float(marker["sparse_weight"])
            indexes = marker.get("indexes")
            self._indexes = (
                {str(name): copy.deepcopy(meta) for name, meta in indexes.items()}
                if isinstance(indexes, dict)
                else {}
            )
            if self._vector_dim <= 0:
                self._vector_dim = _vector_dim_from_schema(self._schema)
            self._loaded = True
            return True

    def get_meta_data(self, *, raise_on_error: bool = False) -> dict[str, Any]:
        if not self._ensure_loaded(raise_on_error=raise_on_error):
            return {}
        return build_meta_view(
            self._schema,
            collection_name=self._collection_name,
            indexes=self._indexes,
        )

    def update(
        self,
        fields: dict[str, Any] | list[dict[str, Any]] | None = None,
        description: str | None = None,
    ):
        if not self._ensure_loaded(raise_on_error=True):
            raise RuntimeError(f"Qdrant collection {self._collection_name!r} is not loaded")
        previous_schema = copy.deepcopy(self._schema)
        if fields:
            if isinstance(fields, list):
                self._merge_fields(fields)
            elif isinstance(fields, dict):
                self._schema.update(copy.deepcopy(fields))
            else:
                raise TypeError("Qdrant collection update fields must be a list or dict")
        if description is not None:
            self._schema["Description"] = description
        try:
            self._write_metadata_marker()
        except Exception:
            self._schema = previous_schema
            raise
        return self.get_meta_data()

    def _merge_fields(self, fields: list[dict[str, Any]]) -> None:
        existing = self._schema.setdefault("Fields", [])
        by_name = {
            field.get("FieldName"): field
            for field in existing
            if isinstance(field, dict) and field.get("FieldName")
        }
        for field in fields:
            if not isinstance(field, dict) or not field.get("FieldName"):
                raise ValueError("Qdrant schema fields require a FieldName")
        for field in fields:
            current = by_name.get(field["FieldName"])
            if current is None:
                copied = copy.deepcopy(field)
                existing.append(copied)
                by_name[copied["FieldName"]] = copied
            else:
                current.update(copy.deepcopy(field))

    def close(self) -> None:
        return None

    def drop(self):
        for name in (self._collection_name, self._metadata_collection_name):
            if self._exists(name):
                self._client.request(
                    "DELETE",
                    collection_path(name),
                    params={"timeout": 30},
                )
        self._schema.clear()
        self._indexes.clear()
        self._loaded = False
        self._sparse_dictionary = None
        return True

    # ------------------------------------------------------------------
    # index lifecycle
    # ------------------------------------------------------------------

    def _field_schema(self, field: str) -> str:
        for item in self._schema.get("Fields", []):
            if item.get("FieldName") != field:
                continue
            field_type = str(item.get("FieldType") or "").lower()
            if field_type.startswith("list<") and field_type.endswith(">"):
                field_type = field_type[5:-1]
            if field_type in {
                "int",
                "int8",
                "int16",
                "int32",
                "int64",
                "uint",
                "uint8",
                "uint16",
                "uint32",
                "uint64",
            }:
                return "integer"
            if field_type in {"float", "float16", "float32", "float64", "double"}:
                return "float"
            if field_type in {"bool", "boolean"}:
                return "bool"
            if field_type in {"date_time", "datetime"}:
                return "datetime"
            if field_type == "text":
                return "text"
            return "keyword"
        return "keyword"

    def _ensure_remote_indexes(self, meta_data: dict[str, Any]) -> None:
        scalar_fields = index_scalar_fields(meta_data)
        scalar_fields.extend(_INTERNAL_SCALAR_FIELDS)
        for field in dict.fromkeys(scalar_fields):
            body = {
                "field_name": field,
                "field_schema": (
                    "integer"
                    if field == "uri_depth"
                    else "keyword"
                    if field == "scope_roots"
                    else self._field_schema(field)
                ),
            }
            try:
                self._client.request(
                    "PUT",
                    collection_path(self._collection_name, "/index"),
                    body,
                    params={"wait": "true"},
                )
            except QdrantError as exc:
                if exc.status != 409:
                    raise

    def _delete_remote_indexes(self, fields: list[str]) -> None:
        for field in dict.fromkeys(fields):
            try:
                self._client.request(
                    "DELETE",
                    collection_path(
                        self._collection_name,
                        f"/index/{quote(str(field), safe='')}",
                    ),
                    params={"wait": "true"},
                )
            except QdrantError as exc:
                if exc.status != 404:
                    raise

    def create_index(self, index_name: str, meta_data: dict[str, Any]):
        if not self._ensure_loaded(raise_on_error=True):
            raise RuntimeError("Qdrant collection is not loaded")
        self._ensure_remote_indexes(meta_data)
        previous = self._indexes
        self._indexes = dict(previous)
        self._indexes[index_name] = copy.deepcopy(meta_data)
        try:
            self._write_metadata_marker()
        except Exception:
            self._indexes = previous
            raise
        return copy.deepcopy(meta_data)

    def has_index(self, index_name: str) -> bool:
        return index_name in self._indexes

    def get_index(self, index_name: str):
        return copy.deepcopy(self._indexes.get(index_name))

    def get_index_meta_data(self, index_name: str) -> dict[str, Any]:
        return copy.deepcopy(self._indexes.get(index_name, {}))

    def list_indexes(self) -> list[str]:
        return list(self._indexes)

    def update_index(
        self,
        index_name: str,
        scalar_index: list[str] | dict[str, Any] | None = None,
        description: str | None = None,
    ):
        if index_name not in self._indexes:
            return None
        previous_meta = copy.deepcopy(self._indexes[index_name])
        previous_schema_index = copy.deepcopy(self._schema.get("ScalarIndex"))
        meta = copy.deepcopy(previous_meta)
        user_fields: list[str] | None = None
        if scalar_index is not None:
            if isinstance(scalar_index, dict):
                user_fields = [str(field) for field in scalar_index]
            elif isinstance(scalar_index, (list, tuple, set)):
                user_fields = [str(field) for field in scalar_index]
            else:
                raise TypeError("update_index scalar_index must be a list or dict")
            meta["ScalarIndex"] = list(user_fields)
        if description is not None:
            meta["Description"] = description

        old_fields = set(index_scalar_fields(previous_meta))
        other_fields = {
            field
            for name, item in self._indexes.items()
            if name != index_name
            for field in index_scalar_fields(item)
        }
        self._ensure_remote_indexes(meta)
        if user_fields is not None:
            removable = [
                field
                for field in old_fields - set(user_fields)
                if field not in other_fields
                and field not in _INTERNAL_SCALAR_FIELDS
            ]
            self._delete_remote_indexes(removable)
            self._schema["ScalarIndex"] = list(user_fields)

        previous_indexes = self._indexes
        self._indexes = dict(previous_indexes)
        self._indexes[index_name] = meta
        try:
            self._write_metadata_marker()
        except Exception:
            self._indexes = previous_indexes
            self._schema["ScalarIndex"] = previous_schema_index
            raise
        return copy.deepcopy(meta)

    def drop_index(self, index_name: str):
        removed = self._indexes.get(index_name)
        if removed is None:
            return True
        other_fields = {
            field
            for name, meta in self._indexes.items()
            if name != index_name
            for field in index_scalar_fields(meta)
        }
        removable = [
            field
            for field in index_scalar_fields(removed)
            if field not in other_fields and field not in _INTERNAL_SCALAR_FIELDS
        ]
        self._delete_remote_indexes(removable)
        previous_indexes = self._indexes
        self._indexes = dict(previous_indexes)
        self._indexes.pop(index_name, None)
        try:
            self._write_metadata_marker()
        except Exception:
            self._indexes = previous_indexes
            raise
        return True

    # ------------------------------------------------------------------
    # low-level data helpers
    # ------------------------------------------------------------------

    def _upsert_points(self, collection_name: str, points: list[dict[str, Any]]) -> None:
        for start in range(0, len(points), _UPSERT_CHUNK_SIZE):
            chunk = points[start : start + _UPSERT_CHUNK_SIZE]
            self._client.request(
                "PUT",
                collection_path(collection_name, "/points"),
                {"points": chunk},
                params={"wait": "true"},
            )

    def _retrieve_points(
        self,
        collection_name: str,
        ids: list[str],
        *,
        with_vectors: bool,
    ) -> list[dict[str, Any]]:
        if not ids:
            return []
        response = self._client.request(
            "POST",
            collection_path(collection_name, "/points"),
            {
                "ids": ids,
                "with_payload": True,
                "with_vector": with_vectors,
            },
        )
        result = self._result(response)
        if not isinstance(result, list):
            raise QdrantError("Qdrant retrieve response did not contain a point list")
        return [point for point in result if isinstance(point, dict)]

    def _iter_points(
        self,
        *,
        filters: dict[str, Any] | None,
        with_vectors: bool = False,
        output_fields: list[str] | None = None,
        order_by: dict[str, Any] | None = None,
        with_payload: bool = True,
        collection_name: str | None = None,
        page_size: int = _SCROLL_PAGE_SIZE,
    ) -> Iterator[dict[str, Any]]:
        cursor: Any = None
        page_size = max(1, min(int(page_size), _SCROLL_PAGE_SIZE))
        while True:
            body: dict[str, Any] = {
                "limit": page_size,
                "with_payload": (
                    payload_selector(output_fields) if with_payload else False
                ),
                "with_vector": with_vectors,
            }
            if filters:
                body["filter"] = filters
            if order_by:
                body["order_by"] = order_by
            if cursor is not None:
                body["offset"] = cursor
            response = self._client.request(
                "POST",
                collection_path(
                    collection_name or self._collection_name,
                    "/points/scroll",
                ),
                body,
            )
            result = self._result(response)
            if not isinstance(result, dict):
                raise QdrantError("Qdrant scroll response did not contain a result object")
            page = result.get("points")
            if not isinstance(page, list):
                raise QdrantError("Qdrant scroll response did not contain a point list")
            for point in page:
                if isinstance(point, dict):
                    yield point
            if not page:
                return
            next_offset = result.get("next_page_offset")
            if next_offset is None or next_offset == cursor:
                return
            cursor = next_offset

    def _scroll(
        self,
        *,
        filters: dict[str, Any] | None,
        limit: int,
        with_vectors: bool = False,
        output_fields: list[str] | None = None,
        order_by: dict[str, Any] | None = None,
        collection_name: str | None = None,
    ) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        points: list[dict[str, Any]] = []
        for point in self._iter_points(
            filters=filters,
            with_vectors=with_vectors,
            output_fields=output_fields,
            order_by=order_by,
            collection_name=collection_name,
            page_size=min(limit, _SCROLL_PAGE_SIZE),
        ):
            points.append(point)
            if len(points) >= limit:
                break
        return points

    def _point_from_record(self, record: dict[str, Any]) -> dict[str, Any]:
        original_id = record.get("id")
        if original_id is None:
            raise ValueError("Qdrant upsert requires an OpenViking record id")
        dense = record.get("vector")
        sparse = record.get("sparse_vector")
        vectors: dict[str, Any] = {}
        if dense is not None:
            self._validate_dense_vector(dense)
            vectors[self._dense_vector_name] = [float(value) for value in dense]
        if sparse:
            if not self._sparse_enabled:
                raise ValueError(
                    "Qdrant collection was created without sparse-vector support"
                )
            encoded = self.encode_sparse_vector(sparse)
            if encoded:
                vectors[self._sparse_vector_name] = encoded
        if not vectors:
            raise ValueError("Qdrant record requires a dense or sparse vector")
        return {
            "id": to_qdrant_point_id(original_id),
            "vector": vectors,
            "payload": build_qdrant_payload(record),
        }

    def _validate_dense_vector(self, vector: Any) -> None:
        if not isinstance(vector, (list, tuple)):
            raise ValueError("Qdrant dense vector must be a list of numbers")
        if self._vector_dim > 0 and len(vector) != self._vector_dim:
            raise ValueError(
                "Qdrant dense vector dimension must be "
                f"{self._vector_dim}, got {len(vector)}"
            )
        for value in vector:
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError("Qdrant dense vector values must be numbers") from exc
            if not math.isfinite(number):
                raise ValueError("Qdrant dense vector values must be finite")

    def _vectors_to_record(self, point: dict[str, Any]) -> dict[str, Any]:
        record = payload_to_record(point)
        vectors = point.get("vector") or point.get("vectors") or {}
        if isinstance(vectors, dict):
            dense = vectors.get(self._dense_vector_name)
            sparse = vectors.get(self._sparse_vector_name)
            if dense is not None:
                record["vector"] = dense
            if sparse is not None:
                record["sparse_vector"] = self._decode_sparse_vector(sparse)
        elif isinstance(vectors, list) and self._dense_vector_name == "vector":
            record["vector"] = vectors
        return record

    def _decode_sparse_vector(self, value: Any) -> dict[str, float]:
        if not isinstance(value, dict):
            raise ValueError("Qdrant sparse vector must be a mapping")
        indices = value.get("indices")
        values = value.get("values")
        if (
            not isinstance(indices, list)
            or not isinstance(values, list)
            or len(indices) != len(values)
        ):
            raise ValueError(
                "Qdrant sparse vector indices and values must be lists of equal length"
            )
        dictionary = self._get_sparse_dictionary()
        result: dict[str, float] = {}
        for index, weight in zip(indices, values, strict=True):
            numeric_index = int(index)
            term = dictionary.term_for(numeric_index)
            if term is None:
                raise ValueError(f"unknown sparse term index: {numeric_index}")
            result[term] = float(weight)
        return result

    # ------------------------------------------------------------------
    # data interface
    # ------------------------------------------------------------------

    def upsert_data(self, data_list: list[dict[str, Any]], ttl: int = 0):
        del ttl  # Qdrant has no per-point TTL.
        if not data_list:
            return {"status": "ok"}
        points = [self._point_from_record(item) for item in data_list]
        self._upsert_points(self._collection_name, points)
        return {"status": "ok"}

    def update_data(self, data_list: list[dict[str, Any]]):
        if not data_list:
            return []
        pending: list[dict[str, Any]] = []
        missing: list[Any] = []
        point_ids: list[str] = []
        for item in data_list:
            if "id" not in item:
                raise ValueError("primary key 'id' is required for update")
            point_ids.append(to_qdrant_point_id(item["id"]))
        points = self._retrieve_points(
            self._collection_name,
            point_ids,
            with_vectors=True,
        )
        by_point_id = {str(point.get("id")): point for point in points}
        for item in data_list:
            record_id = item["id"]
            point = by_point_id.get(to_qdrant_point_id(record_id))
            if point is None:
                missing.append(record_id)
                continue
            merged = self._vectors_to_record(point)
            merged.update(item)
            merged["id"] = record_id
            pending.append(merged)
        if missing:
            raise ValueError(f"record not found for primary key(s): {missing}")
        for record in pending:
            self.upsert_data([record])
        return [str(item["id"]) for item in pending]

    def fetch_data(self, primary_keys: list[Any]) -> FetchDataInCollectionResult:
        if not primary_keys:
            return FetchDataInCollectionResult()
        point_ids = [to_qdrant_point_id(value) for value in primary_keys]
        points = self._retrieve_points(
            self._collection_name,
            point_ids,
            with_vectors=False,
        )
        by_point_id = {str(point.get("id")): point for point in points}
        items: list[DataItem] = []
        missing: list[Any] = []
        for key, point_id in zip(primary_keys, point_ids, strict=True):
            point = by_point_id.get(point_id)
            if point is None:
                missing.append(key)
                continue
            record = payload_to_record(point)
            items.append(DataItem(id=record.get("id"), fields=record))
        return FetchDataInCollectionResult(items=items, ids_not_exist=missing)

    def delete_data(self, primary_keys: list[Any]):
        if not primary_keys:
            return {"status": "ok"}
        self._client.request(
            "POST",
            collection_path(self._collection_name, "/points/delete"),
            {"points": [to_qdrant_point_id(value) for value in primary_keys]},
            params={"wait": "true"},
        )
        return {"status": "ok"}

    def delete_by_filter(self, filters: dict[str, Any]) -> None:
        """Delete all points matching a compiled filter in one server-side call."""
        if filters is None:
            raise ValueError("delete_by_filter requires a filter")
        self._client.request(
            "POST",
            collection_path(self._collection_name, "/points/delete"),
            {"filter": filters},
            params={"wait": "true"},
        )

    def delete_all_data(self):
        self._client.request(
            "POST",
            collection_path(self._collection_name, "/points/delete"),
            {"filter": {}},
            params={"wait": "true"},
        )
        return True

    def aggregate_data(
        self,
        index_name: str,
        op: str = "count",
        field: str | None = None,
        filters: dict[str, Any] | None = None,
        cond: dict[str, Any] | None = None,
    ) -> AggregateResult:
        del index_name
        if op != "count":
            raise NotImplementedError(f"Qdrant aggregate operation is unsupported: {op}")
        if field is not None or cond is not None:
            raise NotImplementedError(
                "Qdrant aggregate supports total count only; grouped/conditional "
                "aggregation is unavailable"
            )
        response = self._client.request(
            "POST",
            collection_path(self._collection_name, "/points/count"),
            {"filter": filters or {}, "exact": True},
        )
        result = self._result(response)
        count = result.get("count", 0) if isinstance(result, dict) else 0
        return AggregateResult(agg={"_total": int(count)}, op="count")

    # ------------------------------------------------------------------
    # search
    # ------------------------------------------------------------------

    def _search_one(
        self,
        *,
        vector: Any,
        using: str,
        filters: dict[str, Any],
        limit: int,
        offset: int,
        output_fields: list[str] | None,
        extra_fields: tuple[str, ...] = (),
    ) -> list[_Hit]:
        if limit <= 0:
            return []
        body: dict[str, Any] = {
            "query": vector,
            "using": using,
            "limit": limit,
            "offset": offset,
            "with_payload": payload_selector(output_fields, extra_fields=extra_fields),
            "with_vector": False,
        }
        if filters:
            body["filter"] = filters
        response = self._client.request(
            "POST",
            collection_path(self._collection_name, "/points/query"),
            body,
        )
        result = self._result(response)
        if isinstance(result, dict):
            result = result.get("points")
        if not isinstance(result, list):
            raise QdrantError("Qdrant search response did not contain a point list")
        hits: list[_Hit] = []
        for point in result:
            if not isinstance(point, dict):
                continue
            record = payload_to_record(point)
            hits.append(
                _Hit(
                    point_id=str(point.get("id")),
                    record=record,
                    score=finite_score(point.get("score")),
                )
            )
        return hits

    def search_by_vector(
        self,
        index_name: str,
        dense_vector: list[float] | None = None,
        limit: int = 10,
        offset: int = 0,
        filters: dict[str, Any] | None = None,
        sparse_vector: dict[str, float] | None = None,
        output_fields: list[str] | None = None,
        advance: dict[str, Any] | None = None,
        return_detail_info: bool = False,
    ) -> SearchResult:
        del index_name, return_detail_info
        self._validate_window(limit, offset)
        if limit == 0:
            return SearchResult()
        spec = self._parse_advance(advance)
        if dense_vector is None and not sparse_vector:
            raise ValueError("Qdrant search_by_vector requires a dense or sparse vector")
        if sparse_vector and not self._sparse_enabled:
            raise ValueError(
                "Qdrant collection was created without sparse-vector support"
            )
        if sparse_vector is not None and not isinstance(sparse_vector, dict):
            raise ValueError("Qdrant sparse query vector must be a mapping")
        if sparse_vector is not None and not sparse_vector:
            raise ValueError("Qdrant sparse query vector must not be empty")
        if dense_vector is not None:
            self._validate_dense_vector(dense_vector)
        compiled = self._normalize_filter(filters)
        extra_fields = (spec["field"],) if spec else ()
        dense_values = (
            [float(value) for value in dense_vector] if dense_vector is not None else None
        )
        sparse_payload = (
            self.encode_sparse_vector(sparse_vector) if sparse_vector else None
        )

        if spec is not None:
            # Match the native engine: amplify the final window by 3x before
            # applying decay, then return only topk.
            candidate_limit = self._candidate_limit((limit + offset) * 3)
            if dense_values is not None and sparse_payload is not None:
                dense_hits = self._search_one(
                    vector=dense_values,
                    using=self._dense_vector_name,
                    filters=compiled,
                    limit=candidate_limit,
                    offset=0,
                    output_fields=output_fields,
                    extra_fields=extra_fields,
                )
                sparse_hits = self._search_one(
                    vector=sparse_payload,
                    using=self._sparse_vector_name,
                    filters=compiled,
                    limit=candidate_limit,
                    offset=0,
                    output_fields=output_fields,
                    extra_fields=extra_fields,
                )
                hits = self._weighted_rank_fusion(dense_hits, sparse_hits)
            elif dense_values is not None:
                hits = self._search_one(
                    vector=dense_values,
                    using=self._dense_vector_name,
                    filters=compiled,
                    limit=candidate_limit,
                    offset=0,
                    output_fields=output_fields,
                    extra_fields=extra_fields,
                )
            else:
                assert sparse_payload is not None
                hits = self._search_one(
                    vector=sparse_payload,
                    using=self._sparse_vector_name,
                    filters=compiled,
                    limit=candidate_limit,
                    offset=0,
                    output_fields=output_fields,
                    extra_fields=extra_fields,
                )
            hits = self._apply_time_decay(hits, spec)
            return SearchResult(
                data=[
                    _hit_to_item(hit, output_fields)
                    for hit in hits[offset : offset + limit]
                ]
            )

        if dense_values is not None and sparse_payload is not None:
            candidate_limit = self._candidate_limit(max(limit + offset, limit * 2))
            dense_hits = self._search_one(
                vector=dense_values,
                using=self._dense_vector_name,
                filters=compiled,
                limit=candidate_limit,
                offset=0,
                output_fields=output_fields,
            )
            sparse_hits = self._search_one(
                vector=sparse_payload,
                using=self._sparse_vector_name,
                filters=compiled,
                limit=candidate_limit,
                offset=0,
                output_fields=output_fields,
            )
            fused = self._weighted_rank_fusion(dense_hits, sparse_hits)
            hits = fused[offset : offset + limit]
        elif dense_values is not None:
            hits = self._search_one(
                vector=dense_values,
                using=self._dense_vector_name,
                filters=compiled,
                limit=limit,
                offset=offset,
                output_fields=output_fields,
            )
        else:
            assert sparse_payload is not None
            hits = self._search_one(
                vector=sparse_payload,
                using=self._sparse_vector_name,
                filters=compiled,
                limit=limit,
                offset=offset,
                output_fields=output_fields,
            )
        return SearchResult(data=[_hit_to_item(hit, output_fields) for hit in hits])

    def _candidate_limit(self, requested: int) -> int:
        return max(1, min(requested, self._query_candidate_cap))

    def _weighted_rank_fusion(
        self,
        dense: list[_Hit],
        sparse: list[_Hit],
    ) -> list[_Hit]:
        alpha = min(max(self._sparse_weight, 0.0), 1.0)
        scores: dict[str, float] = {}
        hits: dict[str, _Hit] = {}
        for rank, hit in enumerate(dense, start=1):
            scores[hit.point_id] = scores.get(hit.point_id, 0.0) + (1.0 - alpha) / (
                60 + rank
            )
            hits[hit.point_id] = hit
        for rank, hit in enumerate(sparse, start=1):
            scores[hit.point_id] = scores.get(hit.point_id, 0.0) + alpha / (60 + rank)
            hits.setdefault(hit.point_id, hit)
        ordered = sorted(hits, key=lambda point_id: scores[point_id], reverse=True)
        return [
            _Hit(
                point_id=point_id,
                record=hits[point_id].record,
                score=scores[point_id],
            )
            for point_id in ordered
        ]

    def _parse_advance(self, advance: dict[str, Any] | None) -> dict[str, Any] | None:
        if advance is None:
            return None
        if not isinstance(advance, dict):
            raise TypeError("advance must be a mapping")
        unknown = sorted(set(advance) - {"time_decay"})
        if unknown:
            raise NotImplementedError(f"Qdrant advanced ranking does not support: {unknown}")
        request = advance.get("time_decay")
        if request is None:
            return None
        if not isinstance(request, dict):
            raise ValueError("time_decay advance must be a mapping")
        if "field" in request:
            field = request["field"]
            if not isinstance(field, str) or not field:
                raise ValueError("time_decay field must be a non-empty string")
        else:
            field = "updated_at"

        if "origin_ms" not in request:
            if "protection" not in request:
                raise ValueError("time_decay requires 'protection' or compiled parameters")
            origin_raw = request.get("origin")
            if isinstance(origin_raw, str):
                origin = parse_iso_datetime(origin_raw)
            elif origin_raw is None:
                origin = datetime.now(timezone.utc)
            elif isinstance(origin_raw, datetime):
                origin = origin_raw
            else:
                raise ValueError("time_decay origin must be an ISO 8601 string")
            spec = build_time_decay_fusion_spec(
                protection=request["protection"],
                origin=origin,
                field=field,
            )
            return {
                "field": spec.field,
                "origin_ms": float(spec.origin_ms),
                "offset_ms": int(spec.offset_ms),
                "scale_ms": int(spec.scale_ms),
                "decay": float(spec.decay),
            }

        values: dict[str, float] = {}
        for key in ("origin_ms", "offset_ms", "scale_ms", "decay"):
            value = request.get(key)
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"time_decay {key} must be a finite number") from exc
            if not math.isfinite(number):
                raise ValueError(f"time_decay {key} must be a finite number")
            values[key] = number
        if values["offset_ms"] < 0:
            raise ValueError("time_decay offset_ms must be non-negative")
        if values["scale_ms"] <= 0:
            raise ValueError("time_decay scale_ms must be positive")
        if not 0.0 < values["decay"] < 1.0:
            raise ValueError("time_decay decay must be in (0, 1)")
        return {
            "field": field,
            "origin_ms": values["origin_ms"],
            "offset_ms": int(values["offset_ms"]),
            "scale_ms": int(values["scale_ms"]),
            "decay": values["decay"],
        }

    def _apply_time_decay(
        self,
        hits: list[_Hit],
        spec: dict[str, Any],
    ) -> list[_Hit]:
        origin = spec["origin_ms"]
        protection = spec["offset_ms"]
        rate = math.log(spec["decay"]) / spec["scale_ms"]
        field = spec["field"]
        for hit in hits:
            updated = _epoch_ms(hit.record.get(field))
            hit.origin_score = hit.score
            if updated is None:
                continue
            time_score = math.exp(rate * max(0.0, abs(origin - updated) - protection))
            if not math.isfinite(time_score):
                continue
            hit.addition_score = time_score
            hit.score = hit.score * time_score
        return sorted(hits, key=lambda hit: hit.score, reverse=True)

    def search_by_random(
        self,
        index_name: str,
        limit: int = 10,
        offset: int = 0,
        filters: dict[str, Any] | None = None,
        output_fields: list[str] | None = None,
        advance: dict[str, Any] | None = None,
        post_filter: Any = None,
    ) -> SearchResult:
        del index_name
        if advance:
            raise NotImplementedError(
                "Qdrant random enumeration has no ranking to advance; "
                "use query() for time-decay ranking"
            )
        self._validate_window(limit, offset)
        if limit == 0:
            return SearchResult()
        if post_filter is not None:
            return self._scan_matching(
                post_filter,
                limit=limit,
                offset=offset,
                output_fields=output_fields,
            )
        points = self._scroll(
            filters=self._normalize_filter(filters),
            limit=limit + offset,
            output_fields=output_fields,
        )
        items = []
        for point in points[offset : offset + limit]:
            record = payload_to_record(point)
            hit = _Hit(point_id=str(point.get("id")), record=record, score=1.0)
            items.append(_hit_to_item(hit, output_fields))
        return SearchResult(data=items)

    def _scan_matching(
        self,
        expression: Any,
        *,
        limit: int,
        offset: int,
        output_fields: list[str] | None,
    ) -> SearchResult:
        matches: list[_Hit] = []
        skipped = 0
        for point in self._iter_points(filters=None, output_fields=None):
            record = payload_to_record(point)
            if not filter_matches_record(expression, record):
                continue
            if skipped < offset:
                skipped += 1
                continue
            matches.append(
                _Hit(point_id=str(point.get("id")), record=record, score=1.0)
            )
            if len(matches) >= limit:
                break
        return SearchResult(data=[_hit_to_item(hit, output_fields) for hit in matches])

    def count_with_post_filter(self, expression: Any) -> int:
        """Count records matching an expression that needs client-side evaluation."""
        count = 0
        for point in self._iter_points(filters=None, output_fields=None):
            if filter_matches_record(expression, payload_to_record(point)):
                count += 1
        return count

    def find_matching_ids(self, expression: Any) -> list[str]:
        """Return every ID matching an expression that needs client-side evaluation."""
        ids: list[str] = []
        for point in self._iter_points(filters=None, output_fields=None):
            record = payload_to_record(point)
            if filter_matches_record(expression, record):
                ids.append(str(record.get("id")))
        return ids

    def search_by_scalar(
        self,
        index_name: str,
        field: str,
        order: str | None = "desc",
        limit: int = 10,
        offset: int = 0,
        filters: dict[str, Any] | None = None,
        output_fields: list[str] | None = None,
    ) -> SearchResult:
        del index_name
        self._validate_window(limit, offset)
        if limit == 0:
            return SearchResult()
        if not isinstance(field, str) or not field:
            raise ValueError("search_by_scalar requires a field name")
        direction = "desc" if str(order or "desc").lower() == "desc" else "asc"
        request_fields = output_fields
        if request_fields is not None:
            request_fields = list(dict.fromkeys([*request_fields, field]))
        points = self._scroll(
            filters=self._normalize_filter(filters),
            limit=limit + offset,
            output_fields=request_fields,
            order_by={"key": field, "direction": direction},
        )
        items: list[SearchItemResult] = []
        for point in points[offset : offset + limit]:
            record = payload_to_record(point)
            score = record.get(field)
            score_value = float(score) if isinstance(score, (int, float)) else 0.0
            hit = _Hit(point_id=str(point.get("id")), record=record, score=score_value)
            items.append(_hit_to_item(hit, output_fields))
        return SearchResult(data=items)

    def search_by_id(
        self,
        index_name: str,
        id: Any,
        limit: int = 10,
        offset: int = 0,
        filters: dict[str, Any] | None = None,
        output_fields: list[str] | None = None,
    ) -> SearchResult:
        if id is None:
            return SearchResult()
        points = self._retrieve_points(
            self._collection_name,
            [to_qdrant_point_id(id)],
            with_vectors=True,
        )
        if not points:
            return SearchResult()
        record = self._vectors_to_record(points[0])
        dense = record.get("vector")
        sparse = record.get("sparse_vector")
        if not dense and not sparse:
            return SearchResult()
        return self.search_by_vector(
            index_name,
            dense_vector=dense,
            limit=limit,
            offset=offset,
            filters=filters,
            sparse_vector=sparse,
            output_fields=output_fields,
        )

    def search_by_keywords(
        self,
        index_name: str,
        keywords: list[str] | None = None,
        query: str | None = None,
        limit: int = 10,
        offset: int = 0,
        filters: dict[str, Any] | None = None,
        output_fields: list[str] | None = None,
        mode: str | None = None,
        fields: list[str] | None = None,
    ) -> SearchResult:
        del index_name, keywords, query, limit, offset, filters, output_fields, mode, fields
        raise NotImplementedError(
            "Qdrant does not store the OpenViking full-text content field; "
            "search_by_keywords is unavailable and grep falls back to the filesystem"
        )

    def search_by_multimodal(
        self,
        index_name: str,
        text: str | None,
        image: Any,
        video: Any,
        limit: int = 10,
        offset: int = 0,
        filters: dict[str, Any] | None = None,
        output_fields: list[str] | None = None,
    ) -> SearchResult:
        del index_name, text, image, video, limit, offset, filters, output_fields
        raise NotImplementedError("Qdrant multimodal search is not supported")

    # ------------------------------------------------------------------
    # sparse dictionary
    # ------------------------------------------------------------------

    def _get_sparse_dictionary(self) -> SparseTermDictionary:
        if not self._sparse_enabled:
            raise ValueError("Qdrant collection was created without sparse-vector support")
        with self._sparse_lock:
            if self._sparse_dictionary is None:
                self._sparse_dictionary = SparseTermDictionary(
                    resolve_term=self._resolve_sparse_term,
                    resolve_index=self._resolve_sparse_index,
                    persist=self._persist_sparse_term,
                )
            return self._sparse_dictionary

    def encode_sparse_vector(self, vector: dict[str, float]) -> dict[str, list[Any]] | None:
        return self._get_sparse_dictionary().encode(vector)

    def _resolve_sparse_term(self, term: str) -> int | None:
        points = self._scroll(
            filters={"must": [{"key": "term", "match": {"value": term}}]},
            limit=1,
            collection_name=self._metadata_collection_name,
        )
        if not points:
            return None
        value = (points[0].get("payload") or {}).get("index")
        if value is None:
            return None
        index = int(value)
        owner = self._resolve_sparse_index(index)
        if owner is not None and owner != term:
            raise ValueError(
                "sparse term index collision: "
                f"index={index} existing_term={owner!r} new_term={term!r}"
            )
        return index

    def _resolve_sparse_index(self, index: int) -> str | None:
        points = self._scroll(
            filters={"must": [{"key": "index", "match": {"value": int(index)}}]},
            limit=2,
            collection_name=self._metadata_collection_name,
        )
        if not points:
            return None
        terms = {
            str(value)
            for point in points
            if (value := (point.get("payload") or {}).get("term")) is not None
        }
        if len(terms) > 1:
            raise ValueError(
                "sparse term index collision: "
                f"index={index} existing_terms={sorted(terms)!r}"
            )
        return next(iter(terms), None)

    def _persist_sparse_term(self, term: str, index: int) -> None:
        self._upsert_points(
            self._metadata_collection_name,
            [
                {
                    "id": to_qdrant_point_id(f"openviking:sparse:{term}"),
                    "vector": {META_VECTOR_NAME: [0.0]},
                    "payload": {
                        "_openviking_sparse_term": True,
                        "term": term,
                        "index": int(index),
                    },
                }
            ],
        )

    # ------------------------------------------------------------------
    # validation / conversion helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_window(limit: int, offset: int) -> None:
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 0:
            raise ValueError("limit must be a non-negative integer")
        if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
            raise ValueError("offset must be a non-negative integer")

    @staticmethod
    def _normalize_filter(filters: dict[str, Any] | None) -> dict[str, Any]:
        if filters is None:
            return {}
        if not isinstance(filters, dict):
            raise TypeError("Qdrant collection expects a compiled filter dictionary")
        return filters


def _hit_to_item(hit: _Hit, output_fields: list[str] | None) -> SearchItemResult:
    fields = project_record(hit.record, output_fields)
    item = SearchItemResult(
        id=hit.record.get("id"),
        fields=fields,
        score=hit.score,
    )
    if hit.origin_score is not None:
        item.origin_score = hit.origin_score
    if hit.addition_score is not None:
        item.addition_score = hit.addition_score
    return item


def _vector_dim_from_schema(schema: dict[str, Any]) -> int:
    for field in schema.get("Fields", []):
        if str(field.get("FieldType", "")).lower() == "vector":
            try:
                return int(field.get("Dim") or 0)
            except (TypeError, ValueError):
                return 0
    return 0


def _epoch_ms(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, str):
        try:
            parsed = parse_iso_datetime(value)
        except (TypeError, ValueError):
            return None
        return _datetime_ms(parsed)
    if isinstance(value, datetime):
        return _datetime_ms(value)
    return None


def _datetime_ms(value: datetime) -> float:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.timestamp() * 1000.0


__all__ = ["QdrantCollection"]
