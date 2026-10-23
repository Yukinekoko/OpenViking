# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Read-only diagnostics for the Qdrant backend binding.

The report answers the questions an operator has before switching a deployment
over to this adapter:

* is the configured Qdrant endpoint reachable? which server version?
* does the physical collection already exist, and how many points does it hold?
* is the OpenViking metadata marker present/valid, and does its schema match
  what v0.5.1.1 expects (vector dim/distance/names, schema additions)?
* can the existing collection be reused as-is, or must it be rebuilt?

``python -m openviking_qdrant.diagnose`` prints the same report and returns:

* ``0`` reachable, and either reusable or nothing to reuse yet (fresh start)
* ``1`` reachable, but the existing collection cannot be reused with this config
* ``2`` Qdrant endpoint unreachable / authentication failed

This module was informed by the ops-focused candidate implementation but is
written against this package's marker layout (no extra marker fields are
written, so old readers stay compatible).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any

from .config import QdrantSettings
from .meta import (
    COMPAT_CONTEXT_FIELDS,
    META_MARKER_ID,
    META_SUFFIX,
    META_VERSION,
    is_valid_metadata,
)
from .rest import QdrantError, QdrantRestClient, collection_path

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_NOT_REUSABLE = 1
EXIT_UNREACHABLE = 2


def run_diagnostics(
    *,
    settings: QdrantSettings,
    client: QdrantRestClient | None = None,
) -> dict[str, Any]:
    """Collect a JSON-serialisable diagnostics report. Never raises for I/O."""
    client = client or QdrantRestClient(
        settings.url,
        api_key=settings.api_key,
        timeout_seconds=settings.timeout_seconds,
        max_retries=settings.max_retries,
        backoff_base_seconds=settings.backoff_base_seconds,
        backoff_max_seconds=settings.backoff_max_seconds,
    )
    physical = f"{settings.project_name or 'default'}__{settings.collection_name}"
    metadata = settings.metadata_collection_name or f"{physical}{META_SUFFIX}"
    report: dict[str, Any] = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "url": settings.url,
        "project": settings.project_name,
        "collection_name": settings.collection_name,
        "physical_collection": physical,
        "metadata_collection": metadata,
        "config": {
            "dimension": settings.dimension,
            "distance_metric": settings.distance_metric,
            "sparse_weight": settings.sparse_weight,
            "dense_vector_name": settings.dense_vector_name,
            "sparse_vector_name": settings.sparse_vector_name,
            "query_candidate_cap": settings.query_candidate_cap,
            "max_retries": settings.max_retries,
        },
        "server": {"reachable": False, "version": None, "title": None},
        "collection": {
            "present": False,
            "points_count": None,
        },
        "metadata": {"present": False, "valid": False},
        "reuse": {"reusable": False, "reasons": [], "warnings": [], "actions": []},
        "status": "unreachable",
        "exit_code": EXIT_UNREACHABLE,
    }
    reuse = report["reuse"]

    # 1. server reachability / version -------------------------------------
    try:
        root = _safe_request(client, "GET", "/")
    except QdrantError as exc:
        report["server"]["error"] = str(exc)
        reuse["reasons"].append("qdrant_unreachable")
        reuse["actions"].append(
            "check storage.vectordb.url / network / api key, then rerun the diagnose command"
        )
        return report
    report["server"].update(
        {
            "reachable": True,
            "version": root.get("version"),
            "title": root.get("title"),
        }
    )
    report["status"] = "ok"

    # 2. collection presence / point count ---------------------------------
    collection_info: dict[str, Any] | None = None
    try:
        collection_info = _safe_request(client, "GET", collection_path(physical))
    except QdrantError as exc:
        if exc.status != 404:
            report["collection"]["error"] = str(exc)
            report["status"] = "degraded"
        collection_info = None
    if collection_info is not None:
        result = collection_info.get("result") or {}
        report["collection"].update(
            {
                "present": True,
                "points_count": result.get("points_count"),
                "status": result.get("status"),
                "vectors_count": result.get("vectors_count"),
            }
        )

    # 3. metadata marker ----------------------------------------------------
    marker: dict[str, Any] | None = None
    try:
        collection_exists = _safe_request(client, "GET", collection_path(metadata)) is not None
    except QdrantError as exc:
        collection_exists = exc.status != 404
    if collection_exists:
        try:
            fetched = _safe_request(
                client,
                "POST",
                collection_path(metadata, "/points"),
                {"ids": [META_MARKER_ID], "with_payload": True, "with_vector": False},
            )
        except QdrantError as exc:
            fetched = None
            report["metadata"]["error"] = str(exc)
        points = (fetched or {}).get("result") or []
        if points:
            payload = points[0].get("payload")
            if isinstance(payload, dict):
                marker = payload
    if marker is not None:
        report["metadata"].update(
            {
                "present": True,
                "valid": is_valid_metadata(marker, physical),
                "meta_version": marker.get("_openviking_meta_version"),
                "expected_meta_version": META_VERSION,
                "collection_name": marker.get("collection_name"),
                "vector_dim": marker.get("vector_dim"),
                "distance": marker.get("distance"),
                "dense_vector_name": marker.get("dense_vector_name"),
                "sparse_vector_name": marker.get("sparse_vector_name"),
                "sparse_enabled": marker.get("sparse_enabled"),
                "fields": _schema_field_names(marker.get("schema")),
                "scalar_index": _scalar_index(marker.get("schema")),
            }
        )
        report["metadata"]["legacy_schema"] = _is_legacy_context_schema(marker.get("schema"))

    # 4. sparse term dictionary size ---------------------------------------
    if collection_exists:
        report["metadata"]["sparse_terms"] = _count_sparse_terms(client, metadata)

    # 5. reuse verdict ------------------------------------------------------
    _assess_reuse(settings=settings, report=report, marker=marker, physical=physical)
    if report["server"]["reachable"]:
        if not report["collection"]["present"]:
            report["exit_code"] = EXIT_OK
            reuse["reasons"].append("collection_absent_fresh_start")
            reuse["actions"].append(
                "nothing to reuse: the server will create the collection on first start"
            )
        elif reuse["reusable"]:
            report["exit_code"] = EXIT_OK
        else:
            report["exit_code"] = EXIT_NOT_REUSABLE
    else:
        report["exit_code"] = EXIT_UNREACHABLE
    return report


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _safe_request(
    client: QdrantRestClient,
    method: str,
    path: str,
    body: dict[str, Any] | None = None,
) -> Any:
    return client.request(method, path, body)


def _schema_field_names(schema: Any) -> list[str]:
    if not isinstance(schema, dict):
        return []
    names = []
    for field in schema.get("Fields") or []:
        if isinstance(field, dict) and field.get("FieldName"):
            names.append(str(field["FieldName"]))
    return names


def _scalar_index(schema: Any) -> list[str]:
    if not isinstance(schema, dict):
        return []
    raw = schema.get("ScalarIndex")
    if isinstance(raw, dict):
        return [str(field) for field in raw]
    if isinstance(raw, (list, tuple, set)):
        return [str(field) for field in raw]
    return []


def _is_legacy_context_schema(schema: Any) -> bool:
    names = set(_schema_field_names(schema))
    if not {"uri", "level", "account_id", "vector"} <= names:
        return False
    return not all(field["FieldName"] in names for field in COMPAT_CONTEXT_FIELDS)


def _count_sparse_terms(client: QdrantRestClient, metadata_collection: str) -> int | None:
    body = {
        "filter": {
            "must": [
                {"key": "_openviking_sparse_term", "match": {"value": True}},
            ]
        },
        "limit": 1000,
        "with_payload": False,
        "with_vector": False,
    }
    total = 0
    try:
        while True:
            fetched = client.request(
                "POST", collection_path(metadata_collection, "/points/scroll"), body
            )
            result = (fetched or {}).get("result") or {}
            points = result.get("points") or []
            total += len(points)
            next_offset = result.get("next_page_offset")
            if not points or next_offset is None:
                return total
            body["offset"] = next_offset
    except QdrantError:
        return None


def _assess_reuse(
    *,
    settings: QdrantSettings,
    report: dict[str, Any],
    marker: dict[str, Any] | None,
    physical: str,
) -> None:
    reuse = report["reuse"]
    collection_present = report["collection"]["present"]
    metadata = report["metadata"]

    if not collection_present:
        return
    if marker is None or not metadata.get("valid"):
        reuse["reasons"].append("metadata_missing_or_invalid")
        reuse["actions"].append(
            "the physical collection exists but has no valid OpenViking marker; "
            "use a different project/name (recommended) or drop/restore the collection"
        )
        return

    marker_name = str(marker.get("collection_name") or "")
    if marker_name and marker_name != physical:
        reuse["reasons"].append("metadata_owner_mismatch")
        reuse["actions"].append(
            f"marker belongs to collection {marker_name!r}, not {physical!r}"
        )

    marker_dim = marker.get("vector_dim")
    try:
        marker_dim_int = int(marker_dim)
    except (TypeError, ValueError):
        marker_dim_int = None
    if marker_dim_int is not None and settings.dimension and marker_dim_int != settings.dimension:
        reuse["reasons"].append("dimension_mismatch")
        reuse["actions"].append(
            f"existing collection dim={marker_dim_int} vs config dim={settings.dimension}; "
            "rebuild the vectors with the configured embedding model"
        )

    marker_distance = str(marker.get("distance") or "").lower()
    # Accept both the Qdrant enum ("Cosine") and the config spelling ("cosine").
    normalized_metric = settings.distance_metric.lower()
    distance_compatible = marker_distance in {"", normalized_metric} or (
        normalized_metric in {"ip", "dot"} and marker_distance == "dot"
    ) or (
        normalized_metric in {"l2", "euclid"} and marker_distance == "euclid"
    ) or (
        normalized_metric in {"l1", "manhattan"} and marker_distance == "manhattan"
    )
    if not distance_compatible:
        reuse["reasons"].append("distance_mismatch")
        reuse["actions"].append(
            f"existing collection distance={marker.get('distance')!r} vs config "
            f"{settings.distance_metric!r}; rebuilding is required"
        )

    marker_dense = str(marker.get("dense_vector_name") or "")
    if marker_dense and marker_dense != settings.dense_vector_name:
        reuse["reasons"].append("dense_vector_name_mismatch")
        reuse["actions"].append(
            f"existing dense vector name {marker_dense!r} != config "
            f"{settings.dense_vector_name!r}"
        )

    marker_sparse_enabled = bool(marker.get("sparse_enabled"))
    config_sparse_enabled = settings.sparse_weight > 0.0
    if marker_sparse_enabled != config_sparse_enabled:
        reuse["reasons"].append("sparse_setting_mismatch")
        reuse["actions"].append(
            "existing collection sparse_enabled="
            f"{marker_sparse_enabled} vs config sparse_weight={settings.sparse_weight}; "
            "align the config or rebuild"
        )

    if metadata.get("legacy_schema"):
        reuse["warnings"].append("legacy_schema_compat_view")
        reuse["actions"].append(
            "legacy 0.4.16 schema detected: md5/acl fields are exposed through an "
            "in-memory compat view; run the reindex to persist the new schema"
        )

    if not reuse["reasons"]:
        reuse["reusable"] = True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m openviking_qdrant.diagnose",
        description=(
            "Check Qdrant reachability and whether the existing collection can be "
            "reused by the OpenViking v0.5.1.1 qdrant backend."
        ),
    )
    parser.add_argument("--url", required=True, help="Qdrant base URL, e.g. http://127.0.0.1:6333")
    parser.add_argument("--api-key", default=None, help="Qdrant api-key header")
    parser.add_argument("--project", default="default", help="storage.vectordb.project")
    parser.add_argument("--collection", default="context", help="storage.vectordb.name")
    parser.add_argument("--dimension", type=int, default=768, help="storage.vectordb.dimension")
    parser.add_argument("--distance", default="cosine", help="storage.vectordb.distance_metric")
    parser.add_argument(
        "--sparse-weight", type=float, default=0.0, help="storage.vectordb.sparse_weight"
    )
    parser.add_argument("--timeout", type=float, default=10.0, help="request timeout seconds")
    parser.add_argument("--max-retries", type=int, default=2, help="retries for transient errors")
    parser.add_argument("--json", action="store_true", help="print the raw JSON report")
    args = parser.parse_args(argv)

    settings = QdrantSettings(
        url=args.url,
        api_key=args.api_key,
        timeout_seconds=args.timeout,
        project_name=args.project,
        collection_name=args.collection,
        index_name="default",
        distance_metric=args.distance,
        dimension=args.dimension,
        sparse_weight=args.sparse_weight,
        dense_vector_name="vector",
        sparse_vector_name="sparse_vector",
        metadata_collection_name=None,
        query_candidate_cap=100_000,
        batch_size=100,
        max_retries=args.max_retries,
    )
    report = run_diagnostics(settings=settings)
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        _print_human(report)
    return int(report["exit_code"])


def _print_human(report: dict[str, Any]) -> None:
    server = report["server"]
    metadata = report["metadata"]
    collection = report["collection"]
    reuse = report["reuse"]
    print(f"Qdrant: {report['url']}")
    if not server["reachable"]:
        print(f"  status: UNREACHABLE ({server.get('error', '')})")
        return
    print(f"  server: reachable, version={server.get('version')}")
    prefix = f"{report['project']}__{report['collection_name']}"
    print(f"  collection: {'present' if collection['present'] else 'absent'} ({prefix})")
    if collection["present"]:
        print(f"  points: {collection.get('points_count')}")
    print(
        "  metadata: "
        + (
            "valid"
            if metadata.get("valid")
            else ("invalid" if metadata.get("present") else "absent")
        )
    )
    if metadata.get("valid"):
        print(
            "  marker: dim={} distance={} sparse_enabled={} fields={}".format(
                metadata.get("vector_dim"),
                metadata.get("distance"),
                metadata.get("sparse_enabled"),
                len(metadata.get("fields") or []),
            )
        )
        if metadata.get("sparse_terms") is not None:
            print(f"  sparse terms: {metadata['sparse_terms']}")
        if metadata.get("legacy_schema"):
            print("  schema: legacy 0.4.16 shape (compat view active)")
    if not collection["present"]:
        print("  reuse: fresh start (collection will be created on first start)")
    elif reuse["reusable"]:
        print("  reuse: reusable with this configuration")
    else:
        print("  reuse: NOT reusable")
        for reason in reuse["reasons"]:
            print(f"    - {reason}")
    for action in reuse["actions"]:
        print(f"  action: {action}")
    print(f"  exit: {report['exit_code']}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())
