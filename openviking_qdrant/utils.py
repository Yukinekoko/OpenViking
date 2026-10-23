# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Pure conversion helpers shared by the Qdrant adapter and collection.

The payload layout is deliberately identical to the historical
``upgrade/v0.4.16-qdrant`` fork so existing collections can be opened without a
migration:

* point IDs are UUIDv5 of the OpenViking record ID;
* the source record ID is retained in ``_openviking_original_id``;
* URIs are stored without the ``viking://`` scheme;
* ``uri_depth`` and ``scope_roots`` back segment-aware path scopes.
"""

from __future__ import annotations

import math
import uuid
from datetime import date, datetime
from typing import Any, Iterable

_OPENVIKING_QDRANT_ID_NAMESPACE = uuid.UUID("4b6bb5a8-7f1f-5b1a-9d4c-b93f29b1d67c")
ORIGINAL_ID_FIELD = "_openviking_original_id"
URI_DEPTH_FIELD = "uri_depth"
SCOPE_ROOTS_FIELD = "scope_roots"
INTERNAL_PAYLOAD_FIELDS = frozenset(
    {ORIGINAL_ID_FIELD, URI_DEPTH_FIELD, SCOPE_ROOTS_FIELD}
)
URI_FIELDS = frozenset({"uri", "parent_uri"})


def to_qdrant_point_id(value: Any) -> str:
    """Return a deterministic UUID point ID for an OpenViking record ID."""
    return str(uuid.uuid5(_OPENVIKING_QDRANT_ID_NAMESPACE, str(value)))


def normalize_uri_path(value: Any) -> Any:
    """Normalize ``viking://...`` and ordinary paths to ``/segment/...``."""
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if stripped.startswith("viking://"):
        stripped = stripped[len("viking://") :]
    if not stripped:
        return "/"
    normalized = "/" + stripped.lstrip("/")
    return normalized.rstrip("/") or "/"


def path_depth(path: str) -> int:
    return len([part for part in path.split("/") if part])


def scope_roots(path: str) -> list[str]:
    """Return ``/``, every ancestor, and the path itself."""
    parts = [part for part in path.split("/") if part]
    roots = ["/"]
    for index in range(1, len(parts) + 1):
        roots.append("/" + "/".join(parts[:index]))
    return roots


def to_qdrant_value(value: Any) -> Any:
    """Make a filter bound JSON/Qdrant compatible without changing its meaning."""
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, tuple):
        return list(value)
    return value


def build_qdrant_payload(record: dict[str, Any]) -> dict[str, Any]:
    """Convert one OpenViking record into a Qdrant payload (no vectors)."""
    payload = {key: _json_safe(value) for key, value in record.items()}
    payload.pop("vector", None)
    payload.pop("sparse_vector", None)
    original_id = payload.pop("id", None)
    if original_id is not None:
        payload[ORIGINAL_ID_FIELD] = str(original_id)

    for field in URI_FIELDS:
        value = payload.get(field)
        if isinstance(value, str):
            payload[field] = normalize_uri_path(value)

    uri = payload.get("uri")
    if isinstance(uri, str):
        payload[URI_DEPTH_FIELD] = path_depth(uri)
        payload[SCOPE_ROOTS_FIELD] = scope_roots(uri)
    return payload


def payload_to_record(point: dict[str, Any]) -> dict[str, Any]:
    """Decode a Qdrant point payload into an OpenViking record."""
    payload = dict(point.get("payload") or {})
    original_id = payload.pop(ORIGINAL_ID_FIELD, None)
    if original_id is None:
        original_id = str(point.get("id"))
    for field in INTERNAL_PAYLOAD_FIELDS:
        payload.pop(field, None)
    payload["id"] = original_id
    return payload


def project_record(record: dict[str, Any], output_fields: list[str] | None) -> dict[str, Any]:
    """Apply an output-field projection while always retaining the record ID."""
    if output_fields is None:
        return dict(record)
    projected = {field: record[field] for field in output_fields if field in record}
    if "id" in record:
        projected["id"] = record["id"]
    return projected


def payload_selector(
    output_fields: list[str] | None,
    *,
    extra_fields: Iterable[str] = (),
) -> bool | dict[str, list[str]]:
    """Build Qdrant's ``with_payload`` selector."""
    if output_fields is None:
        return True
    fields = list(dict.fromkeys([*output_fields, ORIGINAL_ID_FIELD, *extra_fields]))
    return {"include": fields}


def finite_score(value: Any) -> float:
    """Return a finite float for a backend score, defaulting to ``0.0``."""
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return 0.0


def _json_safe(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return value


__all__ = [
    "INTERNAL_PAYLOAD_FIELDS",
    "ORIGINAL_ID_FIELD",
    "SCOPE_ROOTS_FIELD",
    "URI_DEPTH_FIELD",
    "URI_FIELDS",
    "build_qdrant_payload",
    "finite_score",
    "normalize_uri_path",
    "path_depth",
    "payload_selector",
    "payload_to_record",
    "project_record",
    "scope_roots",
    "to_qdrant_point_id",
    "to_qdrant_value",
]
