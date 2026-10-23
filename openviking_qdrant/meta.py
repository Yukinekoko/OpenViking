# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""OpenViking metadata marker and schema-view helpers.

The marker lives in a sidecar Qdrant collection and keeps the historical
``_META_VERSION = 1`` payload shape so collections created by the 0.4.16
Qdrant fork remain readable.
"""

from __future__ import annotations

import copy
from typing import Any

from .utils import to_qdrant_point_id

META_VERSION = 1
META_VECTOR_NAME = "meta"
META_MARKER_ID = to_qdrant_point_id("openviking:metadata")
META_SUFFIX = "__openviking_meta"

# v0.5.1.1 collection schema additions.  Records written by the 0.4.16 fork
# do not contain these payload keys; the upstream ACL filter treats a missing
# acl_mode as the default shared state, and missing md5 as "unknown".
COMPAT_CONTEXT_FIELDS: tuple[dict[str, Any], ...] = (
    {"FieldName": "md5", "FieldType": "string", "DefaultValue": ""},
    {"FieldName": "acl_mode", "FieldType": "string", "DefaultValue": "none"},
    {"FieldName": "acl_direct_grants", "FieldType": "list<string>", "DefaultValue": []},
    {"FieldName": "acl_inherited_grants", "FieldType": "list<string>", "DefaultValue": []},
)
COMPAT_SCALAR_FIELDS = ("acl_mode", "acl_direct_grants", "acl_inherited_grants")
_INTERNAL_SCALAR_FIELDS = frozenset({"uri_depth", "scope_roots"})
_CONTEXT_FIELD_SIGNATURE = frozenset({"uri", "level", "account_id", "vector"})


def metadata_payload(
    *,
    collection_name: str,
    schema: dict[str, Any],
    dense_vector_name: str,
    sparse_vector_name: str,
    vector_dim: int,
    distance: str,
    sparse_enabled: bool,
    sparse_weight: float,
    indexes: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Build the marker payload written to the sidecar collection."""
    return {
        "_openviking_meta_version": META_VERSION,
        "collection_name": collection_name,
        "schema": copy.deepcopy(schema),
        "dense_vector_name": dense_vector_name,
        "sparse_vector_name": sparse_vector_name,
        "vector_dim": int(vector_dim),
        "distance": distance,
        "sparse_enabled": bool(sparse_enabled),
        "sparse_weight": float(sparse_weight),
        "indexes": copy.deepcopy(indexes),
    }


def is_valid_metadata(payload: Any, collection_name: str) -> bool:
    """Validate the minimum marker shape and ownership."""
    return bool(
        isinstance(payload, dict)
        and payload.get("_openviking_meta_version") == META_VERSION
        and payload.get("collection_name") == collection_name
        and isinstance(payload.get("schema"), dict)
    )


def build_meta_view(
    schema: dict[str, Any],
    *,
    collection_name: str,
    indexes: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return a read-only 0.5.1.1-shaped view of a possibly legacy schema.

    The returned view is a deep copy.  It never rewrites the stored marker:
    old collections keep their original schema, but callers (notably
    ``_filter_known_fields`` and the startup compatibility checks) see the
    fields that 0.5.1.1 knows how to read and write.
    """
    view = copy.deepcopy(schema)
    view.setdefault("CollectionName", collection_name)
    fields = view.get("Fields")
    if not isinstance(fields, list):
        fields = []
    view["Fields"] = fields

    names = {
        field.get("FieldName")
        for field in fields
        if isinstance(field, dict) and field.get("FieldName")
    }
    context_schema = _CONTEXT_FIELD_SIGNATURE <= names
    added_compat_field = False
    if context_schema:
        for compat_field in COMPAT_CONTEXT_FIELDS:
            if compat_field["FieldName"] not in names:
                fields.append(copy.deepcopy(compat_field))
                names.add(compat_field["FieldName"])
                added_compat_field = True

    scalar_index = view.get("ScalarIndex")
    if not isinstance(scalar_index, list):
        scalar_index = _derive_scalar_index(indexes)
    else:
        scalar_index = list(scalar_index)
    if context_schema and added_compat_field:
        # Only a legacy schema needs the ACL scalar indexes added to the view.
        # A current schema owns its ScalarIndex, including explicit removals.
        for field_name in COMPAT_SCALAR_FIELDS:
            if field_name not in scalar_index:
                scalar_index.append(field_name)
    view["ScalarIndex"] = scalar_index

    if "Description" not in view:
        view["Description"] = "Unified context collection" if context_schema else ""
    return view


def index_scalar_fields(meta: Any) -> list[str]:
    """Extract scalar-index field names from either a list or legacy dict."""
    raw = meta.get("ScalarIndex") if isinstance(meta, dict) else None
    if isinstance(raw, dict):
        return [str(field) for field in raw]
    if isinstance(raw, (list, tuple, set)):
        return [str(field) for field in raw]
    return []


def _derive_scalar_index(indexes: dict[str, dict[str, Any]] | None) -> list[str]:
    if not indexes:
        return []
    fields: list[str] = []
    for meta in indexes.values():
        for field in index_scalar_fields(meta):
            if field not in _INTERNAL_SCALAR_FIELDS and field not in fields:
                fields.append(field)
    return fields


__all__ = [
    "COMPAT_CONTEXT_FIELDS",
    "COMPAT_SCALAR_FIELDS",
    "META_MARKER_ID",
    "META_SUFFIX",
    "META_VECTOR_NAME",
    "META_VERSION",
    "build_meta_view",
    "index_scalar_fields",
    "is_valid_metadata",
    "metadata_payload",
]
