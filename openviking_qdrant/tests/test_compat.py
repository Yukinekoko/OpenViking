# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Compatibility with the 0.4.16 Qdrant fork's on-disk Qdrant layout."""

from __future__ import annotations

import uuid

from openviking_qdrant.meta import META_MARKER_ID, META_VECTOR_NAME
from openviking_qdrant.rest import collection_path
from openviking_qdrant.sparse import stable_sparse_index
from openviking_qdrant.tests.fake_qdrant import FakeQdrant
from openviking_qdrant.tests.helpers import legacy_schema, make_adapter
from openviking_qdrant.utils import build_qdrant_payload, to_qdrant_point_id

NAMESPACE = uuid.UUID("4b6bb5a8-7f1f-5b1a-9d4c-b93f29b1d67c")


def test_point_id_and_payload_layout_match_the_legacy_fork() -> None:
    assert to_qdrant_point_id("legacy-doc") == "846ce466-a8ec-56c3-9fbd-598fd0b413c4"
    assert to_qdrant_point_id("doc-1") == "0077a8eb-accf-5599-92fe-3586b2ffb18d"
    assert META_MARKER_ID == "400beb5c-1532-581d-9fc1-f8cd88a00243"

    payload = build_qdrant_payload(
        {
            "id": "legacy-doc",
            "uri": "viking://resources/wiki/physics/doc.md",
            "parent_uri": "viking://resources/wiki/physics",
            "account_id": "acct",
        }
    )
    assert payload["_openviking_original_id"] == "legacy-doc"
    assert payload["uri"] == "/resources/wiki/physics/doc.md"
    assert payload["parent_uri"] == "/resources/wiki/physics"
    assert payload["uri_depth"] == 4
    assert payload["scope_roots"] == [
        "/",
        "/resources",
        "/resources/wiki",
        "/resources/wiki/physics",
        "/resources/wiki/physics/doc.md",
    ]


def _seed_legacy_collection(fake: FakeQdrant) -> dict:
    schema = legacy_schema("context")
    physical = "proj__context"
    metadata = f"{physical}__openviking_meta"
    index_meta = {
        "IndexName": "default",
        "VectorIndex": {"IndexType": "hnsw_hybrid", "Distance": "cosine"},
        "ScalarIndex": [
            *schema["ScalarIndex"],
            "uri_depth",
            "scope_roots",
        ],
        "SparseWeight": 0.5,
    }
    marker = {
        "_openviking_meta_version": 1,
        "collection_name": physical,
        "schema": schema,
        "dense_vector_name": "vector",
        "sparse_vector_name": "sparse_vector",
        "vector_dim": 4,
        "distance": "Cosine",
        "sparse_enabled": True,
        "sparse_weight": 0.5,
        "indexes": {"default": index_meta},
    }
    fake.seed(
        physical,
        vectors={"vector": {"size": 4, "distance": "Cosine"}},
        sparse=True,
    )
    fake.seed(
        metadata,
        vectors={META_VECTOR_NAME: {"size": 1, "distance": "Dot"}},
    )
    fake.seed_point(
        metadata,
        point_id=META_MARKER_ID,
        vector={META_VECTOR_NAME: [0.0]},
        payload=marker,
    )
    alpha_index = stable_sparse_index("alpha")
    fake.seed_point(
        metadata,
        point_id=str(uuid.uuid5(NAMESPACE, "openviking:sparse:alpha")),
        vector={META_VECTOR_NAME: [0.0]},
        payload={
            "_openviking_sparse_term": True,
            "term": "alpha",
            "index": alpha_index,
        },
    )
    fake.seed_point(
        physical,
        point_id=to_qdrant_point_id("legacy-doc"),
        vector={
            "vector": [1.0, 0.0, 0.0, 0.0],
            "sparse_vector": {"indices": [alpha_index], "values": [0.7]},
        },
        payload={
            "_openviking_original_id": "legacy-doc",
            "uri": "/resources/legacy/doc.md",
            "uri_depth": 3,
            "scope_roots": ["/", "/resources", "/resources/legacy", "/resources/legacy/doc.md"],
            "context_type": "resource",
            "level": 2,
            "account_id": "acct",
            "abstract": "legacy",
            "updated_at": "2026-01-01T00:00:00Z",
        },
    )
    return {"physical": physical, "metadata": metadata, "alpha_index": alpha_index}


def test_legacy_collection_loads_and_reads_dense_and_sparse() -> None:
    fake = FakeQdrant()
    layout = _seed_legacy_collection(fake)
    # The new config is dense-only; the marker must win so old sparse data is
    # still readable.
    adapter = make_adapter(fake, sparse_weight=0.0)
    assert adapter.collection_exists()
    raw = adapter.get_collection()._Collection__collection
    assert raw._sparse_enabled is True
    assert raw._distance == "Cosine"
    assert raw._sparse_weight == 0.5

    record = adapter.get(["legacy-doc"])[0]
    assert record["uri"] == "viking://resources/legacy/doc.md"
    assert record["abstract"] == "legacy"
    assert "md5" not in record
    assert "acl_mode" not in record

    dense = adapter.query(query_vector=[1, 0, 0, 0], limit=1)
    assert dense[0]["id"] == "legacy-doc"
    sparse = adapter.query(sparse_query_vector={"alpha": 1.0}, limit=1)
    assert sparse[0]["id"] == "legacy-doc"

    scoped = adapter.query(
        filter={
            "op": "must",
            "field": "uri",
            "conds": ["viking://resources/legacy"],
            "para": "-d=-1",
        },
        limit=10,
    )
    assert [item["id"] for item in scoped] == ["legacy-doc"]

    meta = adapter.get_collection().get_meta_data()
    names = {field["FieldName"] for field in meta["Fields"]}
    assert {"md5", "acl_mode", "acl_direct_grants", "acl_inherited_grants"} <= names
    index_meta = adapter.get_collection().get_index_meta_data("default")
    assert "uri_depth" in index_meta["ScalarIndex"]

    # Existing sparse vectors decode through the persisted term dictionary.
    decoded = raw._vectors_to_record(
        fake.collections[layout["physical"]]["points"][to_qdrant_point_id("legacy-doc")]
    )
    assert decoded["sparse_vector"] == {"alpha": 0.7}


def test_legacy_acl_missing_records_are_default_shared() -> None:
    fake = FakeQdrant()
    _seed_legacy_collection(fake)
    adapter = make_adapter(fake)
    assert adapter.collection_exists()
    result = adapter.query(
        filter={
            "op": "must_not",
            "field": "acl_mode",
            "conds": ["inherit", "restricted"],
        },
        limit=10,
    )
    assert [item["id"] for item in result] == ["legacy-doc"]


def test_legacy_marker_distance_and_metadata_path() -> None:
    fake = FakeQdrant()
    layout = _seed_legacy_collection(fake)
    adapter = make_adapter(fake)
    assert adapter.collection_exists()
    assert collection_path(layout["physical"]) == "/collections/proj__context"
    assert adapter._metadata_collection_name == layout["metadata"]
    marker = fake.collections[layout["metadata"]]["points"][META_MARKER_ID]["payload"]
    assert marker["distance"] == "Cosine"
