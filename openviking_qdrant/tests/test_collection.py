# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
from __future__ import annotations

import pytest

from openviking.storage.expr import And, Contains, Eq, In, Or, PathScope, RawDSL
from openviking_qdrant.meta import META_MARKER_ID, META_VERSION
from openviking_qdrant.tests.fake_qdrant import FakeQdrant
from openviking_qdrant.tests.helpers import (
    create_collection,
    make_adapter,
)
from openviking_qdrant.utils import to_qdrant_point_id


def _record(**overrides):
    record = {
        "id": "doc-1",
        "uri": "viking://resources/wiki/doc.md",
        "parent_uri": "viking://resources/wiki",
        "context_type": "resource",
        "level": 2,
        "name": "doc.md",
        "abstract": "abstract",
        "md5": "md5-one",
        "account_id": "acct",
        "owner_user_id": "alice",
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
        "vector": [1.0, 0.0, 0.0, 0.0],
        "acl_mode": "inherit",
        "acl_direct_grants": ["1:user:alice"],
        "acl_inherited_grants": ["3:group:finance"],
    }
    record.update(overrides)
    return record


def test_create_load_and_metadata_marker(fake: FakeQdrant) -> None:
    adapter = make_adapter(fake)
    create_collection(adapter)
    assert adapter.collection_exists()

    meta = adapter.get_collection().get_meta_data(raise_on_error=True)
    assert meta["CollectionName"] == "context"
    assert meta["Description"] == "Unified context collection"
    assert {field["FieldName"] for field in meta["Fields"]} >= {
        "md5",
        "acl_mode",
        "acl_direct_grants",
        "acl_inherited_grants",
    }
    assert "acl_direct_grants" in meta["ScalarIndex"]

    # A second adapter binds the same physical collection.
    reloaded = make_adapter(fake)
    assert reloaded.collection_exists()
    assert reloaded.get_collection().get_meta_data()["Fields"] == meta["Fields"]

    marker = fake.collections["proj__context__openviking_meta"]["points"][META_MARKER_ID]
    payload = marker["payload"]
    assert payload["_openviking_meta_version"] == META_VERSION
    assert payload["collection_name"] == "proj__context"
    assert payload["vector_dim"] == 4
    assert payload["distance"] == "Cosine"
    assert payload["indexes"]["default"]["IndexName"] == "default"


def test_stale_collection_without_metadata_fails_closed(fake: FakeQdrant) -> None:
    fake.seed("proj__context", vectors={"vector": {"size": 4, "distance": "Cosine"}})
    adapter = make_adapter(fake)
    with pytest.raises(RuntimeError, match="metadata is missing"):
        adapter.collection_exists()


def test_drop_removes_both_collections(fake: FakeQdrant) -> None:
    adapter = make_adapter(fake)
    create_collection(adapter)
    assert adapter.drop_collection() is True
    assert "proj__context" not in fake.collections
    assert "proj__context__openviking_meta" not in fake.collections


def test_upsert_get_projection_and_sparse_absent_vector_fields() -> None:
    fake = FakeQdrant()
    adapter = make_adapter(fake)
    create_collection(adapter)
    ids = adapter.upsert([_record()])
    assert ids == ["doc-1"]

    record = adapter.get(["doc-1"])[0]
    assert record["id"] == "doc-1"
    assert record["uri"] == "viking://resources/wiki/doc.md"
    assert record["parent_uri"] == "viking://resources/wiki"
    assert record["md5"] == "md5-one"
    assert record["acl_mode"] == "inherit"
    assert record["acl_direct_grants"] == ["1:user:alice"]
    assert record["acl_inherited_grants"] == ["3:group:finance"]
    assert "vector" not in record
    assert "uri_depth" not in record
    assert "scope_roots" not in record

    projected = adapter.get(["doc-1"])
    assert projected[0]["name"] == "doc.md"

    # Qdrant point IDs are stable UUIDv5 values with the original ID in payload.
    point = fake.collections["proj__context"]["points"][to_qdrant_point_id("doc-1")]
    assert point["payload"]["_openviking_original_id"] == "doc-1"
    assert point["payload"]["uri"] == "/resources/wiki/doc.md"
    assert point["payload"]["uri_depth"] == 3
    assert point["payload"]["scope_roots"] == [
        "/",
        "/resources",
        "/resources/wiki",
        "/resources/wiki/doc.md",
    ]

    first = adapter.get_collection().fetch_data(["doc-1", "missing"])
    assert first.items[0].id == "doc-1"
    assert first.ids_not_exist == ["missing"]


def test_update_data_merges_and_preserves_vector_and_acl() -> None:
    fake = FakeQdrant()
    adapter = make_adapter(fake)
    create_collection(adapter)
    adapter.upsert([_record()])

    updated = adapter.update_data(
        [{"id": "doc-1", "abstract": "new abstract", "md5": "md5-two"}]
    )
    assert updated == ["doc-1"]
    record = adapter.get(["doc-1"])[0]
    assert record["abstract"] == "new abstract"
    assert record["md5"] == "md5-two"
    assert record["acl_direct_grants"] == ["1:user:alice"]
    point = fake.collections["proj__context"]["points"][to_qdrant_point_id("doc-1")]
    assert point["vector"]["vector"] == [1.0, 0.0, 0.0, 0.0]

    with pytest.raises(ValueError, match="not found"):
        adapter.update_data([{"id": "missing", "abstract": "x"}])


def test_dense_query_scores_and_output_projection() -> None:
    fake = FakeQdrant()
    adapter = make_adapter(fake)
    create_collection(adapter)
    adapter.upsert(
        [
            _record(id="a", uri="viking://resources/a", vector=[1, 0, 0, 0], level=0),
            _record(id="b", uri="viking://resources/b", vector=[0.9, 0, 0, 0], level=1),
            _record(id="c", uri="viking://resources/c", vector=[0.5, 0, 0, 0], level=2),
        ]
    )
    result = adapter.query(
        query_vector=[1, 0, 0, 0], limit=2, output_fields=["uri", "level"]
    )
    assert [item["id"] for item in result] == ["a", "b"]
    assert result[0]["uri"] == "viking://resources/a"
    assert "abstract" not in result[0]
    assert result[0]["_score"] == pytest.approx(1.0)

    page = adapter.query(query_vector=[1, 0, 0, 0], limit=1, offset=1)
    assert page[0]["id"] == "b"

    filtered = adapter.query(
        query_vector=[1, 0, 0, 0],
        filter=And([Eq("account_id", "acct"), In("level", [0, 2])]),
        limit=10,
    )
    assert {item["id"] for item in filtered} == {"a", "c"}


def test_path_scope_filter_is_segment_aware() -> None:
    fake = FakeQdrant()
    adapter = make_adapter(fake)
    create_collection(adapter)
    adapter.upsert(
        [
            _record(id="a", uri="viking://resources/wiki/a.md"),
            _record(id="b", uri="viking://resources/wiki/sub/b.md"),
            _record(id="c", uri="viking://resources/other/c.md"),
        ]
    )
    direct = adapter.query(
        filter=PathScope("uri", "viking://resources/wiki", depth=1), limit=10
    )
    assert {item["id"] for item in direct} == {"a"}
    subtree = adapter.query(
        filter=PathScope("uri", "viking://resources/wiki", depth=-1), limit=10
    )
    assert {item["id"] for item in subtree} == {"a", "b"}


def test_filter_only_query_avoids_points_query_under_strict_limit() -> None:
    fake = FakeQdrant(max_query_limit=10)
    adapter = make_adapter(fake)
    create_collection(adapter)
    adapter.upsert(
        [
            _record(id=f"doc-{index}", uri=f"viking://resources/{index}")
            for index in range(25)
        ]
    )
    before = len(fake.requests_to("/points/query"))
    result = adapter.query(filter=Eq("account_id", "acct"), limit=20)
    assert len(result) == 20
    assert len(fake.requests_to("/points/query")) == before
    assert fake.requests_to("/points/scroll")


def test_scalar_order_pagination_and_score() -> None:
    fake = FakeQdrant()
    adapter = make_adapter(fake)
    create_collection(adapter)
    adapter.upsert(
        [
            _record(id="a", uri="viking://resources/a", updated_at="2026-01-03T00:00:00Z"),
            _record(id="b", uri="viking://resources/b", updated_at="2026-01-01T00:00:00Z"),
            _record(id="c", uri="viking://resources/c", updated_at="2026-01-02T00:00:00Z"),
        ]
    )
    ascending = adapter.query(order_by="updated_at", order_desc=False, limit=2)
    assert [item["id"] for item in ascending] == ["b", "c"]
    descending = adapter.query(
        order_by="updated_at",
        order_desc=True,
        limit=3,
        output_fields=["id"],
    )
    assert [item["id"] for item in descending] == ["a", "c", "b"]
    assert descending[0]["_score"] == 0.0
    assert "updated_at" not in descending[0]


def test_random_enumeration_and_count() -> None:
    fake = FakeQdrant()
    adapter = make_adapter(fake)
    create_collection(adapter)
    adapter.upsert(
        [
            _record(id=f"doc-{index}", uri=f"viking://resources/{index}")
            for index in range(5)
        ]
    )
    page = adapter.query(limit=2, offset=1)
    assert len(page) == 2
    assert all("_score" in item for item in page)
    assert adapter.count() == 5
    assert adapter.strict_count(Eq("account_id", "acct")) == 5
    assert adapter.count(Eq("account_id", "other")) == 0


def test_delete_by_filter_uses_native_count_and_delete() -> None:
    fake = FakeQdrant()
    adapter = make_adapter(fake)
    create_collection(adapter)
    adapter.upsert(
        [
            _record(id="a", uri="viking://resources/a", account_id="acct"),
            _record(id="b", uri="viking://resources/b", account_id="other"),
        ]
    )
    before_scroll = len(fake.requests_to("/points/scroll"))
    deleted = adapter.delete(filter=Eq("account_id", "acct"))
    assert deleted == 1
    assert len(fake.requests_to("/points/scroll")) == before_scroll
    assert adapter.count() == 1


def test_contains_filter_only_fallback_and_delete() -> None:
    fake = FakeQdrant(max_query_limit=1)
    adapter = make_adapter(fake)
    create_collection(adapter)
    adapter.upsert(
        [
            _record(id="a", uri="viking://resources/a.md#frag"),
            _record(id="b", uri="viking://resources/a.md"),
            _record(id="c", uri="viking://resources/b.md"),
        ]
    )
    expr = Or(
        [
            Eq("uri", "viking://resources/a.md"),
            PathScope("uri", "viking://resources", depth=-1),
            Contains("uri", "viking://resources/a.md#"),
        ]
    )
    # The PathScope branch is already broad enough for this dataset, but the
    # Contains branch must not be dropped from the client-side predicate.
    result = adapter.query(filter=expr, limit=10)
    assert {item["id"] for item in result} == {"a", "b", "c"}
    assert adapter.count(Contains("uri", "viking://resources/a.md#")) == 1
    deleted = adapter.delete(filter=Contains("uri", "viking://resources/a.md#"))
    assert deleted == 1
    assert adapter.get(["a"]) == []
    assert adapter.get(["b"])


def test_index_lifecycle_update_index_and_schema_update() -> None:
    fake = FakeQdrant()
    adapter = make_adapter(fake)
    create_collection(adapter)
    coll = adapter.get_collection()
    meta = coll.get_index_meta_data("default")
    assert "uri_depth" in meta["ScalarIndex"]
    assert "scope_roots" in meta["ScalarIndex"]
    assert coll.has_index("default")
    assert coll.list_indexes() == ["default"]

    updated = coll.update_index(
        "default",
        scalar_index=["uri", "level"],
        description="primary index",
    )
    assert updated["ScalarIndex"] == ["uri", "level"]
    assert updated["Description"] == "primary index"
    schema = coll.get_meta_data()
    assert schema["ScalarIndex"] == ["uri", "level"]
    assert fake.collections["proj__context"]["indexes"]["level"] == "integer"

    assert coll.drop_index("default") is True
    assert coll.has_index("default") is False
    assert coll.list_indexes() == []


def test_update_fields_list_merges_into_schema() -> None:
    fake = FakeQdrant()
    adapter = make_adapter(fake)
    create_collection(adapter)
    coll = adapter.get_collection()
    coll.update(fields=[{"FieldName": "new_field", "FieldType": "string"}])
    meta = coll.get_meta_data()
    assert any(field["FieldName"] == "new_field" for field in meta["Fields"])

    coll.update(description="custom description")
    assert coll.get_meta_data()["Description"] == "custom description"


def test_aggregate_supports_only_total_count() -> None:
    fake = FakeQdrant()
    adapter = make_adapter(fake)
    create_collection(adapter)
    adapter.upsert([_record()])
    result = adapter.get_collection().aggregate_data("default", op="count")
    assert result.agg == {"_total": 1}
    with pytest.raises(NotImplementedError):
        adapter.get_collection().aggregate_data("default", op="count", field="level")
    with pytest.raises(NotImplementedError):
        adapter.get_collection().aggregate_data("default", op="sum")


def test_get_meta_data_without_marker_and_raise_on_error(fake: FakeQdrant) -> None:
    adapter = make_adapter(fake)
    fake.seed("proj__context", vectors={"vector": {"size": 4, "distance": "Cosine"}})
    coll = adapter._new_collection()
    assert coll.get_meta_data() == {}
    with pytest.raises(RuntimeError, match="missing valid OpenViking metadata"):
        coll.get_meta_data(raise_on_error=True)


def test_raw_acl_filters_match_legacy_and_controlled_records() -> None:
    fake = FakeQdrant()
    adapter = make_adapter(fake)
    create_collection(adapter)
    adapter.upsert(
        [
            _record(
                id="legacy",
                uri="viking://resources/legacy",
                acl_mode=None,
                acl_direct_grants=[],
                acl_inherited_grants=[],
            ),
            _record(
                id="inherit",
                uri="viking://resources/inherit",
                acl_mode="inherit",
                acl_direct_grants=[],
                acl_inherited_grants=["7:user:*"],
            ),
            _record(
                id="restricted",
                uri="viking://resources/restricted",
                acl_mode="restricted",
                acl_direct_grants=["1:user:alice"],
                acl_inherited_grants=[],
            ),
        ]
    )
    # Simulate a legacy record without the ACL payload keys.
    fake.collections["proj__context"]["points"][to_qdrant_point_id("legacy")][
        "payload"
    ].pop("acl_mode", None)

    default_shared = RawDSL(
        {"op": "must_not", "field": "acl_mode", "conds": ["inherit", "restricted"]}
    )
    result = adapter.query(filter=default_shared, limit=10)
    assert {item["id"] for item in result} == {"legacy"}

    direct = adapter.query(
        filter=In("acl_direct_grants", ["1:user:alice"]), limit=10
    )
    assert {item["id"] for item in direct} == {"restricted"}
    inherited = adapter.query(
        filter=And(
            [
                Eq("acl_mode", "inherit"),
                In("acl_inherited_grants", ["7:user:*"]),
            ]
        ),
        limit=10,
    )
    assert {item["id"] for item in inherited} == {"inherit"}


def test_delete_all_data_and_search_by_id_and_multimodal(fake: FakeQdrant) -> None:
    adapter = make_adapter(fake)
    create_collection(adapter)
    adapter.upsert(
        [
            _record(id="a", uri="viking://resources/a", vector=[1, 0, 0, 0]),
            _record(id="b", uri="viking://resources/b", vector=[0, 1, 0, 0]),
        ]
    )
    similar = adapter.get_collection().search_by_id(
        "default", "a", limit=2, output_fields=["id"]
    )
    assert similar.data[0].id == "a"
    with pytest.raises(NotImplementedError, match="multimodal"):
        adapter.get_collection().search_by_multimodal("default", "a", None, None)

    adapter.get_collection().delete_all_data()
    assert adapter.count() == 0
    assert adapter.get_collection().search_by_id("default", "a").data == []


def test_description_persists_across_reload(fake: FakeQdrant) -> None:
    adapter = make_adapter(fake)
    create_collection(adapter)
    adapter.get_collection().update(description="described")
    reloaded = make_adapter(fake)
    assert reloaded.collection_exists()
    assert reloaded.get_collection().get_meta_data()["Description"] == "described"


def test_create_collection_with_old_schema_then_get_meta_view(fake: FakeQdrant) -> None:
    from openviking_qdrant.tests.helpers import legacy_schema

    adapter = make_adapter(fake)
    create_collection(adapter, schema=legacy_schema("context"))
    meta = adapter.get_collection().get_meta_data()
    names = {field["FieldName"] for field in meta["Fields"]}
    assert {"md5", "acl_mode", "acl_direct_grants", "acl_inherited_grants"} <= names
    assert {"acl_mode", "acl_direct_grants", "acl_inherited_grants"} <= set(
        meta["ScalarIndex"]
    )
    # The stored marker keeps the original legacy schema.
    marker = fake.collections["proj__context__openviking_meta"]["points"][META_MARKER_ID]
    stored_names = {field["FieldName"] for field in marker["payload"]["schema"]["Fields"]}
    assert "md5" not in stored_names
