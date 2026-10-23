# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
from __future__ import annotations

import pytest

from openviking_qdrant.tests.fake_qdrant import FakeQdrant
from openviking_qdrant.tests.helpers import create_collection, make_adapter

ORIGIN = "2026-01-10T00:00:00Z"
ADVANCE = {"time_decay": {"protection": "2d", "origin": ORIGIN}}


def _record(record_id: str, score: float, updated_at: str | None) -> dict:
    record = {
        "id": record_id,
        "uri": f"viking://resources/{record_id}",
        "account_id": "acct",
        "level": 2,
        "vector": [score, 0.0, 0.0, 0.0],
    }
    if updated_at is not None:
        record["updated_at"] = updated_at
    return record


def _setup() -> tuple[FakeQdrant, object]:
    fake = FakeQdrant()
    adapter = make_adapter(fake, distance_metric="ip")
    create_collection(adapter)
    adapter.upsert(
        [
            _record("old-high", 1.0, "2026-01-01T00:00:00Z"),
            _record("fresh-mid", 0.9, "2026-01-10T00:00:00Z"),
            _record("protected", 0.8, "2026-01-09T00:00:00Z"),
            _record("missing", 0.7, None),
        ]
    )
    return fake, adapter


def test_time_decay_reorders_and_exposes_scores() -> None:
    fake, adapter = _setup()
    result = adapter.query(
        query_vector=[1, 0, 0, 0],
        limit=2,
        advance=ADVANCE,
        output_fields=["uri", "level"],
    )
    assert [item["id"] for item in result] == ["fresh-mid", "protected"]
    assert [item["_score"] for item in result] == pytest.approx([0.9, 0.8])
    assert [item["_origin_score"] for item in result] == pytest.approx([0.9, 0.8])
    assert [item["_time_score"] for item in result] == pytest.approx([1.0, 1.0])
    assert all("updated_at" not in item for item in result)

    # Candidate amplification matches the native engine: topk * 3, capped.
    query = fake.requests_to("/points/query")[-1]
    assert query["body"]["limit"] == 6

    page = adapter.query(
        query_vector=[1, 0, 0, 0],
        limit=1,
        offset=1,
        advance=ADVANCE,
    )
    assert [item["id"] for item in page] == ["protected"]

    all_results = adapter.query(
        query_vector=[1, 0, 0, 0], limit=4, advance=ADVANCE
    )
    assert [item["id"] for item in all_results] == [
        "fresh-mid",
        "protected",
        "missing",
        "old-high",
    ]
    assert "_time_score" not in all_results[2]
    assert all_results[3]["_time_score"] == pytest.approx(0.5)
    assert all_results[3]["_score"] == pytest.approx(0.5)


def test_time_decay_uses_absolute_distance_for_future_timestamps() -> None:
    fake = FakeQdrant()
    adapter = make_adapter(fake, distance_metric="ip")
    create_collection(adapter)
    adapter.upsert(
        [
            _record("future", 1.0, "2026-01-19T00:00:00Z"),
            _record("fresh", 0.9, "2026-01-10T00:00:00Z"),
        ]
    )
    result = adapter.query(
        query_vector=[1, 0, 0, 0],
        limit=2,
        advance={"time_decay": {"protection": "0", "origin": ORIGIN}},
    )
    assert result[0]["id"] == "fresh"
    assert result[1]["_time_score"] == pytest.approx(0.5 ** (9 / 7))


def test_time_decay_zero_protection_uses_full_curve() -> None:
    fake, adapter = _setup()
    result = adapter.query(
        query_vector=[1, 0, 0, 0],
        limit=4,
        advance={"time_decay": {"protection": "0", "origin": ORIGIN}},
    )
    by_id = {item["id"]: item for item in result}
    assert by_id["old-high"]["_time_score"] == pytest.approx(0.5 ** (9 / 7))
    assert by_id["fresh-mid"]["_time_score"] == pytest.approx(1.0)
    assert by_id["protected"]["_time_score"] == pytest.approx(
        0.5 ** (1 / 7)
    )


def test_time_decay_missing_timestamp_keeps_original_score() -> None:
    fake, adapter = _setup()
    result = adapter.query(
        query_vector=[1, 0, 0, 0],
        limit=1,
        offset=2,
        advance=ADVANCE,
    )
    assert result[0]["id"] == "missing"
    assert result[0]["_score"] == pytest.approx(0.7)
    assert result[0]["_origin_score"] == pytest.approx(0.7)
    assert "_time_score" not in result[0]


def test_compiled_time_decay_spec_is_accepted() -> None:
    fake, adapter = _setup()
    spec = {
        "time_decay": {
            "field": "updated_at",
            "origin_ms": 1768003200000.0,
            "offset_ms": 0,
            "scale_ms": 7 * 24 * 3600 * 1000,
            "decay": 0.5,
        }
    }
    result = adapter.query(query_vector=[1, 0, 0, 0], limit=1, advance=spec)
    assert result[0]["id"] == "fresh-mid"
    assert result[0]["_time_score"] == pytest.approx(1.0)


def test_collection_return_detail_info_sets_origin_and_addition_scores() -> None:
    fake, adapter = _setup()
    coll = adapter.get_collection()
    compiled = {
        "time_decay": {
            "field": "updated_at",
            "origin_ms": 1768003200000.0,
            "offset_ms": 0,
            "scale_ms": 7 * 24 * 3600 * 1000,
            "decay": 0.5,
        }
    }
    result = coll.search_by_vector(
        "default",
        dense_vector=[1, 0, 0, 0],
        limit=4,
        advance=compiled,
        return_detail_info=True,
    )
    by_id = {item.id: item for item in result.data}
    assert by_id["old-high"].origin_score == pytest.approx(1.0)
    assert by_id["old-high"].addition_score == pytest.approx(0.5 ** (9 / 7))
    assert by_id["fresh-mid"].addition_score == pytest.approx(1.0)

    plain = coll.search_by_vector(
        "default", dense_vector=[1, 0, 0, 0], limit=1
    )
    assert plain.data[0].origin_score is None
    assert plain.data[0].addition_score is None


def test_invalid_time_decay_requests_raise_clearly() -> None:
    fake, adapter = _setup()
    with pytest.raises(NotImplementedError, match="does not support"):
        adapter.query(
            query_vector=[1, 0, 0, 0],
            limit=1,
            advance={"post_process_ops": []},
        )
    with pytest.raises(ValueError, match="requires a query vector"):
        adapter.query(filter={"op": "must", "field": "level", "conds": [2]}, advance=ADVANCE)
    with pytest.raises(ValueError, match="scale_ms must be positive"):
        adapter.query(
            query_vector=[1, 0, 0, 0],
            limit=1,
            advance={
                "time_decay": {
                    "origin_ms": 1.0,
                    "offset_ms": 0,
                    "scale_ms": 0,
                    "decay": 0.5,
                }
            },
        )
    # out-of-range compiled decay
    with pytest.raises(ValueError, match="decay must be in"):
        adapter.get_collection().search_by_vector(
            "default",
            dense_vector=[1, 0, 0, 0],
            limit=1,
            advance={
                "time_decay": {
                    "origin_ms": 1.0,
                    "offset_ms": 0,
                    "scale_ms": 1000,
                    "decay": 1.0,
                }
            },
        )


def test_time_decay_with_hybrid_sparse_query() -> None:
    fake = FakeQdrant()
    adapter = make_adapter(fake, distance_metric="ip", sparse_weight=0.5)
    create_collection(adapter, sparse_weight=0.5)
    adapter.upsert(
        [
            {
                "id": "old",
                "uri": "viking://resources/old",
                "vector": [1.0, 0.0, 0.0, 0.0],
                "sparse_vector": {"alpha": 1.0},
                "updated_at": "2026-01-01T00:00:00Z",
            },
            {
                "id": "fresh",
                "uri": "viking://resources/fresh",
                "vector": [0.9, 0.0, 0.0, 0.0],
                "sparse_vector": {"alpha": 0.9},
                "updated_at": "2026-01-10T00:00:00Z",
            },
        ]
    )
    result = adapter.query(
        query_vector=[1, 0, 0, 0],
        sparse_query_vector={"alpha": 1.0},
        limit=2,
        advance=ADVANCE,
    )
    assert result[0]["id"] == "fresh"
    assert result[0]["_time_score"] == pytest.approx(1.0)
    assert result[1]["_time_score"] == pytest.approx(0.5)
