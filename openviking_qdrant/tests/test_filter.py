# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from openviking.storage.expr import (
    And,
    Contains,
    Eq,
    In,
    Or,
    PathScope,
    Range,
    RawDSL,
    TimeRange,
)
from openviking_qdrant.filter import (
    compile_qdrant_filter,
    filter_has_contains,
    filter_matches_record,
    qdrant_filter_matches,
)


def test_empty_filter_dict_is_match_all() -> None:
    assert compile_qdrant_filter({}) == {}
    assert filter_matches_record({}, {"uri": "/resources/a"}) is True


def test_path_scope_depths_are_segment_aware() -> None:
    assert compile_qdrant_filter(PathScope("uri", "viking://resources", depth=0)) == {
        "must": [{"key": "uri", "match": {"value": "/resources"}}]
    }
    assert compile_qdrant_filter(PathScope("uri", "viking://resources", depth=-1)) == {
        "must": [{"key": "scope_roots", "match": {"value": "/resources"}}]
    }
    assert compile_qdrant_filter(
        PathScope("uri", "viking://resources/wiki", depth=2)
    ) == {
        "must": [
            {"key": "scope_roots", "match": {"value": "/resources/wiki"}},
            {"key": "uri_depth", "range": {"lte": 4}},
        ]
    }


def test_parent_uri_scope_is_rejected() -> None:
    with pytest.raises(NotImplementedError, match="uri field only"):
        compile_qdrant_filter(PathScope("parent_uri", "viking://resources", depth=-1))


def test_eq_in_range_time_range_and_uri_normalization() -> None:
    assert compile_qdrant_filter(Eq("uri", "viking://resources/a.md")) == {
        "must": [{"key": "uri", "match": {"value": "/resources/a.md"}}]
    }
    assert compile_qdrant_filter(In("level", [0, 1])) == {
        "must": [{"key": "level", "match": {"any": [0, 1]}}]
    }
    assert compile_qdrant_filter(Range("level", gte=1, lt=3)) == {
        "must": [{"key": "level", "range": {"gte": 1, "lt": 3}}]
    }
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    end = datetime(2026, 1, 2, tzinfo=timezone.utc)
    payload = compile_qdrant_filter(TimeRange("updated_at", start=start, end=end))
    assert payload["must"][0]["key"] == "updated_at"
    assert payload["must"][0]["range"] == {
        "gte": start.isoformat(),
        "lt": end.isoformat(),
    }


def test_and_flattens_positive_clauses() -> None:
    compiled = compile_qdrant_filter(
        And(
            [
                Eq("account_id", "acct"),
                PathScope("uri", "viking://resources", depth=-1),
                Eq("search_tags", "env=prod"),
            ]
        )
    )
    assert compiled == {
        "must": [
            {"key": "account_id", "match": {"value": "acct"}},
            {"key": "scope_roots", "match": {"value": "/resources"}},
            {"key": "search_tags", "match": {"value": "env=prod"}},
        ]
    }


def test_or_keeps_boolean_structure() -> None:
    compiled = compile_qdrant_filter(
        And(
            [
                Eq("account_id", "acct"),
                Or([Eq("level", 0), Eq("level", 1)]),
            ]
        )
    )
    assert compiled == {
        "must": [
            {"key": "account_id", "match": {"value": "acct"}},
            {
                "should": [
                    {"key": "level", "match": {"value": 0}},
                    {"key": "level", "match": {"value": 1}},
                ]
            },
        ]
    }


def test_empty_or_matches_nothing_instead_of_everything() -> None:
    compiled = compile_qdrant_filter(Or([]))
    assert compiled["must"][0]["key"] == "_openviking_never_match_"
    assert filter_matches_record(Or([]), {"uri": "/resources/a"}) is False


def test_empty_in_matches_nothing() -> None:
    compiled = compile_qdrant_filter(In("level", []))
    assert compiled["must"][0]["key"] == "_openviking_never_match_"
    assert filter_matches_record(In("level", []), {"level": 2}) is False


def test_legacy_and_must_not_and_range_filters() -> None:
    compiled = compile_qdrant_filter(
        RawDSL(
            {
                "op": "and",
                "conds": [
                    {"op": "must", "field": "search_tags", "conds": ["team=search"]},
                    {
                        "op": "must_not",
                        "field": "search_tags",
                        "conds": ["memory_type=events"],
                    },
                    {"op": "range", "field": "level", "gte": 1, "lte": 2},
                ],
            }
        )
    )
    assert compiled["must"] == [
        {"key": "search_tags", "match": {"value": "team=search"}},
        {"key": "level", "range": {"gte": 1, "lte": 2}},
    ]
    assert compiled["must_not"] == [
        {"key": "search_tags", "match": {"value": "memory_type=events"}}
    ]


def test_legacy_path_scope_and_prefix() -> None:
    assert compile_qdrant_filter(
        RawDSL(
            {
                "op": "must_not",
                "field": "uri",
                "conds": ["viking://resources/private"],
                "para": "-d=-1",
            }
        )
    ) == {
        "must_not": [
            {"must": [{"key": "scope_roots", "match": {"value": "/resources/private"}}]}
        ]
    }
    assert compile_qdrant_filter(
        RawDSL({"op": "prefix", "field": "uri", "prefix": "viking://resources"})
    ) == {"must": [{"key": "scope_roots", "match": {"value": "/resources"}}]}


def test_legacy_range_out_compiles_to_or() -> None:
    compiled = compile_qdrant_filter(
        RawDSL({"op": "range_out", "field": "level", "gte": 1, "lte": 3})
    )
    assert compiled == {
        "should": [
            {"key": "level", "range": {"lt": 1}},
            {"key": "level", "range": {"gt": 3}},
        ]
    }


def test_native_raw_filter_is_normalized() -> None:
    compiled = compile_qdrant_filter(
        RawDSL(
            {
                "must": [{"key": "uri", "match": {"value": "viking://resources/a"}}],
                "must_not": [{"key": "acl_mode", "match": {"any": ["inherit"]}}],
                "should": [{"key": "level", "match": {"value": 2}}],
            }
        )
    )
    assert compiled == {
        "must": [{"key": "uri", "match": {"value": "/resources/a"}}],
        "must_not": [{"key": "acl_mode", "match": {"any": ["inherit"]}}],
        "should": [{"key": "level", "match": {"value": 2}}],
    }


def test_contains_is_explicitly_unsupported_for_server_compilation() -> None:
    expr = Contains("uri", "viking://resources/a#")
    assert filter_has_contains(expr)
    with pytest.raises(NotImplementedError, match="client-side full scan"):
        compile_qdrant_filter(expr)


def test_contains_fallback_normalizes_uri_and_evaluates_ast() -> None:
    record = {
        "uri": "/resources/a.md#fragment",
        "level": 2,
        "search_tags": ["env=prod", "team=search"],
        "acl_mode": "inherit",
    }
    assert filter_matches_record(Contains("uri", "viking://resources/a.md#"), record)
    assert filter_matches_record(
        And(
            [
                PathScope("uri", "viking://resources", depth=-1),
                In("search_tags", ["env=prod"]),
                Eq("acl_mode", "inherit"),
            ]
        ),
        record,
    )


def test_qdrant_filter_evaluator_handles_missing_acl_as_default_shared() -> None:
    default_shared = RawDSL(
        {"op": "must_not", "field": "acl_mode", "conds": ["inherit", "restricted"]}
    )
    assert filter_matches_record(default_shared, {"uri": "/resources/legacy"})
    assert filter_matches_record(default_shared, {"acl_mode": "none"})
    assert not filter_matches_record(default_shared, {"acl_mode": "inherit"})
    assert qdrant_filter_matches(
        {"must_not": [{"key": "acl_mode", "match": {"any": ["inherit", "restricted"]}}]},
        {},
    )
