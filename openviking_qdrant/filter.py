# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Compile OpenViking filter expressions into Qdrant filter JSON.

Two input shapes are accepted:

* the typed ``openviking.storage.expr`` AST (``And``, ``Or``, ``Eq``, ``In``,
  ``Range``, ``TimeRange``, ``PathScope``, ``Contains``, ``RawDSL``);
* Qdrant-native filter fragments or OpenViking's legacy ``{"op": ...}`` DSL
  embedded in ``RawDSL``/plain dictionaries.

The only AST node without a faithful server-side Qdrant equivalent is
``Contains``.  :func:`compile_qdrant_filter` rejects it explicitly; the adapter
uses :func:`filter_has_contains` to route filter-only queries to a client-side
scan instead of silently changing substring semantics.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any, Iterable

from openviking.storage.expr import (
    And,
    Contains,
    Eq,
    FilterExpr,
    In,
    Or,
    PathScope,
    Range,
    RawDSL,
    TimeRange,
)

from .utils import URI_FIELDS, normalize_uri_path, path_depth, to_qdrant_value

_FILTER_KEYS = frozenset({"must", "should", "must_not", "min_should"})
_CONTAINS_MESSAGE = (
    "Contains has no faithful Qdrant server-side equivalent; the Qdrant adapter "
    "supports it only for filter-only queries via a client-side full scan, "
    "and rejects it for vector/scalar search"
)


def _never_match_filter() -> dict[str, Any]:
    """A sentinel filter that matches no records.

    ``_openviking_never_match_`` is never written into a Qdrant payload, so an
    exact match against it is always false. This expresses an empty ``Or``/
    ``In`` faithfully instead of silently turning a contradiction into a full
    scan.
    """
    return {
        "must": [
            {"key": "_openviking_never_match_", "match": {"value": "__never__"}}
        ]
    }


def compile_qdrant_filter(expr: FilterExpr | dict[str, Any] | None) -> dict[str, Any]:
    """Compile an OpenViking filter into Qdrant filter JSON."""
    if expr is None:
        return {}
    return _compile(expr)


def filter_has_contains(expr: FilterExpr | dict[str, Any] | None) -> bool:
    """Return whether an expression needs the client-side ``Contains`` fallback."""
    if expr is None:
        return False
    if isinstance(expr, Contains):
        return True
    if isinstance(expr, (And, Or)):
        return any(filter_has_contains(item) for item in expr.conds)
    if isinstance(expr, RawDSL):
        return _raw_has_contains(expr.payload)
    if isinstance(expr, dict):
        return _raw_has_contains(expr)
    return False


def _raw_has_contains(payload: Any) -> bool:
    if isinstance(payload, dict):
        op = payload.get("op")
        if isinstance(op, str) and op.lower() == "contains":
            return True
        return any(_raw_has_contains(value) for value in payload.values())
    if isinstance(payload, list):
        return any(_raw_has_contains(value) for value in payload)
    return False


def _compile(expr: Any) -> dict[str, Any]:
    if isinstance(expr, dict):
        if not expr:
            return {}
        if "op" in expr:
            return _compile_legacy(expr)
        if any(key in expr for key in _FILTER_KEYS):
            return _normalize_native_filter(expr)
        if "key" in expr:
            return {"must": [_normalize_native_condition(expr)]}
        raise NotImplementedError(f"Unsupported raw Qdrant filter payload: {expr!r}")
    if isinstance(expr, RawDSL):
        return _compile(expr.payload)
    if isinstance(expr, And):
        clauses = [_compile(item) for item in expr.conds if item is not None]
        clauses = [item for item in clauses if item]
        if not clauses:
            return {}
        if len(clauses) == 1:
            return clauses[0]
        merged: dict[str, list[Any]] = {}
        nested: list[dict[str, Any]] = []
        for item in clauses:
            if set(item) <= {"must", "must_not"}:
                for key, value in item.items():
                    merged.setdefault(key, []).extend(value)
            else:
                nested.append(item)
        if nested:
            # Keep complex clauses as nested filters so a nested `should` cannot
            # be accidentally promoted to an optional top-level clause.
            merged.setdefault("must", []).extend(nested)
        return merged
    if isinstance(expr, Or):
        clauses = [_compile(item) for item in expr.conds if item is not None]
        if any(not item for item in clauses):
            # Or(TRUE, ...) is TRUE.
            return {}
        clauses = [item for item in clauses if item]
        if not clauses:
            return _never_match_filter()
        if len(clauses) == 1:
            return clauses[0]
        return {"should": [_or_condition(clause) for clause in clauses]}
    if isinstance(expr, Eq):
        value = _normalize_uri_value(expr.field, expr.value)
        return {"must": [_match(expr.field, value)]}
    if isinstance(expr, In):
        values = [_normalize_uri_value(expr.field, value) for value in expr.values]
        if not values:
            return _never_match_filter()
        if len(values) == 1:
            return {"must": [_match(expr.field, values[0])]}
        return {"must": [_match_any(expr.field, values)]}
    if isinstance(expr, Range):
        bounds = _range_bounds(expr.gte, expr.gt, expr.lte, expr.lt)
        if not bounds:
            raise ValueError("A Range filter requires at least one bound")
        return {"must": [{"key": expr.field, "range": bounds}]}
    if isinstance(expr, TimeRange):
        bounds: dict[str, Any] = {}
        if expr.start is not None:
            bounds["gte"] = to_qdrant_value(expr.start)
        if expr.end is not None:
            bounds["lt"] = to_qdrant_value(expr.end)
        if not bounds:
            raise ValueError("A TimeRange filter requires start or end")
        return {"must": [{"key": expr.field, "range": bounds}]}
    if isinstance(expr, PathScope):
        return _compile_path_scope(expr)
    if isinstance(expr, Contains):
        raise NotImplementedError(_CONTAINS_MESSAGE)
    raise TypeError(f"Unsupported filter expr type: {type(expr)!r}")


def _normalize_native_filter(payload: dict[str, Any]) -> dict[str, Any]:
    unknown = sorted(set(payload) - _FILTER_KEYS)
    if unknown:
        raise NotImplementedError(
            f"Unsupported Qdrant filter keys in RawDSL: {unknown}"
        )
    result: dict[str, Any] = {}
    for key in ("must", "must_not", "should"):
        if key not in payload:
            continue
        raw_conditions = payload[key]
        if not isinstance(raw_conditions, list):
            raise TypeError(f"Qdrant filter '{key}' must be a list")
        conditions = [_compile_condition(item) for item in raw_conditions if item is not None]
        conditions = [item for item in conditions if item]
        if not conditions:
            if key == "should":
                return _never_match_filter()
            continue
        result[key] = conditions
    if "min_should" in payload:
        result["min_should"] = _normalize_min_should(payload["min_should"])
    return result


def _normalize_min_should(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError("Qdrant min_should must be an object")
    conditions = payload.get("conditions")
    min_count = payload.get("min_count")
    if not isinstance(conditions, list) or not conditions:
        raise ValueError("Qdrant min_should.conditions must be a non-empty list")
    if not isinstance(min_count, int) or min_count <= 0:
        raise ValueError("Qdrant min_should.min_count must be a positive integer")
    normalized = [
        _compile_condition(item) for item in conditions if item is not None
    ]
    normalized = [item for item in normalized if item]
    if len(normalized) < min_count:
        raise ValueError("Qdrant min_should.min_count exceeds the condition count")
    return {"conditions": normalized, "min_count": min_count}


def _compile_condition(node: Any) -> dict[str, Any]:
    if not isinstance(node, dict):
        raise TypeError(f"Qdrant filter conditions must be objects, got {type(node)!r}")
    if "op" in node:
        fragment = _compile_legacy(node)
        # A legacy `must`/`must_not` fragment has exactly one key; flatten it so
        # callers do not build unnecessarily nested filters.
        keys = list(fragment)
        if len(keys) == 1 and isinstance(fragment[keys[0]], list) and len(fragment[keys[0]]) == 1:
            return fragment[keys[0]][0]
        return fragment
    if any(key in node for key in _FILTER_KEYS):
        return _normalize_native_filter(node)
    if "key" in node:
        return _normalize_native_condition(node)
    raise NotImplementedError(f"Unsupported Qdrant filter condition: {node!r}")


def _normalize_native_condition(node: dict[str, Any]) -> dict[str, Any]:
    condition = dict(node)
    field = condition.get("key")
    match = condition.get("match")
    if isinstance(match, dict):
        match = dict(match)
        for key in ("value", "any", "except"):
            if key not in match:
                continue
            if field in URI_FIELDS and isinstance(match[key], list):
                match[key] = [normalize_uri_path(value) for value in match[key]]
            elif field in URI_FIELDS:
                match[key] = normalize_uri_path(match[key])
            elif isinstance(match[key], list):
                match[key] = [to_qdrant_value(value) for value in match[key]]
            else:
                match[key] = to_qdrant_value(match[key])
        condition["match"] = match
    rng = condition.get("range")
    if isinstance(rng, dict):
        condition["range"] = {
            key: to_qdrant_value(value) for key, value in rng.items()
        }
    return condition


def _compile_legacy(payload: dict[str, Any]) -> dict[str, Any]:
    op = str(payload.get("op") or "").lower()
    if op == "and":
        return _compile(And([item for item in payload.get("conds", []) if item]))
    if op == "or":
        return _compile(Or([item for item in payload.get("conds", []) if item]))
    if op in {"must", "must_not"}:
        field = payload.get("field")
        raw_values = payload.get("conds")
        if not isinstance(field, str):
            raise ValueError(f"Legacy {op} filter requires a string field")
        if raw_values is None:
            values: list[Any] = []
        elif isinstance(raw_values, list):
            values = list(raw_values)
        else:
            values = [raw_values]
        if not values:
            return {}
        depth = _legacy_depth(payload)
        if depth is not None and field == "uri":
            scope = PathScope(field, values[0], depth=depth)
            fragment = _compile_path_scope(scope)
            if op == "must_not":
                return {"must_not": [fragment]}
            return fragment
        normalized = [_normalize_uri_value(field, value) for value in values]
        condition = (
            _match(field, normalized[0])
            if len(normalized) == 1
            else _match_any(field, normalized)
        )
        return {op: [condition]}
    if op in {"range", "time_range"}:
        field = payload.get("field")
        if not isinstance(field, str):
            raise ValueError(f"Legacy {op} filter requires a string field")
        bounds = {
            key: to_qdrant_value(payload[key])
            for key in ("gte", "gt", "lte", "lt")
            if payload.get(key) is not None
        }
        if not bounds:
            raise ValueError(f"Legacy {op} filter requires at least one bound")
        return {"must": [{"key": field, "range": bounds}]}
    if op == "range_out":
        field = payload.get("field")
        if not isinstance(field, str):
            raise ValueError("Legacy range_out filter requires a string field")
        branches: list[FilterExpr] = []
        if payload.get("gte") is not None:
            branches.append(Range(field, lt=payload["gte"]))
        if payload.get("lte") is not None:
            branches.append(Range(field, gt=payload["lte"]))
        if not branches:
            raise ValueError("Legacy range_out filter requires gte or lte")
        return _compile(Or(branches))
    if op == "prefix":
        field = payload.get("field")
        if field != "uri":
            raise NotImplementedError(
                "Qdrant adapter only supports prefix filters on the uri field"
            )
        prefix = payload.get("prefix", "")
        return _compile(PathScope("uri", prefix, depth=-1))
    if op == "contains":
        raise NotImplementedError(_CONTAINS_MESSAGE)
    if op:
        raise NotImplementedError(f"Unsupported legacy filter operation: {op}")
    raise NotImplementedError(f"Unsupported raw filter payload: {payload!r}")


def _legacy_depth(payload: dict[str, Any]) -> int | None:
    marker = payload.get("para")
    if not isinstance(marker, str) or not marker.startswith("-d="):
        return None
    try:
        return int(marker[3:])
    except ValueError:
        raise ValueError(f"Invalid legacy path depth: {marker!r}") from None


def _compile_path_scope(expr: PathScope) -> dict[str, Any]:
    if expr.field != "uri":
        raise NotImplementedError(
            "Qdrant PathScope supports the uri field only; parent_uri has no "
            "scope payload"
        )
    if not isinstance(expr.path, str):
        raise ValueError("Qdrant URI path scope requires a string URI path")
    if expr.depth < -1:
        raise ValueError("PathScope depth must be -1, 0, or a positive integer")
    path = normalize_uri_path(expr.path)
    if expr.depth == 0:
        return {"must": [_match("uri", path)]}
    condition = _match("scope_roots", path)
    if expr.depth < 0:
        return {"must": [condition]}
    return {
        "must": [
            condition,
            {
                "key": "uri_depth",
                "range": {"lte": path_depth(path) + expr.depth},
            },
        ]
    }


def _or_condition(clause: dict[str, Any]) -> dict[str, Any]:
    """Flatten a positive single-condition clause inside an Or."""
    if len(clause) == 1:
        key = next(iter(clause))
        values = clause[key]
        if key in {"must", "should"} and isinstance(values, list) and len(values) == 1:
            return values[0]
    return clause


def _match(field: str, value: Any) -> dict[str, Any]:
    return {"key": field, "match": {"value": to_qdrant_value(value)}}


def _match_any(field: str, values: Iterable[Any]) -> dict[str, Any]:
    return {
        "key": field,
        "match": {"any": [to_qdrant_value(value) for value in values]},
    }


def _range_bounds(
    gte: Any = None,
    gt: Any = None,
    lte: Any = None,
    lt: Any = None,
) -> dict[str, Any]:
    return {
        key: to_qdrant_value(value)
        for key, value in (("gte", gte), ("gt", gt), ("lte", lte), ("lt", lt))
        if value is not None
    }


def _normalize_uri_value(field: str, value: Any) -> Any:
    return normalize_uri_path(value) if field in URI_FIELDS else value


# ---------------------------------------------------------------------------
# Python-side evaluation (Contains fallback and test helpers)
# ---------------------------------------------------------------------------


def filter_matches_record(
    expr: FilterExpr | dict[str, Any] | None,
    record: dict[str, Any],
) -> bool:
    """Evaluate an OpenViking filter against one decoded payload record."""
    if expr is None:
        return True
    if isinstance(expr, dict):
        if filter_has_contains(expr):
            return _raw_matches_record(expr, record)
        return qdrant_filter_matches(compile_qdrant_filter(expr), record)
    if isinstance(expr, RawDSL):
        if filter_has_contains(expr):
            return _raw_matches_record(expr.payload, record)
        return qdrant_filter_matches(compile_qdrant_filter(expr), record)
    if isinstance(expr, And):
        return all(filter_matches_record(item, record) for item in expr.conds)
    if isinstance(expr, Or):
        return any(filter_matches_record(item, record) for item in expr.conds)
    if isinstance(expr, Eq):
        return _value_match(
            _record_value(record, expr.field),
            _normalize_uri_value(expr.field, expr.value),
        )
    if isinstance(expr, In):
        return any(
            _value_match(
                _record_value(record, expr.field),
                _normalize_uri_value(expr.field, value),
            )
            for value in expr.values
        )
    if isinstance(expr, Range):
        return _range_matches(
            _record_value(record, expr.field),
            _range_bounds(expr.gte, expr.gt, expr.lte, expr.lt),
        )
    if isinstance(expr, TimeRange):
        bounds = {}
        if expr.start is not None:
            bounds["gte"] = to_qdrant_value(expr.start)
        if expr.end is not None:
            bounds["lt"] = to_qdrant_value(expr.end)
        return _range_matches(_record_value(record, expr.field), bounds)
    if isinstance(expr, PathScope):
        return _path_scope_matches(expr, record)
    if isinstance(expr, Contains):
        substring = expr.substring
        if expr.field in URI_FIELDS:
            substring = normalize_uri_path(substring)
            if not isinstance(substring, str):
                return False
        return _contains_match(_record_value(record, expr.field), substring)
    raise TypeError(f"Unsupported filter expr type: {type(expr)!r}")


def _raw_matches_record(payload: dict[str, Any], record: dict[str, Any]) -> bool:
    """Evaluate a raw legacy/native filter when a Contains node is present."""
    if not isinstance(payload, dict):
        raise TypeError("raw filter payload must be an object")
    op = str(payload.get("op") or "").lower()
    if not op:
        return qdrant_filter_matches(compile_qdrant_filter(payload), record)
    if op == "and":
        return all(
            _raw_matches_record(item, record)
            for item in payload.get("conds", [])
            if isinstance(item, dict)
        )
    if op == "or":
        return any(
            _raw_matches_record(item, record)
            for item in payload.get("conds", [])
            if isinstance(item, dict)
        )
    if op in {"must", "must_not"}:
        field = payload.get("field")
        if not isinstance(field, str):
            raise ValueError(f"Legacy {op} filter requires a string field")
        raw_values = payload.get("conds")
        values = list(raw_values) if isinstance(raw_values, list) else [raw_values]
        values = [value for value in values if value is not None]
        if not values:
            return True
        depth = _legacy_depth(payload)
        if depth is not None and field == "uri":
            matched = all(
                _path_scope_matches(PathScope(field, value, depth=depth), record)
                for value in values
            )
        else:
            normalized = [_normalize_uri_value(field, value) for value in values]
            matched = _value_match_any(_record_value(record, field), normalized)
        return not matched if op == "must_not" else matched
    if op in {"range", "time_range"}:
        field = payload.get("field")
        if not isinstance(field, str):
            raise ValueError(f"Legacy {op} filter requires a string field")
        bounds = {
            key: to_qdrant_value(payload[key])
            for key in ("gte", "gt", "lte", "lt")
            if payload.get(key) is not None
        }
        return _range_matches(_record_value(record, field), bounds)
    if op == "range_out":
        field = payload.get("field")
        if not isinstance(field, str):
            raise ValueError("Legacy range_out filter requires a string field")
        matched_in = _range_matches(
            _record_value(record, field),
            {
                key: to_qdrant_value(payload[key])
                for key in ("gte", "lte")
                if payload.get(key) is not None
            },
        )
        return not matched_in
    if op == "prefix":
        field = payload.get("field")
        if field != "uri":
            raise NotImplementedError(
                "Qdrant adapter only supports prefix filters on the uri field"
            )
        prefix = _normalize_uri_value("uri", payload.get("prefix", ""))
        value = _record_value(record, "uri")
        if not isinstance(value, str):
            return False
        return _scope_prefix_match(normalize_uri_path(value), prefix)
    if op == "contains":
        field = payload.get("field")
        substring = payload.get("substring", payload.get("value", ""))
        if field in URI_FIELDS:
            substring = normalize_uri_path(substring)
        return _contains_match(_record_value(record, field), substring)
    raise NotImplementedError(f"Unsupported legacy filter operation: {op}")


def _scope_prefix_match(uri: str, prefix: str) -> bool:
    path = normalize_uri_path(prefix)
    return uri == path or uri.startswith(path.rstrip("/") + "/")


def _path_scope_matches(expr: PathScope, record: dict[str, Any]) -> bool:
    if expr.field != "uri":
        raise NotImplementedError("Qdrant PathScope supports the uri field only")
    value = _record_value(record, "uri")
    if not isinstance(value, str):
        return False
    path = normalize_uri_path(expr.path)
    uri = normalize_uri_path(value)
    roots = record.get("scope_roots")
    if not isinstance(roots, list):
        roots = _scope_roots(uri)
    depth = record.get("uri_depth")
    if not isinstance(depth, int):
        depth = path_depth(uri)
    if expr.depth == 0:
        return uri == path
    if path not in roots:
        return False
    if expr.depth < 0:
        return True
    return depth <= path_depth(path) + expr.depth


def _scope_roots(path: str) -> list[str]:
    parts = [part for part in path.split("/") if part]
    roots = ["/"]
    for index in range(1, len(parts) + 1):
        roots.append("/" + "/".join(parts[:index]))
    return roots


def qdrant_filter_matches(
    qdrant_filter: dict[str, Any] | None,
    record: dict[str, Any],
) -> bool:
    """Evaluate Qdrant filter JSON against a payload (used by Contains scans)."""
    if not qdrant_filter:
        return True
    must = qdrant_filter.get("must")
    if must is not None and not all(
        _condition_matches(condition, record) for condition in must
    ):
        return False
    should = qdrant_filter.get("should")
    if should is not None and not any(
        _condition_matches(condition, record) for condition in should
    ):
        return False
    must_not = qdrant_filter.get("must_not")
    if must_not is not None and any(
        _condition_matches(condition, record) for condition in must_not
    ):
        return False
    min_should = qdrant_filter.get("min_should")
    if min_should:
        matches = sum(
            1
            for condition in min_should.get("conditions", [])
            if _condition_matches(condition, record)
        )
        if matches < int(min_should.get("min_count", 1)):
            return False
    return True


def _condition_matches(condition: dict[str, Any], record: dict[str, Any]) -> bool:
    if "key" not in condition:
        if any(key in condition for key in _FILTER_KEYS):
            return qdrant_filter_matches(condition, record)
        raise NotImplementedError(f"Unsupported Qdrant condition: {condition!r}")
    field = condition["key"]
    value = _record_value(record, field)
    match = condition.get("match")
    if isinstance(match, dict):
        if "value" in match:
            return _value_match(value, match["value"])
        if "any" in match:
            return _value_match_any(value, match["any"])
        if "except" in match:
            return not _value_match_any(value, match["except"])
        if "text" in match:
            return _contains_match(value, match["text"])
        raise NotImplementedError(f"Unsupported Qdrant match condition: {match!r}")
    if "range" in condition:
        return _range_matches(value, condition["range"])
    if "is_empty" in condition:
        return value is None or value == "" or value == []
    if "is_null" in condition:
        return value is None
    if "values_count" in condition:
        count = len(value) if isinstance(value, (list, tuple, str)) else 0
        return _range_matches(count, condition["values_count"])
    raise NotImplementedError(f"Unsupported Qdrant condition: {condition!r}")


def _record_value(record: dict[str, Any], field: str) -> Any:
    return record.get(field)


def _value_match(stored: Any, expected: Any) -> bool:
    if isinstance(stored, (list, tuple, set)):
        return any(item == expected for item in stored)
    return stored == expected


def _value_match_any(stored: Any, expected: Iterable[Any]) -> bool:
    expected_list = list(expected)
    if isinstance(stored, (list, tuple, set)):
        return any(item in expected_list for item in stored)
    return stored in expected_list


def _range_matches(stored: Any, bounds: dict[str, Any]) -> bool:
    if isinstance(stored, (list, tuple, set)):
        return any(_range_matches(item, bounds) for item in stored)
    if stored is None:
        return False
    candidate = _comparable(stored)
    for key, raw_bound in bounds.items():
        bound = _comparable(raw_bound)
        try:
            if key == "gte" and not candidate >= bound:
                return False
            if key == "gt" and not candidate > bound:
                return False
            if key == "lte" and not candidate <= bound:
                return False
            if key == "lt" and not candidate < bound:
                return False
        except TypeError:
            return False
    return True


def _comparable(value: Any) -> Any:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (datetime, date)):
        return _datetime_ms(value)
    if isinstance(value, str) and len(value) >= 10 and value[4] == "-" and value[7] == "-":
        try:
            return _datetime_ms(datetime.fromisoformat(value.replace("Z", "+00:00")))
        except ValueError:
            return value
    return value


def _datetime_ms(value: datetime | date) -> float:
    if isinstance(value, date) and not isinstance(value, datetime):
        value = datetime(value.year, value.month, value.day, tzinfo=timezone.utc)
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.timestamp() * 1000.0


def _contains_match(stored: Any, substring: str) -> bool:
    if isinstance(stored, (list, tuple, set)):
        return any(_contains_match(item, substring) for item in stored)
    return isinstance(stored, str) and substring in stored


__all__ = [
    "compile_qdrant_filter",
    "filter_has_contains",
    "filter_matches_record",
    "qdrant_filter_matches",
]
