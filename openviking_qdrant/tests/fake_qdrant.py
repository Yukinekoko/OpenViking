# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""In-memory Qdrant REST transport used by the offline test-suite.

This is intentionally independent from the production filter compiler: tests
assert end-to-end adapter behaviour against a second implementation of the
Qdrant filter and scoring semantics.
"""

from __future__ import annotations

import copy
import io
import json
import math
from typing import Any
from urllib.error import HTTPError
from urllib.parse import parse_qs, unquote, urlsplit


class _Response:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *_args: Any) -> bool:
        return False


class FakeQdrant:
    """Small stateful Qdrant HTTP double."""

    def __init__(self, *, max_query_limit: int | None = None) -> None:
        self.collections: dict[str, dict[str, Any]] = {}
        self.requests: list[dict[str, Any]] = []
        self.max_query_limit = max_query_limit

    # ------------------------------------------------------------------
    # transport
    # ------------------------------------------------------------------

    def __call__(self, request: Any, timeout: float) -> _Response:
        parsed = urlsplit(request.full_url)
        path = unquote(parsed.path)
        params = parse_qs(parsed.query)
        body = json.loads(request.data.decode("utf-8")) if request.data else None
        self.requests.append(
            {
                "method": request.method,
                "path": path,
                "params": params,
                "body": body,
                "timeout": timeout,
                "headers": dict(request.header_items()),
            }
        )
        try:
            payload = self._dispatch(request.method, path, body)
        except _FakeError as exc:
            raise HTTPError(
                request.full_url,
                exc.status,
                exc.message,
                {},
                io.BytesIO(json.dumps({"status": {"error": exc.message}}).encode("utf-8")),
            ) from exc
        return _Response({"result": payload, "status": "ok", "time": 0.0})

    def _dispatch(self, method: str, path: str, body: Any) -> Any:
        parts = [part for part in path.split("/") if part]
        if len(parts) < 2 or parts[0] != "collections":
            raise _FakeError(404, f"unknown path {path}")
        name = parts[1]
        suffix = parts[2:]
        if not suffix:
            if method == "GET":
                self._require(name)
                return {"status": "green", "points_count": len(self.collections[name]["points"])}
            if method == "PUT":
                return self._create_collection(name, body or {})
            if method == "DELETE":
                self._require(name)
                del self.collections[name]
                return True
            raise _FakeError(405, "method not allowed")
        if suffix == ["index"] and method == "PUT":
            return self._create_index(name, body or {})
        if len(suffix) == 2 and suffix[0] == "index" and method == "DELETE":
            return self._delete_index(name, suffix[1])
        if suffix == ["points"]:
            if method == "PUT":
                return self._upsert(name, body or {})
            if method == "POST":
                return self._retrieve(name, body or {})
            raise _FakeError(405, "method not allowed")
        if suffix == ["points", "scroll"] and method == "POST":
            return self._scroll(name, body or {})
        if suffix == ["points", "query"] and method == "POST":
            return self._query(name, body or {})
        if suffix == ["points", "count"] and method == "POST":
            return self._count(name, body or {})
        if suffix == ["points", "delete"] and method == "POST":
            return self._delete(name, body or {})
        raise _FakeError(404, f"unknown path {path}")

    # ------------------------------------------------------------------
    # collection / index management
    # ------------------------------------------------------------------

    def _require(self, name: str) -> dict[str, Any]:
        if name not in self.collections:
            raise _FakeError(404, f"collection {name} not found")
        return self.collections[name]

    def _create_collection(self, name: str, body: dict[str, Any]) -> bool:
        if name in self.collections:
            raise _FakeError(409, f"collection {name} already exists")
        vectors_raw = body.get("vectors") or {}
        vectors: dict[str, dict[str, Any]] = {}
        if isinstance(vectors_raw, dict) and "size" in vectors_raw:
            vectors[""] = dict(vectors_raw)
        else:
            for vector_name, config in vectors_raw.items():
                if not isinstance(config, dict) or "size" not in config:
                    raise _FakeError(400, "invalid vector config")
                vectors[vector_name] = dict(config)
        sparse = body.get("sparse_vectors") or {}
        if not isinstance(sparse, dict):
            raise _FakeError(400, "invalid sparse_vectors config")
        self.collections[name] = {
            "vectors": vectors,
            "sparse_vectors": {key: dict(value or {}) for key, value in sparse.items()},
            "points": {},
            "indexes": {},
        }
        return True

    def _create_index(self, name: str, body: dict[str, Any]) -> dict[str, Any]:
        collection = self._require(name)
        field = body.get("field_name")
        if not isinstance(field, str) or not field:
            raise _FakeError(400, "field_name is required")
        if field in collection["indexes"]:
            raise _FakeError(409, f"index for {field} already exists")
        collection["indexes"][field] = body.get("field_schema")
        return {"status": "completed"}

    def _delete_index(self, name: str, field: str) -> dict[str, Any]:
        collection = self._require(name)
        if field not in collection["indexes"]:
            raise _FakeError(404, f"index for {field} not found")
        del collection["indexes"][field]
        return {"status": "completed"}

    # ------------------------------------------------------------------
    # points
    # ------------------------------------------------------------------

    def _upsert(self, name: str, body: dict[str, Any]) -> dict[str, Any]:
        collection = self._require(name)
        points = body.get("points")
        if not isinstance(points, list):
            raise _FakeError(400, "points must be a list")
        for point in points:
            if not isinstance(point, dict) or "id" not in point:
                raise _FakeError(400, "point id is required")
            vectors = point.get("vector") or {}
            if not isinstance(vectors, dict):
                raise _FakeError(400, "point vector must be a named-vector object")
            for vector_name, values in vectors.items():
                if vector_name in collection["sparse_vectors"]:
                    self._validate_sparse(values)
                    continue
                config = collection["vectors"].get(vector_name)
                if config is None:
                    raise _FakeError(400, f"unknown vector name {vector_name}")
                if not isinstance(values, list) or len(values) != int(config["size"]):
                    raise _FakeError(
                        400,
                        f"vector {vector_name} has the wrong dimension",
                    )
                if not all(isinstance(value, (int, float)) for value in values):
                    raise _FakeError(400, "vector values must be numbers")
            collection["points"][str(point["id"])] = copy.deepcopy(point)
        return {"status": "completed"}

    @staticmethod
    def _validate_sparse(values: Any) -> None:
        if not isinstance(values, dict):
            raise _FakeError(400, "sparse vector must be an object")
        indices = values.get("indices")
        weights = values.get("values")
        if (
            not isinstance(indices, list)
            or not isinstance(weights, list)
            or len(indices) != len(weights)
        ):
            raise _FakeError(400, "sparse indices/values must have equal length")

    def _retrieve(self, name: str, body: dict[str, Any]) -> list[dict[str, Any]]:
        collection = self._require(name)
        ids = body.get("ids")
        if not isinstance(ids, list):
            raise _FakeError(400, "ids must be a list")
        with_payload = body.get("with_payload", True)
        with_vector = body.get("with_vector", False)
        points = []
        for point_id in ids:
            point = collection["points"].get(str(point_id))
            if point is not None:
                points.append(
                    self._render_point(
                        point,
                        with_payload=with_payload,
                        with_vector=with_vector,
                    )
                )
        return points

    def _scroll(self, name: str, body: dict[str, Any]) -> dict[str, Any]:
        collection = self._require(name)
        limit = int(body.get("limit", 10))
        if limit <= 0:
            return {"points": [], "next_page_offset": None}
        with_payload = body.get("with_payload", True)
        with_vector = body.get("with_vector", False)
        points = [
            point
            for point in collection["points"].values()
            if _filter_matches(body.get("filter"), point.get("payload") or {})
        ]
        order_by = body.get("order_by")
        if isinstance(order_by, dict):
            field = order_by.get("key")
            direction = str(order_by.get("direction", "asc")).lower()
            if not isinstance(field, str):
                raise _FakeError(400, "order_by key is required")
            points.sort(
                key=lambda point: _sort_key((point.get("payload") or {}).get(field)),
                reverse=direction == "desc",
            )
        else:
            points.sort(key=lambda point: str(point.get("id")))

        cursor = body.get("offset")
        start = 0
        if cursor is not None:
            rendered_ids = [str(point.get("id")) for point in points]
            if str(cursor) in rendered_ids:
                start = rendered_ids.index(str(cursor)) + 1
        page = points[start : start + limit]
        next_offset = (
            str(page[-1].get("id"))
            if page and start + len(page) < len(points)
            else None
        )
        return {
            "points": [
                self._render_point(
                    point, with_payload=with_payload, with_vector=with_vector
                )
                for point in page
            ],
            "next_page_offset": next_offset,
        }

    def _query(self, name: str, body: dict[str, Any]) -> dict[str, Any]:
        collection = self._require(name)
        limit = int(body.get("limit", 10))
        if self.max_query_limit is not None and limit > self.max_query_limit:
            raise _FakeError(
                400,
                f"max_query_limit exceeded: {limit} > {self.max_query_limit}",
            )
        offset = int(body.get("offset", 0))
        using = body.get("using") or ""
        query = body.get("query")
        with_payload = body.get("with_payload", True)
        with_vector = body.get("with_vector", False)
        scored: list[tuple[float, dict[str, Any]]] = []
        for point in collection["points"].values():
            payload = point.get("payload") or {}
            if not _filter_matches(body.get("filter"), payload):
                continue
            vectors = point.get("vector") or {}
            score = self._score(collection, vectors.get(using), query, using)
            if score is None:
                continue
            scored.append((score, point))
        scored.sort(key=lambda item: item[0], reverse=True)
        page = scored[offset : offset + limit]
        return {
            "points": [
                {
                    **self._render_point(
                        point, with_payload=with_payload, with_vector=with_vector
                    ),
                    "score": score,
                    "version": 0,
                }
                for score, point in page
            ]
        }

    def _score(
        self,
        collection: dict[str, Any],
        stored: Any,
        query: Any,
        using: str,
    ) -> float | None:
        if stored is None or query is None:
            return None
        if using in collection["sparse_vectors"]:
            if not isinstance(query, dict) or not isinstance(stored, dict):
                raise _FakeError(400, "invalid sparse query")
            stored_weights = dict(
                zip(stored.get("indices", []), stored.get("values", []), strict=False)
            )
            return float(
                sum(
                    stored_weights.get(int(index), 0.0) * float(value)
                    for index, value in zip(
                        query.get("indices", []), query.get("values", []), strict=False
                    )
                )
            )
        config = collection["vectors"].get(using)
        if config is None:
            return None
        if not isinstance(stored, list) or not isinstance(query, list):
            raise _FakeError(400, "invalid dense query")
        if len(stored) != int(config["size"]) or len(query) != int(config["size"]):
            raise _FakeError(400, "query vector has the wrong dimension")
        distance = str(config.get("distance", "Cosine")).lower()
        if distance == "cosine":
            denominator = _norm(stored) * _norm(query)
            return _dot(stored, query) / denominator if denominator else 0.0
        if distance == "dot":
            return _dot(stored, query)
        if distance == "euclid":
            return -math.dist(stored, query)
        if distance == "manhattan":
            return -sum(abs(a - b) for a, b in zip(stored, query, strict=True))
        raise _FakeError(400, f"unsupported distance {distance}")

    def _count(self, name: str, body: dict[str, Any]) -> dict[str, Any]:
        collection = self._require(name)
        count = sum(
            1
            for point in collection["points"].values()
            if _filter_matches(body.get("filter"), point.get("payload") or {})
        )
        return {"count": count}

    def _delete(self, name: str, body: dict[str, Any]) -> dict[str, Any]:
        collection = self._require(name)
        points = body.get("points")
        if isinstance(points, list):
            for point_id in points:
                collection["points"].pop(str(point_id), None)
            return {"status": "completed"}
        filters = body.get("filter")
        if filters is None:
            raise _FakeError(400, "delete requires points or filter")
        for point_id, point in list(collection["points"].items()):
            if _filter_matches(filters, point.get("payload") or {}):
                del collection["points"][point_id]
        return {"status": "completed"}

    @staticmethod
    def _render_point(
        point: dict[str, Any],
        *,
        with_payload: Any,
        with_vector: bool,
    ) -> dict[str, Any]:
        rendered: dict[str, Any] = {"id": point.get("id")}
        if with_payload is True:
            rendered["payload"] = copy.deepcopy(point.get("payload") or {})
        elif isinstance(with_payload, dict):
            include = with_payload.get("include") or []
            payload = point.get("payload") or {}
            rendered["payload"] = {
                field: copy.deepcopy(payload[field])
                for field in include
                if field in payload
            }
        if with_vector:
            rendered["vector"] = copy.deepcopy(point.get("vector") or {})
        return rendered

    # ------------------------------------------------------------------
    # test helpers
    # ------------------------------------------------------------------

    def seed(self, name: str, *, vectors: dict[str, Any], sparse: bool = False) -> None:
        self.collections[name] = {
            "vectors": vectors,
            "sparse_vectors": {"sparse_vector": {}} if sparse else {},
            "points": {},
            "indexes": {},
        }

    def seed_point(
        self,
        name: str,
        *,
        point_id: str,
        vector: dict[str, Any],
        payload: dict[str, Any],
    ) -> None:
        self._require(name)["points"][point_id] = {
            "id": point_id,
            "vector": copy.deepcopy(vector),
            "payload": copy.deepcopy(payload),
        }

    def requests_to(self, path_suffix: str) -> list[dict[str, Any]]:
        return [
            request
            for request in self.requests
            if request["path"].endswith(path_suffix)
        ]


class _FakeError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


def _norm(values: list[float]) -> float:
    return math.sqrt(sum(float(value) * float(value) for value in values))


def _dot(left: list[float], right: list[float]) -> float:
    return sum(float(a) * float(b) for a, b in zip(left, right, strict=True))


def _sort_key(value: Any) -> tuple[int, Any]:
    if value is None:
        return (1, "")
    return (0, value)


# ---------------------------------------------------------------------------
# Qdrant filter evaluation
# ---------------------------------------------------------------------------


def _filter_matches(qdrant_filter: Any, payload: dict[str, Any]) -> bool:
    if not qdrant_filter:
        return True
    if not isinstance(qdrant_filter, dict):
        raise _FakeError(400, "filter must be an object")
    must = qdrant_filter.get("must")
    if must is not None and not all(
        _condition_matches(condition, payload) for condition in must
    ):
        return False
    should = qdrant_filter.get("should")
    if should is not None and not any(
        _condition_matches(condition, payload) for condition in should
    ):
        return False
    must_not = qdrant_filter.get("must_not")
    if must_not is not None and any(
        _condition_matches(condition, payload) for condition in must_not
    ):
        return False
    min_should = qdrant_filter.get("min_should")
    if min_should:
        conditions = min_should.get("conditions") or []
        matched = sum(
            1 for condition in conditions if _condition_matches(condition, payload)
        )
        if matched < int(min_should.get("min_count", 1)):
            return False
    return True


def _condition_matches(condition: Any, payload: dict[str, Any]) -> bool:
    if not isinstance(condition, dict):
        raise _FakeError(400, "condition must be an object")
    if "key" not in condition:
        return _filter_matches(condition, payload)
    value = payload.get(condition["key"])
    match = condition.get("match")
    if isinstance(match, dict):
        if "value" in match:
            return _value_match(value, match["value"])
        if "any" in match:
            return _value_match_any(value, match["any"])
        if "except" in match:
            return not _value_match_any(value, match["except"])
        if "text" in match:
            if isinstance(value, list):
                return any(
                    isinstance(item, str) and match["text"] in item for item in value
                )
            return isinstance(value, str) and match["text"] in value
        raise _FakeError(400, f"unsupported match {match}")
    if "range" in condition:
        return _range_matches(value, condition["range"])
    if "is_empty" in condition:
        return value is None or value == [] or value == ""
    if "is_null" in condition:
        return value is None
    if "values_count" in condition:
        count = len(value) if isinstance(value, (list, str)) else 0
        return _range_matches(count, condition["values_count"])
    raise _FakeError(400, f"unsupported condition {condition}")


def _value_match(stored: Any, expected: Any) -> bool:
    if isinstance(stored, list):
        return expected in stored
    return stored == expected


def _value_match_any(stored: Any, expected: Any) -> bool:
    if not isinstance(expected, list):
        raise _FakeError(400, "match.any requires a list")
    if isinstance(stored, list):
        return any(item in expected for item in stored)
    return stored in expected


def _range_matches(stored: Any, bounds: dict[str, Any]) -> bool:
    if isinstance(stored, list):
        return any(_range_matches(item, bounds) for item in stored)
    if stored is None:
        return False
    try:
        for key, bound in bounds.items():
            if key == "gte" and not stored >= bound:
                return False
            if key == "gt" and not stored > bound:
                return False
            if key == "lte" and not stored <= bound:
                return False
            if key == "lt" and not stored < bound:
                return False
    except TypeError:
        return False
    return True
