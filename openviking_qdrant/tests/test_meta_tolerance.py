# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""``get_meta_data(raise_on_error=...)`` contract tests (P0 fix).

Upstream probes the collection metadata through ``raise_on_error=False`` (for
example ``_filter_known_fields``).  A Qdrant outage must therefore degrade to an
empty dict instead of raising into unrelated write paths; ``raise_on_error=True``
must still surface the failure.
"""

from __future__ import annotations

import io
import json
from urllib.error import URLError

import pytest

from openviking_qdrant.collection import QdrantCollection
from openviking_qdrant.rest import QdrantError, QdrantRestClient
from openviking_qdrant.tests.fake_qdrant import FakeQdrant
from openviking_qdrant.tests.helpers import create_collection, make_adapter


class _Response:
    def __init__(self, payload: dict) -> None:
        self._payload = json.dumps(payload).encode()

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *_args) -> bool:
        return False


class _ScriptedOpener:
    def __init__(self, *items) -> None:
        self.items = list(items)
        self.calls: list[str] = []

    def __call__(self, request, timeout=None):
        self.calls.append(f"{request.method} {request.full_url}")
        item = self.items.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def _collection(opener, *, max_retries: int = 0) -> QdrantCollection:
    client = QdrantRestClient(
        "http://qdrant.test:6333", opener=opener, max_retries=max_retries
    )
    return QdrantCollection(
        client=client,
        collection_name="proj__context",
        metadata_collection_name="proj__context__openviking_meta",
        dense_vector_name="vector",
        sparse_vector_name="sparse_vector",
        vector_dim=4,
        distance="cosine",
        sparse_enabled=False,
        sparse_weight=0.0,
        query_candidate_cap=100_000,
    )


def _missing_marker_opener() -> _ScriptedOpener:
    # metadata collection exists, but it holds no marker point
    return _ScriptedOpener(_Response({"result": {}}), _Response({"result": []}))


class _AlwaysBoom:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, request, timeout=None):
        self.calls += 1
        raise URLError("unreachable")


def test_transport_error_with_raise_on_error_false_degrades_to_empty() -> None:
    opener = _AlwaysBoom()
    collection = _collection(opener)
    assert collection.get_meta_data() == {}
    assert collection.get_meta_data(raise_on_error=False) == {}
    # A second probe should not crash either.
    assert collection.get_meta_data() == {}
    assert opener.calls >= 3


def test_transport_error_with_raise_on_error_true_raises() -> None:
    opener = _ScriptedOpener(URLError("unreachable"))
    collection = _collection(opener)
    with pytest.raises(QdrantError):
        collection.get_meta_data(raise_on_error=True)


def test_missing_marker_degrades_to_empty_and_raises_on_demand() -> None:
    collection = _collection(_missing_marker_opener())
    assert collection.get_meta_data() == {}
    collection = _collection(_missing_marker_opener())
    with pytest.raises(RuntimeError, match="missing valid OpenViking metadata"):
        collection.get_meta_data(raise_on_error=True)


def test_loaded_collection_meta_is_cached_and_does_not_hit_transport_again() -> None:
    fake = FakeQdrant()
    adapter = make_adapter(fake)
    create_collection(adapter)
    collection = adapter._qdrant
    assert collection is not None
    meta = collection.get_meta_data()
    assert meta["Fields"]

    calls: list[str] = []

    class _Boom:
        def __call__(self, request, timeout=None):
            calls.append(request.full_url)
            raise URLError("connection lost")

    collection._client._opener = _Boom()  # type: ignore[attr-defined]
    assert collection.get_meta_data()["Fields"] == meta["Fields"]
    assert calls == []


def test_error_after_partial_load_keeps_raise_on_error_semantics() -> None:
    """A failure while loading must not leave the instance half-initialised."""
    opener = _ScriptedOpener(_Response({"result": {}}), URLError("dropped"))
    collection = _collection(opener)
    assert collection.get_meta_data() == {}
    assert collection._loaded is False
