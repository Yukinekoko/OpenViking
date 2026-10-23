# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
from __future__ import annotations

import hashlib

import pytest

from openviking_qdrant.sparse import (
    MAX_QDRANT_SPARSE_INDEX,
    SparseTermDictionary,
    stable_sparse_index,
)
from openviking_qdrant.tests.fake_qdrant import FakeQdrant
from openviking_qdrant.tests.helpers import create_collection, make_adapter


def test_stable_sparse_index_is_stable_and_qdrant_compatible() -> None:
    expected = int.from_bytes(hashlib.sha256(b"hello").digest()[:4], "big")
    expected &= MAX_QDRANT_SPARSE_INDEX
    assert stable_sparse_index("hello") == expected
    assert stable_sparse_index("hello") == stable_sparse_index("hello")
    assert 0 < stable_sparse_index("hello") <= MAX_QDRANT_SPARSE_INDEX
    assert stable_sparse_index("hello") != stable_sparse_index("world")


def test_dictionary_persists_new_terms_and_reuses_existing_entries() -> None:
    terms: dict[str, int] = {}
    persisted: list[tuple[str, int]] = []

    def resolve_term(term: str) -> int | None:
        return terms.get(term)

    def resolve_index(index: int) -> str | None:
        for term, value in terms.items():
            if value == index:
                return term
        return None

    def persist(term: str, index: int) -> None:
        terms[term] = index
        persisted.append((term, index))

    dictionary = SparseTermDictionary(
        resolve_term=resolve_term,
        resolve_index=resolve_index,
        persist=persist,
    )
    first = dictionary.index_for("alpha")
    assert dictionary.index_for("alpha") == first
    assert persisted == [("alpha", first)]
    assert dictionary.encode({"alpha": 1.0, "beta": 0.5}) == {
        "indices": [first, dictionary.index_for("beta")],
        "values": [1.0, 0.5],
    }
    assert dictionary.encode({}) is None


def test_dictionary_rejects_hash_collision() -> None:
    dictionary = SparseTermDictionary(
        resolve_term=lambda term: None,
        resolve_index=lambda index: "other",
        persist=lambda term, index: None,
        hash_term=lambda term: 7,
    )
    with pytest.raises(ValueError, match="collision"):
        dictionary.index_for("alpha")


def test_dictionary_rejects_non_finite_weights() -> None:
    dictionary = SparseTermDictionary(
        resolve_term=lambda term: 1,
        resolve_index=lambda index: "alpha",
        persist=lambda term, index: None,
    )
    with pytest.raises(ValueError, match="finite"):
        dictionary.encode({"alpha": float("nan")})


def test_sparse_collection_roundtrip_and_search() -> None:
    fake = FakeQdrant()
    adapter = make_adapter(fake, sparse_weight=0.5)
    create_collection(adapter, sparse_weight=0.5)
    adapter.upsert(
        [
            {
                "id": "doc-1",
                "uri": "viking://resources/doc-1.md",
                "account_id": "acct",
                "level": 2,
                "vector": [1, 0, 0, 0],
                "sparse_vector": {"alpha": 1.0, "beta": 0.2},
            },
            {
                "id": "doc-2",
                "uri": "viking://resources/doc-2.md",
                "account_id": "acct",
                "level": 2,
                "vector": [0, 1, 0, 0],
                "sparse_vector": {"beta": 1.0},
            },
        ]
    )
    sparse_only = adapter.query(sparse_query_vector={"alpha": 1.0}, limit=2)
    assert sparse_only[0]["id"] == "doc-1"
    hybrid = adapter.query(
        query_vector=[1, 0, 0, 0],
        sparse_query_vector={"beta": 1.0},
        limit=2,
    )
    assert {item["id"] for item in hybrid} == {"doc-1", "doc-2"}
    assert all("_score" in item for item in hybrid)

    # search_by_id retrieves and decodes the persisted sparse vector.
    by_id = adapter.get_collection().search_by_id(
        "default", "doc-1", limit=2, output_fields=["id", "uri"]
    )
    assert by_id.data
    assert by_id.data[0].id == "doc-1"

    assert adapter.get_collection()._Collection__collection._decode_sparse_vector(
        {"indices": [stable_sparse_index("alpha")], "values": [1.0]}
    ) == {"alpha": 1.0}


def test_sparse_term_dictionary_survives_reload() -> None:
    fake = FakeQdrant()
    adapter = make_adapter(fake, sparse_weight=0.5)
    create_collection(adapter, sparse_weight=0.5)
    adapter.upsert(
        [
            {
                "id": "doc-1",
                "uri": "viking://resources/doc-1.md",
                "vector": [1, 0, 0, 0],
                "sparse_vector": {"alpha": 1.0},
            }
        ]
    )
    reloaded = make_adapter(fake, sparse_weight=0.5)
    assert reloaded.collection_exists()
    result = reloaded.query(sparse_query_vector={"alpha": 1.0}, limit=1)
    assert result[0]["id"] == "doc-1"


def test_sparse_disabled_collection_rejects_sparse_operations() -> None:
    fake = FakeQdrant()
    adapter = make_adapter(fake)
    create_collection(adapter)
    with pytest.raises(ValueError, match="sparse"):
        adapter.query(sparse_query_vector={"alpha": 1.0}, limit=1)
    with pytest.raises(ValueError, match="sparse-vector support"):
        adapter.upsert(
            [
                {
                    "id": "doc-1",
                    "uri": "viking://resources/doc-1.md",
                    "vector": [1, 0, 0, 0],
                    "sparse_vector": {"alpha": 1.0},
                }
            ]
        )
