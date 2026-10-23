# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Durable string-term -> Qdrant sparse-index mapping.

Qdrant sparse vectors use numeric indices.  OpenViking sparse embeddings expose
string terms, so this module assigns each term a stable positive index derived
from SHA-256 and persists the mapping in the sidecar metadata collection.  The
hash algorithm and the ``0x7FFF_FFFF`` mask match the historical fork, keeping
existing sparse vectors readable.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable
from threading import RLock
from typing import Any

SparseResolveTerm = Callable[[str], int | None]
SparseResolveIndex = Callable[[int], str | None]
SparsePersist = Callable[[str, int], Any]

MAX_QDRANT_SPARSE_INDEX = 0x7FFF_FFFF


def stable_sparse_index(term: str) -> int:
    """Return a positive Qdrant-compatible sparse index derived from a term."""
    digest = hashlib.sha256(term.encode("utf-8")).digest()
    value = int.from_bytes(digest[:4], "big") & MAX_QDRANT_SPARSE_INDEX
    return value or 1


class SparseTermDictionary:
    """Resolve string sparse terms to durable numeric Qdrant indices."""

    def __init__(
        self,
        *,
        resolve_term: SparseResolveTerm,
        resolve_index: SparseResolveIndex,
        persist: SparsePersist,
        hash_term: Callable[[str], int] = stable_sparse_index,
    ) -> None:
        self._resolve_term = resolve_term
        self._resolve_index = resolve_index
        self._persist = persist
        self._hash_term = hash_term
        self._lock = RLock()
        self._term_cache: dict[str, int] = {}
        self._index_cache: dict[int, str] = {}

    def index_for(self, term: str) -> int:
        normalized = str(term)
        if not normalized:
            raise ValueError("sparse terms must not be empty")
        with self._lock:
            cached = self._term_cache.get(normalized)
            if cached is not None:
                return cached

            existing = self._resolve_term(normalized)
            if existing is not None:
                self._remember(normalized, int(existing))
                return int(existing)

            candidate = int(self._hash_term(normalized))
            if not 0 < candidate <= MAX_QDRANT_SPARSE_INDEX:
                raise ValueError(
                    "sparse term index must be a positive Qdrant-compatible uint32"
                )
            owner = self._resolve_index(candidate)
            if owner is not None and owner != normalized:
                raise ValueError(
                    "sparse term index collision: "
                    f"index={candidate} existing_term={owner!r} new_term={normalized!r}"
                )
            self._persist(normalized, candidate)
            owner = self._resolve_index(candidate)
            if owner is not None and owner != normalized:
                raise ValueError(
                    "sparse term index collision after persistence: "
                    f"index={candidate} existing_term={owner!r} new_term={normalized!r}"
                )
            self._remember(normalized, candidate)
            return candidate

    def term_for(self, index: int) -> str | None:
        """Resolve a numeric index back to its term, caching decoded entries."""
        numeric = int(index)
        with self._lock:
            cached = self._index_cache.get(numeric)
            if cached is not None:
                return cached
            term = self._resolve_index(numeric)
            if term is None:
                return None
            self._remember(term, numeric)
            return term

    def encode(self, vector: dict[str, float] | None) -> dict[str, list[Any]] | None:
        """Encode ``{term: weight}`` as Qdrant's sparse indices/values arrays."""
        if not vector:
            return None
        if not isinstance(vector, dict):
            raise ValueError("sparse vector must be a mapping of term to weight")
        indices: list[int] = []
        values: list[float] = []
        for term, weight in vector.items():
            value = float(weight)
            if not math.isfinite(value):
                raise ValueError("sparse vector weights must be finite")
            indices.append(self.index_for(str(term)))
            values.append(value)
        return {"indices": indices, "values": values}

    def _remember(self, term: str, index: int) -> None:
        owner = self._index_cache.get(index)
        if owner is not None and owner != term:
            raise ValueError(
                "sparse term index collision: "
                f"index={index} existing_term={owner!r} new_term={term!r}"
            )
        self._term_cache[term] = index
        self._index_cache[index] = term


__all__ = [
    "MAX_QDRANT_SPARSE_INDEX",
    "SparseTermDictionary",
    "stable_sparse_index",
]
