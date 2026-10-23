# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Byte-for-byte regression tests for the 0.4.16 Qdrant data layout.

Existing collections were written by the historical fork.  Point ids, the
metadata marker id, sparse term ids and the sparse hash function must stay
identical so those collections remain readable after the v0.5.1.1 upgrade.
The expected values below were hard-coded against the 0.4.16 fork.
"""

from __future__ import annotations

from openviking_qdrant.meta import META_MARKER_ID, META_SUFFIX, META_VERSION
from openviking_qdrant.sparse import MAX_QDRANT_SPARSE_INDEX, stable_sparse_index
from openviking_qdrant.utils import to_qdrant_point_id


def test_point_ids_match_the_0416_layout() -> None:
    assert to_qdrant_point_id("doc-1") == "0077a8eb-accf-5599-92fe-3586b2ffb18d"
    assert (
        to_qdrant_point_id("openviking:metadata")
        == "400beb5c-1532-581d-9fc1-f8cd88a00243"
    )
    assert (
        to_qdrant_point_id("openviking:sparse:token")
        == "b5191b7a-21d9-57e1-9f79-3225de572c38"
    )


def test_metadata_marker_constants_match_the_0416_layout() -> None:
    assert META_VERSION == 1
    assert META_SUFFIX == "__openviking_meta"
    assert META_MARKER_ID == "400beb5c-1532-581d-9fc1-f8cd88a00243"


def test_sparse_index_hash_matches_the_0416_layout() -> None:
    assert MAX_QDRANT_SPARSE_INDEX == 0x7FFF_FFFF
    assert stable_sparse_index("token") == 1011261085
