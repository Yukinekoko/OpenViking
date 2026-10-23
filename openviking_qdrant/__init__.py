# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Third-party Qdrant vector backend package for OpenViking v0.5.1.1.

The package is discovered by OpenViking's dynamic adapter factory when
``storage.vectordb.backend`` is set to its dotted class path::

    backend: openviking_qdrant.adapter.QdrantCollectionAdapter

It is additive-only: no tracked OpenViking source file is modified.  See
``README.md`` for configuration, data-layout compatibility, diagnostics and the
offline test instructions.
"""

from .adapter import QdrantCollectionAdapter
from .collection import QdrantCollection
from .compat import check_core_compatibility, core_version
from .config import DEFAULT_QUERY_CANDIDATE_CAP, QdrantSettings
from .filter import compile_qdrant_filter
from .rest import QdrantError, QdrantRestClient
from .sparse import SparseTermDictionary, stable_sparse_index
from .utils import build_qdrant_payload, to_qdrant_point_id

__version__ = "0.2.0"


def __getattr__(name: str):
    # Keep ``python -m openviking_qdrant.diagnose`` importable without the
    # runpy double-import warning: only load this module on demand.
    if name == "run_diagnostics":
        from .diagnose import run_diagnostics

        return run_diagnostics
    raise AttributeError(name)

__all__ = [
    "DEFAULT_QUERY_CANDIDATE_CAP",
    "QdrantCollection",
    "QdrantCollectionAdapter",
    "QdrantError",
    "QdrantRestClient",
    "QdrantSettings",
    "SparseTermDictionary",
    "build_qdrant_payload",
    "check_core_compatibility",
    "compile_qdrant_filter",
    "core_version",
    "run_diagnostics",
    "stable_sparse_index",
    "to_qdrant_point_id",
]
