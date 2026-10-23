# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Configuration parsing for the standalone Qdrant adapter.

OpenViking v0.5.1.1 passes a ``VectorDBBackendConfig`` to ``from_config``.
Third-party backends are selected by a dotted class path, so they cannot add
typed fields to that pydantic model.  The generic connection fields
(``url``, ``project``, ``name``, ``dimension``, ``distance_metric``,
``sparse_weight``) are read directly and the remaining knobs live in
``custom_params``.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

#: The native engine caps the time-decay candidate window at 100000.
NATIVE_QUERY_CANDIDATE_CAP = 100_000
DEFAULT_QUERY_CANDIDATE_CAP = NATIVE_QUERY_CANDIDATE_CAP

_API_KEY_ENV_VARS = ("OPENVIKING_QDRANT_API_KEY", "QDRANT_API_KEY")
_KNOWN_CUSTOM_PARAMS = frozenset(
    {
        "url",
        "api_key",
        "timeout_seconds",
        "dense_vector_name",
        "sparse_vector_name",
        "metadata_collection_name",
        "query_candidate_cap",
        "batch_size",
        "max_retries",
        "backoff_base_seconds",
        "backoff_max_seconds",
    }
)
_KNOWN_DISTANCE_METRICS = {
    "cosine",
    "ip",
    "dot",
    "l2",
    "euclid",
    "l1",
    "manhattan",
}


@dataclass(frozen=True)
class QdrantSettings:
    """Validated, immutable settings for one Qdrant collection binding."""

    url: str
    api_key: str | None
    timeout_seconds: float
    project_name: str
    collection_name: str
    index_name: str
    distance_metric: str
    dimension: int
    sparse_weight: float
    dense_vector_name: str
    sparse_vector_name: str
    metadata_collection_name: str | None
    query_candidate_cap: int
    batch_size: int
    max_retries: int = 2
    backoff_base_seconds: float = 0.2
    backoff_max_seconds: float = 2.0

    @classmethod
    def from_config(cls, config: Any) -> "QdrantSettings":
        custom = dict(getattr(config, "custom_params", None) or {})
        unknown = sorted(set(custom) - _KNOWN_CUSTOM_PARAMS)
        if unknown:
            logger.warning(
                "Ignoring unknown qdrant custom_params keys: %s (known: %s)",
                unknown,
                sorted(_KNOWN_CUSTOM_PARAMS),
            )

        # Compatibility with the 0.4.16 fork's nested `qdrant` config block.
        # v0.5.1.1 third-party backends use `url` + `custom_params`, but reading
        # the nested block keeps existing ov.conf files working.
        nested = getattr(config, "qdrant", None)
        if nested is not None:
            for key in (
                "url",
                "api_key",
                "timeout_seconds",
                "dense_vector_name",
                "sparse_vector_name",
                "metadata_collection_name",
            ):
                value = getattr(nested, key, None)
                if value is not None and key not in custom:
                    custom[key] = value

        url = getattr(config, "url", None) or custom.get("url")
        if not url or not str(url).strip():
            raise ValueError(
                "Qdrant backend requires 'storage.vectordb.url' "
                "(or custom_params.url) to be set"
            )

        api_key = custom.get("api_key")
        if api_key is None:
            for env_name in _API_KEY_ENV_VARS:
                env_value = os.environ.get(env_name)
                if env_value:
                    api_key = env_value
                    break
        if api_key is not None:
            api_key = str(api_key).strip() or None

        timeout = _as_float(custom.get("timeout_seconds", 10.0), "timeout_seconds")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Qdrant timeout_seconds must be a positive finite number")

        dimension = int(getattr(config, "dimension", 0) or 0)
        if dimension < 0:
            raise ValueError("Qdrant dimension must be non-negative")
        if dimension == 0:
            logger.warning(
                "Qdrant dimension is 0; dense-vector writes and vector queries will "
                "fail until storage.vectordb.dimension is configured"
            )

        distance_metric = str(
            getattr(config, "distance_metric", None) or "cosine"
        ).strip().lower()
        if distance_metric not in _KNOWN_DISTANCE_METRICS:
            raise ValueError(
                f"Unsupported Qdrant distance_metric {distance_metric!r}; "
                f"expected one of {sorted(_KNOWN_DISTANCE_METRICS)}"
            )

        sparse_weight = _as_float(
            getattr(config, "sparse_weight", 0.0) or 0.0, "sparse_weight"
        )
        if not math.isfinite(sparse_weight) or not 0.0 <= sparse_weight <= 1.0:
            raise ValueError("Qdrant sparse_weight must be a finite number between 0 and 1")

        cap = _as_int(
            custom.get("query_candidate_cap", DEFAULT_QUERY_CANDIDATE_CAP),
            "query_candidate_cap",
        )
        if cap <= 0:
            raise ValueError("Qdrant query_candidate_cap must be positive")

        batch_size = _as_int(custom.get("batch_size", 100), "batch_size")
        if batch_size <= 0:
            raise ValueError("Qdrant batch_size must be positive")

        max_retries = _as_int(custom.get("max_retries", 2), "max_retries")
        if max_retries < 0:
            raise ValueError("Qdrant max_retries must be non-negative")

        backoff_base = _as_float(
            custom.get("backoff_base_seconds", 0.2), "backoff_base_seconds"
        )
        if not math.isfinite(backoff_base) or backoff_base <= 0:
            raise ValueError("Qdrant backoff_base_seconds must be a positive finite number")
        backoff_max = _as_float(
            custom.get("backoff_max_seconds", 2.0), "backoff_max_seconds"
        )
        if not math.isfinite(backoff_max) or backoff_max < backoff_base:
            raise ValueError(
                "Qdrant backoff_max_seconds must be finite and >= backoff_base_seconds"
            )

        metadata_collection_name = custom.get("metadata_collection_name")
        if metadata_collection_name is not None:
            metadata_collection_name = str(metadata_collection_name).strip() or None

        dense_vector_name = str(custom.get("dense_vector_name") or "vector").strip()
        sparse_vector_name = str(custom.get("sparse_vector_name") or "sparse_vector").strip()
        if not dense_vector_name or not sparse_vector_name:
            raise ValueError("Qdrant vector names must not be empty")
        if dense_vector_name == sparse_vector_name:
            raise ValueError("Qdrant dense and sparse vector names must differ")

        return cls(
            url=_normalize_url(str(url)),
            api_key=api_key,
            timeout_seconds=timeout,
            project_name=str(getattr(config, "project_name", None) or "default"),
            collection_name=str(getattr(config, "name", None) or "context"),
            index_name=str(getattr(config, "index_name", None) or "default"),
            distance_metric=distance_metric,
            dimension=dimension,
            sparse_weight=sparse_weight,
            dense_vector_name=dense_vector_name,
            sparse_vector_name=sparse_vector_name,
            metadata_collection_name=metadata_collection_name,
            query_candidate_cap=cap,
            batch_size=batch_size,
            max_retries=max_retries,
            backoff_base_seconds=backoff_base,
            backoff_max_seconds=backoff_max,
        )


def _normalize_url(raw: str) -> str:
    url = raw.strip().rstrip("/")
    if not url:
        raise ValueError("Qdrant URL must not be empty")
    if "://" not in url:
        url = f"http://{url}"
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(
            "Qdrant URL must be an http(s) URL such as 'http://127.0.0.1:6333'"
        )
    return url


def _as_float(value: Any, name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Qdrant {name} must be a number") from exc


def _as_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"Qdrant {name} must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Qdrant {name} must be an integer") from exc
