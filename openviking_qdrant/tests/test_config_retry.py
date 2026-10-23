# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Config validation tests: defaults, env credentials, retry/backoff knobs."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from openviking_qdrant.config import (
    DEFAULT_QUERY_CANDIDATE_CAP,
    NATIVE_QUERY_CANDIDATE_CAP,
    QdrantSettings,
)


def _config(**overrides):
    values = dict(
        project_name="proj",
        name="context",
        index_name="default",
        distance_metric="cosine",
        dimension=8,
        sparse_weight=0.0,
        custom_params={},
    )
    values.update(overrides)
    return SimpleNamespace(url="http://qdrant.test:6333", **values)


def test_default_candidate_cap_matches_native_engine() -> None:
    assert NATIVE_QUERY_CANDIDATE_CAP == 100_000
    assert DEFAULT_QUERY_CANDIDATE_CAP == NATIVE_QUERY_CANDIDATE_CAP
    settings = QdrantSettings.from_config(_config())
    assert settings.query_candidate_cap == 100_000


def test_candidate_cap_can_still_be_lowered_for_strict_mode() -> None:
    settings = QdrantSettings.from_config(
        _config(custom_params={"query_candidate_cap": 500})
    )
    assert settings.query_candidate_cap == 500


def test_retry_and_backoff_defaults() -> None:
    settings = QdrantSettings.from_config(_config())
    assert settings.max_retries == 2
    assert settings.backoff_base_seconds == 0.2
    assert settings.backoff_max_seconds == 2.0


def test_api_key_falls_back_to_environment(monkeypatch) -> None:
    monkeypatch.delenv("OPENVIKING_QDRANT_API_KEY", raising=False)
    monkeypatch.setenv("QDRANT_API_KEY", "env-secret")
    settings = QdrantSettings.from_config(_config())
    assert settings.api_key == "env-secret"


def test_explicit_api_key_wins_over_environment(monkeypatch) -> None:
    monkeypatch.setenv("QDRANT_API_KEY", "env-secret")
    settings = QdrantSettings.from_config(_config(custom_params={"api_key": "explicit"}))
    assert settings.api_key == "explicit"


def test_unknown_custom_params_are_ignored_with_a_warning(caplog) -> None:
    caplog.set_level(logging.WARNING, logger="openviking_qdrant.config")
    settings = QdrantSettings.from_config(
        _config(custom_params={"totally_unknown": 1, "timeout_seconds": 3})
    )
    assert settings.timeout_seconds == 3
    assert any("unknown qdrant custom_params" in record.message for record in caplog.records)


def test_distance_metric_is_validated() -> None:
    QdrantSettings.from_config(_config(distance_metric="IP"))
    with pytest.raises(ValueError, match="distance_metric"):
        QdrantSettings.from_config(_config(distance_metric="hamming"))


def test_retry_knobs_are_validated() -> None:
    with pytest.raises(ValueError, match="max_retries"):
        QdrantSettings.from_config(_config(custom_params={"max_retries": -1}))
    with pytest.raises(ValueError, match="max_retries"):
        QdrantSettings.from_config(_config(custom_params={"max_retries": True}))
    with pytest.raises(ValueError, match="backoff_base_seconds"):
        QdrantSettings.from_config(_config(custom_params={"backoff_base_seconds": 0}))
    with pytest.raises(ValueError, match="backoff_max_seconds"):
        QdrantSettings.from_config(
            _config(
                custom_params={
                    "backoff_base_seconds": 1.0,
                    "backoff_max_seconds": 0.5,
                }
            )
        )


def test_vector_names_must_be_distinct() -> None:
    with pytest.raises(ValueError, match="must differ"):
        QdrantSettings.from_config(
            _config(
                custom_params={
                    "dense_vector_name": "same",
                    "sparse_vector_name": "same",
                }
            )
        )
