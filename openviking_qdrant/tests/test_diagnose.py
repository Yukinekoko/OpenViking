# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Offline tests for the read-only diagnostics CLI and adapter health check."""

from __future__ import annotations

import json
from urllib.error import URLError
from urllib.parse import urlsplit

from openviking_qdrant import diagnose
from openviking_qdrant.adapter import QdrantCollectionAdapter
from openviking_qdrant.rest import QdrantRestClient
from openviking_qdrant.tests.fake_qdrant import FakeQdrant
from openviking_qdrant.tests.helpers import (
    context_schema,
    legacy_schema,
    settings,
)


class _Response:
    def __init__(self, payload: dict) -> None:
        self._payload = json.dumps(payload).encode()

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *_args) -> bool:
        return False


class _DiagnoseTransport:
    """Adds the Qdrant root / collection-list endpoints to FakeQdrant."""

    def __init__(self, fake: FakeQdrant) -> None:
        self.fake = fake
        self.calls: list[str] = []

    def __call__(self, request, timeout=None):
        path = urlsplit(request.full_url).path
        self.calls.append(f"{request.method} {path}")
        if path == "/":
            return _Response({"title": "qdrant - vector search engine", "version": "1.13.0"})
        if path == "/collections":
            return _Response(
                {
                    "result": {
                        "collections": [
                            {"name": name} for name in sorted(self.fake.collections)
                        ]
                    }
                }
            )
        return self.fake(request, timeout=timeout)


class _Boom:
    def __call__(self, request, timeout=None):
        raise URLError("connection refused")


def _create(fake: FakeQdrant, *, schema=None, dim: int = 4):
    cfg = settings()
    client = QdrantRestClient(cfg.url, opener=_DiagnoseTransport(fake))
    adapter = QdrantCollectionAdapter(cfg, client=client)
    created = adapter.create_collection(
        "context",
        schema or context_schema(dim=dim),
        distance="cosine",
        sparse_weight=0.0,
        index_name="default",
    )
    assert created is True
    return adapter, client


def _run(fake: FakeQdrant, *, cfg=None):
    cfg = cfg or settings()
    client = QdrantRestClient(cfg.url, opener=_DiagnoseTransport(fake))
    return diagnose.run_diagnostics(settings=cfg, client=client)


def test_diagnostics_reports_reusable_collection() -> None:
    fake = FakeQdrant()
    _create(fake)
    report = _run(fake)
    assert report["server"]["reachable"] is True
    assert report["server"]["version"] == "1.13.0"
    assert report["collection"]["present"] is True
    assert report["metadata"]["valid"] is True
    assert report["metadata"]["vector_dim"] == 4
    assert report["reuse"]["reusable"] is True
    assert report["exit_code"] == diagnose.EXIT_OK


def test_diagnostics_fresh_start_when_collection_absent() -> None:
    report = _run(FakeQdrant())
    assert report["collection"]["present"] is False
    assert "collection_absent_fresh_start" in report["reuse"]["reasons"]
    assert report["exit_code"] == diagnose.EXIT_OK


def test_diagnostics_reports_unreachable_endpoint() -> None:
    cfg = settings()
    client = QdrantRestClient(cfg.url, opener=_Boom(), max_retries=0)
    report = diagnose.run_diagnostics(settings=cfg, client=client)
    assert report["server"]["reachable"] is False
    assert report["status"] == "unreachable"
    assert report["exit_code"] == diagnose.EXIT_UNREACHABLE


def test_diagnostics_flags_stale_collection_without_marker() -> None:
    fake = FakeQdrant()
    fake.seed("proj__context", vectors={"vector": {"size": 4, "distance": "Cosine"}})
    report = _run(fake)
    assert report["collection"]["present"] is True
    assert report["metadata"]["valid"] is False
    assert "metadata_missing_or_invalid" in report["reuse"]["reasons"]
    assert report["exit_code"] == diagnose.EXIT_NOT_REUSABLE


def test_diagnostics_flags_dimension_mismatch() -> None:
    fake = FakeQdrant()
    _create(fake, dim=4)
    report = _run(fake, cfg=settings(dimension=8))
    assert "dimension_mismatch" in report["reuse"]["reasons"]
    assert report["reuse"]["reusable"] is False
    assert report["exit_code"] == diagnose.EXIT_NOT_REUSABLE


def test_diagnostics_marks_legacy_schema_as_compatible_with_warning() -> None:
    fake = FakeQdrant()
    _create(fake, schema=legacy_schema())
    report = _run(fake)
    assert report["metadata"]["legacy_schema"] is True
    assert "legacy_schema_compat_view" in report["reuse"]["warnings"]
    assert report["reuse"]["reusable"] is True
    assert report["exit_code"] == diagnose.EXIT_OK


def test_adapter_health_check_uses_same_transport() -> None:
    fake = FakeQdrant()
    adapter, _client = _create(fake)
    report = adapter.health_check()
    assert report["metadata"]["valid"] is True
    assert adapter.reuse_assessment()["reusable"] is True


def test_cli_json_output_and_exit_code(monkeypatch, capsys) -> None:
    fake = FakeQdrant()
    _create(fake)
    # The CLI builds its own client; patch the module-level default opener used by
    # QdrantRestClient so the in-memory fake is reached.
    import openviking_qdrant.rest as rest_mod

    monkeypatch.setattr(rest_mod, "urlopen", _DiagnoseTransport(fake))
    code = diagnose.main(
        [
            "--url",
            "http://qdrant.test:6333",
            "--project",
            "proj",
            "--collection",
            "context",
            "--dimension",
            "4",
            "--json",
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["metadata"]["valid"] is True
    assert code == diagnose.EXIT_OK


def test_cli_returns_two_when_unreachable(monkeypatch, capsys) -> None:
    import openviking_qdrant.rest as rest_mod

    monkeypatch.setattr(rest_mod, "urlopen", _Boom())
    code = diagnose.main(["--url", "http://qdrant.test:6333", "--max-retries", "0"])
    capsys.readouterr()
    assert code == diagnose.EXIT_UNREACHABLE
