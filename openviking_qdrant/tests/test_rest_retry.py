# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Retry / backoff / error-classification tests for the REST transport."""

from __future__ import annotations

import io
import json
from urllib.error import HTTPError, URLError

import pytest

from openviking_qdrant.rest import QdrantError, QdrantRestClient


class _Response:
    def __init__(self, payload: dict | None = None, *, raw: bytes | None = None) -> None:
        self._payload = raw if raw is not None else json.dumps(payload or {}).encode()

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *_args) -> bool:
        return False


class _SequenceOpener:
    def __init__(self, *items) -> None:
        self.items = list(items)
        self.calls: list[str] = []

    def __call__(self, request, timeout=None):
        self.calls.append(request.full_url)
        item = self.items.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


class _Sleeps:
    def __init__(self) -> None:
        self.delays: list[float] = []

    def __call__(self, delay: float) -> None:
        self.delays.append(delay)


def _http_error(url: str, status: int, *, headers: dict | None = None) -> HTTPError:
    return HTTPError(
        url,
        status,
        f"status {status}",
        headers or {},
        io.BytesIO(b'{"status": {"error": "boom"}}'),
    )


def _client(opener, sleeps, **kwargs) -> QdrantRestClient:
    return QdrantRestClient(
        "http://qdrant.test:6333",
        opener=opener,
        sleeper=sleeps,
        backoff_base_seconds=0.2,
        backoff_max_seconds=2.0,
        **kwargs,
    )


def test_transient_server_error_is_retried_then_succeeds() -> None:
    opener = _SequenceOpener(_http_error("http://x", 503), _Response({"result": True}))
    sleeps = _Sleeps()
    client = _client(opener, sleeps)
    assert client.request("GET", "/collections") == {"result": True}
    assert len(opener.calls) == 2
    assert sleeps.delays == [0.2]


def test_retry_after_header_is_honoured() -> None:
    opener = _SequenceOpener(
        _http_error("http://x", 429, headers={"Retry-After": "1.5"}),
        _Response({"result": True}),
    )
    sleeps = _Sleeps()
    client = _client(opener, sleeps)
    assert client.request("GET", "/collections") == {"result": True}
    assert sleeps.delays == [1.5]


def test_client_error_is_not_retried() -> None:
    opener = _SequenceOpener(_http_error("http://x", 400))
    sleeps = _Sleeps()
    client = _client(opener, sleeps)
    with pytest.raises(QdrantError) as excinfo:
        client.request("POST", "/collections/x/points", {"points": []})
    assert excinfo.value.status == 400
    assert excinfo.value.retryable is False
    assert excinfo.value.attempts == 1
    assert len(opener.calls) == 1
    assert sleeps.delays == []


def test_auth_error_is_not_retried() -> None:
    opener = _SequenceOpener(_http_error("http://x", 401))
    sleeps = _Sleeps()
    with pytest.raises(QdrantError) as excinfo:
        _client(opener, sleeps).request("GET", "/collections")
    assert excinfo.value.retryable is False
    assert len(opener.calls) == 1


def test_retries_are_bounded_and_report_attempts() -> None:
    opener = _SequenceOpener(
        _http_error("http://x", 503),
        _http_error("http://x", 503),
        _http_error("http://x", 503),
    )
    sleeps = _Sleeps()
    client = _client(opener, sleeps, max_retries=2)
    with pytest.raises(QdrantError) as excinfo:
        client.request("GET", "/collections")
    assert excinfo.value.attempts == 3
    assert excinfo.value.retryable is True
    assert len(opener.calls) == 3
    # Exponential with cap: 0.2, 0.4
    assert sleeps.delays == [0.2, 0.4]


def test_max_retries_zero_fails_fast() -> None:
    opener = _SequenceOpener(_http_error("http://x", 503))
    sleeps = _Sleeps()
    client = _client(opener, sleeps, max_retries=0)
    with pytest.raises(QdrantError):
        client.request("GET", "/collections")
    assert len(opener.calls) == 1
    assert sleeps.delays == []


def test_timeout_is_retried() -> None:
    opener = _SequenceOpener(TimeoutError(), _Response({"result": True}))
    sleeps = _Sleeps()
    assert _client(opener, sleeps).request("GET", "/collections") == {"result": True}
    assert len(opener.calls) == 2


def test_connection_error_is_retried() -> None:
    opener = _SequenceOpener(URLError("connection refused"), _Response({"result": True}))
    sleeps = _Sleeps()
    assert _client(opener, sleeps).request("GET", "/") == {"result": True}
    assert len(opener.calls) == 2


def test_invalid_json_is_not_retried() -> None:
    opener = _SequenceOpener(_Response(raw=b"not-json"))
    sleeps = _Sleeps()
    with pytest.raises(QdrantError, match="invalid JSON"):
        _client(opener, sleeps).request("GET", "/collections")
    assert len(opener.calls) == 1


def test_permanent_http_error_exposes_action_and_status_code() -> None:
    opener = _SequenceOpener(_http_error("http://x", 404))
    with pytest.raises(QdrantError) as excinfo:
        _client(opener, _Sleeps()).request("GET", "/collections/missing")
    error = excinfo.value
    assert error.status_code == 404
    assert error.status == 404
    assert error.retryable is False
    assert error.action
