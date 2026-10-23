# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Minimal dependency-free Qdrant REST transport.

Only the Python standard library is used.  The opener is injectable so the
whole adapter can be exercised offline with a scripted or in-memory transport.

The client retries transient failures (timeouts, connection errors, HTTP
408/425/429/5xx) with bounded exponential backoff and honours ``Retry-After``.
Permanent failures (400/401/403/404/409/422 and malformed JSON) are raised
immediately.  Every operation performed by the adapter is idempotent (stable
point ids, filter deletes, collection/marker puts), so retrying is safe.

This behaviour was absorbed from the ops-focused candidate implementation and
adapted to this package's existing ``QdrantError`` contract.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

# HTTP statuses worth retrying: timeout / too early / rate limited / server side.
_RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})
# A Retry-After header is trusted but bounded so one bad server cannot stall us.
_RETRY_AFTER_CAP_SECONDS = 30.0


class QdrantError(RuntimeError):
    """A bounded Qdrant REST failure with request context.

    The original constructor signature (``method``/``path``/``status``/``body``)
    is preserved; ``status_code``/``retryable``/``action``/``attempts`` are
    additive so existing callers keep working.
    """

    def __init__(
        self,
        message: str,
        *,
        method: str | None = None,
        path: str | None = None,
        status: int | None = None,
        body: str | None = None,
        status_code: int | None = None,
        retryable: bool = False,
        action: str | None = None,
        attempts: int = 1,
    ) -> None:
        super().__init__(message)
        self.method = method
        self.path = path
        self.status = status if status is not None else status_code
        self.body = body
        self.status_code = self.status
        self.retryable = bool(retryable)
        self.action = action or (
            "retry later" if retryable else "check the Qdrant URL / API key / collection"
        )
        self.attempts = int(attempts)


class QdrantRestClient:
    """JSON REST client for Qdrant's HTTP API."""

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        timeout_seconds: float = 10.0,
        opener: Callable[..., Any] | None = None,
        max_retries: int = 2,
        backoff_base_seconds: float = 0.2,
        backoff_max_seconds: float = 2.0,
        sleeper: Callable[[float], None] | None = None,
    ) -> None:
        normalized = str(base_url).strip().rstrip("/")
        if not normalized:
            raise ValueError("Qdrant URL must not be empty")
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        self._base_url = normalized
        self._api_key = api_key
        self._timeout_seconds = float(timeout_seconds)
        self._opener = opener or urlopen
        self._max_retries = int(max_retries)
        self._backoff_base_seconds = float(backoff_base_seconds)
        self._backoff_max_seconds = float(backoff_max_seconds)
        self._sleeper = sleeper or time.sleep

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def opener(self) -> Callable[..., Any]:
        """Expose the transport so diagnostics can reuse the same injection point."""
        return self._opener

    def request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        *,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Perform one JSON request (with transient-failure retries)."""
        method = method.upper()
        if not path.startswith("/"):
            path = f"/{path}"
        query = _encode_params(params)
        url = f"{self._base_url}{path}{query}"
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        if self._api_key:
            headers["api-key"] = self._api_key

        attempts = self._max_retries + 1
        last_error: QdrantError | None = None
        for attempt in range(1, attempts + 1):
            try:
                return self._request_once(method, path, url, headers, body)
            except QdrantError as exc:
                last_error = exc
                if not exc.retryable or attempt >= attempts:
                    exc.attempts = attempt
                    raise
                delay = self._retry_delay(attempt, exc)
                self._sleeper(delay)

        # Unreachable unless attempts <= 0, kept for type-checkers.
        assert last_error is not None
        raise last_error

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _request_once(
        self,
        method: str,
        path: str,
        url: str,
        headers: dict[str, str],
        body: dict[str, Any] | None,
    ) -> dict[str, Any]:
        request = Request(
            url,
            data=json.dumps(body).encode("utf-8") if body is not None else None,
            headers=headers,
            method=method,
        )
        try:
            with self._opener(request, timeout=self._timeout_seconds) as response:
                raw = response.read()
        except HTTPError as exc:
            raw = exc.read() if exc.fp is not None else b""
            detail = raw.decode("utf-8", errors="replace")[:1000]
            raise _http_error(exc, method=method, path=path, detail=detail) from exc
        except URLError as exc:
            raise QdrantError(
                f"Qdrant transport error {method} {path}: {exc.reason}",
                method=method,
                path=path,
                retryable=True,
            ) from exc
        except TimeoutError as exc:
            raise QdrantError(
                f"Qdrant request timed out {method} {path}",
                method=method,
                path=path,
                retryable=True,
            ) from exc
        except OSError as exc:  # connection reset / broken pipe etc.
            raise QdrantError(
                f"Qdrant connection error {method} {path}: {exc}",
                method=method,
                path=path,
                retryable=True,
            ) from exc

        if not raw:
            return {}
        try:
            decoded = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise QdrantError(
                f"Qdrant returned invalid JSON for {method} {path}",
                method=method,
                path=path,
            ) from exc
        if not isinstance(decoded, dict):
            raise QdrantError(
                f"Qdrant returned a non-object response for {method} {path}",
                method=method,
                path=path,
            )
        return decoded

    def _retry_delay(self, attempt: int, error: QdrantError) -> float:
        retry_after = getattr(error, "retry_after_seconds", None)
        if retry_after is not None:
            return min(float(retry_after), _RETRY_AFTER_CAP_SECONDS)
        delay = self._backoff_base_seconds * (2 ** (attempt - 1))
        return min(delay, self._backoff_max_seconds)


def _http_error(exc: HTTPError, *, method: str, path: str, detail: str) -> QdrantError:
    status = int(exc.code)
    retryable = status in _RETRYABLE_STATUS
    error = QdrantError(
        f"Qdrant HTTP {status} {method} {path}: {detail}",
        method=method,
        path=path,
        status=status,
        body=detail,
        retryable=retryable,
    )
    if status == 429:
        retry_after = _parse_retry_after(exc)
        if retry_after is not None:
            setattr(error, "retry_after_seconds", retry_after)
    return error


def _parse_retry_after(exc: HTTPError) -> float | None:
    headers = getattr(exc, "headers", None)
    if headers is None:
        return None
    try:
        raw = headers.get("Retry-After")
    except Exception:  # noqa: BLE001 - malformed header objects are not fatal
        return None
    if not raw:
        return None
    try:
        value = float(str(raw).strip())
    except (TypeError, ValueError):
        return None
    if value < 0:
        return None
    return value


def collection_path(name: str, suffix: str = "") -> str:
    """Return the REST path for a collection (and optional sub-resource)."""
    return f"/collections/{quote(str(name), safe='')}{suffix}"


def _encode_params(params: dict[str, Any] | None) -> str:
    if not params:
        return ""
    normalized: dict[str, Any] = {}
    for key, value in params.items():
        if value is None:
            continue
        if isinstance(value, bool):
            normalized[key] = "true" if value else "false"
        else:
            normalized[key] = value
    return f"?{urlencode(normalized)}" if normalized else ""


__all__ = ["QdrantError", "QdrantRestClient", "collection_path"]
