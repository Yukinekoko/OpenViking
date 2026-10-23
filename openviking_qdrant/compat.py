# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Runtime compatibility guards for the host OpenViking package.

The adapter is built against the OpenViking v0.5.1.1 adapter interface.  A
future release may change that interface; this module surfaces the mismatch in
the logs early instead of failing deep inside a search path.  Source checkouts
report ``0.0.0+unknown`` and are only warned about (tests intentionally run
from a source tree).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

EXPECTED_CORE_SERIES = ("0.5.",)
_UNKNOWN_VERSIONS = frozenset({"", "0.0.0", "0.0.0+unknown"})


def core_version() -> str:
    """Return the host ``openviking`` package version (best effort)."""
    try:
        import openviking
    except Exception:  # noqa: BLE001 - a broken host install is reported as unknown
        return ""
    return str(getattr(openviking, "__version__", "") or "")


def check_core_compatibility(
    *,
    expected_series: tuple[str, ...] = EXPECTED_CORE_SERIES,
    strict: bool = False,
) -> dict[str, object]:
    """Check the host package series and log a warning on mismatch.

    Returns a small report so diagnostics/tests can assert on it.  ``strict``
    turns an incompatible (but known) version into a ``RuntimeError``.
    """
    version = core_version()
    unknown = version in _UNKNOWN_VERSIONS
    compatible = unknown or version.startswith(expected_series)
    report: dict[str, object] = {
        "version": version,
        "expected_series": list(expected_series),
        "compatible": compatible,
        "unknown": unknown,
    }
    if not compatible:
        message = (
            f"openviking_qdrant was built for OpenViking {expected_series[0]}x, "
            f"but the host package reports {version!r}. The adapter interface may "
            "have changed; run the offline test suite against the new release."
        )
        if strict:
            raise RuntimeError(message)
        logger.warning(message)
    return report


__all__ = ["EXPECTED_CORE_SERIES", "check_core_compatibility", "core_version"]
