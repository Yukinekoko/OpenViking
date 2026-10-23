# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Shared fixtures for the offline Qdrant adapter tests."""

from __future__ import annotations

import pytest

from openviking_qdrant.tests.fake_qdrant import FakeQdrant
from openviking_qdrant.tests.helpers import make_adapter


@pytest.fixture
def fake() -> FakeQdrant:
    return FakeQdrant()


@pytest.fixture
def adapter():
    return make_adapter(FakeQdrant())
