# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
from __future__ import annotations

from types import SimpleNamespace

import pytest

from openviking.storage.vectordb_adapters.base import CollectionAdapter
from openviking.storage.vectordb_adapters.factory import create_collection_adapter
from openviking_cli.utils.config.vectordb_config import VectorDBBackendConfig
from openviking_qdrant.adapter import QdrantCollectionAdapter
from openviking_qdrant.config import QdrantSettings
from openviking_qdrant.rest import QdrantRestClient
from openviking_qdrant.tests.fake_qdrant import FakeQdrant
from openviking_qdrant.tests.helpers import create_collection, make_adapter


def test_factory_loads_dotted_backend_and_injects_dimension() -> None:
    config = VectorDBBackendConfig(
        backend="openviking_qdrant.adapter.QdrantCollectionAdapter",
        name="context",
        project="openviking_test_v3",
        dimension=768,
        distance_metric="cosine",
        url="http://127.0.0.1:6333",
        custom_params={"api_key": None, "timeout_seconds": 1},
    )
    adapter = create_collection_adapter(config)
    assert isinstance(adapter, CollectionAdapter)
    assert isinstance(adapter, QdrantCollectionAdapter)
    assert adapter._dimension == 768
    assert adapter.mode == "qdrant"
    assert adapter._physical_collection_name == "openviking_test_v3__context"
    assert adapter._metadata_collection_name == (
        "openviking_test_v3__context__openviking_meta"
    )
    assert adapter.USE_CONTENT_FIELD is False


def test_settings_parse_url_custom_params_and_legacy_nested_block() -> None:
    config = SimpleNamespace(
        url="qdrant.internal:6333/",
        project_name="proj",
        name="context",
        index_name="default",
        distance_metric="l2",
        dimension=32,
        sparse_weight=0.25,
        custom_params={
            "api_key": "secret",
            "timeout_seconds": "2.5",
            "dense_vector_name": "dense",
            "sparse_vector_name": "lexical",
            "metadata_collection_name": "meta_override",
            "query_candidate_cap": 123,
            "batch_size": 7,
        },
    )
    settings = QdrantSettings.from_config(config)
    assert settings.url == "http://qdrant.internal:6333"
    assert settings.api_key == "secret"
    assert settings.timeout_seconds == 2.5
    assert settings.distance_metric == "l2"
    assert settings.sparse_weight == 0.25
    assert settings.dense_vector_name == "dense"
    assert settings.sparse_vector_name == "lexical"
    assert settings.metadata_collection_name == "meta_override"
    assert settings.query_candidate_cap == 123
    assert settings.batch_size == 7

    legacy = SimpleNamespace(
        url=None,
        project_name="p",
        name="context",
        index_name="default",
        distance_metric="cosine",
        dimension=8,
        sparse_weight=0.0,
        custom_params={},
        qdrant=SimpleNamespace(
            url="http://legacy:6333",
            api_key="legacy-key",
            timeout_seconds=3,
            dense_vector_name="vec",
            sparse_vector_name="sparse_vec",
            metadata_collection_name=None,
        ),
    )
    settings = QdrantSettings.from_config(legacy)
    assert settings.url == "http://legacy:6333"
    assert settings.api_key == "legacy-key"
    assert settings.timeout_seconds == 3.0
    assert settings.dense_vector_name == "vec"
    assert settings.sparse_vector_name == "sparse_vec"


def test_settings_validation_errors() -> None:
    base = dict(
        project_name="p",
        name="context",
        index_name="default",
        distance_metric="cosine",
        dimension=8,
        sparse_weight=0.0,
        custom_params={},
    )
    with pytest.raises(ValueError, match="requires 'storage.vectordb.url'"):
        QdrantSettings.from_config(SimpleNamespace(url=None, **base))
    with pytest.raises(ValueError, match="http"):
        QdrantSettings.from_config(
            SimpleNamespace(url="ftp://host:1", **base)
        )
    with pytest.raises(ValueError, match="timeout_seconds"):
        QdrantSettings.from_config(
            SimpleNamespace(
                url="http://host:1",
                **{**base, "custom_params": {"timeout_seconds": 0}},
            )
        )
    with pytest.raises(ValueError, match="sparse_weight"):
        QdrantSettings.from_config(
            SimpleNamespace(url="http://host:1", **{**base, "sparse_weight": 2.0})
        )
    with pytest.raises(ValueError, match="query_candidate_cap"):
        QdrantSettings.from_config(
            SimpleNamespace(
                url="http://host:1",
                **{**base, "custom_params": {"query_candidate_cap": 0}},
            )
        )


def test_adapter_uses_physical_layout_and_collection_info() -> None:
    fake = FakeQdrant()
    adapter = make_adapter(fake)
    create_collection(adapter)
    info = adapter.get_collection_info()
    assert info["CollectionName"] == "context"
    assert info["Description"] == "Unified context collection"
    assert "proj__context" in fake.collections
    assert "proj__context__openviking_meta" in fake.collections


def test_search_by_keywords_accepts_new_signature_and_raises() -> None:
    adapter = make_adapter(FakeQdrant())
    with pytest.raises(NotImplementedError, match="content field"):
        adapter.search_by_keywords(
            keywords=["alpha"],
            query="alpha",
            limit=5,
            offset=0,
            output_fields=["uri"],
            mode="bm25",
            fields=["content"],
        )


def test_contains_cannot_be_used_with_vector_search() -> None:
    fake = FakeQdrant()
    adapter = make_adapter(fake)
    create_collection(adapter)
    from openviking.storage.expr import Contains

    with pytest.raises(NotImplementedError, match="vector search"):
        adapter.query(
            query_vector=[1, 0, 0, 0],
            filter=Contains("uri", "viking://resources/a#"),
            limit=1,
        )


def test_direct_search_by_random_rejects_advance() -> None:
    fake = FakeQdrant()
    adapter = make_adapter(fake)
    create_collection(adapter)
    with pytest.raises(NotImplementedError, match="random enumeration"):
        adapter.search_by_random(
            limit=1,
            advance={"time_decay": {"protection": "0", "origin": "2026-01-01T00:00:00Z"}},
        )


def test_direct_search_by_random_supports_contains_fallback() -> None:
    fake = FakeQdrant()
    adapter = make_adapter(fake)
    create_collection(adapter)
    adapter.upsert(
        [
            {"id": "a", "uri": "viking://resources/a#frag", "vector": [1, 0, 0, 0]},
            {"id": "b", "uri": "viking://resources/b", "vector": [0, 1, 0, 0]},
        ]
    )
    from openviking.storage.expr import Contains

    result = adapter.search_by_random(
        filter=Contains("uri", "viking://resources/a#"), limit=10
    )
    assert [item["id"] for item in result] == ["a"]


def test_delete_by_ids_and_bulk_ingest_noops() -> None:
    fake = FakeQdrant()
    adapter = make_adapter(fake)
    create_collection(adapter)
    adapter.upsert(
        [
            {"id": "a", "uri": "viking://resources/a", "vector": [1, 0, 0, 0]},
            {"id": "b", "uri": "viking://resources/b", "vector": [0, 1, 0, 0]},
        ]
    )
    assert adapter.delete(ids=["a"]) == 1
    assert adapter.get(["a"]) == []
    adapter.begin_bulk_ingest()
    adapter.end_bulk_ingest()

    adapter.update_data(
        [{"id": "b", "abstract": "updated"}]
    )
    assert adapter.get(["b"])[0]["abstract"] == "updated"


def test_transport_sends_api_key_and_timeout() -> None:
    fake = FakeQdrant()
    settings = QdrantSettings(
        url="http://qdrant.test:6333",
        api_key="secret",
        timeout_seconds=4.5,
        project_name="proj",
        collection_name="context",
        index_name="default",
        distance_metric="cosine",
        dimension=4,
        sparse_weight=0.0,
        dense_vector_name="vector",
        sparse_vector_name="sparse_vector",
        metadata_collection_name=None,
        query_candidate_cap=1000,
        batch_size=100,
    )
    client = QdrantRestClient(
        settings.url,
        api_key="secret",
        timeout_seconds=settings.timeout_seconds,
        opener=fake,
    )
    adapter = QdrantCollectionAdapter(settings, client=client)
    create_collection(adapter)
    request = fake.requests[-1]
    assert request["headers"].get("Api-key") == "secret"
    assert request["timeout"] == 4.5
