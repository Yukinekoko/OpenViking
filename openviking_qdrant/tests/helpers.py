# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Shared fixtures for the offline Qdrant adapter tests."""

from __future__ import annotations

from typing import Any

from openviking_qdrant.adapter import QdrantCollectionAdapter
from openviking_qdrant.config import QdrantSettings
from openviking_qdrant.rest import QdrantRestClient
from openviking_qdrant.tests.fake_qdrant import FakeQdrant

DIM = 4


def context_schema(name: str = "context", *, dim: int = DIM) -> dict[str, Any]:
    fields = [
        {"FieldName": "id", "FieldType": "string", "IsPrimaryKey": True},
        {"FieldName": "uri", "FieldType": "path"},
        {"FieldName": "context_type", "FieldType": "string"},
        {"FieldName": "vector", "FieldType": "vector", "Dim": dim},
        {"FieldName": "sparse_vector", "FieldType": "sparse_vector"},
        {"FieldName": "created_at", "FieldType": "date_time"},
        {"FieldName": "updated_at", "FieldType": "date_time"},
        {"FieldName": "level", "FieldType": "int64"},
        {"FieldName": "name", "FieldType": "string"},
        {"FieldName": "tags", "FieldType": "string"},
        {"FieldName": "search_tags", "FieldType": "list<string>"},
        {"FieldName": "abstract", "FieldType": "string"},
        {"FieldName": "content", "FieldType": "text"},
        {"FieldName": "md5", "FieldType": "string", "DefaultValue": ""},
        {"FieldName": "account_id", "FieldType": "string"},
        {"FieldName": "owner_user_id", "FieldType": "string"},
        {"FieldName": "acl_mode", "FieldType": "string", "DefaultValue": "none"},
        {
            "FieldName": "acl_direct_grants",
            "FieldType": "list<string>",
            "DefaultValue": [],
        },
        {
            "FieldName": "acl_inherited_grants",
            "FieldType": "list<string>",
            "DefaultValue": [],
        },
    ]
    return {
        "CollectionName": name,
        "Description": "Unified context collection",
        "Fields": fields,
        "ScalarIndex": [
            "uri",
            "context_type",
            "created_at",
            "updated_at",
            "level",
            "name",
            "tags",
            "search_tags",
            "account_id",
            "owner_user_id",
            "acl_mode",
            "acl_direct_grants",
            "acl_inherited_grants",
        ],
        "FullText": [{"Field": "content", "Analyzer": {"Tokenizer": "standard"}}],
    }


def legacy_schema(name: str = "context", *, dim: int = DIM) -> dict[str, Any]:
    schema = context_schema(name, dim=dim)
    legacy_names = {
        "md5",
        "acl_mode",
        "acl_direct_grants",
        "acl_inherited_grants",
    }
    schema["Fields"] = [
        field for field in schema["Fields"] if field["FieldName"] not in legacy_names
    ]
    schema["ScalarIndex"] = [
        field for field in schema["ScalarIndex"] if field not in legacy_names
    ]
    return schema


def settings(**overrides: Any) -> QdrantSettings:
    values: dict[str, Any] = {
        "url": "http://qdrant.test:6333",
        "api_key": None,
        "timeout_seconds": 1.0,
        "project_name": "proj",
        "collection_name": "context",
        "index_name": "default",
        "distance_metric": "cosine",
        "dimension": DIM,
        "sparse_weight": 0.0,
        "dense_vector_name": "vector",
        "sparse_vector_name": "sparse_vector",
        "metadata_collection_name": None,
        "query_candidate_cap": 1000,
        "batch_size": 100,
    }
    values.update(overrides)
    return QdrantSettings(**values)


def make_adapter(fake: FakeQdrant, **overrides: Any) -> QdrantCollectionAdapter:
    cfg = settings(**overrides)
    client = QdrantRestClient(cfg.url, api_key=cfg.api_key, opener=fake)
    return QdrantCollectionAdapter(cfg, client=client)


def create_collection(
    adapter: QdrantCollectionAdapter,
    *,
    schema: dict[str, Any] | None = None,
    sparse_weight: float = 0.0,
) -> None:
    meta = schema or context_schema(adapter.collection_name)
    created = adapter.create_collection(
        adapter.collection_name,
        meta,
        distance=adapter._distance_metric,
        sparse_weight=sparse_weight,
        index_name=adapter.index_name,
    )
    assert created is True
