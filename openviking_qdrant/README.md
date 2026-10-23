# openviking_qdrant — third-party Qdrant backend for OpenViking

A standalone, dependency-free Qdrant vector-backend package for **OpenViking
v0.5.1.1**. It plugs into the upstream dynamic adapter factory through a dotted
class path, so **no tracked OpenViking file is modified**.

```yaml
storage:
  vectordb:
    backend: openviking_qdrant.adapter.QdrantCollectionAdapter
    name: context
    project: openviking
    dimension: 1024
    distance_metric: cosine
    sparse_weight: 0.3          # optional; > 0 enables sparse/hybrid
    url: http://127.0.0.1:6333
    custom_params:
      api_key: null             # or OPENVIKING_QDRANT_API_KEY / QDRANT_API_KEY
      timeout_seconds: 10
      dense_vector_name: vector
      sparse_vector_name: sparse_vector
      # metadata_collection_name: null   # default: <physical>__openviking_meta
      # query_candidate_cap: 100000      # time-decay candidate amplification cap
      # batch_size: 100                  # id delete/fetch batch size
      # max_retries: 2                   # transient failures are retried
      # backoff_base_seconds: 0.2
      # backoff_max_seconds: 2.0
```

`storage.vectordb.backend` contains a dot, so upstream
`VectorDBBackendConfig.validate_config` skips the standard-backend whitelist and
`create_collection_adapter()` imports the class dynamically. The class is a
`CollectionAdapter` subclass and is constructed with `from_config(config)`.

## Compatibility

| Component | Supported |
| --- | --- |
| OpenViking | v0.5.1.1 (`c2dea8005`); other 0.5.x only via `check_core_compatibility()` warning |
| Python | 3.10+ (stdlib only) |
| Qdrant | REST API with named dense vectors, sparse vectors, universal `/points/query`, scroll `order_by` (tested against the 1.13 response shapes; 1.10+ expected) |
| New dependencies | none (tests use `pytest` only) |
| Content / BM25 grep | no (`USE_CONTENT_FIELD = False`; upstream falls back to filesystem) |
| Multimodal search | no |
| `search_by_keywords` | explicit `NotImplementedError`; use filesystem grep |
| `Contains` | filter-only client-side fallback; rejected for vector/scalar search |

The package is auto-included by the repository's existing setuptools rule
`[tool.setuptools.packages.find] include = ["openviking*"]`.

### Docker

Place this line **before** the dependency sync/install step so the package is
present in the image, or copy it directly into `site-packages` of the final
image:

```dockerfile
COPY openviking_qdrant/ openviking_qdrant/
```

## Transport: retries, backoff and error classification

All operations performed by the adapter are idempotent (stable point ids,
filter deletes, collection/marker puts), so transient failures are retried:

* retried: connection reset/refused, timeout, HTTP `408/425/429/5xx`;
* not retried: `400/401/403/404/409/422` and malformed JSON responses;
* bounded exponential backoff (`backoff_base_seconds`, `backoff_max_seconds`);
* `Retry-After` on `429` is honoured (bounded to 30 s);
* every raised `QdrantError` carries `status`/`status_code`, `retryable`,
  `action` and `attempts`, which keeps logs actionable;
* retries are configurable per deployment (`custom_params.max_retries`); set
  `0` to fail fast.

## Diagnostics

Read-only pre-flight/reuse check:

```bash
python -m openviking_qdrant.diagnose \
  --url http://127.0.0.1:6333 --project openviking --collection context \
  --dimension 1024 --json
```

The report covers server reachability/version, physical collection presence and
point count, metadata marker validity/schema/vector-dim/distance, sparse term
dictionary size, and a `reuse` verdict with reasons/actions. Exit codes: `0`
reachable (reusable or fresh start), `1` reachable but not reusable with this
config, `2` unreachable.

The adapter exposes the same report programmatically:

```python
adapter.health_check()        # full diagnostics dict
adapter.reuse_assessment()    # report["reuse"]
```

## Data layout (0.4.16-fork compatible)

Collections remain readable by, and interchangeable with, the historical
`upgrade/v0.4.16-qdrant` fork:

| Item | Layout |
| --- | --- |
| Physical collection | `<project>__<collection>` |
| Metadata collection | `<physical>__openviking_meta` (overridable) |
| Metadata marker | point `uuid5("openviking:metadata")`, payload `_openviking_meta_version = 1` |
| Data point ID | `uuid5(namespace, str(openviking_id))` |
| Source ID | payload `_openviking_original_id` |
| URI | payload without `viking://`; `parent_uri` likewise |
| Path scope | `uri_depth` + `scope_roots` (self and every ancestor) |
| Sparse terms | metadata points `{term, index}`; stable SHA-256 / `0x7FFF_FFFF` indices |
| Index metadata | marker payload `indexes`, with `uri_depth`/`scope_roots` in remote payload indexes |

`tests/test_layout_constants.py` hard-codes the 0.4.16 point/marker/sparse IDs
so future refactors cannot silently break the byte-compatible layout.

Legacy collection schemas (created before v0.5.1.1) are **not rewritten**.
`get_meta_data()` returns a read-only compatibility view that adds the
`md5`, `acl_mode`, `acl_direct_grants` and `acl_inherited_grants` field
definitions (and ACL scalar indexes), so upstream's `_filter_known_fields`
accepts new writes on old collections. Records that physically lack those keys
still read as absent: missing `md5` means "unknown", and missing `acl_mode` is
treated by upstream's ACL filter as the default shared state. Run the reindex
when the new schema should be persisted.

## Query semantics

| Path | Behaviour |
| --- | --- |
| Dense vector | Qdrant `/points/query` with the configured named dense vector |
| Sparse vector | Qdrant sparse named vector; terms mapped through the durable dictionary |
| Dense + sparse | client-side weighted RRF (`sparse_weight` as alpha, `k = 60`), then `offset`/`limit` |
| Filter only | Qdrant scroll, **never a synthetic random dense vector** (strict-mode safe) |
| `order_by` | Qdrant scroll `order_by`, page size 256, client-side offset slice |
| `advance.time_decay` | implemented locally; see below |
| `advance.post_process_ops` | explicit `NotImplementedError` |
| `search_by_random` | scroll-based stable enumeration; `advance` is rejected |
| `count` | Qdrant exact count; `Contains` fallback counts client-side |
| `delete(filter)` | Qdrant native count + delete; `Contains` fallback enumerates IDs first |

### Event time decay

The adapter compiles the upstream request with
`build_time_decay_fusion_spec()` and the collection applies the same curve as
the native engine:

```
rate       = ln(decay) / scale_ms                 # fixed 0.5 / 7d
time_score = exp(rate * max(0, abs(origin_ms - updated_at_ms) - protection_ms))
fused      = similarity * time_score
candidates = min((limit + offset) * 3, query_candidate_cap)   # native cap = 100000
```

`query_candidate_cap` defaults to the native `100000`; lower it via
`custom_params` when Qdrant strict mode (`max_query_limit`) would reject the
amplified candidate window. Results expose `_score` (fused), `_origin_score`
and `_time_score` exactly like the local/VikingDB adapters. Missing or
unparsable `updated_at` leaves the original score unchanged and omits
`_time_score`. Future timestamps decay too (`abs`), matching the C++ engine.
Dense-only, sparse-only, and hybrid queries all support decay. A malformed
compiled spec raises `ValueError`; non-decay `advance` keys raise
`NotImplementedError`.

### `Contains`

Qdrant has no faithful substring operator. The adapter therefore:

* rejects `Contains` for vector/scalar search (no lossy top-k approximation);
* for **filter-only** queries, scans scroll pages and evaluates the full filter
  in Python. This is what `vector_migration` uses for `uri + "#"` fragment
  scopes. It is a client-side full scan, not an indexed query;
* supports the same fallback for `count()` and `delete(filter=...)`.

URI substrings are normalized (`viking://a` → `/a`) before comparison.

## ACL and md5

All three ACL columns and `md5` are ordinary Qdrant payload fields. They are
written and read unchanged, including list-valued grants, so upstream
`Eq`/`In`/`RawDSL must_not` ACL filters work. The adapter does not invent ACL
defaults for old records: upstream semantics intentionally treat missing
`acl_mode` as default shared. Qdrant filters work without a payload index; the
package creates keyword payload indexes for ACL fields when they are listed in
the schema's `ScalarIndex` (new collections) and returns them in the legacy
schema compatibility view.

## Qdrant strict mode

Filter-only enumeration uses `/points/scroll`, not `/points/query`, so strict
mode's `max_query_limit` does not affect `limit=100000` migration/cleanup
scans. Vector queries and time-decay amplification still use `/points/query`;
lower `custom_params.query_candidate_cap` if your strict mode uses a lower
`max_query_limit`.

## Testing

The suite is fully offline: `tests/fake_qdrant.py` is an independent in-memory
Qdrant REST double (filter evaluation, dense/sparse scoring, scroll, count,
delete, payload indexes, strict-mode limit enforcement).

```bash
cd <worktree>
PYTHONPATH=. /path/to/python -m pytest openviking_qdrant/tests -q
```

Current integration branch: **100 offline tests** covering the adapter,
collection, filter compiler, sparse dictionary, time decay, retry/backoff,
`get_meta_data(raise_on_error=...)` quadrants, legacy-schema compatibility and
the diagnostics CLI. `openviking_qdrant/pytest.ini` scopes pytest to this
package and avoids the repository-wide `--cov` addopts, which would require
`pytest-cov` in minimal runtime environments. No test requires a live Qdrant
server.

### Pre-production checklist for a real Qdrant

The suite never talks to a real server. Before switching traffic:

1. `python -m openviking_qdrant.diagnose ...` against the real endpoint and
   confirm the expected reuse verdict.
2. Create the collection from the server once and verify the collection schema
   and payload indexes with `GET /collections/<physical>`.
3. Add a resource, query it, filter it, and run one `events_time_decay`
   search; check `_score/_origin_score/_time_score`.
4. Run a reindex smoke on a small namespace; watch retries/errors in logs.
5. Verify ACL filtering with a non-owner user key (missing-`acl_mode` records
   must still be visible as shared).
6. Stop one server briefly / return 503 once to confirm retry behaviour, then
   ensure idempotent upserts do not duplicate points.

## Operational notes

* The first use of a new sparse term writes one metadata point per term. A
  per-process cache avoids repeated metadata round trips; collisions are
  rejected, never merged.
* `delete(filter=...)` without `Contains` is one count + one server-side
  delete, independent of match-set size.
* An existing physical collection without a valid OpenViking marker fails
  closed with an actionable error; drop/rename the stale collection or restore
  the sidecar. `python -m openviking_qdrant.diagnose` reports this case.
* `get_meta_data(raise_on_error=False)` degrades to `{}` on transport failures
  (upstream probes use this mode); `raise_on_error=True` still surfaces the
  error.
* `Ttl` is accepted but ignored (Qdrant has no per-point TTL).
