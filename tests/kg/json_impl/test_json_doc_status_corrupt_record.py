"""A stored doc-status payload that is not a row must be named, not handed
back raw.

``JsonKVStorage`` raises ``CorruptStorageRecordError`` from ``get_by_id`` /
``get_by_ids`` when a key maps to something other than a mapping (a
hand-edited file, a truncated one repaired by hand, a foreign writer) --
see ``test_json_kv_corrupt_record.py``. ``JsonDocStatusStorage`` reimplements
the same shared-memory protocol rather than inheriting it, and its
``get_by_id`` / ``get_by_ids`` / ``get_by_id_strict`` never picked up that
guard: they returned the raw non-mapping value as-is instead of raising or
treating it as absent.

That is reachable through the public contract: ``get_by_id`` is declared to
return ``dict[str, Any] | None`` (``BaseKVStorage.get_by_id``), and
``get_by_id_strict`` additionally promises complete-or-raise semantics
(``supports_strict_point_reads = True``) -- "any state where absence cannot
be positively confirmed MUST raise instead of returning None". Handing back
a bare string instead violates both. Callers across ``lightrag.py`` and
``pipeline.py`` immediately do ``(row or {}).get("status")`` or similar on
the result, so a single damaged doc-status record used to crash with
``AttributeError: 'str' object has no attribute 'get'`` deep inside caller
code instead of a named error at the storage boundary.
"""

import json

import pytest

from lightrag.exceptions import CorruptStorageRecordError
from lightrag.kg.json_doc_status_impl import JsonDocStatusStorage
from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data

pytestmark = pytest.mark.offline


class _DummyEmbeddingFunc:
    embedding_dim = 1
    max_token_size = 1

    async def __call__(self, texts, **kwargs):
        return [[0.0] for _ in texts]


@pytest.fixture(autouse=True)
def setup_shared_data():
    initialize_share_data()
    yield
    finalize_share_data()


async def _storage_over(tmp_path, payload: dict):
    """A storage whose file already holds ``payload``, written behind its back."""
    workspace_dir = tmp_path / "ws"
    workspace_dir.mkdir(parents=True, exist_ok=True)
    (workspace_dir / "kv_store_doc_status.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    storage = JsonDocStatusStorage(
        namespace="doc_status",
        global_config={"working_dir": str(tmp_path)},
        embedding_func=_DummyEmbeddingFunc(),
        workspace="ws",
    )
    await storage.initialize()
    return storage


@pytest.mark.asyncio
async def test_get_by_id_names_the_damaged_record(tmp_path):
    storage = await _storage_over(
        tmp_path,
        {
            "doc-1": "garbage",
            "doc-2": {"status": "pending", "file_path": "fine.pdf"},
        },
    )

    with pytest.raises(CorruptStorageRecordError) as excinfo:
        await storage.get_by_id("doc-1")

    message = str(excinfo.value)
    assert "doc-1" in message, "the damaged key must be named"
    assert "str" in message and "not a mapping" in message
    # The healthy neighbour is unaffected: this is a per-row verdict.
    assert (await storage.get_by_id("doc-2"))["file_path"] == "fine.pdf"


@pytest.mark.asyncio
async def test_get_by_ids_names_the_damaged_record(tmp_path):
    storage = await _storage_over(tmp_path, {"doc-1": ["not", "a", "row"]})

    with pytest.raises(CorruptStorageRecordError):
        await storage.get_by_ids(["doc-1"])


@pytest.mark.asyncio
async def test_get_by_id_strict_names_the_damaged_record(tmp_path):
    """The strict point-read contract promises complete-or-raise: any state
    where absence cannot be positively confirmed MUST raise, never return a
    bare non-mapping value in place of ``dict | None``."""
    storage = await _storage_over(tmp_path, {"doc-1": 42})

    with pytest.raises(CorruptStorageRecordError):
        await storage.get_by_id_strict("doc-1")


@pytest.mark.asyncio
async def test_a_missing_key_is_still_a_confirmed_absence(tmp_path):
    """The guard must not turn a miss into a refusal: absent stays absent."""
    storage = await _storage_over(
        tmp_path, {"doc-1": {"status": "pending", "file_path": "fine.pdf"}}
    )

    assert await storage.get_by_id("nope") is None
    assert await storage.get_by_id_strict("nope") is None
    assert await storage.get_by_ids(["nope"]) == [None]
