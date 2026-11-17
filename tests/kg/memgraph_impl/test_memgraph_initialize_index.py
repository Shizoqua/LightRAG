"""initialize() must backtick-quote the workspace label in CREATE INDEX.

Every other query in this module addresses the workspace label as a
backtick-quoted identifier (``:`{workspace_label}` ``), which is required
whenever the label is not a bare alphanumeric/underscore token -- a workspace
name such as ``my-workspace`` is valid per ``validate_workspace`` (hyphens
and dots are allowed) but is not a legal unquoted Cypher label. The
``CREATE INDEX`` statement in ``initialize()`` built its label without
backticks, so for such a workspace the statement is syntactically invalid:
the index is silently never created (the surrounding ``except`` logs it as
"may already exist"), leaving every subsequent lookup on that workspace
without the entity_id index.
"""

import pytest

from lightrag.kg.memgraph_impl import MemgraphStorage
from lightrag.kg.shared_storage import initialize_share_data


pytestmark = pytest.mark.offline


class _FakeResult:
    async def single(self):
        return None

    async def consume(self):
        return None


class _FakeSession:
    def __init__(self, calls):
        self._calls = calls

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def run(self, query, *args, **kwargs):
        self._calls.append(query)
        return _FakeResult()


class _FakeDriver:
    def __init__(self, calls):
        self._calls = calls

    def session(self, **kwargs):
        return _FakeSession(self._calls)

    async def close(self):
        return None


@pytest.mark.asyncio
async def test_initialize_backtick_quotes_hyphenated_workspace_label(monkeypatch):
    initialize_share_data()
    calls: list[str] = []

    def _fake_driver_factory(*args, **kwargs):
        return _FakeDriver(calls)

    monkeypatch.setattr(
        "lightrag.kg.memgraph_impl.AsyncGraphDatabase.driver",
        _fake_driver_factory,
    )

    storage = MemgraphStorage(
        namespace="chunk_entity_relation",
        global_config={"max_graph_nodes": 1000},
        embedding_func=None,
        workspace="my-workspace",
    )

    await storage.initialize()

    index_queries = [q for q in calls if "CREATE INDEX" in q]
    assert len(index_queries) == 1
    assert "`my-workspace`" in index_queries[0]
    # Guard against a label fragment leaking outside the backticks, which
    # would still be syntactically invalid Cypher.
    assert "ON :my-workspace(" not in index_queries[0]
