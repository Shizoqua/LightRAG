"""llama_index_complete() is the documented entry point for wiring a
LlamaIndex LLM into LightRAG's llm_model_func. LightRAG injects hashing_kv
into every llm_model_func call for caching, and the documented usage
pattern (see examples/unofficial-sample/lightrag_llamaindex_direct_demo.py)
passes the chosen llm_instance through kwargs. Neither reaches
llama_index_complete_if_cache cleanly unless llama_index_complete strips
hashing_kv and forwards llm_instance exactly once.

lightrag/llm/llama_index_impl.py imports llama_index at module level, which
is not a project dependency, so it is stubbed here.
"""

from __future__ import annotations

import importlib
import sys
import types

import pytest

pytestmark = pytest.mark.offline


def install_fake_llama_index(monkeypatch):
    fake_llms = types.ModuleType("llama_index.core.llms")

    class MessageRole:
        SYSTEM = "system"
        USER = "user"
        ASSISTANT = "assistant"

    class ChatMessage:
        def __init__(self, role=None, content=None):
            self.role = role
            self.content = content

    class ChatResponse:
        pass

    fake_llms.MessageRole = MessageRole
    fake_llms.ChatMessage = ChatMessage
    fake_llms.ChatResponse = ChatResponse

    fake_embeddings = types.ModuleType("llama_index.core.embeddings")

    class BaseEmbedding:
        pass

    fake_embeddings.BaseEmbedding = BaseEmbedding

    fake_settings_mod = types.ModuleType("llama_index.core.settings")

    class Settings:
        @classmethod
        def set_global(cls, settings):
            pass

    fake_settings_mod.Settings = Settings

    fake_core = types.ModuleType("llama_index.core")
    fake_core.llms = fake_llms
    fake_core.embeddings = fake_embeddings
    fake_core.settings = fake_settings_mod

    fake_llama_index = types.ModuleType("llama_index")
    fake_llama_index.core = fake_core

    monkeypatch.setitem(sys.modules, "llama_index", fake_llama_index)
    monkeypatch.setitem(sys.modules, "llama_index.core", fake_core)
    monkeypatch.setitem(sys.modules, "llama_index.core.llms", fake_llms)
    monkeypatch.setitem(sys.modules, "llama_index.core.embeddings", fake_embeddings)
    monkeypatch.setitem(sys.modules, "llama_index.core.settings", fake_settings_mod)

    import pipmaster as pm

    monkeypatch.setattr(pm, "is_installed", lambda name: True)


@pytest.fixture
def llama_index_module(monkeypatch):
    install_fake_llama_index(monkeypatch)
    sys.modules.pop("lightrag.llm.llama_index_impl", None)
    return importlib.import_module("lightrag.llm.llama_index_impl")


class FakeChatResponse:
    def __init__(self, content):
        self.message = types.SimpleNamespace(content=content)


class FakeLLM:
    def __init__(self):
        self.calls = []

    async def achat(self, messages, **chat_kwargs):
        self.calls.append({"messages": messages, "chat_kwargs": chat_kwargs})
        return FakeChatResponse("answer")


@pytest.mark.asyncio
async def test_llama_index_complete_forwards_llm_instance_and_strips_hashing_kv(
    llama_index_module,
):
    llm = FakeLLM()

    result = await llama_index_module.llama_index_complete(
        "hello",
        system_prompt="be nice",
        history_messages=[],
        llm_instance=llm,
        hashing_kv=object(),
    )

    assert result == "answer"
    assert len(llm.calls) == 1


@pytest.mark.asyncio
async def test_llama_index_complete_without_llm_instance_reports_the_real_error(
    llama_index_module,
):
    """No llm_instance means model=None reaches llama_index_complete_if_cache,
    which fails on None.achat(...) -- that AttributeError, not a spurious
    TypeError about unexpected keyword arguments, is what callers should see."""
    with pytest.raises(AttributeError):
        await llama_index_module.llama_index_complete("hello", hashing_kv=object())
