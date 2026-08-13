"""Shared test doubles.

Agent and graph tests never call a real LLM provider - they use FakeLLM, a
minimal stand-in that implements just the surface the agents use
(bind_tools/with_structured_output/invoke), so the suite runs with no network
access and no API key.
"""
from types import SimpleNamespace

import pytest


class _FakeToolCallingModel:
    def __init__(self, ai_messages):
        self._ai_messages = list(ai_messages)
        self._index = 0

    def invoke(self, messages):
        message = self._ai_messages[self._index]
        self._index += 1
        return message


class _FakeStructuredModel:
    def __init__(self, response):
        self._response = response

    def invoke(self, messages):
        return self._response


class FakeLLM:
    def __init__(self, tool_ai_messages=None, structured_responses=None, plain_response=None):
        self.tool_ai_messages = tool_ai_messages or []
        self.structured_responses = structured_responses or {}
        self.plain_response = plain_response

    def bind_tools(self, tools):
        return _FakeToolCallingModel(self.tool_ai_messages)

    def with_structured_output(self, schema):
        return _FakeStructuredModel(self.structured_responses[schema])

    def invoke(self, messages):
        return SimpleNamespace(content=self.plain_response)


@pytest.fixture
def fake_llm_factory():
    return FakeLLM
