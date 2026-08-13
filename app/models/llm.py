"""LLM client construction, shared by every agent."""
from langchain_openai import ChatOpenAI

from app import config


def get_llm(temperature: float | None = None) -> ChatOpenAI:
    """Return a chat model configured from environment settings.

    Any OpenAI-compatible provider works by pointing OPENAI_BASE_URL at it
    (see app/config.py) - no code changes needed to switch providers.
    """
    return ChatOpenAI(
        model=config.OPENAI_MODEL,
        api_key=config.OPENAI_API_KEY or "not-needed-for-local-providers",
        base_url=config.OPENAI_BASE_URL,
        temperature=config.LLM_TEMPERATURE if temperature is None else temperature,
    )
