"""Central configuration for the support agent, loaded from environment variables."""
import os

from dotenv import load_dotenv

load_dotenv()

# LLM provider settings. Any OpenAI-compatible endpoint works here (OpenAI itself,
# Groq, or a local Ollama/vLLM server exposing an OpenAI-compatible API) - just
# point OPENAI_BASE_URL at it and set OPENAI_MODEL to a model it serves.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL") or None
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))

# LangSmith tracing. Set LANGCHAIN_TRACING_V2=true and LANGCHAIN_API_KEY to enable.
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "false")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "customer-support-agent")

# Business rules (see app/data/policies.json for the human-readable policy text).
AUTO_REFUND_THRESHOLD = float(os.getenv("AUTO_REFUND_THRESHOLD", "100"))

# Loop guards for the graph's self-correcting cycles.
MAX_REFLECTION_CYCLES = int(os.getenv("MAX_REFLECTION_CYCLES", "2"))
MAX_REINVESTIGATION_CYCLES = int(os.getenv("MAX_REINVESTIGATION_CYCLES", "1"))
