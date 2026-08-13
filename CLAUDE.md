# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Current state

The graph, agents, tools, and data are implemented per the design in `AI Agent Use Case Plan.pdf`. Agent/tool/graph unit tests are green using a fake LLM (no API key required to run the suite). Nothing has been run end-to-end against a real LLM provider in this environment (no credentials configured) — `app/main.py` and `evaluation/evaluate.py` are implemented but unverified against an actual model; treat their control flow as reviewed, not their live behavior.

## Package naming constraint

The Python package is `app/`, not `src/` — `src` is on `langgraph-cli`'s hardcoded reserved-name list (along with `langgraph`, `pydantic`, `fastapi`, etc., see `.venv/Lib/site-packages/langgraph_cli/config.py`) and building the LangGraph Platform Docker image fails immediately if the local dependency directory is named `src`. Don't rename it back.

## Commands

```bash
# Setup (Windows; use source .venv/bin/activate on macOS/Linux)
python -m venv .venv
.venv/Scripts/activate
pip install -r requirements.txt

# Run the whole suite
pytest

# Run one file / one test
pytest tests/test_graph.py
pytest tests/test_graph.py::test_reflection_loop_is_bounded_then_proceeds

# Run a ticket through the graph (needs a configured LLM provider, see .env.example)
python -m app.main "I was charged twice for my subscription, please refund it."

# Run the evaluation dataset end-to-end (needs a configured LLM provider)
python -m evaluation.evaluate

# LangGraph CLI dev server (Studio-compatible, hot reload)
langgraph dev

# Rebuild the deployment Dockerfile/.dockerignore/docker-compose.yml from
# langgraph.json if the graph path, dependencies, python_version, or
# dockerfile_lines change (needs no running Docker daemon for this step -
# only `docker compose up`/`docker build` itself does):
python -m langgraph_cli dockerfile --add-docker-compose -c langgraph.json Dockerfile

# Full containerized stack (verified working end-to-end: builds, migrates,
# imports the support_agent graph, serves /ok and /assistants/search on
# host port 8123 -> container 8000):
docker compose up --build
```

`Dockerfile`, `.dockerignore`, and `docker-compose.yml` are generated output, not hand-authored — regenerate them with the command above rather than hand-editing after a `langgraph.json` change (each regen with `--add-docker-compose` resets `.dockerignore` to `langgraph_cli`'s generic template, so re-add the repo-specific excludes — `.venv`, `.langgraph_api`, `.pytest_cache`, `tests`, the PDF — if you rerun it). The `langgraph-api` image is **not** standalone: it hard-requires `REDIS_URI`/`POSTGRES_URI` env vars pointing at live services, which only `docker-compose.yml`'s three-service stack (`langgraph-redis`, `langgraph-postgres`, `langgraph-api`) provides — a bare `docker run` on the built image fails with `KeyError: "Config 'REDIS_URI' is missing..."`. Always use `docker compose up`, never `docker run` directly on this image.

CORS (`langgraph.json`'s `"http.cors"` → baked in as `ENV LANGGRAPH_HTTP`, read by `langgraph_api` at startup into `CORS_CONFIG`, applied via Starlette's `CORSMiddleware` — traced through `.venv/Lib/site-packages/langgraph_api/{server.py,config/__init__.py}` to confirm, not guessed) only solves *cross-origin* blocking. It does **not** solve mixed-content blocking: Studio (`https://smith.langchain.com`) is HTTPS and browsers refuse to even attempt a fetch to a plain `http://` resource, CORS headers or not. That's what `docker-compose.override.yml` + `nginx/nginx.conf` + `scripts/generate_self_signed_cert.sh` are for — a TLS-terminating reverse proxy in front of `langgraph-api`, kept in a separate override file specifically so regenerating `docker-compose.yml` never deletes it. The cert's SAN must match whatever host/IP the browser actually connects to (`scripts/generate_self_signed_cert.sh <ip-or-hostname>` handles both `IP:` and `DNS:` SAN forms) — a mismatch means the browser rejects it outright, not just warns. Self-signed also means a human has to visit `https://<host>` directly once and accept the browser warning before Studio's background fetches will succeed; there is no config knob to skip that. On Windows dev boxes specifically, ports 80/443 may be in Windows' *own* excluded TCP port range (Hyper-V/WSL dynamic ports) independent of Docker — `netsh interface ipv4 show excludedportrange protocol=tcp` shows this; it's a host quirk, not a compose/proxy bug, and doesn't reproduce on a normal Linux host.

`pytest.ini` sets `pythonpath = .` so `app...` imports resolve regardless of how pytest is invoked — don't remove it or add a conflicting `rootdir`/import-mode setting without checking imports still work.

## Project purpose

This is a learning project for the LangChain / LangGraph / LangSmith agentic stack, built around a deliberately small use case: a **Customer Support Ticket Triage & Resolution Agent** for a fictional SaaS company ("AcmeCloud"). Explicitly out of scope: RAG, vector DBs, web search, long-term memory, auth, a frontend, real payment integration — don't add these without the user asking.

## Domain model

Three JSON-backed data types in `app/data/`, loaded fresh from disk on every tool call (no caching, no DB):

- **Customers** (`customers.json`): `customer_id`, `name`, `subscription`, `account_status`
- **Payments** (`payments.json`): `payment_id`, `customer_id`, `amount`, `date`, `status`
- **Support policies** (`policies.json`): `policy_id`, `category`, `policy_text`

Business rules encoded in `app/config.py` / `app/agents/resolution_agent.py` (not just left to the LLM's judgment — see "Deterministic rule enforcement" below): duplicate payment → auto-refund eligible; refund > `AUTO_REFUND_THRESHOLD` ($100 default) → requires human approval; account cancellation → always requires human approval; purely informational requests → no approval.

## Architecture

LangGraph state machine (`app/graph/support_graph.py`), ticket in, customer-facing string out:

```
START → ticket_analyzer → {billing_agent, account_agent}   (fan-out, run in parallel)
                              ↓ (fan-in once both finish)
                          policy_agent → resolution_agent → reflection_agent
                                                                  │
                             ┌────────────────────────────────────┤
                       not approved,                         approved, or
                    reflection_count < MAX_REFLECTION_CYCLES   cap reached
                             │                                    │
                             ▼                                    ▼
                       resolution_agent                requires_human_approval?
                        (retry, loops                    │              │
                         back into reflection)           yes            no
                                                          ▼              │
                                                     human_review        │
                                              (interrupt()/Command       │
                                               pause-resume)             │
                                        ┌──────────┴──────────┐          │
                                  request_more_investigation   approve/reject
                                        │                          │
                             {billing_agent, account_agent}        │
                             (re-fan-out, bounded by               │
                              MAX_REINVESTIGATION_CYCLES,          │
                              re-runs the whole downstream         │
                              pipeline)                            │
                                                                    ▼
                                                               summarizer → END
```

Key distinctions the code preserves — don't collapse these when touching agents:
- **Investigation vs. decision**: `billing_agent`/`account_agent` only investigate; `policy_agent` decides what's *allowed*; `resolution_agent` decides what action to *take*.
- **Reasoning state vs. customer-facing output**: `summarizer_agent` only renders the already-decided resolution (plus the human's decision, if any) into text — it never investigates or re-decides.
- **Deterministic rule enforcement**: `resolution_agent.py`'s `_enforce_business_rules()` overrides `requires_human_approval` for refunds over threshold and all cancellations after the LLM proposes a resolution, as a safety net independent of the LLM's own judgment and the Reflection Agent's critique.
- **Bounded loops**: `reflection_count` (cap: `MAX_REFLECTION_CYCLES`, default 2) and `reinvestigation_count` (cap: `MAX_REINVESTIGATION_CYCLES`, default 1) are incremented inside the node functions (`app/graph/nodes.py`), not the conditional-edge routing functions (`app/graph/edges.py`) — LangGraph routing functions only read state, they can't write to it.

### Human-in-the-loop mechanics

`human_review_node` calls `langgraph.types.interrupt()`, which suspends the graph (persisted via the `MemorySaver` checkpointer passed to `build_graph()`). The caller (`app/main.py`, `evaluation/evaluate.py`) checks `"__interrupt__" in result` and resumes with `graph.invoke(Command(resume=decision), config=config)`, where `decision` is one of `"approve"`, `"reject"`, `"request_more_investigation"`. Every invoke/resume in a single ticket's lifecycle must reuse the same `thread_id` in `config` — a new thread_id starts a fresh, unrelated run.

### Directory-to-role mapping

- `app/state/support_state.py` — the `SupportState` TypedDict threaded through the graph, plus every Pydantic schema (`TicketAnalysis`, `BillingInvestigation`, `AccountInvestigation`, `PolicyDecision`, `Resolution`, `ReflectionResult`) used as `llm.with_structured_output(...)` targets
- `app/graph/nodes.py` — node functions wrapping each agent; owns the loop counters
- `app/graph/edges.py` — `route_after_reflection`, `route_after_human_review`: pure functions reading state, returning next node name(s)
- `app/graph/support_graph.py` — wires nodes + edges, compiles with a checkpointer
- `app/agents/_common.py` — `run_tool_calling_agent` (bounded bind_tools loop) and `structured_synthesis` (second LLM call that turns a tool-calling transcript into a Pydantic result); shared by `billing_agent.py`, `account_agent.py`, `policy_agent.py`
- `app/agents/ticket_analyzer.py`, `resolution_agent.py`, `reflection_agent.py`, `summarizer_agent.py` — no tools; single structured (or plain, for the summarizer) LLM call
- `app/tools/*.py` — `@tool`-decorated functions reading the JSON fixtures directly (`find_duplicate_payments` does the same-amount/within-N-days matching; `refund_eligibility` applies the $-threshold rule)
- `app/models/llm.py` — `get_llm()`, a `ChatOpenAI` pointed at whatever `OPENAI_BASE_URL` is set to, so any OpenAI-compatible provider (OpenAI, Groq, a local Ollama/vLLM server) works without code changes
- `evaluation/dataset.json` + `evaluate.py` — ~16 tickets with expected category/resolution_type/requires_human_approval; `evaluate.py` auto-approves any interrupt so batches run unattended, and scores policy compliance by whether approval *was requested*, not by emulating a human's actual decision

## Testing approach

Every agent function takes an optional `llm=` parameter for dependency injection. Tests never call a real provider:
- `tests/conftest.py`'s `FakeLLM` stands in for a chat model (`bind_tools`, `with_structured_output`, `invoke`) and is injected via the `fake_llm_factory` fixture.
- `tests/test_graph.py` monkeypatches the agent functions *imported into `app.graph.nodes`* (e.g. `nodes_module.propose_resolution`), not the underlying agent modules — node functions resolve those names from `nodes.py`'s own module globals at call time, so patching anywhere else won't take effect.
- `tests/test_tools.py` hits the real JSON fixtures in `app/data/` with no mocking.

When adding a new agent or node, follow this pattern rather than reaching for a mocking framework.
