# Design Document — Agentic RAG over Indian IT Financials

## 1. What the system does

This agent answers natural language questions about Infosys, TCS, and Wipro by deciding which
of three tools to call, calling them in sequence, and composing a grounded answer with
citations. It refuses gracefully when it cannot or should not answer.

---

## 2. Agent Architecture

```
User Question
     │
     ├─── Cache check (agent/cache.py)
     │    └── Cache hit → return immediately, no API calls
     │
     ▼
┌─────────────┐
│   Planner   │  1-3 sentence plan via call_llm() or rule-based (Bonus A)
└──────┬──────┘
       │ plan string
       ▼
┌──────────────────────────────────────────────────────┐
│                  Agent Loop (MAX_STEPS = 8)           │
│                                                      │
│  ┌───────────────────────────────────────────────┐   │
│  │ Decision Engine (agent/decision_engine.py)    │   │
│  │  call_llm() → Groq / Gemini / rule-based      │   │
│  └────────┬──────────────────────────────────────┘   │
│           │ AgentAction: tool | final | refuse       │
│           ▼                                          │
│  ┌───────────────────────────────────────────────┐   │
│  │ Tool Registry                                 │   │
│  │  search_docs  │  query_data  │  web_search    │   │
│  └───────────────────────────────────────────────┘   │
│           │ ToolResult + Telemetry (Bonus B)         │
│           ▼                                          │
│  context accumulates across steps                    │
└──────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────┐
│   Compose    │  call_llm() synthesis → raw format fallback
│   Answer     │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Reflector   │  Critiques answer once; may trigger one more retrieval (Bonus C)
└──────┬───────┘
       │
       ▼
  AgentResponse
  (answer + citations + trace + plan + reflection + telemetry)
       │
       ▼
┌──────────────┐
│    Cache     │  Stores response for future identical questions (excludes web_search)
└──────────────┘
```

---

## 3. LLM Interface (`agent/llm.py`)

All LLM calls in the system go through a single `call_llm(prompt, temperature)` function.
This provides automatic provider fallback based on `LLM_TYPE` in `.env`:

| `LLM_TYPE` | Primary | Fallback |
|---|---|---|
| `GROQ` (recommended) | Groq (Llama 3.3 70B) | Gemini |
| `GEMINI` | Gemini | Groq |

If both fail → raises `LLMUnavailableError`. Callers catch this and fall back to rule-based logic.

**Adding a new provider:** implement `_call_<provider>(prompt, temperature) -> str` and add it
to `PROVIDER_REGISTRY`. Nothing else changes anywhere in the codebase.

---

## 4. Tool Contracts

### `search_docs`

| Field | Value |
|---|---|
| Purpose | Semantic search over annual report PDFs |
| Input | Natural language query string |
| Output | Top-3 chunks: `{text, source, page}` |
| Use when | Qualitative info, explanations, management commentary, strategy |
| Do NOT use when | Exact numbers (use `query_data`) or live news (use `web_search`) |
| Index | FAISS (`data/faiss_index.bin`) built by `ingest.py` |
| Fallback | Mock chunks with keyword scoring when FAISS index not available |
| PDFs available | TCS Annual Report 2024-2025, Wipro Integrated Annual Report 2024-2025, Infosys Annual Report 2024-2025 |

### `query_data`

| Field | Value |
|---|---|
| Purpose | Query structured financial data from SQLite |
| Input | Natural language question about financial data |
| Output | `{sql, columns, rows, row_count}` |
| Use when | Numbers, comparisons, trends, EPS, headcount, margins, revenue |
| Do NOT use when | Qualitative reasons (use `search_docs`) or live data (use `web_search`) |
| Database | `data/financials.db` — FY15–FY24 annual + FY22–FY24 quarterly for all 3 companies |
| NL→SQL | `call_llm()` when any API key set; rule-based keyword fallback otherwise |
| Safety | Only `SELECT` statements permitted — any other statement is rejected |

### `web_search`

| Field | Value |
|---|---|
| Purpose | Live web search via Tavily API |
| Input | Short search query (under 10 words) |
| Output | Top-3 results: `{title, snippet, url, published_date}` |
| Use when | Current stock prices, recent news, live information, CEO/CFO names |
| Do NOT use when | Historical data available in DB or annual reports |
| Fallback | Mock results with keyword scoring when `TAVILY_API_KEY` not set |

---

## 5. Decision Engine (`agent/decision_engine.py`)

**LLM path** (any API key set):
- Sends question + accumulated context + tool descriptions to `call_llm()`
- Receives JSON: `{"type": "tool"|"final"|"refuse", "tool_name": ..., "input": ..., "reasoning": ...}`
- Strips markdown fences before JSON parsing

**Rule-based fallback** (all LLM providers fail):
- Checks `REFUSE_PATTERNS` (explicit investment advice) → `type="refuse"`
- Checks `WEB_KEYWORDS` (stock, current, news, nse, bse, etc.) → `web_search`
- Checks `DATA_KEYWORDS` (revenue, margin, eps, etc.) → `query_data`
- Checks `DOCS_KEYWORDS` (why, strategy, reason, etc.) → `search_docs`
- No keyword match → `type="refuse"` (never defaults to a tool blindly)

**Refusal policy:** Only explicit buy/sell/invest advice is refused. Questions about stock
prices, financial data, and company performance are always answered.

---

## 6. Preventing Infinite Loops

- `MAX_STEPS = 8` enforced in `while step < MAX_STEPS` — hard cap, not a suggestion
- Loop counter increments on every iteration including the reflection retry
- Cap reached → `status="cap_reached"` with a structured refusal message
- Reflector runs **at most once** per run (guarded by `_reflected = False` flag)
- Decision engine instructed never to call the same tool twice unless the first call failed

---

## 7. Bonus Features

**Planning step (Bonus A) — `agent/planner.py`:**
Generates a 1-3 sentence plan before the loop starts. Uses `call_llm()` for questions longer
than 15 characters; falls back to rule-based keyword plan for short inputs (greetings, typos).
Printed at the top of every trace under `PLAN:`.

**Per-tool telemetry (Bonus B) — `agent/telemetry.py`:**
`TelemetryCollector` records wall-clock latency (ms) and call count per tool per run.
Attached to `AgentResponse.telemetry`. Printed at the bottom of each trace. Aggregated across
all 20 eval questions in `evaluate.py` into a summary table saved to `evaluation_results.json`.

**Reflection step (Bonus C) — `agent/reflector.py`:**
After `_compose_answer`, critiques the answer via `call_llm()`. Returns
`{"passes": bool, "issue": str|null}`. If `passes=False` and cap not hit, one more retrieval
round is triggered using the **original question** (not the issue string — avoids the LLM
sending raw SQL as a retry query). Skipped when no API key is set.

**Response cache — `agent/cache.py`:**
File-backed JSON cache at `traces/response_cache.json`. Key: normalised question (lowercased,
whitespace-collapsed). Web-search responses excluded (time-sensitive data). Cache hits skip
all API calls and return instantly. Shown in `main.py` on startup.

---

## 8. Data Flow — Question to Answer

1. `main.py` receives question from CLI
2. Cache checked — if hit, return immediately
3. `run_agent(question)` called in `agent/agent_loop.py`
4. `generate_plan(question)` → plan string
5. Loop: `decide_next_action(question, context)` → `AgentAction`
6. If `type="tool"`: tool called, `ToolResult` + telemetry added to `context` and `trace`
7. If `type="final"`: `_compose_answer` synthesises via `call_llm()`; reflector critiques
8. If `type="refuse"`: structured refusal returned immediately
9. If loop exhausts `MAX_STEPS`: cap-reached refusal returned
10. `AgentResponse` returned with `final_answer`, `citations`, `trace`, `plan`, `reflection`, `telemetry`
11. `print_trace(response)` formats and prints the full trace
12. Response cached (if not web-search-based)
