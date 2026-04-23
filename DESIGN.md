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
     ▼
┌─────────────┐
│   Planner   │  Generates a 1-3 sentence plan before any tool is called (Bonus A)
└──────┬──────┘
       │ plan string
       ▼
┌─────────────────────────────────────────────────────┐
│                   Agent Loop (MAX_STEPS = 8)         │
│                                                     │
│  ┌──────────────────────────────────────────────┐   │
│  │ Decision Engine (agent/decision_engine.py)   │   │
│  │  call_llm() → Groq / Gemini / rule-based     │   │
│  └────────┬─────────────────────────────────────┘   │
│           │ AgentAction (tool / final / refuse)     │
│           ▼                                         │
│  ┌──────────────────────────────────────────────┐   │
│  │ Tool Registry                                │   │
│  │  search_docs  │  query_data  │  web_search   │   │
│  └──────────────────────────────────────────────┘   │
│           │ ToolResult + Telemetry (Bonus B)        │
│           ▼                                         │
│  context accumulates across steps                   │
└─────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────┐
│ Compose      │  call_llm() synthesis → raw format fallback
│ Answer       │
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
│    Cache     │  Stores response for identical future questions (agent/cache.py)
└──────────────┘
```

---

## 3. LLM Interface (`agent/llm.py`)

All LLM calls in the system go through a single `call_llm(prompt, temperature)` function.
This provides automatic provider fallback based on `LLM_TYPE` in `.env`:

- `LLM_TYPE=GROQ` → tries Groq first, falls back to Gemini
- `LLM_TYPE=GEMINI` → tries Gemini first, falls back to Groq
- If both fail → raises `LLMUnavailableError`, caller falls back to rule-based logic

Adding a new provider requires only implementing `_call_<provider>(prompt, temperature) -> str`
and registering it in `PROVIDER_REGISTRY`. Nothing else changes.

---

## 4. Tool Contracts

### `search_docs`

| Field | Value |
|---|---|
| Purpose | Semantic search over annual report PDFs |
| Input | Natural language query string |
| Output | Top-3 chunks: `{text, source, page}` |
| Use when | Question asks for qualitative info, explanations, management commentary, strategy |
| Do NOT use when | Question asks for exact numbers (use `query_data`) or live news (use `web_search`) |
| Fallback | Mock chunks with keyword scoring when FAISS index not available |

### `query_data`

| Field | Value |
|---|---|
| Purpose | Query structured financial data from SQLite (`data/financials.db`) |
| Input | Natural language question about financial data |
| Output | `{sql, columns, rows, row_count}` |
| Use when | Question asks for specific numbers, comparisons, trends, EPS, headcount, margins |
| Do NOT use when | Question asks for qualitative reasons (use `search_docs`) or live data (use `web_search`) |
| NL→SQL | LLM-generated SQL via `call_llm()` when any API key is set; rule-based keyword fallback otherwise |
| Safety | Only `SELECT` statements permitted — any other statement is rejected |

### `web_search`

| Field | Value |
|---|---|
| Purpose | Live web search via Tavily API |
| Input | Short search query (under 10 words) |
| Output | Top-3 results: `{title, snippet, url, published_date}` |
| Use when | Question asks for current stock prices, recent news, live information, CEO/CFO names |
| Do NOT use when | Historical data is available in the DB or annual reports |
| Fallback | Mock results with keyword scoring when `TAVILY_API_KEY` not set |

---

## 5. Decision Engine (`agent/decision_engine.py`)

**LLM path** (any API key set):
- Sends the question + accumulated context + tool descriptions to `call_llm()`
- Receives JSON: `{"type": "tool"|"final"|"refuse", "tool_name": ..., "input": ..., "reasoning": ...}`
- Strips markdown fences before JSON parsing

**Rule-based fallback** (all LLM providers fail):
- Checks `REFUSE_PATTERNS` (explicit investment advice) → returns `type="refuse"`
- Checks `WEB_KEYWORDS` (stock, current, news, etc.) → routes to `web_search`
- Checks `DATA_KEYWORDS` (revenue, margin, eps, etc.) → routes to `query_data`
- Checks `DOCS_KEYWORDS` (why, strategy, reason, etc.) → routes to `search_docs`
- If no keyword matches → returns `type="refuse"` (never defaults to a tool blindly)

**Refusal policy:** Only explicit buy/sell/invest advice is refused. Questions about stock
prices, financial data, and company performance are always answered.

---

## 6. Preventing Infinite Loops

- `MAX_STEPS = 8` is a hard cap enforced in the `while step < MAX_STEPS` condition
- The loop counter increments on every iteration including the reflection retry
- When the cap is reached, the agent returns `status="cap_reached"` with a structured refusal
- The reflector runs **at most once** per run (guarded by `_reflected` boolean flag)
- The decision engine is instructed never to call the same tool twice for the same question

---

## 7. Bonus Features

**Planning step (Bonus A) — `agent/planner.py`:**
Generates a 1-3 sentence plan before the loop starts. Uses `call_llm()` for questions longer
than 15 characters; falls back to rule-based keyword plan for short inputs (greetings, typos).
Printed at the top of every trace under `PLAN:`.

**Per-tool telemetry (Bonus B) — `agent/telemetry.py`:**
`TelemetryCollector` records wall-clock latency (ms) and call count for every tool invocation.
Attached to `AgentResponse.telemetry` and printed at the bottom of each trace. Aggregated
across all 20 eval questions in `evaluate.py` into a summary table.

**Reflection step (Bonus C) — `agent/reflector.py`:**
After `_compose_answer`, critiques the answer once using `call_llm()`. Returns
`{"passes": bool, "issue": str|null}`. If `passes=False` and the cap hasn't been hit, one
more retrieval round is triggered using the original question (not the issue string, to avoid
the LLM sending raw SQL as a query). Skipped when no API key is set.

**Response cache — `agent/cache.py`:**
File-backed JSON cache at `traces/response_cache.json`. Key: normalised question (lowercased,
whitespace-collapsed). Web-search responses are excluded (time-sensitive). Cache hits skip all
API calls and return instantly.

---

## 8. Data Flow — Question to Answer

1. `main.py` receives question from CLI; checks cache first
2. `run_agent(question)` called in `agent/agent_loop.py`
3. `generate_plan(question)` → plan string (via `call_llm()` or rule-based)
4. Loop: `decide_next_action(question, context)` → `AgentAction`
5. If `type="tool"`: tool called, `ToolResult` + telemetry added to `context` and `trace`
6. If `type="final"`: `_compose_answer` synthesises via `call_llm()`; reflector critiques
7. If `type="refuse"`: structured refusal returned immediately
8. If loop exhausts `MAX_STEPS`: cap-reached refusal returned
9. `AgentResponse` returned with `final_answer`, `citations`, `trace`, `plan`, `reflection`, `telemetry`
10. `print_trace(response)` formats and prints the full trace
11. Response cached (if not web-search-based)
