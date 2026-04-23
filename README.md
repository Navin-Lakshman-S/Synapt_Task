# Agentic RAG — Indian IT Financials

Answers natural language questions about Infosys, TCS, and Wipro by routing to the right tool,
composing a grounded answer with citations, and refusing gracefully when it cannot help.

No black-box agent frameworks. Every decision is traceable and explainable line by line.

---

## What it does

- Takes a natural language question
- Generates a pre-loop plan describing which tools it intends to use (Bonus A)
- Decides which of 3 tools to call: document search, financial data, or live web search
- Calls tools in sequence, up to a hard cap of 8 steps
- Composes a cited answer using the LLM, with a self-critique pass (Bonus C)
- Records per-tool latency and estimated cost (Bonus B)
- Caches responses to avoid redundant API calls on repeated questions
- Refuses gracefully for investment advice, out-of-scope questions, and unanswerable queries

---

## Project structure

```
├── agent/
│   ├── agent_loop.py       # Core agent loop — read this first
│   ├── decision_engine.py  # LLM-powered tool routing with rule-based fallback
│   ├── llm.py              # Unified LLM interface (Groq + Gemini with fallback)
│   ├── planner.py          # Pre-loop planning step (Bonus A)
│   ├── telemetry.py        # Per-tool latency + cost tracking (Bonus B)
│   ├── reflector.py        # Post-answer self-critique (Bonus C)
│   └── cache.py            # File-backed response cache
├── tools/
│   ├── search_docs.py      # Semantic search over PDFs (FAISS)
│   ├── query_data.py       # LLM-driven SQL over financial data (SQLite)
│   └── web_search.py       # Live web search (Tavily)
├── utils/
│   ├── types.py            # Shared data structures
│   └── logger.py           # Trace formatter + JSON export
├── data/
│   ├── docs/               # Annual report PDFs (Infosys, TCS, Wipro)
│   ├── build_db.py         # Builds financials.db from hardcoded published data
│   └── financials.db       # SQLite DB with FY15–FY24 annual + quarterly data
├── traces/                 # Auto-created — stores trace JSON and cache
├── main.py                 # CLI entry point
├── ingest.py               # PDF → FAISS index builder
├── evaluate.py             # Runs 20-question eval set with Jaccard scoring
├── DESIGN.md               # Agent architecture and tool contracts
├── EVALUATION.md           # Evaluation methodology and failure analysis
├── .env.example            # API key template
└── requirements.txt
```

---

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Copy and fill in your API keys
cp .env.example .env

# 3. Build the SQLite database (one-time)
python data/build_db.py

# 4. Ingest PDFs into FAISS index (one-time, takes 2-5 mins)
python ingest.py
```

---

## API keys

| Key | Where to get | Required? |
|---|---|---|
| `GEMINI_API_KEY` | [aistudio.google.com](https://aistudio.google.com) | Yes (or Groq) |
| `GROQ_API_KEY` | [console.groq.com](https://console.groq.com) | Yes (or Gemini) |
| `TAVILY_API_KEY` | [tavily.com](https://tavily.com) | Yes (for live web search) |
| `HF_TOKEN` | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) | Optional |

---

## LLM routing

Set `LLM_TYPE` in `.env` to control which provider is tried first:

```
LLM_TYPE=GROQ    # Groq first → Gemini fallback (recommended: higher free-tier RPM)
LLM_TYPE=GEMINI  # Gemini first → Groq fallback
```

If both fail, the agent falls back to rule-based routing — it still works, just with reduced accuracy.

To switch models:
```
GEMINI_MODEL=gemini-2.0-flash-lite   # default (higher free RPM)
GROQ_MODEL=llama-3.3-70b-versatile   # default
```

---

## Run the agent

```bash
python main.py
```

Type any question. The agent prints a full trace showing the plan, each tool call with reasoning,
the final answer with citations, a self-critique, and per-tool telemetry.

Repeated questions are served from cache instantly — no API calls.

---

## Run the evaluation set

```bash
python evaluate.py
```

Runs all 20 questions across 4 categories and prints:
- Per-question Jaccard tool-routing score and status
- Per-category accuracy summary
- Per-tool telemetry table (latency, call count, estimated cost)

Results saved to `traces/evaluation_results.json`.

---

## Known failure modes

1. **Rule-based NL→SQL fails on complex queries** — when both LLM providers are unavailable, `_rule_based_nl_to_sql` in `tools/query_data.py` uses keyword matching and cannot handle multi-company aggregations or year-range queries. Mitigation: LLM-driven SQL is the primary path when any API key is set.

2. **LLM over-refuses stock-related questions** — some LLMs (especially Llama via Groq) treat "Tell me about Infosys stocks" as investment advice and refuse. Fixed in the system prompt: refusal is now restricted to explicit buy/sell/invest advice only.

3. **FAISS retrieval quality on short/ambiguous queries** — queries like "Why why why?" produce low-quality embedding vectors. The mock fallback uses keyword scoring as a safety net.

4. **Multi-tool questions approaching the hard cap** — questions requiring 3 tools + reflection can use 5-6 of the 8 steps. By design — the cap is a safety feature, not a bug.

5. **LLM rate limits during bulk evaluation** — running all 20 eval questions back-to-back can exhaust free-tier quotas. Mitigation: the cache skips API calls for repeated questions; switch `LLM_TYPE` between Groq and Gemini to spread load.

6. **Chunking at word boundaries** — `ingest.py` now snaps chunk boundaries to word edges, but very long words at chunk boundaries may still be split across chunks in edge cases.

---

## AI assistance disclosure

This project was built with AI coding assistance. All design decisions, architecture choices,
and the agent loop logic are understood and can be explained line by line.
