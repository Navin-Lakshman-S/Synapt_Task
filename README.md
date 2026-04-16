# Agentic RAG — Indian IT Financials

A minimal, explainable LLM agent that answers questions over mixed data sources:
annual report PDFs, structured financial data, and live web search.

Built for the AI & Data Science internship assignment. No black-box frameworks.
Every decision the agent makes is traceable and explainable.

---

## What it does

- Takes a natural language question
- Decides which of 3 tools to call (document search, financial data, web search)
- Calls tools in sequence, up to a hard cap of 8 steps
- Returns a grounded answer with citations and a full execution trace
- Refuses gracefully when it cannot or should not answer

---

## Project structure

```
agentic-rag/
├── agent/
│   ├── agent_loop.py       # Core agent loop (~80 lines) — read this first
│   └── decision_engine.py  # Mock LLM brain — replace with real LLM here
├── tools/
│   ├── search_docs.py      # Semantic search over PDFs (FAISS)
│   ├── query_data.py       # SQL queries over financial data (SQLite)
│   └── web_search.py       # Live web search (Tavily)
├── utils/
│   ├── types.py            # Shared data structures
│   └── logger.py           # Trace formatter
├── data/
│   └── docs/               # Place annual report PDFs here
├── traces/                 # Auto-created — stores trace JSON files
├── main.py                 # CLI entry point
├── ingest.py               # PDF → FAISS index builder
├── evaluate.py             # Runs 20-question eval set
├── .env.example            # API key template
└── requirements.txt
```

---

## Setup

```bash
cd agentic-rag
pip install -r requirements.txt
cp .env.example .env
# Fill in your API keys in .env
```

---

## Run the agent

```bash
python main.py
```

Type any question. The agent prints a full trace showing every decision it made.

---

## Run the evaluation set

```bash
python evaluate.py
```

Runs all 20 questions and saves results to `traces/evaluation_results.json`.

---

## Add real data

**Structured data:**
1. Download quarterly financials from [Screener.in](https://www.screener.in) for TCS, Infosys, Wipro
2. Save as `data/financials.csv`
3. Run `python ingest.py` — it auto-loads the CSV into SQLite

**Unstructured data (PDFs):**
1. Download annual reports from company investor relations pages
2. Place in `data/docs/`
3. Install: `pip install faiss-cpu sentence-transformers pypdf`
4. Run `python ingest.py` — builds the FAISS index

**Web search:**
1. Sign up at [tavily.com](https://tavily.com) (free tier)
2. Add `TAVILY_API_KEY` to `.env`
3. Uncomment the real implementation in `tools/web_search.py`

---

## Plug in a real LLM

Open `agent/decision_engine.py` and replace the body of `decide_next_action()` with an LLM API call.
The function signature and return type (`AgentAction`) stay the same — nothing else changes.

---

## Known failure modes

1. **Wrong tool routing on ambiguous questions** — questions that mix keywords from multiple categories
   can confuse the rule-based decision engine. Fix: real LLM with tool descriptions.

2. **NL→SQL translation failures** — the rule-based SQL generator misses complex queries.
   Fix: replace `_nl_to_sql()` in `query_data.py` with an LLM call.

3. **Mock data limitations** — mock chunks don't cover all possible questions.
   Fix: ingest real PDFs.

4. **Hard cap fires on complex multi-source questions** — questions requiring 4+ tools hit the 8-step cap.
   This is by design — the cap is a safety feature, not a bug.

---

## AI assistance disclosure

This project was built with AI coding assistance. All design decisions, architecture choices,
and the agent loop logic are understood and can be explained line by line.
