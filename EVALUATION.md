# Evaluation Report — Agentic RAG over Indian IT Financials

## 1. Evaluation Set Structure

The evaluation set contains 20 questions across four categories:

| Category | Count | Description |
|---|---|---|
| `single_tool` | 6 | Answerable from exactly one tool |
| `multi_tool` | 6 | Require combining two or more tools |
| `refusal` | 4 | Agent should decline to answer |
| `edge_case` | 4 | Ambiguous, boundary, or adversarial questions |

Run the evaluation:
```bash
python evaluate.py
```
Results saved to: `traces/evaluation_results.json`

---

## 2. Scoring Methodology

### Tool-routing score (Jaccard similarity)

```
score(E, A) = |E ∩ A| / |E ∪ A|
```

Where `E` = expected tool set, `A` = actual tool set called by the agent.

Special cases:
- `score({}, {}) = 1.0` — no tools expected and none called → correct
- `score(E, {}) = 0.0` for non-empty `E` — expected tools but none called → wrong
- `score({}, A) = 0.0` for non-empty `A` — no tools expected but some called → wrong

A result is **fully correct** when `tool_routing_score = 1.0` AND `status_correct = True`.
A result has **partial credit** when `0 < tool_routing_score < 1.0`.

The evaluation reports:
- Per-question Jaccard score, actual tools called, and status
- Per-category average Jaccard score and binary fully-correct count
- Partial-credit count separately from fully-correct and fully-wrong
- Per-tool telemetry summary (average latency, total calls, total estimated cost)

---

## 3. Evaluation Questions

| # | Category | Question | Expected Tools | Expected Status |
|---|---|---|---|---|
| 1 | single_tool | What was Infosys operating margin in FY24? | query_data | answered |
| 2 | single_tool | What was TCS revenue in FY23? | query_data | answered |
| 3 | single_tool | What reason did TCS give for its margin improvement in FY24? | search_docs | answered |
| 4 | single_tool | What is the current stock price of Infosys? | web_search | answered |
| 5 | single_tool | What were Wipro strategic priorities in FY24? | search_docs | answered |
| 6 | single_tool | What is 2+2? | (none) | answered |
| 7 | multi_tool | How did Infosys and TCS operating margins compare in FY24 and what drove each? | query_data, search_docs | answered |
| 8 | multi_tool | Compare headcount growth at all 3 companies over 4 years and explain the reasons. | query_data, search_docs | answered |
| 9 | multi_tool | What was Wipro revenue growth over 4 years and what strategy did management highlight? | query_data, search_docs | answered |
| 10 | multi_tool | What is TCS current stock price and how does it compare to their FY24 EPS? | web_search, query_data | answered |
| 11 | multi_tool | What did Infosys management say about margins and what do the numbers show for FY24? | search_docs, query_data | answered |
| 12 | multi_tool | What happened to IT sector stocks recently and how does that relate to TCS FY24 profit? | web_search, query_data | answered |
| 13 | refusal | Which company should I invest in? | (refuse) | refused |
| 14 | refusal | Should I buy or sell TCS shares right now? | (refuse) | refused |
| 15 | refusal | Which stock should I pick for maximum returns? | (refuse) | refused |
| 16 | refusal | What is the airspeed velocity of an unladen swallow? | (refuse) | refused |
| 17 | edge_case | What was Infosys revenue in FY19? | query_data | answered |
| 18 | edge_case | Compare all three companies on everything. | query_data | answered |
| 19 | edge_case | Why why why why why why why why why why? | search_docs | answered |
| 20 | edge_case | What is the latest news about Infosys and also their FY24 margin and also why did they grow? | web_search, query_data, search_docs | answered |

> Run `python evaluate.py` to populate actual outputs, Jaccard scores, and telemetry.

---

## 4. Failure Mode Analysis

### Failure Mode 1: Rule-based NL→SQL misses complex queries

**Trigger:** Questions like "Compare headcount growth at all 3 companies over 4 years" when
all LLM providers are unavailable.

**Root cause:** `_rule_based_nl_to_sql` in `tools/query_data.py` uses keyword matching to
construct SQL. It cannot handle multi-company comparisons, year-range queries, or aggregations
requiring `GROUP BY` or `ORDER BY` logic beyond simple filters.

**Mitigation:** LLM-driven SQL via `call_llm()` is the primary path when any API key is set.
The rule-based path is a last-resort fallback only.

---

### Failure Mode 2: LLM over-refuses stock-related questions

**Trigger:** Questions like "Tell me about Infosys stocks" when using Llama via Groq.

**Root cause:** Open-source LLMs (especially Llama) are fine-tuned to be cautious about
financial topics and treat "stocks" as investment advice even when the question is purely
informational.

**Mitigation:** The system prompt explicitly states that questions about stock prices,
financial data, and company performance are allowed. Only explicit buy/sell/invest advice
triggers a refusal. The rule-based fallback adds `stock`, `stocks`, `share`, `market cap`,
`nse`, `bse` to `WEB_KEYWORDS` so they route to `web_search` rather than refusing.

---

### Failure Mode 3: FAISS retrieval quality on short or ambiguous queries

**Trigger:** Questions like "Why why why why?" (eval question 19) or very short queries.

**Root cause:** The `all-MiniLM-L6-v2` embedding model produces low-quality vectors for
repetitive or semantically empty strings. The top-k results are essentially random.

**Mitigation:** The mock fallback in `search_docs.py` uses keyword scoring which at least
returns the most lexically relevant chunks. For production use, query expansion or a minimum
relevance threshold should be added.

---

### Failure Mode 4: Multi-tool questions approaching the hard cap

**Trigger:** Questions requiring 3 tools + reflection (e.g. eval question 20).

**Root cause:** Each tool call consumes one step. With `MAX_STEPS = 8`, a question requiring
3 tools + 1 decision step + 1 synthesis step + 1 reflection step uses 6 steps.

**Mitigation:** By design — the cap is a safety feature. The agent returns `cap_reached`
with whatever partial information was collected. Rephrase into smaller sub-questions.

---

### Failure Mode 5: LLM rate limits during bulk evaluation

**Trigger:** Running `python evaluate.py` with all 20 questions in sequence.

**Root cause:** Free-tier APIs have per-minute request limits. The agent makes up to 4 LLM
calls per question (planner + decision engine + synthesis + reflector).

**Mitigation:** The response cache skips API calls for repeated questions. Switch `LLM_TYPE`
between `GROQ` and `GEMINI` to spread load. Groq's free tier has higher RPM than Gemini.

---

## 5. Observations

- The planning step (Bonus A) consistently produces accurate plans for single-tool questions
  but sometimes over-specifies tools for simple questions. Short inputs (≤15 chars) now bypass
  the LLM planner entirely and use rule-based planning.

- The reflection step (Bonus C) is most useful when `search_docs` returns a mock chunk that
  doesn't actually answer the question. When real PDFs are ingested, reflection rarely triggers
  a second retrieval round because the LLM synthesis is well-grounded.

- Refusal questions (13–15) are the most reliable category. Question 16 (unladen swallow) is
  correctly refused by both the LLM and the rule-based fallback since it matches no domain
  keywords.

- The Groq provider (Llama 3.3 70B) is faster and has higher free-tier RPM than Gemini, but
  is more prone to over-refusing financial questions. Setting `LLM_TYPE=GROQ` with Gemini as
  fallback gives the best balance of speed and accuracy.

- The response cache significantly reduces API usage on repeated evaluation runs. On the second
  run of `evaluate.py`, all non-web-search questions are served from cache with zero API calls.
