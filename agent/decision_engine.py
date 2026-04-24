# decision_engine.py — The agent's brain.
# Routes questions to the right tool using call_llm() (Groq/Gemini with fallback).
# Falls back to rule-based keyword routing if all LLM providers fail.

import os
import json
from dotenv import load_dotenv
from utils.types import AgentAction

load_dotenv()

# ── Tool schemas passed to Gemini so it knows what tools exist ─────────────────
# These descriptions are written FOR the LLM — they tell it when to use each tool
# and critically when NOT to. This is what drives correct routing.
TOOL_SCHEMAS = [
    {
        "name": "search_docs",
        "description": (
            "Use this tool when the question asks about qualitative information, explanations, "
            "strategies, management commentary, or reasons found inside company annual reports. "
            "Examples: 'What reason did TCS give for margin improvement?', "
            "'What are Infosys strategic priorities?'. "
            "Do NOT use for exact numbers — use query_data. "
            "Do NOT use for recent/live news — use web_search."
        ),
        "input_description": "Natural language query string about document content.",
    },
    {
        "name": "query_data",
        "description": (
            "Use this tool when the question asks for specific numbers, statistics, comparisons, "
            "or trends from structured financial data. "
            "Examples: 'What was Infosys revenue in FY24?', 'Compare TCS and Wipro margins over 4 years'. "
            "Do NOT use for qualitative explanations — use search_docs. "
            "Do NOT use for live/current data — use web_search."
        ),
        "input_description": "Natural language question about financial data.",
    },
    {
        "name": "web_search",
        "description": (
            "Use this tool when the question asks for recent, live, or current information "
            "not found in static documents or historical database. "
            "Examples: 'What is the current stock price of Infosys?', 'Latest IT sector news'. "
            "Do NOT use for historical financial data — use query_data. "
            "Do NOT use for information clearly inside annual reports — use search_docs. "
            "Keep the search query short (under 10 words)."
        ),
        "input_description": "Short search query string (under 10 words).",
    },
]

# ── System prompt for the LLM ──────────────────────────────────────────────────
SYSTEM_PROMPT = """You are the decision engine of an agentic RAG system about Indian IT companies (Infosys, TCS, Wipro).

Your job is to decide the NEXT action given a user question and the context already collected.

You have access to three tools:
{tool_descriptions}

Rules:
1. REFUSE ONLY IF the question explicitly asks for investment advice, buy/sell recommendations, or price predictions. Example: "Should I buy TCS?" or "Which stock will go up?" → REFUSE.
2. REFUSE if the question has absolutely no connection to Infosys, TCS, Wipro or Indian IT.
3. Questions about stock prices, financial data, EPS, revenue, margins, or company performance are ALLOWED — use query_data or web_search. Example: "Tell me about Infosys stocks" → use query_data + web_search.
4. If the question is trivial (greetings, simple arithmetic) → answer directly with type="final" and no tool call.
5. If you already have enough context to answer → compose the final answer.
6. Otherwise → call the most appropriate tool. Never call the same tool twice unless the first call failed.
7. Only reply to the question if it is about the companies Infosys, TCS and WIPRO. If other companies or anyother thing is asked don't call any tool and refuse to answer with type=final

Respond ONLY with valid JSON in this exact format:
{{
  "type": "tool" | "final" | "refuse",
  "tool_name": "<tool name or null>",
  "input": "<query to pass to the tool, or null>",
  "reasoning": "<one sentence explaining why you made this choice>"
}}
"""


def _build_prompt(question: str, context: list[dict]) -> str:
    """Build the prompt showing the question and all context collected so far."""
    tool_descriptions = "\n".join(
        f"- {t['name']}: {t['description']}" for t in TOOL_SCHEMAS
    )

    prompt = SYSTEM_PROMPT.format(tool_descriptions=tool_descriptions)
    prompt += f"\n\nUser question: {question}\n"

    if context:
        prompt += "\nContext already collected:\n"
        for i, entry in enumerate(context, 1):
            result = entry["result"]
            prompt += f"\nStep {i} — Tool: {entry['tool']}\n"
            if result.success:
                output_str = str(result.output)
                # Truncate long outputs to keep prompt manageable
                if len(output_str) > 2000:
                    output_str = output_str[:2000] + "... [truncated]"
                prompt += f"Result: {output_str}\n"
            else:
                prompt += f"Result: ERROR — {result.error}\n"
    else:
        prompt += "\nNo context collected yet. This is the first step.\n"

    prompt += "\nWhat should the agent do next? Respond with JSON only."
    return prompt


def _call_llm(prompt: str) -> AgentAction:
    """Call the LLM (with fallback) and parse the JSON response into an AgentAction."""
    from agent.llm import call_llm, LLMUnavailableError

    try:
        raw = call_llm(prompt, temperature=0.1)
    except LLMUnavailableError:
        raise  # let decide_next_action catch this and fall back to rule-based

    # Strip markdown code fences if model wraps the JSON
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()
    data = json.loads(raw)
    return AgentAction(
        type=data.get("type", "refuse"),
        tool_name=data.get("tool_name"),
        input=data.get("input"),
        reasoning=data.get("reasoning", ""),
    )


# ── Fallback rule-based routing (used if Gemini call fails) ───────────────────
REFUSE_PATTERNS = [
    "should i invest", "which stock should i", "buy or sell", "investment advice",
    "recommend a stock", "should i buy", "should i sell", "will the stock go up",
    "price prediction", "stock forecast", "which company to invest"
]
DIRECT_PATTERNS   = ["what is 2+2", "what is 2 + 2", "hello", "hi there"]
WEB_KEYWORDS      = ["current", "today", "latest", "recent", "last week", "now", "stock price", "share price", "news", "ceo", "cfo", "right now", "2025", "stocks", "stock", "share", "market cap", "nse", "bse"]
DATA_KEYWORDS     = ["revenue", "profit", "margin", "eps", "earnings per share", "headcount", "employees", "fy2", "how much", "how many", "compare", "growth", "financial", "figures"]
DOCS_KEYWORDS     = ["why", "reason", "explain", "strategy", "strategic", "priority", "management", "commentary", "said", "highlight", "drove", "driven", "cause", "attributed", "approach", "plan", "outlook"]


def _rule_based_fallback(question: str, context: list[dict]) -> AgentAction:
    """Simple keyword-based routing used when Gemini is unavailable."""
    q = query = question.lower()

    for p in REFUSE_PATTERNS:
        if p in q:
            return AgentAction(type="refuse", reasoning=f"Matches refusal pattern: '{p}'")

    for p in DIRECT_PATTERNS:
        if p in q:
            return AgentAction(type="final", reasoning="Trivial question, no tool needed.")

    if context:
        tools_called = [c["tool"] for c in context]
        needs_data = any(kw in q for kw in DATA_KEYWORDS)
        needs_docs = any(kw in q for kw in DOCS_KEYWORDS)
        needs_web  = any(kw in q for kw in WEB_KEYWORDS)

        if needs_data and "query_data" not in tools_called:
            return AgentAction(type="tool", tool_name="query_data", input=question, reasoning="Need structured data, not yet fetched.")
        if needs_docs and "search_docs" not in tools_called:
            return AgentAction(type="tool", tool_name="search_docs", input=question, reasoning="Need document context, not yet fetched.")
        if needs_web and "web_search" not in tools_called:
            return AgentAction(type="tool", tool_name="web_search", input=question, reasoning="Need live data, not yet fetched.")
        return AgentAction(type="final", reasoning=f"Sufficient context from {tools_called}.")

    if any(kw in q for kw in WEB_KEYWORDS):
        return AgentAction(type="tool", tool_name="web_search", input=question, reasoning="Live/recent keywords detected.")
    if any(kw in q for kw in DATA_KEYWORDS):
        return AgentAction(type="tool", tool_name="query_data", input=question, reasoning="Numerical/financial keywords detected.")
    if any(kw in q for kw in DOCS_KEYWORDS):
        return AgentAction(type="tool", tool_name="search_docs", input=question, reasoning="Qualitative/explanation keywords detected.")

    return AgentAction(type="refuse", reasoning="No domain signal detected — refusing.")


def decide_next_action(question: str, context: list[dict]) -> AgentAction:
    """
    Decide what the agent should do next.

    Tries LLM providers in order (based on LLM_TYPE in .env).
    Falls back to rule-based routing if all LLM providers fail.
    This design means the agent always works — even without any API key.
    """
    from agent.llm import LLMUnavailableError

    if os.getenv("GEMINI_API_KEY") or os.getenv("GROQ_API_KEY"):
        try:
            prompt = _build_prompt(question, context)
            return _call_llm(prompt)
        except LLMUnavailableError:
            print("[decision_engine] All LLM providers failed — using rule-based routing.")
        except Exception as e:
            print(f"[decision_engine] LLM error: {str(e)[:120]} — falling back to rule-based routing.")

    return _rule_based_fallback(question, context)
