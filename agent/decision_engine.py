# decision_engine.py — The agent's brain.
# Uses Gemini to decide which tool to call next (or whether to answer/refuse).
# Falls back to rule-based routing if GEMINI_API_KEY is not set.

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
1. If the question asks for investment advice, stock recommendations, or buy/sell decisions → REFUSE immediately, call no tools.
2. If the question is trivial (math, greetings) → answer directly with no tool call.
3. If you already have enough context to answer → compose the final answer.
4. Otherwise → call the most appropriate tool.
5. Never call the same tool twice for the same question unless the first call failed.

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
                if len(output_str) > 800:
                    output_str = output_str[:800] + "... [truncated]"
                prompt += f"Result: {output_str}\n"
            else:
                prompt += f"Result: ERROR — {result.error}\n"
    else:
        prompt += "\nNo context collected yet. This is the first step.\n"

    prompt += "\nWhat should the agent do next? Respond with JSON only."
    return prompt


def _call_gemini(prompt: str) -> AgentAction:
    """Call Gemini API and parse the JSON response into an AgentAction."""
    import time
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.1),
            )
            raw = response.text.strip()
            # Strip markdown code fences if Gemini wraps the JSON
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
        except Exception as e:
            err = str(e)
            if "429" in err or "RESOURCE_EXHAUSTED" in err:
                # Extract retry delay from error if available, else wait 5s
                wait = 5
                import re
                match = re.search(r"retryDelay.*?(\d+)s", err)
                if match:
                    wait = min(int(match.group(1)), 15)  # cap at 15s
                if attempt < 2:
                    time.sleep(wait)
                    continue
            raise  # re-raise on non-rate-limit errors or after 3 attempts


# ── Fallback rule-based routing (used if Gemini call fails) ───────────────────
REFUSE_PATTERNS   = ["should i invest", "which stock should", "buy or sell", "investment advice", "recommend a stock"]
DIRECT_PATTERNS   = ["what is 2+2", "what is 2 + 2", "hello", "hi there"]
WEB_KEYWORDS      = ["current", "today", "latest", "recent", "last week", "now", "stock price", "share price", "news", "ceo", "cfo", "right now", "2025"]
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

    return AgentAction(type="tool", tool_name="search_docs", input=question, reasoning="No clear signal — defaulting to search_docs.")


def decide_next_action(question: str, context: list[dict]) -> AgentAction:
    """
    Decide what the agent should do next.

    Tries Gemini first. Falls back to rule-based routing on any error.
    This design means the agent always works — even without an API key.

    To swap LLM provider: replace _call_gemini() with your provider's call.
    The AgentAction return type is the contract — nothing else changes.
    """
    api_key = os.getenv("GEMINI_API_KEY")

    if api_key:
        try:
            prompt = _build_prompt(question, context)
            return _call_gemini(prompt)
        except Exception as e:
            err = str(e)
            if "429" in err or "RESOURCE_EXHAUSTED" in err:
                print("[decision_engine] Gemini rate limit hit — using rule-based routing.")
            else:
                print(f"[decision_engine] Gemini error: {err[:120]} — falling back to rule-based routing.")

    return _rule_based_fallback(question, context)
