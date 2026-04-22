# planner.py — Pre-loop planning step (Bonus A).
# Generates a 1-3 sentence plan describing which tools will be used and why,
# before any tool is called. Makes agent intent visible in the trace.

import os
from dotenv import load_dotenv

load_dotenv()

# Reuse the same keyword lists as decision_engine for rule-based fallback
WEB_KEYWORDS  = ["current", "today", "latest", "recent", "last week", "now",
                  "stock price", "share price", "news", "ceo", "cfo", "right now", "2025",
                  "stocks", "stock", "share", "market cap", "nse", "bse"]
DATA_KEYWORDS = ["revenue", "profit", "margin", "eps", "earnings per share",
                  "headcount", "employees", "fy2", "how much", "how many",
                  "compare", "growth", "financial", "figures"]
DOCS_KEYWORDS = ["why", "reason", "explain", "strategy", "strategic", "priority",
                  "management", "commentary", "said", "highlight", "drove", "driven",
                  "cause", "attributed", "approach", "plan", "outlook"]
REFUSE_PATTERNS = [
    "should i invest", "which stock should i", "buy or sell", "investment advice",
    "recommend a stock", "should i buy", "should i sell", "will the stock go up",
    "price prediction", "stock forecast", "which company to invest"
]

TOOL_DESCRIPTIONS = (
    "- search_docs: semantic search over annual report PDFs for qualitative info\n"
    "- query_data: SQL queries over structured financial data (numbers, trends)\n"
    "- web_search: live web search for recent/current information"
)


def generate_plan(question: str) -> str:
    """
    Generate a 1-3 sentence plan describing which tools to use and why.
    Uses Gemini when available; falls back to rule-based plan otherwise.

    Args:
        question: The user's natural language question.

    Returns:
        A short plan string shown at the top of the trace.
    """
    # Skip LLM for very short inputs — rule-based is more reliable for greetings/typos
    if len(question.strip()) <= 15:
        return _rule_based_plan(question)

    if os.getenv("GEMINI_API_KEY") or os.getenv("GROQ_API_KEY"):
        try:
            return _gemini_plan(question)
        except Exception as e:
            print(f"[planner] LLM error: {str(e)[:80]} — using rule-based plan.")
    return _rule_based_plan(question)


def _gemini_plan(question: str) -> str:
    """Ask the LLM to write a short tool-use plan for the question."""
    from agent.llm import call_llm

    prompt = (
        f"You are planning how to answer a question using these tools:\n"
        f"{TOOL_DESCRIPTIONS}\n\n"
        f"Question: {question}\n\n"
        f"Write a plan of 1-3 sentences describing which tool(s) you will call "
        f"and why. Be specific. Do not answer the question itself."
    )
    return call_llm(prompt, temperature=0.1)


def _rule_based_plan(question: str) -> str:
    """Construct a plan string from keyword matching — no API needed."""
    q = question.lower()

    for p in REFUSE_PATTERNS:
        if p in q:
            return "This question asks for investment advice, which is outside my scope. I will refuse without calling any tool."

    needs_web  = any(kw in q for kw in WEB_KEYWORDS)
    needs_data = any(kw in q for kw in DATA_KEYWORDS)
    needs_docs = any(kw in q for kw in DOCS_KEYWORDS)

    tools = []
    if needs_web:
        tools.append("web_search (live/recent information needed)")
    if needs_data:
        tools.append("query_data (structured financial numbers needed)")
    if needs_docs:
        tools.append("search_docs (qualitative explanation from annual reports needed)")

    if not tools:
        return "This question appears to be trivial or out of scope. I will answer directly or refuse without calling any tool."

    tool_str = " and ".join(tools)
    return f"I will call {tool_str} to answer this question."
