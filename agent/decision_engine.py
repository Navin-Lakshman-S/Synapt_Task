# decision_engine.py — The "brain" of the agent.
# Right now it's a rule-based mock that simulates LLM reasoning.
# To plug in a real LLM: replace `decide_next_action` with an API call.
# Everything else in the system stays exactly the same.

from utils.types import AgentAction


# ── Refusal triggers ───────────────────────────────────────────────────────────
# Questions the agent should decline without calling any tool.
# In a real LLM, this would be handled by the model's safety layer + your system prompt.
REFUSE_PATTERNS = [
    "should i invest",
    "which stock should",
    "buy or sell",
    "investment advice",
    "recommend a stock",
    "which company is better to invest",
]

# ── Direct answer triggers ─────────────────────────────────────────────────────
# Questions that don't need any tool — agent answers from general knowledge.
DIRECT_ANSWER_PATTERNS = [
    "what is 2+2",
    "what is 2 + 2",
    "hello",
    "hi",
    "what can you do",
    "what are you",
]

# ── Tool routing keywords ──────────────────────────────────────────────────────
# Maps question keywords → which tool to call.
# Order matters: more specific patterns should come first.
# In a real LLM, the model reads tool descriptions and decides — no hardcoding needed.

WEB_SEARCH_KEYWORDS = [
    "current", "today", "latest", "recent", "last week", "now",
    "stock price", "share price", "news", "ceo", "cfo",
    "analyst rating", "2025", "this year", "right now",
]

QUERY_DATA_KEYWORDS = [
    "revenue", "profit", "margin", "eps", "earnings per share",
    "headcount", "employees", "fy21", "fy22", "fy23", "fy24",
    "how much", "how many", "compare", "growth", "year", "financial",
    "numbers", "data", "statistics", "figure",
]

SEARCH_DOCS_KEYWORDS = [
    "why", "reason", "explain", "strategy", "strategic", "priority",
    "management", "commentary", "annual report", "said", "stated",
    "highlight", "describe", "what did", "how did they", "approach",
    "initiative", "plan", "outlook", "guidance", "drove", "driven",
    "cause", "behind", "factor", "attributed", "due to",
]


def decide_next_action(question: str, context: list[dict]) -> AgentAction:
    """
    Decide what the agent should do next given the question and what it already knows.

    Args:
        question: The original user question.
        context:  List of previous tool results already collected this run.
                  Each entry is {"tool": tool_name, "result": ToolResult}.

    Returns:
        AgentAction with type, tool_name, input, and reasoning.

    ── HOW TO REPLACE WITH A REAL LLM ────────────────────────────────────────
    Replace the body of this function with:

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": build_prompt(question, context)},
        ]
        response = llm_client.chat(messages, tools=TOOL_SCHEMAS)
        return parse_llm_response(response)

    The AgentAction dataclass is the contract — your parser just needs to
    return one of those. Nothing else in the system changes.
    ──────────────────────────────────────────────────────────────────────────
    """
    q = question.lower()

    # ── Step 1: Check if this is a refusal case ────────────────────────────────
    # Agent should refuse without calling any tool
    for pattern in REFUSE_PATTERNS:
        if pattern in q:
            return AgentAction(
                type="refuse",
                reasoning=f"Question matches refusal pattern '{pattern}'. "
                          "This is an investment advice question — agent does not provide this.",
            )

    # ── Step 2: Check if no tool is needed ────────────────────────────────────
    for pattern in DIRECT_ANSWER_PATTERNS:
        if pattern in q:
            return AgentAction(
                type="final",
                reasoning="Question is trivial and answerable from general knowledge. No tool needed.",
            )

    # ── Step 3: Check if we already have enough context to answer ──────────────
    # If we've already called tools and have results, decide if we need more.
    if context:
        tools_called = [c["tool"] for c in context]

        # Determine what this question needs
        needs_data = any(kw in q for kw in QUERY_DATA_KEYWORDS)
        needs_docs = any(kw in q for kw in SEARCH_DOCS_KEYWORDS)
        needs_web  = any(kw in q for kw in WEB_SEARCH_KEYWORDS)

        # Multi-tool logic: fetch whatever is still missing
        if needs_data and "query_data" not in tools_called:
            return AgentAction(
                type="tool",
                tool_name="query_data",
                input=question,
                reasoning="Question needs structured numbers. "
                          f"Already have {tools_called}, now fetching data.",
            )
        if needs_docs and "search_docs" not in tools_called:
            return AgentAction(
                type="tool",
                tool_name="search_docs",
                input=question,
                reasoning="Question needs document explanation. "
                          f"Already have {tools_called}, now fetching docs.",
            )
        if needs_web and "web_search" not in tools_called:
            return AgentAction(
                type="tool",
                tool_name="web_search",
                input=question,
                reasoning="Question needs live/recent information. "
                          f"Already have {tools_called}, now fetching web.",
            )

        # We have everything we need — compose the answer
        return AgentAction(
            type="final",
            reasoning=f"Sufficient context collected from {tools_called}. Composing final answer.",
        )

    # ── Step 4: First tool call — route based on question type ────────────────
    # Check web search first (most time-sensitive)
    if any(kw in q for kw in WEB_SEARCH_KEYWORDS):
        return AgentAction(
            type="tool",
            tool_name="web_search",
            input=question,
            reasoning="Question contains live/recent keywords. Routing to web_search.",
        )

    # Check structured data
    if any(kw in q for kw in QUERY_DATA_KEYWORDS):
        return AgentAction(
            type="tool",
            tool_name="query_data",
            input=question,
            reasoning="Question asks for specific numbers or statistics. Routing to query_data.",
        )

    # Check document search
    if any(kw in q for kw in SEARCH_DOCS_KEYWORDS):
        return AgentAction(
            type="tool",
            tool_name="search_docs",
            input=question,
            reasoning="Question asks for qualitative explanation or document content. Routing to search_docs.",
        )

    # ── Step 5: Fallback — default to document search ─────────────────────────
    # If we can't determine the right tool, try search_docs as a safe default.
    return AgentAction(
        type="tool",
        tool_name="search_docs",
        input=question,
        reasoning="Could not determine tool from keywords. Defaulting to search_docs as safe fallback.",
    )
