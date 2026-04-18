# agent_loop.py — The core agent loop. This is the heart of the system.
# Read this top to bottom in under 3 minutes. That's the goal.
#
# Flow:
#   question → decide action → call tool → store result → decide again → ...
#   → compose answer (or refuse, or hit hard cap)

from utils.types import AgentAction, AgentResponse, TraceStep, ToolResult
from agent.decision_engine import decide_next_action
from tools.search_docs import search_docs
from tools.query_data import query_data
from tools.web_search import web_search

# Hard cap: agent MUST stop after this many tool calls. Non-negotiable.
MAX_STEPS = 8

# Registry maps tool names → actual functions.
# To add a new tool: add one line here. Nothing else changes.
TOOL_REGISTRY: dict[str, callable] = {
    "search_docs": search_docs,
    "query_data":  query_data,
    "web_search":  web_search,
}


def run_agent(question: str) -> AgentResponse:
    """
    Run the agent on a single question and return a fully traced response.

    Args:
        question: The user's natural language question.

    Returns:
        AgentResponse containing the answer, citations, trace, and status.
    """

    # context accumulates everything the agent has learned this run.
    # It's passed to the decision engine each step so it can reason about
    # what it already knows before deciding what to do next.
    context: list[dict] = []
    trace:   list[TraceStep] = []
    step = 0

    # ── Main agent loop ────────────────────────────────────────────────────────
    while step < MAX_STEPS:
        step += 1

        # ── Decide what to do next ─────────────────────────────────────────────
        # The decision engine looks at the question + everything collected so far.
        action: AgentAction = decide_next_action(question, context)

        # ── Handle: refuse ─────────────────────────────────────────────────────
        if action.type == "refuse":
            trace.append(TraceStep(step_number=step, action=action, result=None))
            return AgentResponse(
                question=question,
                final_answer="I'm not able to answer this question. "
                             "It falls outside the scope of what this agent is designed to do "
                             "(e.g., investment advice, opinion-based queries).",
                citations=[],
                trace=trace,
                steps_used=step,
                status="refused",
            )

        # ── Handle: final answer ───────────────────────────────────────────────
        if action.type == "final":
            trace.append(TraceStep(step_number=step, action=action, result=None))
            answer, citations = _compose_answer(question, context)
            return AgentResponse(
                question=question,
                final_answer=answer,
                citations=citations,
                trace=trace,
                steps_used=step,
                status="answered",
            )

        # ── Handle: tool call ──────────────────────────────────────────────────
        if action.type == "tool":
            tool_fn = TOOL_REGISTRY.get(action.tool_name)

            # Guard: unknown tool name (shouldn't happen, but handle it cleanly)
            if tool_fn is None:
                result = ToolResult(
                    tool_name=action.tool_name,
                    input_query=action.input,
                    output={},
                    source_citations=[],
                    success=False,
                    error=f"Tool '{action.tool_name}' not found in registry.",
                )
            else:
                # Call the tool — all exceptions are caught inside each tool
                result = tool_fn(action.input)

            # Record this step in the trace
            trace.append(TraceStep(step_number=step, action=action, result=result))

            # Add result to context so the decision engine can use it next step
            context.append({"tool": action.tool_name, "result": result})

    # ── Hard cap reached ───────────────────────────────────────────────────────
    # We exhausted all 8 steps without reaching a final answer.
    # Return a structured refusal — never guess.
    return AgentResponse(
        question=question,
        final_answer=(
            f"I was unable to compose a complete answer within the {MAX_STEPS}-step limit. "
            "The question may require more sources than available, or the tools did not return "
            "sufficient information. Please try rephrasing or narrowing the question."
        ),
        citations=_collect_all_citations(context),
        trace=trace,
        steps_used=MAX_STEPS,
        status="cap_reached",
    )


# ── Helper: compose the final answer from collected context ────────────────────
def _compose_answer(question: str, context: list[dict]) -> tuple[str, list[str]]:
    """
    Build a final answer string from all tool results in context.
    Also collects all citations.

    In a real LLM setup: replace this with an LLM call that synthesises
    the context into a coherent answer. The context list is already formatted
    for easy injection into a prompt.
    """
    if not context:
        # No tools were called — answer from general knowledge
        return _direct_answer(question), []

    # Build answer by summarising each tool's contribution
    parts = []
    all_citations = []

    for entry in context:
        tool_name = entry["tool"]
        result: ToolResult = entry["result"]

        if not result.success:
            parts.append(f"[{tool_name} returned an error: {result.error}]")
            continue

        # Format each tool's output into a readable paragraph
        if tool_name == "search_docs":
            chunks = result.output
            for chunk in chunks:
                parts.append(f"From {chunk['source']} (p.{chunk['page']}): {chunk['text']}")

        elif tool_name == "query_data":
            data = result.output
            cols = data.get("columns", [])
            rows = data.get("rows", [])
            table_lines = [" | ".join(str(v) for v in row) for row in rows]
            parts.append(
                f"Structured data ({data.get('row_count', 0)} rows):\n"
                f"Columns: {cols}\n" +
                "\n".join(table_lines)
            )

        elif tool_name == "web_search":
            for item in result.output:
                parts.append(
                    f"[Web] {item['title']} ({item['published_date']}): {item['snippet']}"
                )

        all_citations.extend(result.source_citations)

    answer = "\n\n".join(parts) if parts else "No relevant information found."
    return answer, all_citations


def _collect_all_citations(context: list[dict]) -> list[str]:
    """Flatten all citations from all tool results in context."""
    citations = []
    for entry in context:
        result: ToolResult = entry["result"]
        if result and result.success:
            citations.extend(result.source_citations)
    return citations


def _direct_answer(question: str) -> str:
    """Handle trivial questions that don't need any tool."""
    q = question.lower()
    if "2+2" in q or "2 + 2" in q:
        return "4"
    if q in ("hello", "hi"):
        return "Hello! Ask me anything about Infosys, TCS, or Wipro."
    return "I can answer questions about Infosys, TCS, and Wipro using financial data, annual reports, and web search."
