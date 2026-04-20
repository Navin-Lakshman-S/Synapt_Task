# agent_loop.py — The core agent loop. Read top to bottom in under 3 minutes.
#
# Flow:
#   question → decide action → call tool → store result → decide again → ...
#   → compose answer (or refuse, or hit hard cap)

import os
import json
from dotenv import load_dotenv
from utils.types import AgentAction, AgentResponse, TraceStep, ToolResult
from agent.decision_engine import decide_next_action
from tools.search_docs import search_docs
from tools.query_data import query_data
from tools.web_search import web_search

load_dotenv()

# Hard cap — agent MUST stop after this many tool calls. Non-negotiable.
MAX_STEPS = 8

# Registry maps tool names → functions.
# To add a new tool: one line here, nothing else changes.
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
        AgentResponse with answer, citations, trace, and status.
    """
    context: list[dict] = []   # accumulates tool results across steps
    trace:   list[TraceStep] = []
    step = 0

    # ── Main agent loop ────────────────────────────────────────────────────────
    while step < MAX_STEPS:
        step += 1

        # Ask the decision engine what to do next
        action: AgentAction = decide_next_action(question, context)

        # ── Refuse ────────────────────────────────────────────────────────────
        if action.type == "refuse":
            trace.append(TraceStep(step_number=step, action=action, result=None))
            return AgentResponse(
                question=question,
                final_answer=(
                    "I'm not able to answer this question. It falls outside the scope "
                    "of what this agent is designed to do (e.g. investment advice, "
                    "opinion-based queries, or topics unrelated to Infosys/TCS/Wipro)."
                ),
                citations=[],
                trace=trace,
                steps_used=step,
                status="refused",
            )

        # ── Final answer ───────────────────────────────────────────────────────
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

        # ── Tool call ──────────────────────────────────────────────────────────
        if action.type == "tool":
            tool_fn = TOOL_REGISTRY.get(action.tool_name)

            if tool_fn is None:
                result = ToolResult(
                    tool_name=action.tool_name, input_query=action.input,
                    output={}, source_citations=[], success=False,
                    error=f"Tool '{action.tool_name}' not found in registry.",
                )
            else:
                result = tool_fn(action.input)

            trace.append(TraceStep(step_number=step, action=action, result=result))
            context.append({"tool": action.tool_name, "result": result})

    # ── Hard cap reached ───────────────────────────────────────────────────────
    # Never guess — return a structured refusal with whatever was collected
    return AgentResponse(
        question=question,
        final_answer=(
            f"I was unable to compose a complete answer within the {MAX_STEPS}-step limit. "
            "The question may require more sources than available, or the tools did not "
            "return sufficient information. Please try rephrasing or narrowing the question."
        ),
        citations=_collect_citations(context),
        trace=trace,
        steps_used=MAX_STEPS,
        status="cap_reached",
    )


# ── Answer composition ─────────────────────────────────────────────────────────

def _compose_answer(question: str, context: list[dict]) -> tuple[str, list[str]]:
    """
    Synthesise a final answer from all collected tool results.
    Uses Gemini if available, otherwise formats results directly.
    """
    if not context:
        return _direct_answer(question), []

    all_citations = _collect_citations(context)

    # Try Gemini synthesis first
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        try:
            answer = _gemini_synthesise(question, context)
            return answer, all_citations
        except Exception as e:
            err = str(e)
            if "429" in err or "RESOURCE_EXHAUSTED" in err:
                print("[agent_loop] Gemini rate limit hit — using raw format.")
            else:
                print(f"[agent_loop] Gemini synthesis error: {err[:120]} — using raw format.")

    # Fallback: format each tool's output directly
    return _format_raw_answer(context), all_citations


def _gemini_synthesise(question: str, context: list[dict]) -> str:
    """Ask Gemini to write a coherent answer from the collected tool results."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    # Build context summary for the prompt
    context_text = ""
    for i, entry in enumerate(context, 1):
        result = entry["result"]
        context_text += f"\n[Source {i} — {entry['tool']}]\n"
        if result.success:
            output_str = str(result.output)
            if len(output_str) > 1000:
                output_str = output_str[:1000] + "... [truncated]"
            context_text += output_str + "\n"
            if result.source_citations:
                context_text += f"Citations: {', '.join(result.source_citations)}\n"
        else:
            context_text += f"Error: {result.error}\n"

    prompt = f"""You are a financial analyst assistant. Using ONLY the information provided below, 
answer the user's question clearly and concisely. 
Cite the specific source for each claim (document name + page, or database row).
Do not hallucinate or add information not present in the sources.

Question: {question}

Retrieved information:
{context_text}

Write a clear, well-structured answer with inline citations."""

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.2),
    )
    return response.text.strip()


def _format_raw_answer(context: list[dict]) -> str:
    """Format tool results into a readable answer without LLM synthesis."""
    parts = []
    for entry in context:
        tool_name = entry["tool"]
        result: ToolResult = entry["result"]

        if not result.success:
            parts.append(f"[{tool_name} returned an error: {result.error}]")
            continue

        if tool_name == "search_docs":
            for chunk in result.output:
                parts.append(f"From {chunk['source']} (p.{chunk['page']}):\n{chunk['text']}")

        elif tool_name == "query_data":
            data = result.output
            cols = data.get("columns", [])
            rows = data.get("rows", [])
            header = " | ".join(str(c) for c in cols)
            rows_str = "\n".join(" | ".join(str(v) for v in row) for row in rows)
            parts.append(f"Financial data ({data.get('row_count', 0)} rows):\n{header}\n{rows_str}")

        elif tool_name == "web_search":
            for item in result.output:
                parts.append(f"[Web — {item['title']} ({item['published_date']})]:\n{item['snippet']}\nSource: {item['url']}")

    return "\n\n---\n\n".join(parts) if parts else "No relevant information found."


def _collect_citations(context: list[dict]) -> list[str]:
    """Flatten all citations from all tool results."""
    citations = []
    for entry in context:
        result: ToolResult = entry["result"]
        if result and result.success:
            citations.extend(result.source_citations)
    return citations


def _direct_answer(question: str) -> str:
    """Handle trivial questions that need no tool."""
    q = question.lower()
    if "2+2" in q or "2 + 2" in q:
        return "4"
    if q.strip() in ("hello", "hi", "hey"):
        return "Hello! Ask me anything about Infosys, TCS, or Wipro."
    return "I can answer questions about Infosys, TCS, and Wipro using their annual reports, financial data, and live web search."
