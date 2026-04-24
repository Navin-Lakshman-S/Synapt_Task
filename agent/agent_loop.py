# agent_loop.py — The core agent loop. Read top to bottom in under 3 minutes.
#
# Flow:
#   question → plan → decide action → call tool → store result → decide again
#   → compose answer → reflect → return (or refuse, or hit hard cap)

import os
import re
import time
from dotenv import load_dotenv
from utils.types import AgentAction, AgentResponse, TraceStep, ToolResult
from agent.decision_engine import decide_next_action
from agent.planner import generate_plan
from agent.telemetry import TelemetryCollector
from agent.reflector import reflect
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
        AgentResponse with answer, citations, trace, plan, reflection, telemetry.
    """
    context: list[dict] = []
    trace:   list[TraceStep] = []
    step = 0
    _reflected = False  # reflection runs at most once per run

    # ── Pre-loop: generate a plan ──────────────────────────────────────────────
    plan = generate_plan(question)
    telemetry = TelemetryCollector()

    # ── Main agent loop ────────────────────────────────────────────────────────
    while step < MAX_STEPS:

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
                plan=plan,
                telemetry=telemetry.to_dict(),
            )

        # ── Final answer ───────────────────────────────────────────────────────
        if action.type == "final":
            trace.append(TraceStep(step_number=step, action=action, result=None))
            answer, citations = _compose_answer(question, context)

            # Reflection — runs once, may push one more retrieval if answer fails
            reflection_text = None
            if not _reflected and context:
                _reflected = True
                critique = reflect(question, answer, context)
                if critique:
                    reflection_text = critique.get("issue") or "Answer passed self-critique."
                    if not critique.get("passes") and step < MAX_STEPS:
                        # One more retrieval round — use original question, not the issue string
                        step += 1
                        retry_action = decide_next_action(question, context)
                        if retry_action.type == "tool":
                            tool_fn = TOOL_REGISTRY.get(retry_action.tool_name)
                            if tool_fn:
                                t0 = time.monotonic()
                                result = tool_fn(retry_action.input)
                                telemetry.record_tool_call(
                                    retry_action.tool_name,
                                    (time.monotonic() - t0) * 1000,
                                )
                                trace.append(TraceStep(step_number=step, action=retry_action, result=result))
                                context.append({"tool": retry_action.tool_name, "result": result})
                                answer, citations = _compose_answer(question, context)

            return AgentResponse(
                question=question,
                final_answer=answer,
                citations=citations,
                trace=trace,
                steps_used=step,
                status="answered",
                plan=plan,
                reflection=reflection_text,
                telemetry=telemetry.to_dict(),
            )

        # ── Tool call ──────────────────────────────────────────────────────────
        if action.type == "tool":
            step += 1
            tool_fn = TOOL_REGISTRY.get(action.tool_name)

            if tool_fn is None:
                result = ToolResult(
                    tool_name=action.tool_name, input_query=action.input,
                    output={}, source_citations=[], success=False,
                    error=f"Tool '{action.tool_name}' not found in registry.",
                )
            else:
                t0 = time.monotonic()
                result = tool_fn(action.input)
                telemetry.record_tool_call(
                    action.tool_name,
                    (time.monotonic() - t0) * 1000,
                )

            trace.append(TraceStep(step_number=step, action=action, result=result))
            context.append({"tool": action.tool_name, "result": result})

    # ── Hard cap reached ───────────────────────────────────────────────────────
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
        plan=plan,
        telemetry=telemetry.to_dict(),
    )


# ── Answer composition ─────────────────────────────────────────────────────────

def _compose_answer(question: str, context: list[dict]) -> tuple[str, list[str]]:
    """Synthesise a final answer from all collected tool results."""
    if not context:
        return _direct_answer(question), []

    all_citations = _collect_citations(context)

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GROQ_API_KEY")
    if api_key:
        try:
            answer = _synthesise(question, context)
            return answer, all_citations
        except Exception as e:
            err = str(e)
            if "429" in err or "RESOURCE_EXHAUSTED" in err:
                print("[agent_loop] LLM rate limit hit — using raw format.")
            else:
                print(f"[agent_loop] LLM synthesis error: {err[:120]} — using raw format.")

    return _format_raw_answer(context), all_citations


def _synthesise(question: str, context: list[dict]) -> str:
    """Ask the LLM to write a coherent answer from the collected tool results."""
    from agent.llm import call_llm

    context_text = ""
    for i, entry in enumerate(context, 1):
        result = entry["result"]
        context_text += f"\n[Source {i} — {entry['tool']}]\n"
        if result.success:
            output_str = str(result.output)
            if len(output_str) > 3000:
                output_str = output_str[:3000] + "... [truncated]"
            context_text += output_str + "\n"
            if result.source_citations:
                context_text += f"Citations: {', '.join(result.source_citations)}\n"
        else:
            context_text += f"Error: {result.error}\n"

    prompt = (
        f"You are a financial analyst assistant. Using ONLY the information provided below, "
        f"answer the user's question clearly and concisely. "
        f"Cite the specific source for each claim (document name + page, or database row). "
        f"Do not hallucinate or add information not present in the sources.\n\n"
        f"Question: {question}\n\n"
        f"Retrieved information:\n{context_text}\n\n"
        f"Write a clear, well-structured answer with inline citations."
    )

    return call_llm(prompt, temperature=0.2)


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
                parts.append(
                    f"[Web — {item['title']} ({item['published_date']})]:\n"
                    f"{item['snippet']}\nSource: {item['url']}"
                )

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
    # Arithmetic: integers and decimals, operators +, -, *, /
    arith_match = re.match(
        r'^\s*(-?\d+\.?\d*)\s*([+\-*/])\s*(-?\d+\.?\d*)\s*$',
        question.strip()
    )
    if arith_match:
        a = float(arith_match.group(1))
        op = arith_match.group(2)
        b = float(arith_match.group(3))
        if op == '+':
            result = a + b
        elif op == '-':
            result = a - b
        elif op == '*':
            result = a * b
        elif op == '/':
            if b == 0:
                return "Division by zero is undefined."
            result = a / b
        return str(int(result) if result == int(result) else result)

    # Greetings
    q = question.lower()
    if any(g in q for g in ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"]):
        return "Hello! Ask me anything about Infosys, TCS, or Wipro financials."

    return "I can answer questions about Infosys, TCS, and Wipro using their annual reports, financial data, and live web search."
