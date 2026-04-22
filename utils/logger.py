# logger.py — Formats and prints the agent trace in a readable, structured way.
# This is your primary debugging tool. Every run produces a full trace.

import json
from utils.types import AgentResponse, TraceStep


def print_trace(response: AgentResponse) -> None:
    """Print the full agent trace to stdout in the format required by the assignment."""

    divider = "=" * 60

    print(f"\n{divider}")
    print(f"QUESTION: {response.question}")
    print(divider)

    # Print plan if present (Bonus A)
    if getattr(response, "plan", None):
        print(f"\nPLAN: {response.plan}")

    for step in response.trace:
        action = step.action
        print(f"\nStep {step.step_number}:")
        print(f"  Reasoning : {action.reasoning}")
        print(f"  Action    : {action.type.upper()}", end="")

        if action.type == "tool":
            print(f" → {action.tool_name}")
            print(f"  Input     : {action.input}")

        if step.result:
            result = step.result
            if result.success:
                output_str = str(result.output)
                if len(output_str) > 600:
                    output_str = output_str[:600] + "... [truncated]"
                print(f"  Output    : {output_str}")
                if result.source_citations:
                    print(f"  Sources   : {', '.join(result.source_citations)}")
            else:
                print(f"  ERROR     : {result.error}")
        else:
            print()

    print(f"\n{divider}")
    print(f"STATUS       : {response.status.upper()}")
    print(f"FINAL ANSWER : {response.final_answer}")

    if response.citations:
        print(f"\nCITATIONS:")
        for i, cite in enumerate(response.citations, 1):
            print(f"  [{i}] {cite}")

    # Print reflection if present (Bonus C)
    if getattr(response, "reflection", None):
        print(f"\nREFLECTION: {response.reflection}")

    print(f"\nSTEPS USED   : {response.steps_used} / 8 max")

    # Print telemetry summary if present (Bonus B)
    telemetry = getattr(response, "telemetry", {})
    if telemetry:
        print(f"\nTELEMETRY:")
        for tool, metrics in telemetry.items():
            print(f"  {tool:20s} calls={metrics['call_count']}  "
                  f"latency={metrics['latency_ms']:.1f}ms  "
                  f"cost≈${metrics['estimated_token_cost']:.4f}")

    print(divider)


def _make_json_safe(value) -> object:
    """Convert non-JSON-serialisable values to a safe representation."""
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        pass

    # sqlite3.Row → list
    try:
        import sqlite3
        if isinstance(value, sqlite3.Row):
            return list(value)
    except ImportError:
        pass

    # numpy array → list
    try:
        import numpy as np
        if isinstance(value, np.ndarray):
            return value.tolist()
    except ImportError:
        pass

    # list/tuple of rows
    if isinstance(value, (list, tuple)):
        return [_make_json_safe(item) for item in value]

    # dict
    if isinstance(value, dict):
        return {k: _make_json_safe(v) for k, v in value.items()}

    return str(value)


def export_trace_to_dict(response: AgentResponse) -> dict:
    """Export the trace as a plain dict — useful for saving to JSON for the eval report."""
    steps = []
    for step in response.trace:
        s = {
            "step": step.step_number,
            "reasoning": step.action.reasoning,
            "action_type": step.action.type,
            "tool_name": step.action.tool_name,
            "input": step.action.input,
        }
        if step.result:
            s["output"] = _make_json_safe(step.result.output)
            s["sources"] = step.result.source_citations
            s["success"] = step.result.success
            s["error"] = step.result.error
        steps.append(s)

    return {
        "question": response.question,
        "status": response.status,
        "final_answer": response.final_answer,
        "citations": response.citations,
        "steps_used": response.steps_used,
        "plan": getattr(response, "plan", None),
        "reflection": getattr(response, "reflection", None),
        "telemetry": getattr(response, "telemetry", {}),
        "trace": steps,
    }
