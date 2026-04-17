# logger.py — Formats and prints the agent trace in a readable, structured way.
# This is your primary debugging tool. Every run produces a full trace.

from utils.types import AgentResponse, TraceStep


def print_trace(response: AgentResponse) -> None:
    """Print the full agent trace to stdout in the format required by the assignment."""

    divider = "=" * 60

    print(f"\n{divider}")
    print(f"QUESTION: {response.question}")
    print(divider)

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
                # Truncate long outputs so the trace stays readable
                output_str = str(result.output)
                if len(output_str) > 300:
                    output_str = output_str[:300] + "... [truncated]"
                print(f"  Output    : {output_str}")
                if result.source_citations:
                    print(f"  Sources   : {', '.join(result.source_citations)}")
            else:
                print(f"  ERROR     : {result.error}")
        else:
            print()  # newline for final/refuse steps

    print(f"\n{divider}")
    print(f"STATUS       : {response.status.upper()}")
    print(f"FINAL ANSWER : {response.final_answer}")

    if response.citations:
        print(f"\nCITATIONS:")
        for i, cite in enumerate(response.citations, 1):
            print(f"  [{i}] {cite}")

    print(f"\nSTEPS USED   : {response.steps_used} / 8 max")
    print(divider)


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
            s["output"] = str(step.result.output)
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
        "trace": steps,
    }
