# main.py — CLI entry point. Run: python main.py
# Accepts a question from the user, runs the agent, prints the full trace.
# Repeated questions are served from cache — no API calls, no cost.

import sys
import json
import os

sys.path.insert(0, os.path.dirname(__file__))

from agent.agent_loop import run_agent
from agent.cache import get as cache_get, put as cache_put, stats as cache_stats
from utils.logger import print_trace, export_trace_to_dict


def main():
    print("\n=== Agentic RAG — Indian IT Financials ===")
    print("Type your question and press Enter.")
    print("Type 'quit' to exit.\n")

    cs = cache_stats()
    if cs["entries"] > 0:
        print(f"[cache] {cs['entries']} cached response(s) loaded from {cs['path']}\n")

    while True:
        try:
            question = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break

        # ── Check cache first ──────────────────────────────────────────────────
        cached = cache_get(question)
        if cached:
            print("\n[cache] Returning cached response (no API call made).")
            print(f"\n{'='*60}")
            print(f"QUESTION: {cached['question']}")
            print(f"{'='*60}")
            if cached.get("plan"):
                print(f"\nPLAN: {cached['plan']}")
            print(f"\nFINAL ANSWER : {cached['final_answer']}")
            if cached.get("citations"):
                print(f"\nCITATIONS:")
                for i, c in enumerate(cached["citations"], 1):
                    print(f"  [{i}] {c}")
            print(f"\nSTEPS USED   : {cached['steps_used']} / 8 max  [cached]")
            print(f"{'='*60}")
            print()
            continue

        # ── Run the agent ──────────────────────────────────────────────────────
        response = run_agent(question)
        print_trace(response)

        # Cache the response for future identical questions
        trace_dict = export_trace_to_dict(response)
        cache_put(question, trace_dict)

        # Optionally save trace to JSON
        save = input("\nSave trace to JSON? (y/n): ").strip().lower()
        if save == "y":
            os.makedirs("traces", exist_ok=True)
            filename = f"traces/trace_{len(os.listdir('traces')) + 1}.json"
            with open(filename, "w") as f:
                json.dump(trace_dict, f, indent=2)
            print(f"Saved to {filename}")

        print()


if __name__ == "__main__":
    main()
