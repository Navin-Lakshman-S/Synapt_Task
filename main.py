# main.py — CLI entry point. Run: python main.py
# Accepts a question from the user, runs the agent, prints the full trace.

import sys
import json
import os

# Make sure imports resolve from project root
sys.path.insert(0, os.path.dirname(__file__))

from agent.agent_loop import run_agent
from utils.logger import print_trace, export_trace_to_dict


def main():
    print("\n=== Agentic RAG — Indian IT Financials ===")
    print("Type your question and press Enter.")
    print("Type 'quit' to exit.\n")

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

        # Run the agent
        response = run_agent(question)

        # Print the structured trace to terminal
        print_trace(response)

        # Optionally save trace to JSON for the evaluation report
        save = input("\nSave trace to JSON? (y/n): ").strip().lower()
        if save == "y":
            os.makedirs("traces", exist_ok=True)
            filename = f"traces/trace_{len(os.listdir('traces')) + 1}.json"
            with open(filename, "w") as f:
                json.dump(export_trace_to_dict(response), f, indent=2)
            print(f"Saved to {filename}")

        print()


if __name__ == "__main__":
    main()
