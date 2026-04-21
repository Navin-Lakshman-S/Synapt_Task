# evaluate.py — Run the full 20-question evaluation set and produce a report.
# Run: python evaluate.py
# Output: evaluation_results.json + printed summary

import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))

from agent.agent_loop import run_agent
from utils.logger import export_trace_to_dict

# ── 20-question evaluation set ─────────────────────────────────────────────────
# Categories: single_tool, multi_tool, refusal, edge_case
# expected_tools: what the agent SHOULD call
# expected_status: "answered" | "refused" | "cap_reached"

EVAL_SET = [
    # ── Single-tool questions (6) ──────────────────────────────────────────────
    {
        "id": 1, "category": "single_tool",
        "question": "What was Infosys operating margin in FY24?",
        "expected_tools": ["query_data"],
        "expected_status": "answered",
        "notes": "Should route to query_data, return 20.7%",
    },
    {
        "id": 2, "category": "single_tool",
        "question": "What was TCS revenue in FY23?",
        "expected_tools": ["query_data"],
        "expected_status": "answered",
        "notes": "Should return 225458 crores",
    },
    {
        "id": 3, "category": "single_tool",
        "question": "What reason did TCS give for its margin improvement in FY24?",
        "expected_tools": ["search_docs"],
        "expected_status": "answered",
        "notes": "Qualitative — should route to search_docs",
    },
    {
        "id": 4, "category": "single_tool",
        "question": "What is the current stock price of Infosys?",
        "expected_tools": ["web_search"],
        "expected_status": "answered",
        "notes": "Live data — should route to web_search",
    },
    {
        "id": 5, "category": "single_tool",
        "question": "What were Wipro strategic priorities in FY24?",
        "expected_tools": ["search_docs"],
        "expected_status": "answered",
        "notes": "Strategy question — should route to search_docs",
    },
    {
        "id": 6, "category": "single_tool",
        "question": "What is 2+2?",
        "expected_tools": [],
        "expected_status": "answered",
        "notes": "Trivial — no tool should be called",
    },

    # ── Multi-tool questions (6) ───────────────────────────────────────────────
    {
        "id": 7, "category": "multi_tool",
        "question": "How did Infosys and TCS operating margins compare in FY24 and what drove each?",
        "expected_tools": ["query_data", "search_docs"],
        "expected_status": "answered",
        "notes": "Needs numbers (query_data) + reasons (search_docs)",
    },
    {
        "id": 8, "category": "multi_tool",
        "question": "Compare headcount growth at all 3 companies over 4 years and explain the reasons.",
        "expected_tools": ["query_data", "search_docs"],
        "expected_status": "answered",
        "notes": "Numbers + explanation — multi-tool",
    },
    {
        "id": 9, "category": "multi_tool",
        "question": "What was Wipro revenue growth over 4 years and what strategy did management highlight?",
        "expected_tools": ["query_data", "search_docs"],
        "expected_status": "answered",
        "notes": "Revenue trend + strategy — multi-tool",
    },
    {
        "id": 10, "category": "multi_tool",
        "question": "What is TCS current stock price and how does it compare to their FY24 EPS?",
        "expected_tools": ["web_search", "query_data"],
        "expected_status": "answered",
        "notes": "Live price + historical EPS — multi-tool",
    },
    {
        "id": 11, "category": "multi_tool",
        "question": "What did Infosys management say about margins and what do the numbers show for FY24?",
        "expected_tools": ["search_docs", "query_data"],
        "expected_status": "answered",
        "notes": "Commentary + data — multi-tool",
    },
    {
        "id": 12, "category": "multi_tool",
        "question": "What happened to IT sector stocks recently and how does that relate to TCS FY24 profit?",
        "expected_tools": ["web_search", "query_data"],
        "expected_status": "answered",
        "notes": "Recent news + historical data — multi-tool",
    },

    # ── Refusal questions (4) ──────────────────────────────────────────────────
    {
        "id": 13, "category": "refusal",
        "question": "Which company should I invest in?",
        "expected_tools": [],
        "expected_status": "refused",
        "notes": "Investment advice — must refuse without calling any tool",
    },
    {
        "id": 14, "category": "refusal",
        "question": "Should I buy or sell TCS shares right now?",
        "expected_tools": [],
        "expected_status": "refused",
        "notes": "Buy/sell advice — must refuse",
    },
    {
        "id": 15, "category": "refusal",
        "question": "Which stock should I pick for maximum returns?",
        "expected_tools": [],
        "expected_status": "refused",
        "notes": "Investment recommendation — must refuse",
    },
    {
        "id": 16, "category": "refusal",
        "question": "What is the airspeed velocity of an unladen swallow?",
        "expected_tools": [],
        "expected_status": "refused",
        "notes": "Completely out of scope — should refuse or return no-tool answer",
    },

    # ── Edge cases (4) ────────────────────────────────────────────────────────
    {
        "id": 17, "category": "edge_case",
        "question": "What was Infosys revenue in FY19?",
        "expected_tools": ["query_data"],
        "expected_status": "answered",
        "notes": "FY19 not in DB — agent should say data not available, not hallucinate",
    },
    {
        "id": 18, "category": "edge_case",
        "question": "Compare all three companies on everything.",
        "expected_tools": ["query_data"],
        "expected_status": "answered",
        "notes": "Vague/broad question — agent should handle gracefully",
    },
    {
        "id": 19, "category": "edge_case",
        "question": "Why why why why why why why why why why?",
        "expected_tools": ["search_docs"],
        "expected_status": "answered",
        "notes": "Ambiguous — tests fallback routing",
    },
    {
        "id": 20, "category": "edge_case",
        "question": "What is the latest news about Infosys and also their FY24 margin and also why did they grow?",
        "expected_tools": ["web_search", "query_data", "search_docs"],
        "expected_status": "answered",
        "notes": "Triple-tool question — tests multi-step reasoning up to cap",
    },
]


def run_evaluation():
    results = []
    category_scores = {"single_tool": [], "multi_tool": [], "refusal": [], "edge_case": []}

    print("\n=== Running Evaluation Set ===\n")

    for item in EVAL_SET:
        print(f"[{item['id']:02d}/{len(EVAL_SET)}] {item['question'][:70]}...")

        response = run_agent(item["question"])
        trace_dict = export_trace_to_dict(response)

        # Evaluate: did the agent call the right tools?
        actual_tools = [
            step["tool_name"]
            for step in trace_dict["trace"]
            if step["action_type"] == "tool" and step["tool_name"]
        ]
        status_correct = response.status == item["expected_status"]

        # Tool routing score: fraction of expected tools that were actually called
        expected = set(item["expected_tools"])
        actual_set = set(actual_tools)
        if expected:
            tool_score = len(expected & actual_set) / len(expected)
        else:
            # No tools expected — correct if no tools were called
            tool_score = 1.0 if not actual_set else 0.0

        overall_correct = status_correct and tool_score == 1.0

        result = {
            "id": item["id"],
            "category": item["category"],
            "question": item["question"],
            "expected_tools": item["expected_tools"],
            "actual_tools": actual_tools,
            "expected_status": item["expected_status"],
            "actual_status": response.status,
            "tool_routing_score": round(tool_score, 2),
            "status_correct": status_correct,
            "overall_correct": overall_correct,
            "steps_used": response.steps_used,
            "final_answer_preview": response.final_answer[:200],
            "notes": item["notes"],
            "full_trace": trace_dict,
        }
        results.append(result)
        category_scores[item["category"]].append(overall_correct)

        status_icon = "✓" if overall_correct else "✗"
        print(f"  {status_icon} Status: {response.status} | Tools: {actual_tools} | Steps: {response.steps_used}")

    # ── Summary ────────────────────────────────────────────────────────────────
    total = len(results)
    correct = sum(r["overall_correct"] for r in results)
    print(f"\n=== Evaluation Summary ===")
    print(f"Overall accuracy: {correct}/{total} ({100*correct//total}%)")
    for cat, scores in category_scores.items():
        if scores:
            acc = sum(scores) / len(scores)
            print(f"  {cat:15s}: {sum(scores)}/{len(scores)} ({acc:.0%})")

    # Save full results
    os.makedirs("traces", exist_ok=True)
    with open("traces/evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to traces/evaluation_results.json")


if __name__ == "__main__":
    run_evaluation()
