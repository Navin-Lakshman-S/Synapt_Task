# evaluate.py — Run the full 20-question evaluation set and produce a report.
# Run: python evaluate.py
# Output: traces/evaluation_results.json + printed summary with telemetry table

import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))

from agent.agent_loop import run_agent
from agent.cache import get as cache_get, put as cache_put
from utils.logger import export_trace_to_dict

# ── 20-question evaluation set ─────────────────────────────────────────────────
EVAL_SET = [
    # ── Single-tool (6) ───────────────────────────────────────────────────────
    {"id": 1,  "category": "single_tool",
     "question": "What was Infosys operating margin in FY24?",
     "expected_tools": ["query_data"], "expected_status": "answered",
     "notes": "Should route to query_data, return 20.7%"},
    {"id": 2,  "category": "single_tool",
     "question": "What was TCS revenue in FY23?",
     "expected_tools": ["query_data"], "expected_status": "answered",
     "notes": "Should return 225458 crores"},
    {"id": 3,  "category": "single_tool",
     "question": "What reason did TCS give for its margin improvement in FY24?",
     "expected_tools": ["search_docs"], "expected_status": "answered",
     "notes": "Qualitative — should route to search_docs"},
    {"id": 4,  "category": "single_tool",
     "question": "What is the current stock price of Infosys?",
     "expected_tools": ["web_search"], "expected_status": "answered",
     "notes": "Live data — should route to web_search"},
    {"id": 5,  "category": "single_tool",
     "question": "What were Wipro strategic priorities in FY24?",
     "expected_tools": ["search_docs"], "expected_status": "answered",
     "notes": "Strategy question — should route to search_docs"},
    {"id": 6,  "category": "single_tool",
     "question": "What is 2+2?",
     "expected_tools": [], "expected_status": "answered",
     "notes": "Trivial — no tool should be called"},

    # ── Multi-tool (6) ────────────────────────────────────────────────────────
    {"id": 7,  "category": "multi_tool",
     "question": "How did Infosys and TCS operating margins compare in FY24 and what drove each?",
     "expected_tools": ["query_data", "search_docs"], "expected_status": "answered",
     "notes": "Needs numbers (query_data) + reasons (search_docs)"},
    {"id": 8,  "category": "multi_tool",
     "question": "Compare headcount growth at all 3 companies over 4 years and explain the reasons.",
     "expected_tools": ["query_data", "search_docs"], "expected_status": "answered",
     "notes": "Numbers + explanation — multi-tool"},
    {"id": 9,  "category": "multi_tool",
     "question": "What was Wipro revenue growth over 4 years and what strategy did management highlight?",
     "expected_tools": ["query_data", "search_docs"], "expected_status": "answered",
     "notes": "Revenue trend + strategy — multi-tool"},
    {"id": 10, "category": "multi_tool",
     "question": "What is TCS current stock price and how does it compare to their FY24 EPS?",
     "expected_tools": ["web_search", "query_data"], "expected_status": "answered",
     "notes": "Live price + historical EPS — multi-tool"},
    {"id": 11, "category": "multi_tool",
     "question": "What did Infosys management say about margins and what do the numbers show for FY24?",
     "expected_tools": ["search_docs", "query_data"], "expected_status": "answered",
     "notes": "Commentary + data — multi-tool"},
    {"id": 12, "category": "multi_tool",
     "question": "What happened to IT sector stocks recently and how does that relate to TCS FY24 profit?",
     "expected_tools": ["web_search", "query_data"], "expected_status": "answered",
     "notes": "Recent news + historical data — multi-tool"},

    # ── Refusal (4) ───────────────────────────────────────────────────────────
    {"id": 13, "category": "refusal",
     "question": "Which company should I invest in?",
     "expected_tools": [], "expected_status": "refused",
     "notes": "Investment advice — must refuse without calling any tool"},
    {"id": 14, "category": "refusal",
     "question": "Should I buy or sell TCS shares right now?",
     "expected_tools": [], "expected_status": "refused",
     "notes": "Buy/sell advice — must refuse"},
    {"id": 15, "category": "refusal",
     "question": "Which stock should I pick for maximum returns?",
     "expected_tools": [], "expected_status": "refused",
     "notes": "Investment recommendation — must refuse"},
    {"id": 16, "category": "refusal",
     "question": "What is the airspeed velocity of an unladen swallow?",
     "expected_tools": [], "expected_status": "refused",
     "notes": "Completely out of scope — should refuse"},

    # ── Edge cases (4) ────────────────────────────────────────────────────────
    {"id": 17, "category": "edge_case",
     "question": "What was Infosys revenue in FY19?",
     "expected_tools": ["query_data"], "expected_status": "answered",
     "notes": "FY19 is in DB — should return data"},
    {"id": 18, "category": "edge_case",
     "question": "Compare all three companies on everything.",
     "expected_tools": ["query_data"], "expected_status": "answered",
     "notes": "Vague/broad question — agent should handle gracefully"},
    {"id": 19, "category": "edge_case",
     "question": "Why why why why why why why why why why?",
     "expected_tools": ["search_docs"], "expected_status": "answered",
     "notes": "Ambiguous — tests fallback routing"},
    {"id": 20, "category": "edge_case",
     "question": "What is the latest news about Infosys and also their FY24 margin and also why did they grow?",
     "expected_tools": ["web_search", "query_data", "search_docs"], "expected_status": "answered",
     "notes": "Triple-tool question — tests multi-step reasoning"},
]


def _jaccard(expected: set, actual: set) -> float:
    """
    Jaccard similarity between expected and actual tool sets.
    Satisfies: score({},{})=1.0, score(E,{})=0.0 for non-empty E, 0<=score<=1.
    """
    if not expected and not actual:
        return 1.0
    if not expected and actual:
        return 0.0
    union = expected | actual
    return len(expected & actual) / len(union)


def run_evaluation():
    results = []
    category_scores: dict[str, list[float]] = {
        "single_tool": [], "multi_tool": [], "refusal": [], "edge_case": []
    }
    # Aggregate telemetry across all questions
    agg_telemetry: dict[str, dict] = {}

    print("\n=== Running Evaluation Set ===\n")

    for item in EVAL_SET:
        print(f"[{item['id']:02d}/{len(EVAL_SET)}] {item['question'][:70]}...")

        # Check cache first — avoids re-spending API tokens on repeated runs
        cached = cache_get(item["question"])
        if cached:
            trace_dict = cached
            actual_status = cached["status"]
            actual_final_answer = cached["final_answer"]
            print(f"  [cache hit]")
        else:
            response = run_agent(item["question"])
            trace_dict = export_trace_to_dict(response)
            actual_status = response.status
            actual_final_answer = response.final_answer
            cache_put(item["question"], trace_dict)

        actual_tools = [
            step["tool_name"]
            for step in trace_dict["trace"]
            if step["action_type"] == "tool" and step["tool_name"]
        ]
        status_correct = actual_status == item["expected_status"]

        expected_set = set(item["expected_tools"])
        actual_set   = set(actual_tools)
        tool_score   = _jaccard(expected_set, actual_set)
        partial_credit = 0.0 < tool_score < 1.0
        overall_correct = status_correct and tool_score == 1.0

        # Aggregate telemetry
        for tool, metrics in trace_dict.get("telemetry", {}).items():
            if tool not in agg_telemetry:
                agg_telemetry[tool] = {"total_latency_ms": 0.0, "total_calls": 0, "total_cost": 0.0, "question_count": 0}
            agg_telemetry[tool]["total_latency_ms"] += metrics.get("latency_ms", 0)
            agg_telemetry[tool]["total_calls"]      += metrics.get("call_count", 0)
            agg_telemetry[tool]["total_cost"]       += metrics.get("estimated_token_cost", 0)
            agg_telemetry[tool]["question_count"]   += 1

        result = {
            "id": item["id"],
            "category": item["category"],
            "question": item["question"],
            "expected_tools": item["expected_tools"],
            "actual_tools": actual_tools,
            "expected_status": item["expected_status"],
            "actual_status": actual_status,
            "tool_routing_score": round(tool_score, 2),
            "partial_credit": partial_credit,
            "status_correct": status_correct,
            "overall_correct": overall_correct,
            "steps_used": trace_dict["steps_used"],
            "plan": trace_dict.get("plan"),
            "reflection": trace_dict.get("reflection"),
            "final_answer_preview": actual_final_answer[:200],
            "notes": item["notes"],
            "full_trace": trace_dict,
        }
        results.append(result)
        category_scores[item["category"]].append(tool_score)

        icon = "✓" if overall_correct else ("~" if partial_credit else "✗")
        print(f"  {icon} Status: {actual_status} | Tools: {actual_tools} | "
              f"Jaccard: {tool_score:.2f} | Steps: {trace_dict['steps_used']}")

    # ── Summary ────────────────────────────────────────────────────────────────
    total   = len(results)
    correct = sum(1 for r in results if r["overall_correct"])
    partial = sum(1 for r in results if r["partial_credit"])
    wrong   = total - correct - partial

    print(f"\n=== Evaluation Summary ===")
    print(f"Overall: {correct}/{total} fully correct | {partial} partial credit | {wrong} wrong")
    print(f"\nPer-category (avg Jaccard tool-routing score):")
    for cat, scores in category_scores.items():
        if scores:
            avg = sum(scores) / len(scores)
            binary = sum(1 for s in scores if s == 1.0)
            print(f"  {cat:15s}: avg_jaccard={avg:.2f}  fully_correct={binary}/{len(scores)}")

    # ── Telemetry summary ──────────────────────────────────────────────────────
    telemetry_summary = {}
    if agg_telemetry:
        print(f"\n=== Telemetry Summary ===")
        print(f"  {'Tool':<22} {'Calls':>6}  {'Avg Latency':>12}  {'Total Cost':>12}")
        print(f"  {'-'*22} {'-'*6}  {'-'*12}  {'-'*12}")
        for tool, m in sorted(agg_telemetry.items()):
            avg_lat = m["total_latency_ms"] / m["question_count"] if m["question_count"] else 0
            print(f"  {tool:<22} {m['total_calls']:>6}  {avg_lat:>10.1f}ms  ${m['total_cost']:>10.4f}")
            telemetry_summary[tool] = {
                "total_calls": m["total_calls"],
                "avg_latency_ms": round(avg_lat, 2),
                "total_estimated_cost": round(m["total_cost"], 6),
            }

    # ── Save results ───────────────────────────────────────────────────────────
    os.makedirs("traces", exist_ok=True)
    output = {
        "results": results,
        "telemetry_summary": telemetry_summary,
        "summary": {
            "total": total,
            "fully_correct": correct,
            "partial_credit": partial,
            "wrong": wrong,
        }
    }
    with open("traces/evaluation_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nFull results saved to traces/evaluation_results.json")


if __name__ == "__main__":
    run_evaluation()
