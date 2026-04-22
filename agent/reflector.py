# reflector.py — Post-answer self-critique step (Bonus C).
# After composing a final answer, the agent critiques it once.
# If the answer fails the critique and the hard cap hasn't been hit,
# the agent triggers one more retrieval round.

import os
import json
import re
from dotenv import load_dotenv

load_dotenv()


def reflect(question: str, answer: str, context: list[dict]) -> dict | None:
    """
    Critique the composed answer against the question and retrieved sources.

    Returns:
        {"passes": bool, "issue": str | None}  — when an LLM is available
        None                                    — when no API key is set
    """
    if not (os.getenv("GEMINI_API_KEY") or os.getenv("GROQ_API_KEY")):
        return None

    try:
        return _llm_reflect(question, answer, context)
    except Exception as e:
        print(f"[reflector] LLM error: {str(e)[:80]} — skipping reflection.")
        return None


def _llm_reflect(question: str, answer: str, context: list[dict]) -> dict | None:
    """Call the LLM to critique the answer."""
    from agent.llm import call_llm

    # Summarise retrieved sources for the prompt
    sources_summary = ""
    for i, entry in enumerate(context, 1):
        result = entry["result"]
        sources_summary += f"\n[Source {i} — {entry['tool']}]: "
        if result.success:
            out = str(result.output)
            sources_summary += out[:1500] + ("..." if len(out) > 1500 else "")
        else:
            sources_summary += f"ERROR: {result.error}"

    prompt = (
        f"You are reviewing an answer for quality and grounding.\n\n"
        f"Question: {question}\n\n"
        f"Answer: {answer}\n\n"
        f"Retrieved sources used:{sources_summary}\n\n"
        f"Does the answer directly address the question? "
        f"Is every factual claim supported by the retrieved sources above?\n\n"
        f'Respond with ONLY valid JSON: {{"passes": true/false, "issue": "<one sentence describing the problem, or null if passes>"}}'
    )

    raw = call_llm(prompt, temperature=0.1)

    # Strip markdown fences
    raw = re.sub(r"^```json\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"^```\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    raw = raw.strip()

    try:
        data = json.loads(raw)
        return {
            "passes": bool(data.get("passes", True)),
            "issue": data.get("issue"),
        }
    except json.JSONDecodeError:
        print(f"[reflector] Could not parse LLM response as JSON: {raw[:100]}")
        return None
