# llm.py — Unified LLM interface with automatic fallback.
#
# Usage:
#   from agent.llm import call_llm
#   response_text = call_llm(prompt, temperature=0.1)
#
# Reads from .env:
#   LLM_TYPE    = GEMINI | GROQ   (default: GEMINI)
#   GEMINI_API_KEY, GEMINI_MODEL  (default model: gemini-2.0-flash-lite)
#   GROQ_API_KEY, GROQ_MODEL      (default model: llama-3.3-70b-versatile)
#
# Fallback chain:
#   LLM_TYPE=GEMINI  →  Gemini first, then Groq, then raises LLMUnavailableError
#   LLM_TYPE=GROQ    →  Groq first, then Gemini, then raises LLMUnavailableError
#
# Adding a new provider: implement _call_<provider>(prompt, temperature) -> str
# and add it to PROVIDER_REGISTRY below. Nothing else changes.

import os
from dotenv import load_dotenv

load_dotenv()


class LLMUnavailableError(Exception):
    """Raised when all configured LLM providers fail."""
    pass


# ── Provider implementations ───────────────────────────────────────────────────

def _call_gemini(prompt: str, temperature: float) -> str:
    from google import genai
    from google.genai import types

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set")

    model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-lite")
    client = genai.Client(api_key=api_key)

    import time
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=temperature),
            )
            return response.text.strip()
        except Exception as e:
            err = str(e)
            if "429" in err or "RESOURCE_EXHAUSTED" in err:
                wait = 5
                import re
                m = re.search(r"retryDelay.*?(\d+)s", err)
                if m:
                    wait = min(int(m.group(1)), 15)
                if attempt < 2:
                    time.sleep(wait)
                    continue
            raise


def _call_groq(prompt: str, temperature: float) -> str:
    try:
        from groq import Groq
    except ImportError:
        raise ImportError("groq package not installed. Run: pip install groq")

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set")

    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    client = Groq(api_key=api_key)

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()


# ── Provider registry — add new providers here ────────────────────────────────
PROVIDER_REGISTRY = {
    "gemini": _call_gemini,
    "groq":   _call_groq,
}


def _get_ordered_providers() -> list[str]:
    """Return providers in priority order based on LLM_TYPE env var."""
    llm_type = os.getenv("LLM_TYPE", "GEMINI").upper()
    if llm_type == "GROQ":
        return ["groq", "gemini"]
    return ["gemini", "groq"]  # default: GEMINI first


def call_llm(prompt: str, temperature: float = 0.1) -> str:
    """
    Call the LLM with automatic fallback.

    Tries providers in order based on LLM_TYPE. Falls through to the next
    provider on any error. Raises LLMUnavailableError if all fail.

    Args:
        prompt:      The full prompt string to send.
        temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative).

    Returns:
        The model's response text, stripped of leading/trailing whitespace.

    Raises:
        LLMUnavailableError: if all providers fail.
    """
    providers = _get_ordered_providers()
    errors = []

    for provider in providers:
        fn = PROVIDER_REGISTRY.get(provider)
        if fn is None:
            continue
        try:
            result = fn(prompt, temperature)
            if errors:
                print(f"[llm] Used fallback provider: {provider} "
                      f"(primary failed: {errors[0][0]})")
            return result
        except ImportError as e:
            # Package not installed — skip silently, no point retrying
            errors.append((provider, str(e)[:80]))
        except Exception as e:
            err_msg = str(e)[:100]
            errors.append((provider, err_msg))
            print(f"[llm] {provider} failed: {err_msg} — trying next provider.")

    raise LLMUnavailableError(
        f"All LLM providers failed: {[(p, e) for p, e in errors]}"
    )
