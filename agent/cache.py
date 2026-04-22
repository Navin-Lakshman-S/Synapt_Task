# cache.py — File-backed response cache for the agent.
#
# Purpose: avoid re-running the full agent pipeline (and spending API tokens)
# when the same question is asked more than once in a session or across runs.
#
# Cache key: normalised question string (lowercased, stripped, collapsed whitespace).
# Cache store: JSON file at traces/response_cache.json (auto-created).
# Cache scope: static questions only — web_search results are NOT cached because
#              they are time-sensitive (stock prices, news). Any response that used
#              web_search is excluded from caching.

import json
import os
import re

CACHE_PATH = "traces/response_cache.json"


def _normalise(question: str) -> str:
    """Normalise a question to a stable cache key."""
    return re.sub(r'\s+', ' ', question.strip().lower())


def _load() -> dict:
    """Load the cache from disk. Returns empty dict if file doesn't exist."""
    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save(cache: dict) -> None:
    """Persist the cache to disk."""
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)


def get(question: str) -> dict | None:
    """
    Return the cached response dict for this question, or None if not cached.

    Args:
        question: The raw user question.

    Returns:
        The cached export_trace_to_dict output, or None.
    """
    key = _normalise(question)
    cache = _load()
    return cache.get(key)


def put(question: str, trace_dict: dict) -> None:
    """
    Store a response in the cache.

    Web-search responses are excluded because they contain time-sensitive data.

    Args:
        question:   The raw user question.
        trace_dict: The export_trace_to_dict output to cache.
    """
    # Don't cache responses that used web_search — results change over time
    tools_used = [
        step.get("tool_name")
        for step in trace_dict.get("trace", [])
        if step.get("action_type") == "tool"
    ]
    if "web_search" in tools_used:
        return

    key = _normalise(question)
    cache = _load()
    cache[key] = trace_dict
    _save(cache)


def clear() -> None:
    """Delete all cached responses."""
    if os.path.exists(CACHE_PATH):
        os.remove(CACHE_PATH)
    print("[cache] Cache cleared.")


def stats() -> dict:
    """Return cache statistics."""
    cache = _load()
    return {"entries": len(cache), "path": CACHE_PATH}
