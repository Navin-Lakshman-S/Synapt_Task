# telemetry.py — Per-tool latency, call count, and token cost tracking (Bonus B).
# One TelemetryCollector instance lives for the duration of a single run_agent call.

from dataclasses import dataclass, field


@dataclass
class ToolTelemetry:
    """Metrics for a single tool across one agent run."""
    latency_ms: float = 0.0
    call_count: int = 0
    estimated_token_cost: float = 0.0


class TelemetryCollector:
    """
    Collects per-tool metrics during one agent run.

    Usage:
        tc = TelemetryCollector()
        tc.record_tool_call("query_data", 87.3)
        tc.record_token_cost("query_data", 450, 120)
        print(tc.to_dict())
    """

    def __init__(self):
        self._data: dict[str, ToolTelemetry] = {}

    def _get(self, tool_name: str) -> ToolTelemetry:
        if tool_name not in self._data:
            self._data[tool_name] = ToolTelemetry()
        return self._data[tool_name]

    def record_tool_call(self, tool_name: str, latency_ms: float) -> None:
        """Record one tool invocation with its wall-clock latency."""
        entry = self._get(tool_name)
        entry.call_count += 1
        entry.latency_ms += latency_ms

    def record_token_cost(self, tool_name: str, prompt_tokens: int, response_tokens: int) -> None:
        """
        Add estimated token cost for a Gemini API call.
        Formula: (prompt + response tokens) / 1000 * 0.001
        """
        cost = (prompt_tokens + response_tokens) / 1000 * 0.001
        self._get(tool_name).estimated_token_cost += cost

    def to_dict(self) -> dict[str, dict]:
        """Return a JSON-serialisable dict of all recorded metrics."""
        return {
            name: {
                "latency_ms": round(entry.latency_ms, 2),
                "call_count": entry.call_count,
                "estimated_token_cost": round(entry.estimated_token_cost, 6),
            }
            for name, entry in self._data.items()
        }
