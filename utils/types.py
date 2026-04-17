# types.py — Shared data structures used across the entire system.
# Keeping types in one place makes it easy to understand what flows between components.

from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class ToolResult:
    """What every tool must return. Consistent shape = agent can handle any tool uniformly."""
    tool_name: str
    input_query: str
    output: list | dict | str      # raw result from the tool
    source_citations: list[str]    # e.g. ["Infosys_AR24.pdf p.47", "financials.csv row 12"]
    success: bool
    error: Optional[str] = None    # populated if the tool failed


@dataclass
class AgentAction:
    """What the decision engine returns each step.
    type='tool'   → call a tool
    type='final'  → compose and return the answer
    type='refuse' → agent cannot / should not answer
    """
    type: Literal["tool", "final", "refuse"]
    tool_name: Optional[str] = None   # only when type='tool'
    input: Optional[str] = None       # the query to pass to the tool
    reasoning: str = ""               # WHY the agent made this choice (key for explainability)


@dataclass
class TraceStep:
    """One recorded step in the agent's execution trace."""
    step_number: int
    action: AgentAction
    result: Optional[ToolResult] = None


@dataclass
class AgentResponse:
    """The final object returned after the agent finishes."""
    question: str
    final_answer: str
    citations: list[str]
    trace: list[TraceStep]
    steps_used: int
    status: Literal["answered", "refused", "cap_reached"]
