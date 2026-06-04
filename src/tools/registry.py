"""Tool Registry for the ReAct Agent.

Manages tool specifications and provides formatted descriptions for LLM prompts.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional


@dataclass
class ToolSpec:
    """Specification for a tool that the LLM can select."""
    name: str           # e.g., "calculator"
    description: str    # e.g., "Evaluate arithmetic expressions safely."
    parameters: str     # e.g., 'a mathematical expression, e.g., "(8 + 3) * 2 - 1"'
    fn: Callable[[str], str]  # Takes input string, returns observation string


class ToolRegistry:
    """Registry of tools available to the ReAct executor."""

    def __init__(self):
        self._tools: Dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        """Register a tool specification."""
        self._tools[spec.name] = spec

    def get(self, name: str) -> Optional[ToolSpec]:
        """Get a tool by name (case-insensitive)."""
        return self._tools.get(name) or self._tools.get(name.lower())

    def all_specs(self) -> List[ToolSpec]:
        """Return all registered tool specs."""
        return list(self._tools.values())

    def format_for_prompt(self) -> str:
        """Generate a numbered tool list for inclusion in the ReAct prompt."""
        lines = []
        for i, spec in enumerate(self._tools.values(), 1):
            lines.append(
                f"{i}. {spec.name}({spec.parameters})\n"
                f"   {spec.description}"
            )
        # Always include FINISH as the last "tool"
        lines.append(
            f"{len(self._tools) + 1}. FINISH(\"your complete solution\")\n"
            f"   Return your final answer. Use this when you have enough information to answer."
        )
        return "\n".join(lines)
