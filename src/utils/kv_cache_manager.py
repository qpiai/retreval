"""
KV Cache Manager — Manages prompt prefix caching for efficient vLLM inference.

vLLM supports automatic prefix caching: when multiple requests share the same
prefix tokens, the KV cache is reused. This manager structures all prompts so
they share a stable prefix (system context + problem + memory), and each node
only appends its task-specific suffix.

Architecture (system / user message split):
  - System message: static role instruction (identical across ALL calls → max
    KV cache reuse).
  - User message: [PROBLEM + PLAN + MEMORY]  (stable per problem → prefix
    reused across nodes)  +  [task-specific suffix]  (varies per node).

Usage:
    cache_mgr = KVCacheManager(memory=reflexion_memory)
    cache_mgr.set_problem_context(problem)

    # Each node builds its prompt using the shared prefix:
    sys_msg, user_msg = cache_mgr.build_expand_messages(thought, num_children)
    response = vllm._call_api(user_msg, system_prompt=sys_msg)
"""

from __future__ import annotations

import hashlib
from typing import Optional, List, Dict, Any, Tuple

from src.utils.memory import ReflexionMemory

# ---------------------------------------------------------------------------
# Static system prompt — identical across every call for max KV cache reuse.
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = (
    "You are an expert math problem solver. "
    "Solve the PROBLEM stated in the user message directly. "
    "Show your step-by-step work, then put your final answer in \\boxed{answer} format. "
    "Do NOT analyze the prompt structure or discuss what you should do — just solve the problem."
)


class KVCacheManager:
    """Manages prompt prefix caching for vLLM prefix-cache-friendly inference."""

    def __init__(self, memory: ReflexionMemory, verbose: bool = True):
        self.memory = memory
        self.verbose = verbose

        # Cached components
        self._problem: str = ""
        self._plan: str = ""
        self._cached_memory_summary: str = ""
        self._memory_version: int = -1
        self._cached_user_prefix: str = ""
        self._prefix_dirty: bool = True

        # Stats
        self._total_calls: int = 0
        self._prefix_builds: int = 0

    # ------------------------------------------------------------------
    # Problem context (set once per problem, reused across all nodes)
    # ------------------------------------------------------------------
    def set_problem_context(self, problem: str, plan: str = "") -> None:
        """Set the problem context. Called once at the start of each problem."""
        self._problem = problem
        self._plan = plan
        self._prefix_dirty = True

    def update_plan(self, plan: str) -> None:
        """Update plan after planning node completes."""
        if plan != self._plan:
            self._plan = plan
            self._prefix_dirty = True

    # ------------------------------------------------------------------
    # Memory summary caching
    # ------------------------------------------------------------------
    def _refresh_memory_if_needed(self) -> str:
        """Regenerate memory summary only if memory state has changed."""
        version = self.memory.iteration_count + len(self.memory.entries)
        if version != self._memory_version:
            self._cached_memory_summary = self.memory.get_summary()
            self._memory_version = version
            self._prefix_dirty = True
        return self._cached_memory_summary

    def refresh_memory_summary(self) -> str:
        """Public API: refresh and return cached memory summary."""
        return self._refresh_memory_if_needed()

    def get_cached_memory_summary(self) -> str:
        """Return cached memory summary, refreshing only if needed."""
        return self._refresh_memory_if_needed()

    # ------------------------------------------------------------------
    # User-message prefix (stable per problem, shared across all nodes)
    # ------------------------------------------------------------------
    def _build_user_prefix(self) -> str:
        """Build the user-message prefix shared across all LLM calls.

        Contains: PROBLEM + PLAN + MEMORY (in that order for cache locality).
        """
        if not self._prefix_dirty and self._cached_user_prefix:
            return self._cached_user_prefix

        parts = [f"PROBLEM:\n{self._problem}"]

        if self._plan:
            parts.append(f"PLAN:\n{self._plan}")

        mem = self._refresh_memory_if_needed()
        if mem and mem != "No memory insights yet.":
            parts.append(f"MEMORY INSIGHTS:\n{mem[:500]}")

        self._cached_user_prefix = "\n\n".join(parts)
        self._prefix_dirty = False
        self._prefix_builds += 1
        return self._cached_user_prefix

    # ------------------------------------------------------------------
    # Legacy string-based prompt (backward compat)
    # ------------------------------------------------------------------
    def _build_prefix(self) -> str:
        """Legacy: return system + user prefix concatenated as one string."""
        return f"{_SYSTEM_PROMPT}\n\n{self._build_user_prefix()}"

    # ------------------------------------------------------------------
    # Message-based prompt builders  (system, user) tuples
    # ------------------------------------------------------------------
    def build_expand_messages(
        self, parent_thought: str, num_children: int = 3
    ) -> Tuple[str, str]:
        """Expansion prompt as (system_msg, user_msg)."""
        self._total_calls += 1
        prefix = self._build_user_prefix()

        relevant = self.memory.get_relevant(parent_thought, max_entries=2)
        mem_block = f"\nRelevant insights:\n{relevant}\n" if relevant else ""

        suffix = f"""
---
TASK: Generate {num_children} approaches to solve the PROBLEM above.

Current reasoning so far:
{parent_thought}
{mem_block}
IMPORTANT: All approaches MUST be about solving the PROBLEM stated above — \
do NOT suggest approaches for any other problem.

Suggest {num_children} different ways to continue solving this problem. \
Each should take a different approach.

Format as a numbered list:
1. ...
2. ...
3. ...

Provide ONLY the numbered list."""

        return _SYSTEM_PROMPT, f"{prefix}\n{suffix}"

    def build_refine_messages(
        self, thought: str, critique_history: List[str]
    ) -> Tuple[str, str]:
        """Refinement prompt as (system_msg, user_msg)."""
        self._total_calls += 1
        prefix = self._build_user_prefix()

        relevant = self.memory.get_relevant(thought, max_entries=2)

        parts = [f"Here is a solution attempt for the PROBLEM above:\n\n{thought}"]

        if critique_history:
            critique_text = "\n".join(f"- {c}" for c in critique_history[-3:])
            parts.append(f"Previous feedback:\n{critique_text}")

        if relevant:
            parts.append(f"Relevant insights:\n{relevant}")

        parts.append(
            "Rewrite this solution to fix any errors and fill gaps. "
            "Focus ONLY on the PROBLEM above. "
            "Show your work step by step.\n\nIMPROVED:"
        )

        suffix = "\n\n".join(parts)
        return _SYSTEM_PROMPT, f"{prefix}\n\n---\nTASK: Refine the solution.\n\n{suffix}"

    def build_react_messages(
        self,
        current_thought: str,
        tool_list: str,
        history_str: str,
    ) -> Tuple[str, str]:
        """ReAct prompt as (system_msg, user_msg)."""
        self._total_calls += 1
        prefix = self._build_user_prefix()

        relevant = self.memory.get_relevant(current_thought, max_entries=2)
        mem_block = f"RELEVANT MEMORY:\n{relevant}\n" if relevant else ""

        user_msg = f"""{prefix}

---
TASK: Solve the PROBLEM above step-by-step using available tools.

CURRENT REASONING: {current_thought}
{mem_block}
AVAILABLE TOOLS:
{tool_list}

PREVIOUS STEPS:
{history_str}

Think about what to do next to solve the PROBLEM above, then choose ONE action.

Output format (follow EXACTLY):
THOUGHT: <your reasoning>
ACTION: <chosen_tool>("<argument>")

Example — if you want to compute 2+3:
THOUGHT: I need to add 2 and 3.
ACTION: calculator("2 + 3")

When you have a final answer to the PROBLEM above, use:
THOUGHT: <summary of reasoning>
ACTION: FINISH("\\boxed{{your_answer}}")

Rules:
- Pick a tool from AVAILABLE TOOLS above or use FINISH. Do NOT invent tool names.
- Focus ONLY on the PROBLEM stated above. Do NOT analyze the prompt — just solve the problem.
- Use tools only when they add value; if you can reason the answer directly, use FINISH.
- Do NOT repeat the same tool call with the same input.
- Put your final answer in \\boxed{{answer}} format."""

        return _SYSTEM_PROMPT, user_msg

    def build_cross_critique_messages(
        self,
        node_a_id: str, thought_a: str,
        node_b_id: str, thought_b: str,
    ) -> Tuple[str, str]:
        """Cross-critique prompt as (system_msg, user_msg)."""
        self._total_calls += 1
        prefix = self._build_user_prefix()
        user_msg = f"""{prefix}

---
TASK: Compare and score these two approaches to the PROBLEM above.

NODE {node_a_id}:
{thought_a[:500]}

NODE {node_b_id}:
{thought_b[:500]}

Rate each from 0.0 to 1.0 based on correctness, efficiency, and clarity \
for solving the PROBLEM above.

Respond ONLY in this exact format:
SCORE_A: 0.X
SCORE_B: 0.X"""

        return _SYSTEM_PROMPT, user_msg

    def build_synthesis_messages(
        self,
        best_strategy: str,
        reasoning_path: List[str],
    ) -> Tuple[str, str]:
        """Synthesis prompt as (system_msg, user_msg)."""
        self._total_calls += 1
        prefix = self._build_user_prefix()
        path_text = "\n".join(
            f"Step {i+1}: {step}" for i, step in enumerate(reasoning_path[-3:])
        )

        has_retrieved_data = "RELEVANT DATA RETRIEVED:" in best_strategy

        if has_retrieved_data:
            user_msg = f"""{prefix}

---
TASK: Provide the FINAL, COMPLETE solution to the PROBLEM above.

BEST STRATEGY AND RETRIEVED DATA:
{best_strategy}

REASONING PATH:
{path_text}

SOLVE the PROBLEM above using the retrieved data. Show your work step-by-step.
Provide SPECIFIC answers. Do NOT say "cannot be solved"."""
        else:
            user_msg = f"""{prefix}

---
TASK: Provide the FINAL, COMPLETE solution to the PROBLEM above.

BEST STRATEGY:
{best_strategy}

REASONING PATH:
{path_text}

IMPLEMENT this strategy and provide the CONCRETE solution to the PROBLEM above.
- If it's code, write the actual code
- If it's analysis, provide detailed analysis with numbers
- Show calculations and reasoning step-by-step

Provide the complete solution:"""

        return _SYSTEM_PROMPT, user_msg

    # ------------------------------------------------------------------
    # Legacy string-based prompt builders (backward compat for non-vLLM)
    # ------------------------------------------------------------------
    def build_prompt(
        self,
        task_instruction: str,
        dynamic_content: str = "",
        relevance_query: str = "",
    ) -> str:
        """Build a prompt with shared prefix + task-specific suffix (string)."""
        self._total_calls += 1
        prefix = self._build_prefix()

        suffix_parts = [f"\n\n---\nTASK: {task_instruction}"]

        if relevance_query:
            relevant = self.memory.get_relevant(relevance_query, max_entries=3)
            if relevant:
                suffix_parts.append(f"\nRelevant insights:\n{relevant}")

        if dynamic_content:
            suffix_parts.append(f"\n{dynamic_content}")

        return prefix + "\n".join(suffix_parts)

    def build_expand_prompt(self, parent_thought: str, num_children: int = 3) -> str:
        """Expansion prompt (legacy string version)."""
        sys_msg, user_msg = self.build_expand_messages(parent_thought, num_children)
        return f"{sys_msg}\n\n{user_msg}"

    def build_refine_prompt(self, thought: str, critique_history: List[str]) -> str:
        """Refinement prompt (legacy string version)."""
        sys_msg, user_msg = self.build_refine_messages(thought, critique_history)
        return f"{sys_msg}\n\n{user_msg}"

    def build_react_prompt(
        self,
        current_thought: str,
        tool_list: str,
        history_str: str,
    ) -> str:
        """ReAct prompt (legacy string version)."""
        sys_msg, user_msg = self.build_react_messages(current_thought, tool_list, history_str)
        return f"{sys_msg}\n\n{user_msg}"

    def build_score_prompt(self, thought: str) -> str:
        """Score prompt: short, no full prefix needed (independent evaluation)."""
        self._total_calls += 1
        return f"""Evaluate this reasoning on a scale of 0-1.

{thought}

Score based on: correctness (most important), completeness, and clarity.

SCORE: """

    def build_cross_critique_prompt(
        self,
        node_a_id: str, thought_a: str,
        node_b_id: str, thought_b: str,
    ) -> str:
        """Cross-critique prompt (legacy string version)."""
        sys_msg, user_msg = self.build_cross_critique_messages(
            node_a_id, thought_a, node_b_id, thought_b
        )
        return f"{sys_msg}\n\n{user_msg}"

    def build_synthesis_prompt(
        self,
        best_strategy: str,
        reasoning_path: List[str],
    ) -> str:
        """Synthesis prompt (legacy string version)."""
        sys_msg, user_msg = self.build_synthesis_messages(best_strategy, reasoning_path)
        return f"{sys_msg}\n\n{user_msg}"

    # ------------------------------------------------------------------
    # Stats / diagnostics
    # ------------------------------------------------------------------
    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_calls": self._total_calls,
            "prefix_builds": self._prefix_builds,
            "prefix_reuse_ratio": (
                f"{(self._total_calls - self._prefix_builds) / max(self._total_calls, 1):.1%}"
            ),
            "cached_memory_len": len(self._cached_memory_summary),
            "memory_entries": len(self.memory.entries),
        }

    def log_stats(self) -> None:
        if self.verbose:
            s = self.get_stats()
            print(f"\n📊 KV Cache Stats: {s['total_calls']} prompts built, "
                  f"prefix rebuilt {s['prefix_builds']}x "
                  f"(reuse {s['prefix_reuse_ratio']}), "
                  f"memory={s['cached_memory_len']} chars")
