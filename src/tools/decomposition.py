"""Task Decomposition Tools for ReTreVal Agent.

Decomposes complex problems into sub-tasks, solves them with lightweight
sub-agents, and aggregates results before synthesis.
"""

import re
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, TYPE_CHECKING

from src.clients.llm_client import LLMClient
from src.utils.memory import ReflexionMemory

if TYPE_CHECKING:
    from src.agents.state import AgentState


@dataclass
class SubTask:
    """A single sub-task produced by decomposition."""
    id: str                        # "sub_1", "sub_2", ...
    description: str               # Self-contained sub-problem
    depends_on: List[str] = field(default_factory=list)  # IDs of prerequisites
    context: str = ""              # Inherited context from parent
    status: str = "pending"        # pending | completed | failed
    result: str = ""
    score: float = 0.0
    attempts: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "depends_on": self.depends_on,
            "context": self.context,
            "status": self.status,
            "result": self.result,
            "score": self.score,
            "attempts": self.attempts,
        }


def topological_sort(subtasks: List[SubTask]) -> List[SubTask]:
    """Kahn's algorithm — orders sub-tasks respecting dependencies."""
    id_to_task = {t.id: t for t in subtasks}
    in_degree: Dict[str, int] = {t.id: 0 for t in subtasks}
    adj: Dict[str, List[str]] = {t.id: [] for t in subtasks}

    for t in subtasks:
        for dep in t.depends_on:
            if dep in adj:
                adj[dep].append(t.id)
                in_degree[t.id] += 1

    queue: deque[str] = deque(tid for tid, deg in in_degree.items() if deg == 0)
    ordered: List[SubTask] = []

    while queue:
        tid = queue.popleft()
        ordered.append(id_to_task[tid])
        for neighbour in adj[tid]:
            in_degree[neighbour] -= 1
            if in_degree[neighbour] == 0:
                queue.append(neighbour)

    # If cycle detected, return original order as fallback
    if len(ordered) != len(subtasks):
        return subtasks
    return ordered


class TaskDecomposerTool:
    """Decomposes a complex problem into 2-4 self-contained sub-tasks."""

    def __init__(self, llm: LLMClient, memory: ReflexionMemory, verbose: bool = True):
        self.llm = llm
        self.memory = memory
        self.verbose = verbose

    # ---------- ReAct call() wrapper ----------
    def call(self, problem: str, state: dict = None) -> str:
        """Decompose *problem* into subtasks and return a formatted list."""
        try:
            best_thought = (state or {}).get('best_thought', (state or {}).get('current_thought', ''))
            plan = (state or {}).get('plan', '')
            memory_summary = self.memory.get_summary() if hasattr(self.memory, 'get_summary') else ''

            prompt = f"""Given this problem and the best approach, decompose into 2-4 sub-tasks.

Problem: {problem}
Best Approach: {best_thought[:800]}
Plan: {plan[:400]}
Memory insights: {memory_summary[:300]}

Rules:
- Each sub-task must be self-contained and solvable independently
- Specify dependencies (which sub-tasks must complete first)
- If the problem is already atomic (can't be meaningfully split), return ATOMIC

Format (one line per sub-task):
SUB_1: [description] | DEPENDS: none | CONTEXT: [info needed]
SUB_2: [description] | DEPENDS: sub_1 | CONTEXT: [info needed]
"""
            response = self.llm._call_api(prompt)
            if "ATOMIC" in response.upper():
                return "Problem is ATOMIC - cannot be meaningfully decomposed."
            return response[:600]
        except Exception as exc:
            return f"Decomposition error: {exc}"

    def invoke(self, state: "AgentState") -> "AgentState":
        problem = state.get("problem", "")
        best_thought = state.get("best_thought", state.get("current_thought", ""))
        plan = state.get("plan", "")
        memory_summary = self.memory.get_summary() if hasattr(self.memory, "get_summary") else ""

        prompt = f"""Given this problem and the best approach, decompose into 2-4 sub-tasks.

Problem: {problem}
Best Approach: {best_thought[:800]}
Plan: {plan[:400]}
Memory insights: {memory_summary[:300]}

Rules:
- Each sub-task must be self-contained and solvable independently
- Specify dependencies (which sub-tasks must complete first)
- If the problem is already atomic (can't be meaningfully split), return ATOMIC

Format (one line per sub-task):
SUB_1: [description] | DEPENDS: none | CONTEXT: [info needed]
SUB_2: [description] | DEPENDS: sub_1 | CONTEXT: [info needed]
"""
        response = self.llm._call_api(prompt)

        if "ATOMIC" in response.upper():
            if self.verbose:
                print("\n🔬 DECOMPOSE — Problem is ATOMIC, skipping decomposition")
            state["is_decomposed"] = False
            state["subtasks"] = []
            return state

        subtasks = self._parse_subtasks(response, best_thought)

        if len(subtasks) < 2:
            if self.verbose:
                print("\n🔬 DECOMPOSE — Could not parse sub-tasks, treating as ATOMIC")
            state["is_decomposed"] = False
            state["subtasks"] = []
            return state

        if self.verbose:
            print(f"\n🔬 DECOMPOSE — Split into {len(subtasks)} sub-tasks:")
            for st in subtasks:
                deps = ", ".join(st.depends_on) if st.depends_on else "none"
                print(f"   {st.id}: {st.description[:80]}  [deps: {deps}]")

        state["is_decomposed"] = True
        state["subtasks"] = [st.to_dict() for st in subtasks]
        state["subtask_results"] = {}
        return state

    def _parse_subtasks(self, response: str, parent_context: str) -> List[SubTask]:
        """Parse LLM response into SubTask objects."""
        subtasks: List[SubTask] = []
        pattern = re.compile(
            r"SUB_(\d+)\s*:\s*(.+?)\s*\|\s*DEPENDS\s*:\s*(.+?)\s*\|\s*CONTEXT\s*:\s*(.+)",
            re.IGNORECASE,
        )
        for line in response.strip().splitlines():
            m = pattern.match(line.strip())
            if not m:
                continue
            idx = m.group(1)
            desc = m.group(2).strip()
            deps_raw = m.group(3).strip().lower()
            ctx = m.group(4).strip()

            deps: List[str] = []
            if deps_raw and deps_raw != "none":
                deps = [d.strip() for d in re.split(r"[,;]", deps_raw) if d.strip()]

            subtasks.append(SubTask(
                id=f"sub_{idx}",
                description=desc,
                depends_on=deps,
                context=f"{ctx}\n\nParent context: {parent_context[:300]}",
            ))

        # Cap at 4
        return subtasks[:4]


class SubAgentRunner:
    """Solves individual sub-tasks with focused, single-call LLM invocations."""

    MAX_ATTEMPTS = 2
    MIN_SCORE = 0.4

    def __init__(self, llm: LLMClient, memory: ReflexionMemory, verbose: bool = True):
        self.llm = llm
        self.memory = memory
        self.verbose = verbose

    def solve_subtask(
        self,
        subtask: SubTask,
        parent_context: str,
        sibling_results: Dict[str, str],
    ) -> SubTask:
        """Solve a single sub-task (with up to one retry)."""
        dep_context = ""
        for dep_id in subtask.depends_on:
            if dep_id in sibling_results:
                dep_context += f"\nResult of {dep_id}: {sibling_results[dep_id][:400]}\n"

        memory_summary = self.memory.get_summary() if hasattr(self.memory, "get_summary") else ""

        for attempt in range(self.MAX_ATTEMPTS):
            subtask.attempts = attempt + 1
            retry_note = ""
            if attempt > 0:
                retry_note = (
                    f"\nPREVIOUS ATTEMPT FAILED (score {subtask.score:.2f}). "
                    f"Previous answer:\n{subtask.result[:300]}\n"
                    "Try a different approach.\n"
                )

            prompt = f"""Solve this sub-task completely and concisely.

Sub-task: {subtask.description}
Context: {subtask.context[:400]}
{dep_context}
Memory insights: {memory_summary[:200]}
{retry_note}
Provide a clear, self-contained solution.
"""
            response = self.llm._call_api(prompt)
            score_res = self.llm.score_node(response)
            score = float(score_res.get("score", 0.5))

            subtask.result = response
            subtask.score = score

            if self.verbose:
                status = "✓" if score >= self.MIN_SCORE else "✗"
                print(f"   {status} {subtask.id} attempt {attempt + 1}: score={score:.2f}")

            if score >= self.MIN_SCORE:
                subtask.status = "completed"
                return subtask

        # Exhausted retries
        subtask.status = "failed" if subtask.score < self.MIN_SCORE else "completed"
        return subtask


class ResultAggregatorTool:
    """Aggregates sub-task results into a coherent answer."""

    def __init__(self, llm: LLMClient, verbose: bool = True):
        self.llm = llm
        self.verbose = verbose

    def invoke(self, state: "AgentState") -> "AgentState":
        subtasks = state.get("subtasks", [])
        subtask_results = state.get("subtask_results", {})
        problem = state.get("problem", "")

        if not subtasks:
            return state

        results_block = ""
        failed_ids: List[str] = []
        for st in subtasks:
            sid = st["id"]
            result = subtask_results.get(sid, st.get("result", ""))
            status = st.get("status", "pending")
            results_block += f"\n--- {sid} ({status}) ---\n{result[:600]}\n"
            if status == "failed":
                failed_ids.append(sid)

        gap_note = ""
        if failed_ids:
            gap_note = (
                f"\nNote: Sub-tasks {', '.join(failed_ids)} failed or scored low. "
                "Fill in the gaps using your reasoning."
            )

        prompt = f"""Combine these sub-task results into a single coherent answer.

Problem: {problem}

Sub-task results:
{results_block}
{gap_note}
Produce a complete, well-structured final answer. Show your work clearly.
"""
        aggregated = self.llm._call_api(prompt)

        if self.verbose:
            n_ok = sum(1 for st in subtasks if st.get("status") == "completed")
            print(f"\n🔗 AGGREGATE — Combined {n_ok}/{len(subtasks)} sub-task results")

        state["aggregated_result"] = aggregated
        state["current_thought"] = aggregated
        return state
