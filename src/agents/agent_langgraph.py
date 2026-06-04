"""ReTreVal v2+ — ReAct Agent with LLM-Driven Tool Selection.

The outer loop (tree expansion, critique, scoring, memory) stays
infrastructure-driven.  The inner loop (what happens during refinement)
is a ReAct loop where the LLM chooses which tools to invoke.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langgraph.graph import StateGraph, START, END

from src.utils.memory import Memory
from src.utils.kv_cache_manager import KVCacheManager
from src.clients.llm_client import LLMClient
from src.agents.state import AgentState
from src.agents.tool_augmented_executor import ToolAugmentedExecutor

from src.tools.registry import ToolSpec, ToolRegistry
from src.tools.feedback import ValidationTool, FailureAnalyzer, FeedbackLoopController, BacktrackingTool
from src.tools.search import WebSearchTool, SearchTool, WikipediaLookupTool, ArxivSearchTool
from src.tools.planning import PlannerTool
from src.tools.refinement import RefinerTool, CritiqueTool
from src.tools.scoring import ScoringTool, MemoryTool, ComplexityTool
from src.tools.computation import CalculatorTool, LinearProgrammingTool, PythonREPLTool, CombinatoricsTool, PythonExecTool
from src.tools.synthesis import SynthesizerTool
from src.tools.decomposition import TaskDecomposerTool, SubAgentRunner, ResultAggregatorTool, SubTask, topological_sort


# ----------------------------------------------------------------------------
# Answer extraction / comparison helpers (used by the eval harness, kept at
# module level so callers can `from src.agents.agent_langgraph import ...`).
# ----------------------------------------------------------------------------
def extract_answer(text: str) -> str:
    """Extract the final answer from a model response."""
    if not text:
        return ""
    boxed = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)
    if boxed:
        return boxed[-1].strip()
    patterns = [
        r"(?:final answer|answer)\s*(?:is|:)\s*([^\n]+)",
        r"therefore,?\s*([^\n]+)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            return matches[-1].strip().strip(".")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1].strip().strip(".") if lines else text.strip()


def normalize_answer(value: str) -> str:
    """Normalize an answer for comparison (strip LaTeX/whitespace/wrappers)."""
    value = extract_answer(value or "")
    value = value.strip()
    value = re.sub(r"\\boxed\{(.+)\}", r"\1", value)
    value = value.replace("\\left", "").replace("\\right", "")
    value = value.replace("\\,", "")
    value = value.replace("$", "")
    value = re.sub(r"\s+", "", value)
    value = value.strip("{}().,")
    return value.lower()


def answers_match(predicted: str, expected: str) -> bool:
    """Compare a prediction against an expected answer after normalization."""
    return normalize_answer(predicted) == normalize_answer(expected)


class LangGraphAgent:
    """ReTreVal agent: ReAct inside Tree-of-Thoughts."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        max_loops: int = 2,
        verbose: bool = True,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        memory_file: Optional[str] = None,
        use_memory: bool = True,
        oracle: bool = False,
    ):
        load_dotenv()

        # Provider/model selection — LLMClient reads provider-specific model env vars.
        if model:
            provider_name = (provider or os.getenv("LLM_PROVIDER", "gemini")).lower()
            model_env = {
                "gemini": "GEMINI_MODEL", "openai": "OPENAI_MODEL",
                "ollama": "OLLAMA_MODEL", "vllm": "VLLM_MODEL",
            }.get(provider_name)
            if model_env:
                os.environ[model_env] = model

        # oracle=False (default) → the reference answer is withheld from the agent;
        # validation/backtracking run on the agent's own score instead.
        self.oracle = bool(oracle)
        self.use_memory = bool(use_memory)

        # Persistent cross-problem memory. Stays integral to the tools (never None);
        # use_memory only controls whether prior memory is loaded / saved.
        if memory_file is None:
            memory_file = os.getenv("MEMORY_FILE") or None
        self.memory = Memory(memory_file=memory_file, load_on_init=use_memory)
        self.llm = LLMClient(provider=provider)
        self.max_loops = max(1, int(max_loops))
        self.verbose = bool(verbose)

        # KV Cache Manager — structures prompts for vLLM prefix caching
        self.cache_mgr = KVCacheManager(memory=self.memory, verbose=self.verbose)

        # Attach cache manager to vLLM backend if available
        backend = getattr(self.llm, 'backend', None)
        if hasattr(backend, 'set_cache_manager'):
            backend.set_cache_manager(self.cache_mgr)
            if self.verbose:
                print("✓ KV Cache Manager wired to LLM backend")

        # Instantiate tools once
        self.calc = CalculatorTool(verbose=self.verbose)
        self.python_repl = PythonREPLTool(verbose=self.verbose)
        self.combo = CombinatoricsTool(verbose=self.verbose)
        self.lp = LinearProgrammingTool(self.llm, verbose=self.verbose)
        self.web_search = WebSearchTool(verbose=self.verbose)
        self.refiner = RefinerTool(self.llm, self.memory, verbose=self.verbose)
        self.decomposer = TaskDecomposerTool(self.llm, self.memory, verbose=self.verbose)
        self.python_exec = PythonExecTool(verbose=self.verbose)
        self.wikipedia = WikipediaLookupTool(verbose=self.verbose)
        self.arxiv = ArxivSearchTool(verbose=self.verbose)

        self._build_graph()

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------
    def _build_graph(self):
        sg = StateGraph(AgentState)

        planner = PlannerTool(self.llm, self.memory, verbose=self.verbose, cache_mgr=self.cache_mgr)
        search = SearchTool(self.llm, self.memory, verbose=self.verbose)
        critic = CritiqueTool(self.llm, verbose=self.verbose)
        scorer = ScoringTool(self.llm, self.memory, verbose=self.verbose)
        memo = MemoryTool(self.llm, self.memory, verbose=self.verbose, cache_mgr=self.cache_mgr)
        complexifier = ComplexityTool(verbose=self.verbose)
        synth = SynthesizerTool(self.llm, verbose=self.verbose)

        # Feedback loop tools
        validator = ValidationTool(verbose=self.verbose)
        failure_analyzer = FailureAnalyzer(self.llm, self.memory, verbose=self.verbose)
        feedback_controller = FeedbackLoopController(memory=self.memory, verbose=self.verbose)
        backtracker = BacktrackingTool(verbose=self.verbose)

        # Decomposition tools
        decomposer = TaskDecomposerTool(self.llm, self.memory, verbose=self.verbose)
        sub_runner = SubAgentRunner(self.llm, self.memory, verbose=self.verbose)
        aggregator = ResultAggregatorTool(self.llm, verbose=self.verbose)

        # ----- Nodes -----
        sg.add_node('plan', planner.invoke)
        sg.add_node('expand_tree', search.invoke)

        # ReAct refinement replaces the fixed refine -> calculate -> math_compute -> lp_solve chain
        sg.add_node('react_refine', lambda state: self._react_refine_node(state))

        sg.add_node('critique', critic.invoke)
        sg.add_node('score', scorer.invoke)
        sg.add_node('memory', memo.invoke)
        sg.add_node('complexity', complexifier.invoke)
        sg.add_node('synthesize', synth.invoke)
        sg.add_node('validate', validator.invoke)
        sg.add_node('analyze_failure', failure_analyzer.invoke)
        sg.add_node('feedback_control', feedback_controller.invoke)
        sg.add_node('backtrack', backtracker.invoke)

        # Backtrack refine node: uses failure context for better refinement
        sg.add_node('backtrack_refine', lambda state: self._backtrack_refine_node(state))

        sg.add_node('decompose', decomposer.invoke)
        sg.add_node('run_subtasks', lambda state: self._run_subtasks_node(state, sub_runner, aggregator))

        # ----- Edges -----
        sg.add_edge(START, 'plan')
        sg.add_edge('plan', 'expand_tree')
        sg.add_edge('expand_tree', 'react_refine')
        sg.add_edge('react_refine', 'critique')
        sg.add_edge('critique', 'score')
        sg.add_edge('score', 'memory')

        # Loop a couple of times then decide
        def loop_or_finish(state: AgentState) -> str:
            if state.get('iterations', 0) >= self.max_loops:
                complexity = state.get('complexity', 1)
                best_score = state.get('best_score', 0.0)
                is_subtask = state.get('is_subtask', False)
                if complexity >= 4 and best_score < 0.7 and not is_subtask:
                    return 'decompose'
                return 'synthesize'
            return 'complexity'

        sg.add_conditional_edges('memory', loop_or_finish, {
            'complexity': 'complexity',
            'synthesize': 'synthesize',
            'decompose': 'decompose',
        })
        sg.add_edge('complexity', 'expand_tree')
        sg.add_edge('decompose', 'run_subtasks')
        sg.add_edge('run_subtasks', 'synthesize')
        sg.add_edge('synthesize', 'validate')

        # Feedback loop edges
        sg.add_edge('validate', 'feedback_control')

        def should_retry_feedback(state: AgentState) -> str:
            is_correct = state.get('is_correct', False)
            should_retry = state.get('should_retry', False)
            if is_correct:
                return 'END'
            elif should_retry:
                return 'backtrack'
            else:
                return 'END'

        sg.add_conditional_edges('feedback_control', should_retry_feedback, {
            'backtrack': 'backtrack',
            'END': END,
        })

        def after_backtrack(state: AgentState) -> str:
            if state.get('backtrack_available', False):
                return 'backtrack_refine'
            return 'analyze_failure'

        sg.add_conditional_edges('backtrack', after_backtrack, {
            'backtrack_refine': 'backtrack_refine',
            'analyze_failure': 'analyze_failure',
        })

        sg.add_edge('backtrack_refine', 'critique')
        sg.add_edge('analyze_failure', 'expand_tree')

        self.app = sg.compile()

    # ------------------------------------------------------------------
    # ReAct refinement node (replaces fixed refine->calc->math->lp chain)
    # ------------------------------------------------------------------
    def _react_refine_node(self, state: AgentState) -> AgentState:
        """Run a ReAct loop where the LLM selects tools to refine the current thought."""
        if self.verbose:
            print("\n\U0001f916 REACT REFINE — LLM selects tools")

        registry = self._build_tool_registry(state)
        executor = ToolAugmentedExecutor(
            self.llm, registry,
            max_steps=int(os.getenv('REACT_MAX_STEPS', '10')),
            verbose=self.verbose,
            cache_mgr=self.cache_mgr,
        )

        improved_thought, trace = executor.run(
            problem=state.get('problem', ''),
            plan=state.get('plan', ''),
            current_thought=state.get('current_thought', ''),
            memory_summary=state.get('memory_summary', ''),
        )

        # If the ReAct loop produced a FINISH with a \boxed{} answer,
        # preserve both the reasoning and the boxed answer in current_thought.
        # This ensures best_thought (used by synthesis) contains the answer.
        finish_answer = ''
        for step in reversed(trace):
            if step.get('action', '').startswith('FINISH'):
                finish_answer = step.get('input', '')
                break

        if finish_answer and '\\boxed' in finish_answer:
            # The FINISH answer has boxed format — use it directly
            state['current_thought'] = finish_answer
        elif finish_answer:
            # FINISH answer exists but no \boxed — append it to reasoning
            state['current_thought'] = improved_thought + '\n\n' + finish_answer
        else:
            state['current_thought'] = improved_thought

        # Track traces
        state.setdefault('react_traces', []).append(trace)

        # Track which tools were used
        tools = []
        for step in trace:
            action = step.get('action', '')
            if action and not action.startswith('FINISH'):
                tool_name = action.split('(')[0].strip()
                if tool_name:
                    tools.append(tool_name)
        state['tools_used'] = state.get('tools_used', []) + tools

        # Update tree node
        nid = state.get('current_node_id')
        nodes_map = state.get('tree', {}).get('nodes', {})
        if nid and nid in nodes_map:
            nodes_map[nid]['thought'] = state['current_thought']
            nodes_map[nid]['refinement_count'] = int(nodes_map[nid].get('refinement_count', 0)) + 1

        if self.verbose:
            unique_tools = sorted(set(tools)) if tools else ['(none)']
            print(f"   Tools used: {', '.join(unique_tools)}")
            print(f"   Steps: {len(trace)}")

        return state

    # ------------------------------------------------------------------
    # Build tool registry (binds tools to current state)
    # ------------------------------------------------------------------
    def _build_tool_registry(self, state: AgentState) -> ToolRegistry:
        registry = ToolRegistry()

        registry.register(ToolSpec(
            name="calculator",
            description="Evaluate arithmetic expressions safely. Returns the numerical result.",
            parameters='a mathematical expression, e.g., "(8 + 3) * 2 - 1"',
            fn=self.calc.call,
        ))
        registry.register(ToolSpec(
            name="equation_solver",
            description="Solve algebraic equations or systems of equations using SymPy. Returns solutions.",
            parameters='one equation (e.g., "2*x + 3 = 7") or a system separated by commas (e.g., "x + y = 5, x - y = 1")',
            fn=self.python_repl.call,
        ))
        registry.register(ToolSpec(
            name="combo",
            description="Combinatorics and exact-fraction calculator. Returns exact fractions, not decimals. Use for probability, counting, permutations, combinations.",
            parameters='an expression, e.g., "C(7,4) * (1/5)**4 * (4/5)**3" or "35 * (1/5)**4 * (4/5)**3"',
            fn=self.combo.call,
        ))
        registry.register(ToolSpec(
            name="lp_solver",
            description="Formulate and solve linear programming/optimization problems. Returns optimal solution.",
            parameters="natural language description of the optimization problem with objective and constraints",
            fn=lambda inp: self.lp.call(inp, state),
        ))
        registry.register(ToolSpec(
            name="refine",
            description="Improve the current reasoning by addressing a specific weakness or critique.",
            parameters="description of what to improve or the critique to address",
            fn=lambda inp: self.refiner.call(inp, state),
        ))
        registry.register(ToolSpec(
            name="decompose",
            description="Break a complex problem into 2-4 independent sub-tasks, solve each, and combine.",
            parameters="the problem or sub-problem to decompose",
            fn=lambda inp: self.decomposer.call(inp, state),
        ))
        registry.register(ToolSpec(
            name="web_search",
            description="Search the web for factual information, definitions, theorems, or domain knowledge. Use this when the problem requires specific facts you are unsure about.",
            parameters='a search query, e.g., "Arrhenius impossibility theorem conditions"',
            fn=self.web_search.call,
        ))
        registry.register(ToolSpec(
            name="python_exec",
            description="Execute arbitrary Python code and return the output. Use for complex computations, matrix operations, number theory, string processing, graph algorithms, or anything that doesn't fit calculator/equation_solver. Available libraries: math, sympy, numpy, itertools, collections, re, fractions. Always use print() to output results.",
            parameters='Python code as a string, e.g., "import sympy; print(sympy.isprime(997))"',
            fn=self.python_exec.call,
        ))
        registry.register(ToolSpec(
            name="wikipedia",
            description="Look up a Wikipedia article by title or search term. Use for facts about people, places, concepts, historical events, scientific theories, laws, philosophical concepts, or any established knowledge. Best for humanities, social science, biology, and general knowledge questions.",
            parameters='article title or search term, e.g., "Arrhenius impossibility theorem" or "CRISPR gene editing"',
            fn=self.wikipedia.call,
        ))
        registry.register(ToolSpec(
            name="arxiv_search",
            description="Search arxiv.org for academic papers by topic, title, or author. Use for questions about specific research papers, algorithms, ML models, theorems, or recent scientific results. Returns titles, authors, abstracts, and links.",
            parameters='search query, e.g., "GELU activation function" or "attention is all you need"',
            fn=self.arxiv.call,
        ))

        return registry

    # ------------------------------------------------------------------
    # Backtrack refine (uses failure context + ReAct)
    # ------------------------------------------------------------------
    def _backtrack_refine_node(self, state: AgentState) -> AgentState:
        if self.verbose:
            print("\n\U0001f504 BACKTRACK REFINE — Using failure context + ReAct")

        failure_contexts = state.get('failure_contexts', [])
        if failure_contexts:
            last_failure = failure_contexts[-1]
            failure_summary = (
                f"\nPREVIOUS ATTEMPT FAILED:\n"
                f"- Failure Type: {last_failure.get('failure_type', 'unknown')}\n"
                f"- Failed Answer: {last_failure.get('failed_answer', '')}\n"
                f"- Error Analysis: {last_failure.get('error_analysis', '')[:300]}\n"
                f"- Failed Approach: {last_failure.get('failure_thought', '')[:200]}\n"
                f"\nLEARN FROM THIS: Avoid the same mistake.\n"
            )
            state.setdefault('critique_history', []).append(failure_summary)
            if self.verbose:
                print(f"   Added failure context from {len(failure_contexts)} previous attempt(s)")

        # Run the ReAct refine node (with failure context now in critique_history)
        return self._react_refine_node(state)

    # ------------------------------------------------------------------
    # Run subtasks (same as before)
    # ------------------------------------------------------------------
    def _run_subtasks_node(
        self,
        state: AgentState,
        sub_runner: SubAgentRunner,
        aggregator: ResultAggregatorTool,
    ) -> AgentState:
        if not state.get('is_decomposed', False):
            return state

        raw_subtasks = state.get('subtasks', [])
        if not raw_subtasks:
            return state

        subtasks = [
            SubTask(
                id=st['id'],
                description=st['description'],
                depends_on=st.get('depends_on', []),
                context=st.get('context', ''),
                status=st.get('status', 'pending'),
                result=st.get('result', ''),
                score=st.get('score', 0.0),
                attempts=st.get('attempts', 0),
            )
            for st in raw_subtasks
        ]

        ordered = topological_sort(subtasks)
        parent_context = state.get('best_thought', state.get('current_thought', ''))
        results: Dict[str, str] = {}

        if self.verbose:
            print(f"\n\U0001f3c3 RUN SUB-TASKS — Executing {len(ordered)} sub-tasks")

        for st in ordered:
            st = sub_runner.solve_subtask(st, parent_context, results)
            results[st.id] = st.result

            if st.score >= 0.6:
                self.memory.add_insight(
                    f"Sub-task '{st.description[:80]}' solved (score={st.score:.2f}): {st.result[:200]}"
                )
            elif st.score < 0.4:
                self.memory.add_failure(
                    f"Sub-task '{st.description[:80]}' failed (score={st.score:.2f}): {st.result[:200]}"
                )

        state['subtask_results'] = results
        state['subtasks'] = [st.to_dict() for st in ordered]

        return aggregator.invoke(state)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    def run(self, problem: str, expected_answer: str = None, problem_id: str = None) -> AgentState:
        # Initialize KV cache for this problem
        self.cache_mgr.set_problem_context(problem)

        state: AgentState = {
            'problem': problem,
            'critique_history': [],
            'iterations': 0,
            'max_depth': int(os.getenv('MAX_DEPTH', '5')),  # Cap; planner sets adaptive value
            'children_per_expansion': int(os.getenv('CHILDREN_PER_EXPANSION', '3')),
            'problem_id': problem_id or '',
            # The reference answer only enters the agent when oracle=True; otherwise
            # ValidationTool gates retry/backtracking on the agent's own score.
            'expected_answer': (expected_answer or '') if self.oracle else '',
            'is_correct': False,
            'predicted_answer': '',
            'error_analysis': '',
            'failure_type': '',
            'approach_type': '',
            'feedback_loops': 0,
            'should_retry': False,
            # Backtracking fields
            'failed_nodes': [],
            'backtrack_history': [],
            'failure_contexts': [],
            'explored_root_children': [],
            'backtrack_available': False,
            'backtracking_iteration': 0,
            # Decomposition fields
            'subtasks': [],
            'subtask_results': {},
            'is_decomposed': False,
            'is_subtask': False,
            'aggregated_result': '',
            # ReAct fields
            'react_traces': [],
            'tools_used': [],
        }
        default_limit = 100 + 12 * int(self.max_loops)
        recursion_limit = int(os.getenv('RECURSION_LIMIT', str(default_limit)))
        result = self.app.invoke(state, config={"recursion_limit": recursion_limit})

        if self.use_memory:
            self.memory.save()

        # Log KV cache stats
        self.cache_mgr.log_stats()

        return result


def run_agent(
    problem: str,
    expected_answer: str = None,
    problem_id: str = None,
    iterations: int = 2,
    verbose: bool = True,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    memory_file: Optional[str] = None,
    use_memory: bool = True,
    oracle: bool = False,
) -> Dict[str, Any]:
    """Convenience function: create agent, run problem, return results dict.

    oracle=False (default): the `expected_answer` is not shown to the agent during
    solving; the caller scores the prediction afterwards. Pass oracle=True to let
    the agent use the reference answer during solving.
    """
    agent = LangGraphAgent(
        max_loops=iterations, verbose=verbose, provider=provider, model=model,
        memory_file=memory_file, use_memory=use_memory, oracle=oracle,
    )
    final_state = agent.run(problem, expected_answer=expected_answer, problem_id=problem_id)
    return {
        'problem': problem,
        'problem_id': final_state.get('problem_id', ''),
        'expected_answer': final_state.get('expected_answer', ''),
        'predicted_answer': final_state.get('predicted_answer', final_state.get('final_output', '')),
        'is_correct': final_state.get('is_correct', False),
        'plan': final_state.get('plan', ''),
        'best_thought': final_state.get('best_thought', final_state.get('current_thought', '')),
        'best_node_id': final_state.get('best_node_id', ''),
        'score': final_state.get('best_score', final_state.get('score', 0.0)),
        'final_output': final_state.get('final_output', ''),
        'final_outputs': final_state.get('final_outputs', []),
        'feasible': final_state.get('feasible', True),
        'feasibility_reasons': final_state.get('feasibility_reasons', ''),
        'error_analysis': final_state.get('error_analysis', ''),
        'failure_type': final_state.get('failure_type', ''),
        'approach_type': final_state.get('approach_type', ''),
        'feedback_loops': final_state.get('feedback_loops', 0),
        'memory_summary': agent.memory.to_dict(),
        'tree': final_state.get('tree', {}),
        'lp_solution': final_state.get('lp_solution', None),
        # Backtracking info
        'backtrack_history': final_state.get('backtrack_history', []),
        'failed_nodes': final_state.get('failed_nodes', []),
        'total_backtracks': len(final_state.get('backtrack_history', [])),
        # Decomposition info
        'is_decomposed': final_state.get('is_decomposed', False),
        'subtask_results': final_state.get('subtask_results', {}),
        # ReAct info
        'react_traces': final_state.get('react_traces', []),
        'trace': final_state.get('react_traces', []),
        'tools_used': final_state.get('tools_used', []),
    }
