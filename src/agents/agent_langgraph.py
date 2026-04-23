"""Minimal ReTreVal agent built on LangGraph.

The public OSS path keeps one small agent surface:

1. draft a step-by-step solution,
2. optionally refine it,
3. extract a final answer.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.clients.llm_client import LLMClient


class AgentState(TypedDict, total=False):
    problem: str
    expected_answer: str
    problem_id: str
    draft: str
    refinement: str
    final_output: str
    predicted_answer: str
    iterations: int
    max_iterations: int
    trace: List[Dict[str, str]]


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


class LangGraphAgent:
    """Small math-oriented reasoning agent."""

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        max_iterations: int = 1,
        verbose: bool = False,
    ) -> None:
        load_dotenv()
        if model:
            # LLMClient reads provider-specific model env vars.
            import os

            provider_name = (provider or os.getenv("LLM_PROVIDER", "gemini")).lower()
            if provider_name == "gemini":
                os.environ["GEMINI_MODEL"] = model
            elif provider_name == "openai":
                os.environ["OPENAI_MODEL"] = model
            elif provider_name == "ollama":
                os.environ["OLLAMA_MODEL"] = model
            elif provider_name == "vllm":
                os.environ["VLLM_MODEL"] = model

        self.llm = LLMClient(provider=provider)
        self.max_iterations = max(0, int(max_iterations))
        self.verbose = verbose
        self.app = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("draft", self._draft)
        graph.add_node("refine", self._refine)
        graph.add_node("finalize", self._finalize)

        graph.add_edge(START, "draft")

        def next_step(state: AgentState) -> str:
            if state.get("iterations", 0) < state.get("max_iterations", 0):
                return "refine"
            return "finalize"

        graph.add_conditional_edges("draft", next_step, {"refine": "refine", "finalize": "finalize"})
        graph.add_conditional_edges("refine", next_step, {"refine": "refine", "finalize": "finalize"})
        graph.add_edge("finalize", END)
        return graph.compile()

    def _call_llm(self, prompt: str, max_tokens: int = 2048) -> str:
        return self.llm._call_api(prompt, max_tokens=max_tokens)

    def _draft(self, state: AgentState) -> AgentState:
        problem = state["problem"]
        prompt = f"""Solve this math problem step by step.

Problem:
{problem}

Return a concise solution and put the final answer in \\boxed{{...}}."""
        draft = self._call_llm(prompt)
        state["draft"] = draft
        state["final_output"] = draft
        state["trace"] = state.get("trace", []) + [{"step": "draft", "output": draft}]
        if self.verbose:
            print(draft)
        return state

    def _refine(self, state: AgentState) -> AgentState:
        problem = state["problem"]
        current = state.get("final_output", state.get("draft", ""))
        prompt = f"""Check the solution for mathematical errors. If it is correct, keep it concise.
If it is wrong, fix it. Always end with the final answer in \\boxed{{...}}.

Problem:
{problem}

Current solution:
{current}"""
        refined = self._call_llm(prompt)
        state["iterations"] = state.get("iterations", 0) + 1
        state["refinement"] = refined
        state["final_output"] = refined
        state["trace"] = state.get("trace", []) + [{"step": "refine", "output": refined}]
        if self.verbose:
            print(refined)
        return state

    def _finalize(self, state: AgentState) -> AgentState:
        final_output = state.get("final_output", "")
        state["predicted_answer"] = extract_answer(final_output)
        state["trace"] = state.get("trace", []) + [
            {"step": "finalize", "output": state["predicted_answer"]}
        ]
        return state

    def run(
        self,
        problem: str,
        expected_answer: Optional[str] = None,
        problem_id: Optional[str] = None,
    ) -> AgentState:
        state: AgentState = {
            "problem": problem,
            "expected_answer": expected_answer or "",
            "problem_id": problem_id or "",
            "iterations": 0,
            "max_iterations": self.max_iterations,
            "trace": [],
        }
        return self.app.invoke(state)


def run_agent(
    problem: str,
    expected_answer: Optional[str] = None,
    problem_id: Optional[str] = None,
    iterations: int = 1,
    verbose: bool = False,
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the simplified ReTreVal agent and return a serializable result."""
    agent = LangGraphAgent(
        provider=provider,
        model=model,
        max_iterations=iterations,
        verbose=verbose,
    )
    result = agent.run(problem, expected_answer=expected_answer, problem_id=problem_id)
    return dict(result)
