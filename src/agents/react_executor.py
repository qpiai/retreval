"""ReAct Executor — LLM-driven tool selection loop.

Implements the ReAct (Reasoning + Acting) pattern where the LLM sees tool
descriptions and chooses which to invoke, exactly as in the original ReAct paper.
The LLM output is parsed for THOUGHT + ACTION text; no native function-calling is used.
"""

import re
from typing import Dict, List, Tuple

from src.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Output parser
# ---------------------------------------------------------------------------

_ACTION_PATTERN = re.compile(
    r'ACTION\s*:\s*(\w+)\s*\(\s*["\']?(.*?)["\']?\s*\)\s*$',
    re.IGNORECASE | re.MULTILINE | re.DOTALL,
)


def _strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks from thinking models (e.g., Qwen, DeepSeek-R1)."""
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()


def parse_react_output(text: str) -> Tuple[str, str, str]:
    """Parse LLM output into (thought, tool_name, tool_input).

    Falls back to FINISH with the entire text as the answer if parsing fails.
    """
    text = _strip_think_tags(text)

    thought = ""
    tool_name = "FINISH"
    tool_input = text.strip()

    # Extract THOUGHT block
    thought_match = re.search(
        r'THOUGHT\s*:\s*(.*?)(?=ACTION\s*:|$)', text, re.IGNORECASE | re.DOTALL
    )
    if thought_match:
        thought = thought_match.group(1).strip()

    # Extract ACTION block
    action_match = _ACTION_PATTERN.search(text)
    if action_match:
        tool_name = action_match.group(1).strip()
        tool_input = action_match.group(2).strip()
    else:
        # Try a more lenient pattern: ACTION: tool_name(anything)
        lenient = re.search(
            r'ACTION\s*:\s*(\w+)\s*\(([\s\S]*?)\)\s*$',
            text, re.IGNORECASE | re.MULTILINE,
        )
        if lenient:
            tool_name = lenient.group(1).strip()
            tool_input = lenient.group(2).strip().strip("\"'")

    return thought, tool_name, tool_input


# ---------------------------------------------------------------------------
# ReAct prompt template — used ONLY as fallback when no cache_mgr is present
# ---------------------------------------------------------------------------

_REACT_PROMPT = """\
You are solving a problem step-by-step using available tools.

PROBLEM: {problem}
PLAN: {plan}
CURRENT REASONING: {current_thought}
{memory_block}
AVAILABLE TOOLS:
{tool_list}

PREVIOUS STEPS:
{history}

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
- Put your final answer in \\boxed{{answer}} format.
"""


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

class ReactExecutor:
    """Execute a ReAct loop: LLM chooses tools, agent dispatches them."""

    def __init__(
        self,
        llm,  # LLMClient (has ._call_api)
        registry: ToolRegistry,
        max_steps: int = 5,
        verbose: bool = True,
        cache_mgr=None,  # KVCacheManager for prefix reuse
    ):
        self.llm = llm
        self.registry = registry
        self.max_steps = max_steps
        self.verbose = verbose
        self.cache_mgr = cache_mgr

    # ----- public API -----

    def run(
        self,
        problem: str,
        plan: str,
        current_thought: str,
        memory_summary: str,
    ) -> Tuple[str, List[Dict]]:
        """Execute the ReAct loop.

        Returns:
            (final_thought, trace)  where trace is a list of step dicts.
        """
        history: List[Dict] = []
        last_thought = current_thought
        all_calls: List[Tuple[str, str]] = []  # (tool, input_prefix) for loop detection
        error_count = 0  # consecutive errors
        tool_calls = 0  # only count actual tool calls, not FINISH

        # No hard cap on tool calls — let the LLM use as many as it needs.
        # Safety comes from loop detection + absolute ceiling.
        max_total = self.max_steps + 2  # absolute ceiling on LLM calls

        for step_idx in range(max_total):
            prompt, system_prompt = self._build_prompt(
                problem, plan, last_thought, memory_summary, history
            )
            raw = _strip_think_tags(
                self._call_llm(prompt, system_prompt=system_prompt) or ""
            )

            thought, tool_name, tool_input = parse_react_output(raw)
            if not thought:
                thought = last_thought

            if self.verbose:
                print(f"\n  [ReAct step {step_idx + 1}]")
                print(f"    THOUGHT: {thought[:200]}")
                print(f"    ACTION:  {tool_name}({tool_input[:120]})")

            # FINISH
            if tool_name.upper() == "FINISH":
                final = tool_input if tool_input else thought
                history.append({
                    "step": step_idx + 1,
                    "thought": thought,
                    "action": "FINISH",
                    "input": tool_input,
                    "observation": "",
                })
                return final, history

            # --- Loop detection (multi-strategy) ---
            call_sig = (tool_name.lower(), tool_input.lower()[:80])
            all_calls.append(call_sig)

            force_finish = False

            # Strategy 1: exact duplicate — same tool+input 2 times in a row
            if len(all_calls) >= 2 and all_calls[-1] == all_calls[-2]:
                if self.verbose:
                    print("    Loop detected (identical repeat) — forcing FINISH")
                force_finish = True

            # Strategy 2: same tool called with error 3+ times total (not necessarily consecutive)
            if not force_finish and error_count >= 3:
                if self.verbose:
                    print("    Loop detected (3 consecutive errors) — forcing FINISH")
                force_finish = True

            if force_finish:
                loop_answer = thought
                for prev in reversed(history):
                    prev_input = prev.get('input', '')
                    if '\\boxed' in prev_input:
                        loop_answer = prev_input
                        break
                    prev_obs = prev.get('observation', '')
                    if prev_obs and prev_obs.strip() and not prev_obs.startswith('Error'):
                        loop_answer = prev_obs
                        break
                history.append({
                    "step": step_idx + 1,
                    "thought": thought,
                    "action": "FINISH (forced — loop detected)",
                    "input": loop_answer,
                    "observation": "",
                })
                return loop_answer, history

            # Dispatch tool
            observation = self._dispatch(tool_name, tool_input)
            tool_calls += 1

            # Track consecutive errors for loop detection
            if observation.startswith('Error') or observation.startswith('Could not'):
                error_count += 1
            else:
                error_count = 0  # reset on success

            if self.verbose:
                print(f"    OBSERVATION: {observation[:200]}")

            history.append({
                "step": step_idx + 1,
                "thought": thought,
                "action": f"{tool_name}({tool_input[:200]})",
                "input": tool_input,
                "observation": observation,
            })
            last_thought = thought

        # Absolute ceiling reached — force one final FINISH call
        if self.verbose:
            print(f"    Max steps ({max_total}) reached — requesting FINISH")
        prompt, system_prompt = self._build_prompt(
            problem, plan, last_thought, memory_summary, history
        )
        prompt += "\n\nYou have used all available tool calls. You MUST now use FINISH to provide your final answer.\nTHOUGHT: <summarize>\nACTION: FINISH(\"\\boxed{your_answer}\")"
        raw = _strip_think_tags(
            self._call_llm(prompt, system_prompt=system_prompt) or ""
        )
        thought, tool_name, tool_input = parse_react_output(raw)
        if not thought:
            thought = last_thought
        final = tool_input if tool_input else thought
        history.append({
            "step": max_total + 1,
            "thought": thought,
            "action": "FINISH",
            "input": tool_input,
            "observation": "",
        })
        return final, history

    # ----- internals -----

    def _call_llm(self, prompt: str, system_prompt: str = None) -> str:
        """Call the LLM, routing through the backend's system_prompt support."""
        backend = getattr(self.llm, 'backend', self.llm)
        call_fn = getattr(backend, '_call_api', None)
        if call_fn is None:
            call_fn = self.llm._call_api

        # If the backend supports system_prompt kwarg (vllm_client), use it
        if system_prompt:
            try:
                return call_fn(prompt, system_prompt=system_prompt)
            except TypeError:
                # Backend doesn't support system_prompt — concat and fall back
                return call_fn(f"{system_prompt}\n\n{prompt}")
        return call_fn(prompt)

    def _build_prompt(
        self,
        problem: str,
        plan: str,
        current_thought: str,
        memory_summary: str,
        history: List[Dict],
    ) -> Tuple[str, str]:
        """Build the prompt. Returns (user_msg, system_msg_or_None)."""
        if history:
            hist_lines = []
            for h in history:
                hist_lines.append(f"Step {h['step']}:")
                hist_lines.append(f"  THOUGHT: {h['thought'][:300]}")
                hist_lines.append(f"  ACTION: {h['action']}")
                if h.get("observation"):
                    hist_lines.append(f"  OBSERVATION: {h['observation'][:400]}")
            history_str = "\n".join(hist_lines)
        else:
            history_str = "(none yet)"

        tool_list = self.registry.format_for_prompt()

        # Use KV cache manager if available — shares prefix with other nodes
        if self.cache_mgr:
            sys_msg, user_msg = self.cache_mgr.build_react_messages(
                current_thought=current_thought or "(no reasoning yet)",
                tool_list=tool_list,
                history_str=history_str,
            )
            return user_msg, sys_msg

        # Fallback: original prompt construction (no system/user split)
        mem = (memory_summary or "").strip()
        memory_block = f"MEMORY INSIGHTS: {mem[:500]}\n" if mem and mem != "No memory insights yet." else ""

        prompt = _REACT_PROMPT.format(
            problem=problem,
            plan=plan or "(no plan)",
            current_thought=current_thought or "(no reasoning yet)",
            memory_block=memory_block,
            tool_list=tool_list,
            history=history_str,
        )
        return prompt, None

    def _dispatch(self, tool_name: str, tool_input: str) -> str:
        """Look up tool in registry and call it. Returns observation string."""
        spec = self.registry.get(tool_name)
        if spec is None:
            return f"Error: unknown tool '{tool_name}'. Available tools: {', '.join(s.name for s in self.registry.all_specs())}"
        try:
            return spec.fn(tool_input)
        except Exception as exc:
            return f"Error running {tool_name}: {exc}"
