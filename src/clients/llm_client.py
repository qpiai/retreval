"""
Unified LLM Client
Allows selecting different providers (Gemini, OpenAI, Ollama, vLLM) using only environment variables.

Environment variables:
  LLM_PROVIDER   = gemini | openai | ollama | vllm   (default: gemini)

  # For Gemini
  GEMINI_API_KEY = ...
  GEMINI_MODEL   = gemini-2.5-flash (default)
  TEMPERATURE    = 0.7 (optional)
  TOP_P          = 0.95 (optional)
  TOP_K          = 40 (optional)
  MAX_OUTPUT_TOKENS = 2048 (optional)
  GENAI_TIMEOUT  = 20 (seconds, optional)

  # For OpenAI
  OPENAI_API_KEY = ...
  OPENAI_MODEL   = gpt-4o-mini (default)
  OPENAI_TIMEOUT = 20 (seconds, optional)

  # For vLLM
  VLLM_MODEL     = Qwen/Qwen3.5-9B (default)
  VLLM_BASE_URL  = http://localhost:8000/v1 (default)
  VLLM_TIMEOUT   = 120 (seconds, optional)

No API keys are stored in code; they are read from .env via python-dotenv.
"""
from __future__ import annotations

import os
import re
import sys
from typing import Any, Dict, List, Optional
from pathlib import Path

from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Gemini backend — imported lazily to avoid loading google.generativeai when not needed
GeminiClient = None

def _get_gemini_client_class():
    global GeminiClient
    if GeminiClient is None:
        from src.clients.gemini_client import GeminiClient as _GC
        GeminiClient = _GC
    return GeminiClient


def _load_env_all():
    """Load .env and config.env from common locations (cwd and module dir)."""
    try:
        load_dotenv()  # default search (cwd upward)
        cwd = Path.cwd()
        load_dotenv(cwd / ".env")
        load_dotenv(cwd / "config.env")
        here = Path(__file__).resolve().parent
        load_dotenv(here / ".env")
        load_dotenv(here / "config.env")
    except Exception:
        # Best-effort; proceed even if loading fails
        pass


class _OpenAIBackend:
    """Minimal OpenAI backend implementing the same interface as GeminiClient methods used."""

    def __init__(self):
        _load_env_all()
        try:
            from openai import OpenAI  # type: ignore
        except Exception as e:
            raise RuntimeError("OpenAI backend requested but 'openai' package is not installed. 'pip install openai'.") from e

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment or .env file")

        self.client = OpenAI(api_key=api_key)
        _load_env_all()
        # Model selection with support for 'auto'/'latest' alias and preferences
        self.model_env = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.model_preferences = [
            m.strip() for m in os.getenv(
                "OPENAI_MODEL_PREFERENCES",
                # Preference order: try stronger models first, then smaller
                "gpt-4o,gpt-4o-mini"
            ).split(",") if m.strip()
        ]
        self._resolved_model: Optional[str] = None
        self.temperature = float(os.getenv("TEMPERATURE", "0.7"))
        self.top_p = float(os.getenv("TOP_P", "0.95"))
        self.max_tokens = int(os.getenv("MAX_OUTPUT_TOKENS", "2048"))
        self.timeout = int(os.getenv("OPENAI_TIMEOUT", "20"))

    # -------------- Core call --------------
    def _call_api(self, prompt: str, retry_count: int = 3, system_prompt: str = None, max_tokens: int = None) -> str:
        tokens = max_tokens or self.max_tokens
        last_error = None

        def _try_model(model_name: str, temp: float) -> Optional[str]:
            nonlocal last_error
            try:
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})
                resp = self.client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=temp,
                    top_p=self.top_p,
                    max_tokens=tokens,
                    timeout=self.timeout,
                )
                if resp.choices and resp.choices[0].message and resp.choices[0].message.content:
                    return resp.choices[0].message.content
            except Exception as e:
                last_error = str(e)
                return None
            return None

        for attempt in range(retry_count):
            temp = max(0.3, self.temperature - 0.2 * attempt)

            # If we've already resolved a working model, use it
            if self._resolved_model:
                out = _try_model(self._resolved_model, temp)
                if out:
                    return out
                # If previously working model fails now, clear and re-resolve
                self._resolved_model = None

            model_env = (self.model_env or "").strip().lower()
            if model_env in ("auto", "latest"):
                # Try preferred models in order
                for candidate in self.model_preferences + ["gpt-4o-mini"]:
                    out = _try_model(candidate, temp)
                    if out:
                        self._resolved_model = candidate
                        return out
                # If none worked this attempt, continue retry loop
            else:
                out = _try_model(self.model_env, temp)
                if out:
                    self._resolved_model = self.model_env
                    return out

        error_msg = f"Unable to generate response after {retry_count} attempts."
        if last_error:
            print(f"⚠️ API Error: {last_error}")
            error_msg += f" Last error: {last_error[:200]}"
        return error_msg

    # -------------- High-level methods --------------
    def expand(self, node_id: str, thought: str, memory_summary: str, num_children: int = 3, diversity_mode: bool = False) -> List[str]:
        mem = (memory_summary or "")[:500]
        mem_block = f"\nRelevant insights:\n{mem}\n" if mem and mem != "No memory insights yet." else ""

        prompt = f"""Given this reasoning so far:

{thought}
{mem_block}
Suggest {num_children} different ways to continue solving this problem. Each should take a different approach.

Format as a numbered list:
1. ...
2. ...
3. ...

Provide ONLY the numbered list."""

        response = self._call_api(prompt)
        return self._parse_numbered_list(response, num_children)

    def self_refine(self, thought: str, critique_history: List[str], memory_summary: str) -> Dict[str, str]:
        critique_text = "\n".join(f"- {c}" for c in critique_history) if critique_history else ""
        mem = (memory_summary or "")[:500]

        parts = [f"Here is a solution attempt:\n\n{thought}"]
        if critique_text:
            parts.append(f"Previous feedback:\n{critique_text}")
        if mem and mem != "No memory insights yet.":
            parts.append(f"Relevant insights:\n{mem}")
        parts.append("Rewrite this solution to fix any errors and fill gaps. Show your work step by step.\n\nIMPROVED:")

        prompt = "\n\n".join(parts)
        response = self._call_api(prompt)
        return self._parse_refinement(response)

    def cross_critique(self, node_a_id: str, node_a_thought: str, node_b_id: str, node_b_thought: str) -> Dict[str, float]:
        prompt = f"""Compare these two approaches and rate each from 0.0 to 1.0.

Approach 1: {node_a_thought[:500]}

Approach 2: {node_b_thought[:500]}

Respond ONLY in this exact format:
SCORE_A: 0.X
SCORE_B: 0.X"""
        response = self._call_api(prompt)
        if not response or "Unable to generate" in response or len(response) < 10:
            score_a = min(0.5 + len(node_a_thought) / 2000, 0.9)
            score_b = min(0.5 + len(node_b_thought) / 2000, 0.9)
            return {node_a_id: score_a, node_b_id: score_b}
        return self._parse_cross_critique(response, node_a_id, node_b_id)

    def score_node(self, thought: str) -> Dict[str, Any]:
        prompt = f"""Evaluate this reasoning on a scale of 0-1.

{thought}

Score based on: correctness (most important), completeness, and clarity.

SCORE: """
        response = self._call_api(prompt)
        return self._parse_score(response)

    def update_memory(self, recent_nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        nodes_text = "\n\n".join([
            f"Node {n['id']} (score: {n.get('combined_score', 0):.2f}):\n{n['thought']}"
            for n in recent_nodes
        ])
        prompt = f"""Summarize key takeaways from the last reasoning cycle:

{nodes_text}

What reasoning strategies worked well?
What patterns failed?
What insights should be remembered?

Format your response as:
INSIGHTS:
- [Insight 1]
- [Insight 2]

FAILURES:
- [Failure 1]
- [Failure 2]

BEST_PATHS: [Comma-separated node IDs of top performers]"""
        response = self._call_api(prompt)
        return self._parse_memory_update(response)

    def generate_implementation(self, original_problem: str, best_strategy: str, reasoning_path: List[str]) -> str:
        # Check if retrieved data is present
        has_retrieved_data = "RELEVANT DATA RETRIEVED:" in best_strategy
        
        if has_retrieved_data:
            prompt = f"""CURRENT DATE: November 2025

You have completed deep reasoning about this problem:

ORIGINAL PROBLEM:
{original_problem}

BEST STRATEGY AND RETRIEVED DATA:
{best_strategy}

Now, SOLVE the problem and provide a CONCRETE answer using available information.

CRITICAL INSTRUCTIONS:
- ONLY use data provided in the "RELEVANT DATA RETRIEVED" section - this is current, real data
- Do NOT use your training data from 2023 or earlier - it is outdated
- If web search failed and no retrieved data is shown, clearly state you're using estimates
- Show your work step-by-step with clear explanations
- Provide SPECIFIC answers with the actual retrieved data
- State your assumptions clearly when making estimates
- Do NOT say "cannot be solved" or "insufficient data" - provide a best-effort answer
- Be thorough and answer ALL parts of the question

Provide the complete solution:"""
        else:
            prompt = f"""CURRENT DATE: November 2025

You have completed deep reasoning about this problem:

ORIGINAL PROBLEM:
{original_problem}

BEST STRATEGY IDENTIFIED:
{best_strategy}

Now, IMPLEMENT this strategy and provide the CONCRETE, ACTIONABLE OUTPUT.

IMPORTANT:
- We are in November 2025 - do NOT use outdated data from your 2023 training
- If external data is needed and wasn't retrieved, clearly state you're using estimates
- Provide the actual solution, not explanations or advice
- State assumptions clearly when using estimated values
- Show calculations and reasoning step-by-step
- Do NOT say "cannot be determined" - provide a best-effort answer with stated assumptions
- If it's code, write the actual code
- If it's analysis, provide detailed analysis with numbers

Provide the complete solution:"""
        
        response = self._call_api(prompt)
        if not response or "Unable to generate" in response or "Error generating" in response:
            # Minimal fallback: echo best strategy
            return best_strategy
        return response.strip()

    # -------------- Parsers --------------
    def _parse_numbered_list(self, text: str, expected: int) -> List[str]:
        lines = text.strip().split('\n') if text else []
        thoughts = []
        for line in lines:
            m = re.match(r'^\d+[\.|\)]\s*(.+)', line.strip())
            if m:
                thoughts.append(m.group(1).strip())
        if len(thoughts) < expected:
            thoughts = [t.strip() for t in (text or '').split('\n') if t.strip()][:expected]
        return thoughts[:expected]

    def _parse_refinement(self, text: str) -> Dict[str, str]:
        improved = ""
        critique = ""
        for line in (text or '').split('\n'):
            if line.startswith("IMPROVED:"):
                improved = line.replace("IMPROVED:", "").strip()
            elif line.startswith("CRITIQUE:"):
                critique = line.replace("CRITIQUE:", "").strip()
        if not improved:
            improved = (text or '').strip()
        if not critique:
            critique = "Refined based on previous context"
        return {"improved_thought": improved, "critique": critique}

    def _parse_cross_critique(self, text: str, node_a_id: str, node_b_id: str) -> Dict[str, float]:
        scores = {node_a_id: 0.5, node_b_id: 0.5}
        for line in (text or '').split('\n'):
            if "SCORE_A:" in line:
                try:
                    scores[node_a_id] = float(re.findall(r'[\d\.]+', line)[0])
                except Exception:
                    pass
            elif "SCORE_B:" in line:
                try:
                    scores[node_b_id] = float(re.findall(r'[\d\.]+', line)[0])
                except Exception:
                    pass
        return scores

    def _parse_score(self, text: str) -> Dict[str, Any]:
        score = 0.5
        rationale = "Default scoring"
        for line in (text or '').split('\n'):
            if "SCORE:" in line:
                try:
                    score = float(re.findall(r'[\d\.]+', line)[0])
                except Exception:
                    pass
            elif "RATIONALE:" in line:
                rationale = line.replace("RATIONALE:", "").strip()
        return {"score": score, "rationale": rationale}

    def _parse_memory_update(self, text: str) -> Dict[str, Any]:
        insights: List[str] = []
        failures: List[str] = []
        best_paths: List[str] = []
        current_section = None
        for raw in (text or '').split('\n'):
            line = raw.strip()
            if line.startswith("INSIGHTS:"):
                current_section = "insights"
            elif line.startswith("FAILURES:"):
                current_section = "failures"
            elif line.startswith("BEST_PATHS:"):
                best_paths_text = line.replace("BEST_PATHS:", "").strip()
                best_paths = [p.strip() for p in best_paths_text.split(',') if p.strip()]
            elif line.startswith('-') and current_section:
                item = line[1:].strip()
                if current_section == "insights":
                    insights.append(item)
                elif current_section == "failures":
                    failures.append(item)
        return {"insights": insights, "failures": failures, "best_paths": best_paths}


class LLMClient:
    """Provider-agnostic client that delegates to Gemini, OpenAI, or Ollama based on env."""

    def __init__(self, provider: Optional[str] = None):
        _load_env_all()
        
        # Check if USE_OLLAMA is set first
        use_ollama = os.getenv("USE_OLLAMA", "").lower() in ("1", "true", "yes")
        
        if use_ollama:
            # Use Ollama backend
            try:
                from ollama_client import OllamaClient
                self.backend = OllamaClient()
                self.provider = "ollama"
                print("✓ Using Ollama backend")
            except Exception as e:
                print(f"⚠️  Ollama initialization failed: {e}")
                print("   Falling back to default provider...")
                self.provider = (provider or os.getenv("LLM_PROVIDER", "gemini")).strip().lower()
                self._init_backend()
        else:
            self.provider = (provider or os.getenv("LLM_PROVIDER", "gemini")).strip().lower()
            self._init_backend()

    def _init_backend(self):
        """Initialize the backend based on provider."""
        if self.provider == "vllm":
            try:
                from src.clients.vllm_client import VLLMClient
                self.backend = VLLMClient()
            except Exception as e:
                raise ValueError(f"vLLM initialization failed: {e}")
        elif self.provider == "openai":
            try:
                self.backend = _OpenAIBackend()
            except Exception as e:
                # Fallback: if OpenAI is not configured, try Gemini instead
                gem_key = os.getenv("GEMINI_API_KEY")
                if gem_key:
                    print("⚠️  OPENAI_API_KEY missing or invalid; falling back to Gemini provider.")
                    self.backend = _get_gemini_client_class()()
                    self.provider = "gemini"
                else:
                    # Re-raise with clearer guidance
                    raise ValueError(
                        "OpenAI selected but OPENAI_API_KEY not found. Set it in .env or set LLM_PROVIDER=gemini or USE_OLLAMA=1."
                    ) from e
        elif self.provider == "ollama":
            try:
                from src.clients.ollama_client import OllamaClient
                self.backend = OllamaClient()
            except Exception as e:
                raise ValueError(f"Ollama initialization failed: {e}")
        else:
            # default to gemini
            self.backend = _get_gemini_client_class()()

    # Public methods mirror GeminiClient
    def _call_api(self, prompt: str, retry_count: int = 3, **kwargs) -> str:
        """Call the backend API, forwarding extra kwargs (system_prompt, max_tokens)."""
        try:
            return self.backend._call_api(prompt, retry_count=retry_count, **kwargs)
        except TypeError:
            # Backend doesn't support extra kwargs — strip them and retry
            return self.backend._call_api(prompt, retry_count=retry_count)

    def expand(self, *args, **kwargs):
        return self.backend.expand(*args, **kwargs)

    def self_refine(self, *args, **kwargs):
        return self.backend.self_refine(*args, **kwargs)

    def cross_critique(self, *args, **kwargs):
        return self.backend.cross_critique(*args, **kwargs)

    def score_node(self, *args, **kwargs):
        return self.backend.score_node(*args, **kwargs)

    def update_memory(self, *args, **kwargs):
        return self.backend.update_memory(*args, **kwargs)

    def generate_implementation(self, *args, **kwargs):
        return self.backend.generate_implementation(*args, **kwargs)

