"""
vLLM Client for local LLM inference via OpenAI-compatible API.
Mimics the interface of GeminiClient/OllamaClient for compatibility with agent_langgraph.py.

vLLM serves at http://localhost:8000/v1 by default with full OpenAI API compatibility.

KV Cache Integration:
  When a KVCacheManager is attached (via set_cache_manager()), high-level methods
  (expand, self_refine, cross_critique, generate_implementation) use prefix-structured
  prompts for maximum KV cache reuse. The shared prefix (problem + plan + memory) is
  computed once and reused across all calls within a problem, so vLLM's automatic
  prefix caching avoids recomputing those tokens.

  To enable prefix caching on the vLLM server, start it with:
    --enable-prefix-caching
"""
import os
import re
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

try:
    from openai import OpenAI
except ImportError:
    raise RuntimeError("Please install openai: pip install openai")


class VLLMClient:
    """Client for vLLM OpenAI-compatible API with KV cache support."""

    def __init__(self, model_name: str = None, base_url: str = None, verbose: bool = True):
        load_dotenv()

        self.verbose = verbose
        self.model_name = model_name or os.getenv("VLLM_MODEL", "Qwen/Qwen3.5-9B")
        self.base_url = base_url or os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")

        # Generation config
        self.temperature = float(os.getenv("TEMPERATURE", "0.7"))
        self.top_p = float(os.getenv("TOP_P", "0.95"))
        self.max_tokens = int(os.getenv("MAX_OUTPUT_TOKENS", "2048"))
        self.timeout = int(os.getenv("VLLM_TIMEOUT", "120"))

        # KV Cache Manager (set via set_cache_manager)
        self._cache_mgr = None

        self.client = OpenAI(api_key="not-needed", base_url=self.base_url)

        if self.verbose:
            print(f"🚀 Using vLLM backend: {self.model_name}")
            print(f"   Endpoint: {self.base_url}")
            print(f"   Temperature: {self.temperature}")

        self._verify_connection()

    def set_cache_manager(self, cache_mgr) -> None:
        """Attach a KVCacheManager for prefix-structured prompts."""
        self._cache_mgr = cache_mgr
        if self.verbose:
            print("   ✓ KV Cache Manager attached")

    def _verify_connection(self):
        """Check that vLLM server is reachable and serving the expected model."""
        try:
            models = self.client.models.list()
            available = [m.id for m in models.data]
            if self.model_name not in available:
                print(f"⚠️  Model '{self.model_name}' not found on vLLM server.")
                print(f"   Available: {', '.join(available)}")
                if available:
                    self.model_name = available[0]
                    print(f"   Auto-selected: {self.model_name}")
            else:
                if self.verbose:
                    print(f"   ✓ Model verified on server")
        except Exception as e:
            print(f"⚠️  Could not connect to vLLM at {self.base_url}: {e}")

    # ---- Core API call ----
    def _call_api(self, prompt: str, retry_count: int = 3,
                  system_prompt: str = None, max_tokens: int = None) -> str:
        """Call the vLLM API.

        Args:
            prompt: The user message content.
            retry_count: Number of retries on failure.
            system_prompt: Optional system message.  When provided the request
                uses a two-message layout (system + user) which both improves
                instruction-following and maximises vLLM prefix-cache reuse
                (the system message tokens are identical across calls).
            max_tokens: Per-call override for max output tokens.
        """
        tokens = max_tokens or self.max_tokens
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        last_error = None
        for attempt in range(retry_count):
            try:
                temp = max(0.3, self.temperature - 0.1 * attempt)
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temp,
                    top_p=self.top_p,
                    max_tokens=tokens,
                    timeout=self.timeout,
                )
                if resp.choices and resp.choices[0].message and resp.choices[0].message.content:
                    return resp.choices[0].message.content.strip()
                print(f"⚠️  Empty response from vLLM (attempt {attempt + 1})")
            except Exception as e:
                last_error = str(e)
                print(f"⚠️  vLLM error (attempt {attempt + 1}): {last_error[:200]}")
                if attempt < retry_count - 1:
                    print("   Retrying...")

        msg = "Unable to generate response after multiple attempts."
        if last_error:
            msg += f" Last error: {last_error[:200]}"
        return msg

    def complete(self, prompt: str) -> str:
        return self._call_api(prompt)

    def generate(self, prompt: str, max_tokens: int = None) -> str:
        if max_tokens:
            old = self.max_tokens
            self.max_tokens = max_tokens
            result = self._call_api(prompt)
            self.max_tokens = old
            return result
        return self._call_api(prompt)

    # ---- Helper: strip thinking model tags ----
    @staticmethod
    def _strip_think(text: str) -> str:
        """Remove <think>...</think> blocks from thinking models."""
        import re as _re
        return _re.sub(r'<think>.*?</think>', '', text, flags=_re.DOTALL).strip()

    # ---- High-level methods (same interface as OllamaClient / _OpenAIBackend) ----

    def expand(self, node_id: str, thought: str, memory_summary: str,
               num_children: int = 3, diversity_mode: bool = False) -> List[str]:
        # Use KV cache manager if available for prefix reuse
        if self._cache_mgr:
            sys_msg, user_msg = self._cache_mgr.build_expand_messages(thought, num_children)
            response = self._strip_think(
                self._call_api(user_msg, system_prompt=sys_msg, max_tokens=1024)
            )
        else:
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

            response = self._strip_think(self._call_api(prompt, max_tokens=1024))
        if not response or "Unable to generate" in response:
            return [f"Approach {i+1}: Consider alternative for: {thought[:100]}..."
                    for i in range(num_children)]
        return self._parse_numbered_list(response, num_children)

    def self_refine(self, thought: str, critique_history: List[str], memory_summary: str) -> Dict[str, str]:
        # Use KV cache manager if available
        if self._cache_mgr:
            sys_msg, user_msg = self._cache_mgr.build_refine_messages(thought, critique_history)
            response = self._strip_think(self._call_api(user_msg, system_prompt=sys_msg))
        else:
            critique_text = "\n".join(f"- {c}" for c in critique_history) if critique_history else ""
            mem = (memory_summary or "")[:500]

            parts = [f"Here is a solution attempt:\n\n{thought}"]
            if critique_text:
                parts.append(f"Previous feedback:\n{critique_text}")
            if mem and mem != "No memory insights yet.":
                parts.append(f"Relevant insights:\n{mem}")
            parts.append("Rewrite this solution to fix any errors and fill gaps. Show your work step by step.\n\nIMPROVED:")

            prompt = "\n\n".join(parts)
            response = self._strip_think(self._call_api(prompt))
        parsed = self._parse_refinement(response)
        if not parsed.get("improved_thought"):
            parsed["improved_thought"] = thought
        if not parsed.get("critique"):
            parsed["critique"] = "Refined based on feedback"
        return parsed

    def score_node(self, thought: str) -> Dict[str, Any]:
        prompt = f"""Evaluate this reasoning on a scale of 0-1.

{thought}

Score based on: correctness (most important), completeness, and clarity.

SCORE: """

        response = self._strip_think(self._call_api(prompt, max_tokens=256))
        parsed = self._parse_score(response)
        if "score" in parsed:
            parsed["score"] = min(1.0, max(0.0, parsed["score"]))
        return parsed

    def update_memory(self, recent_nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        nodes_text = "\n\n".join(
            f"Node {n.get('id', '?')} (score: {n.get('combined_score', n.get('score', 0)):.2f}):\n{n.get('thought', '')[:300]}"
            for n in recent_nodes[:5]
        )
        prompt = f"""Analyze these recent reasoning nodes and extract insights.

RECENT NODES:
{nodes_text}

What reasoning strategies worked well?
What patterns failed?
What insights should be remembered?

Format your response as:
INSIGHTS:
- [insight 1]
- [insight 2]

FAILURES:
- [failure 1]
- [failure 2]

BEST_PATHS: [Comma-separated node IDs of top performers]"""

        response = self._call_api(prompt)
        return self._parse_memory_update(response)

    def cross_critique(self, node_a_id: str, node_a_thought: str,
                       node_b_id: str, node_b_thought: str) -> Dict[str, float]:
        # Use KV cache manager if available for prefix reuse
        if self._cache_mgr:
            sys_msg, user_msg = self._cache_mgr.build_cross_critique_messages(
                node_a_id, node_a_thought, node_b_id, node_b_thought
            )
            response = self._strip_think(
                self._call_api(user_msg, system_prompt=sys_msg, max_tokens=256)
            )
        else:
            prompt = f"""Compare and score these two approaches.

NODE {node_a_id}:
{node_a_thought[:500]}

NODE {node_b_id}:
{node_b_thought[:500]}

Rate each from 0.0 to 1.0 based on correctness, efficiency, and clarity.

Respond ONLY in this exact format:
SCORE_A: 0.X
SCORE_B: 0.X"""

            response = self._strip_think(self._call_api(prompt, max_tokens=256))
        if not response or "Unable to generate" in response or len(response) < 10:
            score_a = min(0.5 + len(node_a_thought) / 2000, 0.9)
            score_b = min(0.5 + len(node_b_thought) / 2000, 0.9)
            return {node_a_id: score_a, node_b_id: score_b}
        return self._parse_cross_critique(response, node_a_id, node_b_id)

    def generate_implementation(self, original_problem: str, best_strategy: str, reasoning_path: List[str]) -> str:
        # Use KV cache manager if available for prefix reuse
        if self._cache_mgr:
            sys_msg, user_msg = self._cache_mgr.build_synthesis_messages(best_strategy, reasoning_path)
            response = self._call_api(user_msg, system_prompt=sys_msg, max_tokens=4096)
            if not response or "Unable to generate" in response:
                return best_strategy
            return response.strip()
        else:
            has_retrieved_data = "RELEVANT DATA RETRIEVED:" in best_strategy
            path_text = "\n".join(f"Step {i+1}: {step}" for i, step in enumerate(reasoning_path[-3:]))

            if has_retrieved_data:
                prompt = f"""You have completed deep reasoning about this problem:

ORIGINAL PROBLEM:
{original_problem}

BEST STRATEGY AND RETRIEVED DATA:
{best_strategy}

REASONING PATH:
{path_text}

SOLVE the problem using the retrieved data. Show your work step-by-step.
Provide SPECIFIC answers. Do NOT say "cannot be solved"."""
            else:
                prompt = f"""You have completed deep reasoning about this problem:

ORIGINAL PROBLEM:
{original_problem}

BEST STRATEGY:
{best_strategy}

REASONING PATH:
{path_text}

IMPLEMENT this strategy and provide the CONCRETE solution.
- If it's code, write the actual code
- If it's analysis, provide detailed analysis with numbers
- Show calculations and reasoning step-by-step

Provide the complete solution:"""

        response = self._call_api(prompt)
        if not response or "Unable to generate" in response:
            return best_strategy
        return response.strip()

    # ---- Parsers (same as OllamaClient) ----

    def _parse_numbered_list(self, text: str, expected: int) -> List[str]:
        text = self._strip_think(text or "")
        lines = text.strip().split('\n') if text else []
        thoughts = []
        for line in lines:
            m = re.match(r'^\d+[\.|\)]\s*(.+)', line.strip())
            if m:
                thoughts.append(m.group(1).strip())
        if len(thoughts) < expected:
            thoughts = [t.strip() for t in (text or '').split('\n') if t.strip()][:expected]
        return thoughts[:expected] if thoughts else [
            f"Approach {i+1}: Alternative reasoning path" for i in range(expected)
        ]

    def _parse_refinement(self, text: str) -> Dict[str, str]:
        text = self._strip_think(text or "")
        improved = ""
        critique = ""
        for line in text.split('\n'):
            if line.startswith("IMPROVED_THOUGHT:") or line.startswith("IMPROVED:"):
                improved = line.replace("IMPROVED_THOUGHT:", "").replace("IMPROVED:", "").strip()
            elif line.startswith("SELF_CRITIQUE:") or line.startswith("CRITIQUE:"):
                critique = line.replace("SELF_CRITIQUE:", "").replace("CRITIQUE:", "").strip()
        if not improved:
            improved = (text or '').strip()
        if not critique:
            critique = "Refined based on previous context"
        return {"improved_thought": improved, "critique": critique}

    def _parse_score(self, text: str) -> Dict[str, Any]:
        text = self._strip_think(text or "")
        score = 0.5
        rationale = "Default scoring"
        for line in text.split('\n'):
            if "SCORE:" in line:
                try:
                    score = float(re.findall(r'[\d\.]+', line)[0])
                except Exception:
                    pass
            elif "RATIONALE:" in line:
                rationale = line.replace("RATIONALE:", "").strip()
        return {"score": min(1.0, max(0.0, score)), "rationale": rationale}

    def _parse_memory_update(self, text: str) -> Dict[str, Any]:
        insights, failures, best_paths = [], [], []
        current_section = None
        for raw in (text or '').split('\n'):
            line = raw.strip()
            if line.startswith("INSIGHTS:"):
                current_section = "insights"
            elif line.startswith("FAILURES:"):
                current_section = "failures"
            elif line.startswith("BEST_PATHS:"):
                best_paths = [p.strip() for p in line.replace("BEST_PATHS:", "").strip().split(',') if p.strip()]
            elif line.startswith('-') and current_section:
                item = line[1:].strip()
                if current_section == "insights":
                    insights.append(item)
                elif current_section == "failures":
                    failures.append(item)
        return {"insights": insights[:3], "failures": failures[:3], "best_paths": best_paths}

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
