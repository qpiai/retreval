"""
Ollama Client for local LLM inference
Supports Gemma 3:1b and other Ollama models
Mimics the interface of GeminiClient for compatibility with agent_langgraph.py
"""
import os
import json
import re
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

try:
    import requests
except ImportError:
    raise RuntimeError("Please install requests: pip install requests")


class OllamaClient:
    """Client for Ollama API - compatible with GeminiClient interface."""

    def complete(self, prompt: str) -> str:
        """Generate a completion for a prompt (simple LLM call)."""
        return self._call_api(prompt)
    
    def generate(self, prompt: str, max_tokens: int = None) -> str:
        """Generate a completion for a prompt (alias for complete with optional max_tokens)."""
        if max_tokens:
            old_max = self.max_tokens
            self.max_tokens = max_tokens
            result = self._call_api(prompt)
            self.max_tokens = old_max
            return result
        return self._call_api(prompt)

    def __init__(self, model_name: str = None, base_url: str = None, verbose: bool = True):
        """
        Initialize Ollama client.
        
        Args:
            model_name: Ollama model to use (overrides .env if provided)
            base_url: Ollama API endpoint (overrides .env if provided)
            verbose: Enable verbose output
        """
        load_dotenv()
        
        self.verbose = verbose
        
        # Prioritize environment variables, then fall back to parameters, then defaults
        self.model_name = os.getenv("OLLAMA_MODEL") or model_name or "qwen2.5:3b"
        self.base_url = os.getenv("OLLAMA_BASE_URL") or base_url or "http://localhost:11434"
        self.api_endpoint = f"{self.base_url}/api/generate"
        
        # Generation config
        self.temperature = float(os.getenv("TEMPERATURE", "0.7"))
        self.top_p = float(os.getenv("TOP_P", "0.95"))
        self.top_k = int(os.getenv("TOP_K", "40"))
        self.max_tokens = int(os.getenv("MAX_OUTPUT_TOKENS", "2048"))
        
        if self.verbose:
            print(f"🤖 Using Ollama model: {self.model_name}")
            print(f"   Endpoint: {self.base_url}")
            print(f"   Temperature: {self.temperature}")
        
        # Verify connection
        self._verify_connection()
    
    def _verify_connection(self):
        """Verify Ollama is running and model is available via HTTP."""
        # Use HTTP API only (for remote Ollama servers)
        try:
            for path in ("/api/models", "/api/list", "/api/tags"):
                try:
                    response = requests.get(f"{self.base_url}{path}", timeout=5)
                except Exception:
                    response = None
                if response is not None and response.status_code == 200:
                    try:
                        data = response.json()
                        # attempt common keys
                        models = data.get("models") or data.get("tags") or data.get("items") or []
                        model_names = [m.get("name", "") if isinstance(m, dict) else str(m) for m in models]
                        if not any(self.model_name in name for name in model_names):
                            print(f"⚠️  Warning: Model '{self.model_name}' not found in Ollama (via HTTP).")
                            print(f"   Available models: {', '.join(model_names)}")
                            print(f"   Run: ollama pull {self.model_name}")
                        return
                    except Exception:
                        # not JSON or unexpected shape
                        pass
            # If we get here, HTTP probes failed
            print(f"⚠️  Warning: Could not verify Ollama models at {self.base_url} via HTTP.")
            print(f"   Make sure Ollama is running and accessible at {self.base_url}")
        except Exception as e:
            print(f"⚠️  Warning: Could not connect to Ollama at {self.base_url}")
            print(f"   Error: {e}")
            print(f"   Make sure Ollama is running and accessible at {self.base_url}")
    
    def _call_api(self, prompt: str, retry_count: int = 3) -> str:
        """
        Make API call to Ollama with error handling and retries.
        
        Args:
            prompt: The prompt to send
            retry_count: Number of retries on failure
            
        Returns:
            Response text from the model
        """
        # Use HTTP API only (for remote Ollama servers)
        for attempt in range(retry_count):
            try:
                payload = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": max(0.3, self.temperature - 0.1 * attempt),
                        "top_p": self.top_p,
                        "top_k": self.top_k,
                        "num_predict": self.max_tokens,
                    }
                }

                response = requests.post(self.api_endpoint, json=payload, timeout=120)
                
                if response.status_code == 200:
                    result = response.json()
                    # Ollama HTTP outputs vary; try common keys
                    text = (result.get("response") or result.get("text") or result.get("out") or "").strip()
                    if text:
                        return text
                    # If empty response, treat as error and retry
                    print(f"⚠️  Empty response from Ollama (attempt {attempt + 1})")
                else:
                    print(f"⚠️  Ollama HTTP error (attempt {attempt + 1}): Status {response.status_code}")
                    if response.text:
                        print(f"   Response: {response.text[:200]}")
                
                if attempt < retry_count - 1:
                    print("   Retrying...")

            except requests.exceptions.Timeout:
                print(f"⚠️  Request timeout (attempt {attempt + 1})")
                if attempt < retry_count - 1:
                    print("   Retrying...")

            except requests.exceptions.ConnectionError as e:
                print(f"⚠️  Connection error (attempt {attempt + 1}): Cannot connect to {self.base_url}")
                print(f"   Make sure Ollama is running and accessible")
                if attempt < retry_count - 1:
                    print("   Retrying...")

            except Exception as e:
                print(f"⚠️  Error (attempt {attempt + 1}): {e}")
                if attempt < retry_count - 1:
                    print("   Retrying...")

        return "Unable to generate response after multiple attempts."
    
    def expand(self, node_id: str, thought: str, memory_summary: str, num_children: int = 3, diversity_mode: bool = False) -> List[str]:
        """Generate expansion thoughts for a node."""
        if diversity_mode:
            prompt = f"""You are expanding reasoning node {node_id}. Generate {num_children} DIVERSE next-step reasoning paths.

CURRENT REASONING:
{thought}

LEARNED INSIGHTS:
{memory_summary}

Generate {num_children} distinct reasoning paths that explore DIFFERENT optimization approaches.
Each thought should be specific, actionable, and different from the others.

Format as:
1. [First optimization approach]
2. [Second optimization approach]  
3. [Third optimization approach]

Provide ONLY the numbered list."""
        else:
            prompt = f"""You are expanding reasoning node {node_id}. Generate {num_children} DIVERSE next-step reasoning paths.

CURRENT REASONING:
{thought}

LEARNED INSIGHTS:
{memory_summary}

EXPANSION STRATEGY - Generate diverse approaches using these angles:

1. **Decomposition Approach**: Break the problem into smaller sub-problems or components
2. **Alternative Perspectives**: Consider different solution strategies  
3. **Critical Analysis**: Challenge assumptions and identify gaps
4. **Concrete Steps**: Propose specific, actionable next steps

Generate {num_children} distinct reasoning paths that explore DIFFERENT aspects.
Each thought should be:
- Substantive (3-5 sentences minimum)
- Specific and actionable
- Different in approach from the others
- Logically sound and grounded

Format as:
1. [First comprehensive thought exploring one angle]
2. [Second comprehensive thought exploring a different angle]
3. [Third comprehensive thought exploring yet another angle]

Provide ONLY the numbered list."""

        response = self._call_api(prompt)
        if not response or "Unable to generate" in response:
            return self._fallback_expand(thought, num_children)
        return self._parse_numbered_list(response, num_children)
    
    def self_refine(self, thought: str, critique_history: List[str], memory_summary: str) -> Dict[str, str]:
        """Refine a thought based on critiques and memory."""
        critique_text = "\n".join(f"- {c}" for c in critique_history) if critique_history else "None yet"
        
        prompt = f"""Improve the following reasoning based on critiques and insights.

CURRENT THOUGHT:
{thought}

PREVIOUS CRITIQUES:
{critique_text}

MEMORY INSIGHTS:
{memory_summary}

Provide:
1. IMPROVED_THOUGHT: Enhanced reasoning addressing the critiques
2. SELF_CRITIQUE: Identify remaining weaknesses or gaps

Format:
IMPROVED_THOUGHT:
[your improved thought]

SELF_CRITIQUE:
[your critique]"""

        response = self._call_api(prompt)
        parsed = self._parse_refinement(response)
        if not parsed.get("improved_thought"):
            parsed["improved_thought"] = thought
        if not parsed.get("critique"):
            parsed["critique"] = "Refined based on feedback"
        return parsed
    
    def score_node(self, thought: str) -> Dict[str, Any]:
        """Score a reasoning node."""
        prompt = f"""Evaluate this code optimization reasoning on a scale of 0-1.

REASONING:
{thought}

Scoring criteria:
1. Correctness of optimization (0.4 weight)
2. Time/space complexity improvement (0.3 weight)
3. Code clarity and readability (0.2 weight)
4. Completeness of solution (0.1 weight)

Provide a score between 0 and 1 (higher for correct, complete optimizations).

Format:
SCORE: [0.0-1.0]
RATIONALE: [Brief explanation of the score]"""

        response = self._call_api(prompt)
        parsed = self._parse_score(response)
        # Ensure score is in valid range
        if "score" in parsed:
            parsed["score"] = min(1.0, max(0.0, parsed["score"]))
        return parsed
    
    def update_memory(self, recent_nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Update memory based on recent reasoning."""
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
        """Cross-critique two reasoning nodes."""
        prompt = f"""Compare and score these two optimization approaches.

NODE {node_a_id}:
{node_a_thought[:500]}

NODE {node_b_id}:
{node_b_thought[:500]}

Rate each approach from 0.0 to 1.0 based on:
- Correctness
- Efficiency improvement
- Code clarity

Respond ONLY in this exact format:
SCORE_A: 0.X
SCORE_B: 0.X"""

        response = self._call_api(prompt)
        if not response or "Unable to generate" in response or len(response) < 10:
            # Heuristic fallback based on length
            score_a = min(0.5 + len(node_a_thought) / 2000, 0.9)
            score_b = min(0.5 + len(node_b_thought) / 2000, 0.9)
            return {node_a_id: score_a, node_b_id: score_b}
        return self._parse_cross_critique(response, node_a_id, node_b_id)
    
    def generate_implementation(self, original_problem: str, best_strategy: str, reasoning_path: List[str]) -> str:
        """Generate optimized code implementation."""
        path_text = "\n".join(f"Step {i+1}: {step}" for i, step in enumerate(reasoning_path[-3:]))
        
        prompt = f"""Generate optimized Python code based on this analysis.

ORIGINAL PROBLEM:
{original_problem}

OPTIMIZATION STRATEGY:
{best_strategy}

REASONING PATH:
{path_text}

Provide:
1. OPTIMIZED_CODE: The improved Python code
2. EXPLANATION: Brief explanation of improvements
3. COMPLEXITY: Time and space complexity analysis

Format your response as:
OPTIMIZED_CODE:
```python
[your optimized code]
```

EXPLANATION:
[explanation of improvements]

COMPLEXITY:
Time: O(...)
Space: O(...)"""

        response = self._call_api(prompt)
        return response
    
    # Helper methods
    def _fallback_expand(self, thought: str, num_children: int) -> List[str]:
        """Fallback expansion when API fails."""
        return [
            f"Approach {i+1}: Consider alternative optimization for: {thought[:100]}..."
            for i in range(num_children)
        ]
    
    def _parse_numbered_list(self, text: str, expected: int) -> List[str]:
        """Parse numbered list from response."""
        lines = text.strip().split('\n') if text else []
        thoughts = []
        for line in lines:
            m = re.match(r'^\d+[\.|\)]\s*(.+)', line.strip())
            if m:
                thoughts.append(m.group(1).strip())
        if len(thoughts) < expected:
            thoughts = [t.strip() for t in (text or '').split('\n') if t.strip()][:expected]
        return thoughts[:expected] if thoughts else self._fallback_expand("", expected)
    
    def _parse_refinement(self, text: str) -> Dict[str, str]:
        """Parse refinement response."""
        improved = ""
        critique = ""
        for line in (text or '').split('\n'):
            if line.startswith("IMPROVED_THOUGHT:") or line.startswith("IMPROVED:"):
                improved = line.replace("IMPROVED_THOUGHT:", "").replace("IMPROVED:", "").strip()
            elif line.startswith("SELF_CRITIQUE:") or line.startswith("CRITIQUE:"):
                critique = line.replace("SELF_CRITIQUE:", "").replace("CRITIQUE:", "").strip()
        
        if not improved:
            improved = (text or '').strip()
        if not critique:
            critique = "Refined based on previous context"
        
        return {
            "improved_thought": improved,
            "critique": critique
        }
    
    def _parse_score(self, text: str) -> Dict[str, Any]:
        """Parse scoring response."""
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
        
        return {
            "score": min(1.0, max(0.0, score)),
            "rationale": rationale
        }
    
    def _parse_memory_update(self, text: str) -> Dict[str, Any]:
        """Parse memory update response."""
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
        
        return {
            "insights": insights[:3],
            "failures": failures[:3],
            "best_paths": best_paths
        }
    
    def _parse_cross_critique(self, text: str, node_a_id: str, node_b_id: str) -> Dict[str, float]:
        """Parse cross-critique response."""
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
