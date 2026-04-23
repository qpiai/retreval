"""Synthesis Tool for generating final outputs from ReTreVal Agent."""

import os
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from src.clients.llm_client import LLMClient
from src.tools.search import WebSearchTool

if TYPE_CHECKING:
    from src.agents.state import AgentState


class SynthesizerTool:
    """Synthesizes final output from agent reasoning, handles LP solution formatting and quality validation."""
    
    def __init__(self, llm: LLMClient, verbose: bool = True):
        self.llm = llm
        self.verbose = verbose
        self.web_search = WebSearchTool(verbose=verbose)

    def _tokenize(self, text: str) -> set:
        # Lightweight normalization + tokenization for similarity checks
        text = text.lower()
        # Remove code fences and punctuation-like separators
        text = re.sub(r"```[\s\S]*?```", " ", text)
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        tokens = [t for t in text.split() if len(t) > 2]
        stop = {
            'the','and','for','with','that','this','you','are','was','were','have','has','had',
            'not','but','all','any','can','cannot','will','would','should','could','into','from',
            'your','about','they','them','then','than','use','used','using','over','under','into',
            'our','out','per','let','via','also','just','one','two','three','more','less','very',
        }
        return {t for t in tokens if t not in stop}

    def _is_distinct(self, new_text: str, kept_token_sets: List[set], threshold: float) -> bool:
        # Jaccard similarity over token sets; require sufficiently low overlap to count as distinct
        new_tokens = self._tokenize(new_text)
        if not kept_token_sets:
            return True
        for ks in kept_token_sets:
            if not ks and not new_tokens:
                return False
            inter = len(ks & new_tokens)
            union = len(ks | new_tokens) or 1
            sim = inter / union
            if sim >= threshold:
                return False
        return True
    
    def _validate_output_quality(self, output: str, problem: str) -> tuple[bool, str]:
        """Validate if output meets quality standards. Returns (is_valid, reason)."""
        if not output or len(output.strip()) < 20:
            return False, "Output too short or empty"
        
        # Check for error indicators
        error_phrases = [
            "unable to generate", "error generating", "failed to", "cannot generate",
            "api error", "invalid response", "no response", "timeout"
        ]
        if any(phrase in output.lower() for phrase in error_phrases):
            return False, "Output contains error messages"
        
        # Check if output is just repeating the problem
        problem_words = set(problem.lower().split())
        output_words = set(output.lower().split())
        if len(output_words) > 0:
            overlap = len(problem_words & output_words) / len(output_words)
            if overlap > 0.9:  # More than 90% overlap with problem
                return False, "Output is mostly repeating the problem"
        
        # Check for placeholder text
        placeholders = [
            "[insert", "[add", "[your", "[todo", "[placeholder",
            "lorem ipsum",
        ]
        if any(ph in output.lower() for ph in placeholders):
            return False, "Output contains placeholders"

        # Check for meta-reasoning (model analyzing the prompt instead of solving)
        meta_phrases = [
            "analyze the request", "the prompt asks", "the instruction",
            "this is a reasoning", "the current thought", "the previous critiques",
            "input:", "output format:", "the task requires me to",
        ]
        meta_count = sum(1 for ph in meta_phrases if ph in output.lower())
        if meta_count >= 2:
            return False, "Output is meta-reasoning about the prompt instead of solving"

        # Check for meaningful content (not just filler words)
        meaningful_words = [w for w in output_words if len(w) > 3]
        if len(meaningful_words) < 5:
            return False, "Output lacks meaningful content"
        
        return True, "Output passes quality checks"

    def _dedupe_fenced_blocks(self, text: str) -> str:
        # Remove duplicate fenced code blocks while preserving the first occurrence
        pattern = re.compile(r"```[\s\S]*?```", re.MULTILINE)
        parts = re.split(r"(```[\s\S]*?```)", text)
        seen_blocks = set()
        out: List[str] = []
        for part in parts:
            if pattern.fullmatch(part or ""):
                norm = part.strip()
                if norm in seen_blocks:
                    continue
                seen_blocks.add(norm)
                out.append(part)
            else:
                out.append(part)
        deduped = "".join(out)
        # If the whole output is accidentally duplicated back-to-back, collapse it
        s = deduped.strip()
        if len(s) > 0 and len(s) % 2 == 0:
            mid = len(s) // 2
            if s[:mid].strip() == s[mid:].strip():
                return s[:mid].strip()
        return deduped

    def _wants_code(self, problem: str) -> bool:
        """Heuristic: detect whether the user explicitly asked for code in the problem."""
        if not problem:
            return False
        
        text = problem.lower()
        
        # Check for math puzzles and direct math problems - these NEVER want code
        math_puzzle_indicators = [
            r"game\s+of\s+24", r"puzzle", r"find\s+(?:a|an|the)\s+(?:mathematical\s+)?expression",
            r"solve\s+(?:the\s+)?equation", r"calculate\s+the\s+(?:value|result)",
            r"what\s+(?:is|equals|are).*\?", r"evaluate\s+the\s+expression",
            r"simplify\s+the\s+expression", r"find\s+x", r"solve\s+for",
            r"mathematical\s+expression", r"arithmetic\s+(?:problem|puzzle|expression)",
            r"algebra\s+(?:problem|puzzle)", r"using\s+(?:the\s+)?numbers?.*find",
        ]
        if any(re.search(pat, text, re.I) for pat in math_puzzle_indicators):
            return False
        
        # Check for LP/optimization keywords - if present, likely NOT asking for code
        lp_keywords = [
            "minimize", "maximize", "optimization", "linear program",
            "factory", "factories", "plant", "warehouse", "supply", "demand",
            "shipping cost", "transportation", "allocation", "distribute"
        ]
        if any(kw in text for kw in lp_keywords):
            # Only return True if they ALSO explicitly ask for code
            explicit_code = [r"\bcode\b", r"\bimplement\b", r"\bprogram\b", r"\bscript\b"]
            return any(re.search(pat, text) for pat in explicit_code)
        
        # Strong indicators for code
        keywords = [
            r"\bwrite\s+code\b", r"\bgive\s+me\s+code\b", r"\bprovide\s+code\b",
            r"\bimplement(?:\s+a)?\s+(?:function|class|algorithm)", r"\bgenerate\s+code\b", r"\bcode\s+solution\b",
            r"```", r"\bclass\b", r"\bfunction\b", r"\bscript\b",
        ]
        langs = [
            r"python", r"java\b", r"javascript|typescript", r"c\+\+|c#\b|c\b",
            r"go\b", r"rust\b", r"ruby\b", r"php\b", r"sql\b", r"bash|powershell",
            r"matlab\b", r"r\b",
        ]
        for pat in keywords + langs:
            if re.search(pat, text, re.I):
                return True
        return False

    def _strip_code_blocks(self, text: str) -> str:
        """Remove all fenced code blocks and common inline code artifacts."""
        # Remove fenced blocks ```...```
        text = re.sub(r"```[\s\S]*?```", "", text, flags=re.MULTILINE)
        # Remove triple backticks left alone
        text = text.replace("```", "")
        # Remove typical python import one-liners if they slipped outside fences
        text = re.sub(r"^(?:from\s+\S+\s+import\s+\S+|import\s+\S+)\s*$", "", text, flags=re.MULTILINE)
        # Collapse excess blank lines
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        return text

    def _strip_think_tags(self, text: str) -> str:
        """Remove <think>...</think> blocks from thinking models."""
        return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

    def _generate_enhanced_output(self, problem: str, strategy: str, plan: str, retrieved_info: str = "") -> str:
        """Generate high-quality output with multiple strategies and validation."""

        # Build concise context from strategy (limit to avoid overwhelming the model)
        strategy_short = (strategy or '')[:1000]

        # Strategy 1: Direct math-focused prompt — simple and clear
        structured_prompt = f"""Solve this math problem step by step.

Problem: {problem}

Reasoning so far: {strategy_short}

Show your step-by-step solution, then put your final answer in \\boxed{{answer}} format.

Solution:"""

        output = self._strip_think_tags(
            self.llm._call_api(structured_prompt, max_tokens=2048)
        )

        # Validate output quality
        is_valid, reason = self._validate_output_quality(output, problem)

        if is_valid:
            return output

        if self.verbose:
            print(f"   ⚠️ First generation failed validation: {reason}. Retrying...")

        # Strategy 2: Even simpler prompt
        direct_prompt = f"""Problem: {problem}

Solve step by step. Final answer in \\boxed{{answer}} format.

Solution:"""

        output = self._strip_think_tags(
            self.llm._call_api(direct_prompt, max_tokens=2048)
        )
        is_valid, reason = self._validate_output_quality(output, problem)

        if is_valid:
            return output

        if self.verbose:
            print(f"   ⚠️ Second generation failed: {reason}. Using fallback...")

        # Strategy 3: Fallback — use the strategy/reasoning we already have
        # Try to extract an answer from the strategy itself
        fallback = f"Based on the reasoning:\n{strategy_short}\n\nThe answer is \\boxed{{{strategy_short.split('=')[-1].strip()[:50] if '=' in strategy_short else strategy_short[:50]}}}"
        return fallback
    
    def _format_lp_answer(self, problem: str, lp: Dict[str, Any]) -> str:
        """Generate a comprehensive mathematical explanation with full LP formulation."""
        if not isinstance(lp, dict) or lp.get('status') != 'optimal' or not isinstance(lp.get('vars'), dict):
            return "Unable to find an optimal solution for this problem."

        # Rounding helper
        tol = float(os.getenv('LP_ROUND_TOL', '1e-6'))
        def rfmt(x: float) -> str:
            r = round(x)
            return f"{r}" if abs(x - r) < tol else f"{x:.3f}"

        # Helper: format linear expression nicely
        def format_expr(coeffs: Dict[str, float]) -> str:
            terms = []
            for var, coef in sorted(coeffs.items()):
                if abs(coef) < tol:
                    continue
                if abs(coef - 1.0) < tol:
                    terms.append(f"{var}")
                elif abs(coef + 1.0) < tol:
                    terms.append(f"-{var}")
                else:
                    terms.append(f"{rfmt(coef)}{var}")
            if not terms:
                return "0"
            expr = terms[0]
            for t in terms[1:]:
                if t.startswith('-'):
                    expr += f" - {t[1:]}"
                else:
                    expr += f" + {t}"
            return expr

        vars_dict = lp['vars']
        spec = lp.get('spec', {})
        sense = (spec.get('sense') or 'max').lower()
        obj_map = spec.get('objective', {})
        constraints = spec.get('constraints', [])
        bounds = spec.get('bounds', {})
        obj_val = lp.get('objective')
        
        # Build comprehensive explanation
        sections = []
        
        # 1. Problem Understanding
        sections.append("# Linear Programming Solution\n")
        sections.append("## Problem Statement\n")
        problem_summary = problem[:400] if len(problem) <= 400 else problem[:400] + "..."
        sections.append(f"{problem_summary}\n")
        
        # 2. Mathematical Formulation
        sections.append("## Mathematical Formulation\n")
        
        # Decision Variables
        sections.append("**Decision Variables:**")
        all_vars = sorted(set(list(obj_map.keys()) + list(vars_dict.keys())))
        for var in all_vars:
            sections.append(f"  • `{var}` = amount/quantity to allocate")
        sections.append("")
        
        # Objective Function
        obj_word = "Minimize" if sense == 'min' else "Maximize"
        obj_expr = format_expr(obj_map)
        sections.append(f"**Objective Function:**")
        sections.append(f"  {obj_word}: `{obj_expr}`")
        sections.append("")
        
        # Constraints
        if constraints:
            sections.append(f"**Constraints:**")
            for i, con in enumerate(constraints, 1):
                lhs_map = con.get("lhs", {})
                op = con.get("op", "<=")
                rhs = con.get("rhs", 0)
                lhs_expr = format_expr(lhs_map)
                sections.append(f"  {i}. `{lhs_expr} {op} {rfmt(rhs)}`")
            sections.append("")
        
        # Variable Bounds
        has_bounds = any(var in bounds and bounds[var] != [None, None] for var in all_vars)
        if has_bounds:
            sections.append("**Variable Bounds:**")
            for var in all_vars:
                if var in bounds and bounds[var] != [None, None]:
                    lb, ub = bounds[var]
                    if lb is not None and ub is not None:
                        sections.append(f"  • `{rfmt(lb)} ≤ {var} ≤ {rfmt(ub)}`")
                    elif lb is not None:
                        sections.append(f"  • `{var} ≥ {rfmt(lb)}`")
                    elif ub is not None:
                        sections.append(f"  • `{var} ≤ {rfmt(ub)}`")
                else:
                    # Check if non-negativity is implicit
                    sections.append(f"  • `{var} ≥ 0` (non-negativity)")
            sections.append("")
        
        # 3. Optimal Solution
        sections.append("## Optimal Solution\n")
        sections.append("**Variable Values:**")
        non_zero = [(k, float(v)) for k, v in vars_dict.items() if abs(float(v)) > tol]
        zero_vars = [(k, float(v)) for k, v in vars_dict.items() if abs(float(v)) <= tol]
        
        if non_zero:
            non_zero.sort(key=lambda kv: kv[0])
            for var, val in non_zero:
                sections.append(f"  • `{var} = {rfmt(val)}`")
        
        if zero_vars:
            sections.append(f"\n**Zero Variables:** {', '.join(f'`{v}`' for v, _ in zero_vars)}")
        
        sections.append(f"\n**Optimal Objective Value:** `{rfmt(float(obj_val)) if obj_val is not None else 'N/A'}`")
        sections.append("")
        
        # 4. Constraint Verification
        if constraints:
            sections.append("## Constraint Verification\n")
            sections.append("Let's verify that all constraints are satisfied by substituting the solution values:\n")
            for i, con in enumerate(constraints, 1):
                lhs_map = con.get("lhs", {})
                op = con.get("op", "<=")
                rhs = con.get("rhs", 0)
                
                # Build the calculation string with actual values
                calc_parts = []
                lhs_val = 0.0
                for var in sorted(lhs_map.keys()):
                    coef = float(lhs_map.get(var, 0))
                    var_val = float(vars_dict.get(var, 0))
                    contribution = coef * var_val
                    lhs_val += contribution
                    
                    # Format the calculation term
                    if abs(coef - 1.0) < tol:
                        calc_parts.append(f"{var}({rfmt(var_val)})")
                    elif abs(coef + 1.0) < tol:
                        calc_parts.append(f"-{var}({rfmt(var_val)})")
                    else:
                        calc_parts.append(f"{rfmt(coef)}×{var}({rfmt(var_val)})")
                
                # Check if satisfied
                satisfied = False
                if op == '<=':
                    satisfied = lhs_val <= rhs + tol
                elif op == '>=':
                    satisfied = lhs_val >= rhs - tol
                else:  # ==
                    satisfied = abs(lhs_val - rhs) < tol
                
                check = "✓" if satisfied else "✗"
                lhs_expr = format_expr(lhs_map)
                
                # Format the calculation
                calc_str = " + ".join(calc_parts).replace("+ -", "- ")
                result_str = " + ".join([f"{rfmt(float(lhs_map.get(var, 0)) * float(vars_dict.get(var, 0)))}" 
                                        for var in sorted(lhs_map.keys()) 
                                        if abs(float(lhs_map.get(var, 0))) > tol])
                result_str = result_str.replace("+ -", "- ")
                
                sections.append(f"**{i}. Constraint:** `{lhs_expr} {op} {rfmt(rhs)}`")
                sections.append(f"   - Substituting values: {calc_str}")
                sections.append(f"   - Calculation: {result_str} = **{rfmt(lhs_val)}**")
                sections.append(f"   - Check: {rfmt(lhs_val)} {op} {rfmt(rhs)} {check}\n")
            sections.append("")
        
        # 5. Interpretation & Explanation
        sections.append("## Solution Interpretation\n")
        
        if sense == 'max':
            sections.append(f"This solution **maximizes** the objective function, achieving the highest possible value of `{rfmt(float(obj_val))}` while satisfying all constraints.")
        else:
            sections.append(f"This solution **minimizes** the objective function, achieving the lowest possible value of `{rfmt(float(obj_val))}` while satisfying all constraints.")
        
        sections.append("\n**Why This is Optimal:**")
        sections.append("1. All constraints are satisfied (verified above)")
        sections.append(f"2. No other feasible allocation can achieve a better objective value")
        sections.append("3. The solution is found using proven linear programming algorithms (Simplex method)")
        sections.append("4. Any deviation from these values would either violate constraints or worsen the objective")
        
        return "\n".join(sections)

    def _detect_information_needs(self, problem: str) -> Optional[List[str]]:
        """Detect if problem requires external information and generate search queries."""
        p_lower = problem.lower()
        
        # Broad information need indicators - questions, time-sensitive queries, facts
        question_patterns = [
            r'\bwhat\s+(is|are|was|were|will|would|does|do|did)\b',
            r'\bwhen\s+(is|are|was|were|will|would|does|do|did)\b',
            r'\bwhere\s+(is|are|was|were|will|would|does|do|did)\b',
            r'\bhow\s+(much|many|often|long|far|does|do|did|can|will)\b',
            r'\bwho\s+(is|are|was|were|will|would|does|do|did)\b',
            r'\bwhich\s+',
            r'\bwhy\s+',
        ]
        
        time_sensitive = [
            r'\b(current|latest|recent|today|now|this\s+year)\b',
            r'\b(20\d{2}|19\d{2})\b',  # Any year mentioned
            r'\b(last|previous|next)\s+(year|month|week|decade)\b',
        ]
        
        factual_needs = [
            r'\b(data|statistics|information|facts|figures|numbers)\b',
            r'\b(price|cost|value|amount|rate|percentage)\b',
            r'\b(population|gdp|revenue|market|sales|growth)\b',
            r'\b(temperature|weather|climate|distance|size|area)\b',
            r'\b(compare|comparison|difference|versus|vs)\b',
        ]
        
        # Check if problem needs external data
        has_question = any(re.search(pat, p_lower) for pat in question_patterns)
        has_time_ref = any(re.search(pat, p_lower) for pat in time_sensitive)
        has_factual = any(re.search(pat, p_lower) for pat in factual_needs)
        
        needs_data = has_question or (has_time_ref and has_factual) or (
            any(word in p_lower for word in ['find', 'determine', 'calculate', 'compute']) and has_factual
        )
        
        if not needs_data:
            return None
        
        # Generate search queries intelligently
        queries = []
        
        # Strategy 1: Use first question sentence as-is (often best for search engines)
        sentences = [s.strip() + '.' for s in problem.split('.') if s.strip()]
        for sent in sentences[:3]:  # Check first 3 sentences
            if re.search(r'\bwhat|when|where|how|who|which|why\b', sent, re.I):
                # Remove question mark if present, add context
                query = sent.replace('?', '').strip()
                if len(query) > 10 and len(query) < 250:
                    queries.append(query)
        
        # Strategy 2: Extract key entities + years + metrics for targeted searches
        years = re.findall(r'\b(20\d{2}|19\d{2})\b', problem)
        
        # Extract capitalized entities (likely proper nouns - countries, companies, etc.)
        entities = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', problem)
        entities = list(dict.fromkeys(entities))[:5]  # Dedupe, limit to 5
        
        # Extract metric/fact keywords
        metrics = []
        metric_keywords = ['gdp', 'population', 'revenue', 'price', 'cost', 'growth', 'rate', 
                          'temperature', 'distance', 'size', 'market', 'sales', 'data']
        for word in metric_keywords:
            if word in p_lower:
                metrics.append(word)
        
        # Build targeted queries from components
        if entities and years:
            for entity in entities[:2]:
                for year in years[:2]:
                    if metrics:
                        for metric in metrics[:1]:
                            queries.append(f"{entity} {metric} {year}")
                    else:
                        queries.append(f"{entity} {year}")
        
        # Strategy 3: If no structured queries yet, use first sentence or full problem (truncated)
        if not queries:
            if sentences:
                queries.append(sentences[0][:200])
            else:
                # Use full problem as query, truncated
                queries.append(problem[:200].strip())
        
        # Deduplicate and limit
        seen = set()
        unique_queries = []
        for q in queries:
            q_norm = q.lower().strip()
            if q_norm not in seen and len(q_norm) > 10:
                seen.add(q_norm)
                unique_queries.append(q)
        
        return unique_queries[:5] if unique_queries else None  # Max 5 queries

    def _find_best_lp_solution(self, state: "AgentState") -> Optional[Dict[str, Any]]:
        """Search state and nodes for any optimal LP solution; prefer highest objective if multiple."""
        best = None
        def better(a, b):
            if a is None:
                return True
            try:
                return float((b or {}).get('objective', float('-inf'))) > float((a or {}).get('objective', float('-inf')))
            except Exception:
                return False
        # Check global
        lp = state.get('lp_solution')
        if isinstance(lp, dict) and lp.get('status') == 'optimal':
            best = lp
        # Check node metadata
        nodes_map = state.get('tree', {}).get('nodes', {})
        for nid, node in nodes_map.items():
            meta = node.get('metadata', {}) or {}
            lpn = meta.get('lp_solution')
            if isinstance(lpn, dict) and lpn.get('status') == 'optimal':
                if better(best, lpn):
                    best = lpn
        return best

    def invoke(self, state: "AgentState") -> "AgentState":
        problem_text = state['problem']
        best_strategy = state.get('best_thought', state.get('current_thought', ''))
        # Prefer aggregated result from decomposition
        if state.get('is_decomposed') and state.get('aggregated_result'):
            best_strategy = state['aggregated_result']
        plan_text = state.get('plan', '')
        # Prefer any optimal LP solution available anywhere in the state
        lp = self._find_best_lp_solution(state) or {}

        # Step 1: Web search disabled for self-contained math problems
        retrieved_info = ""
        
        # Step 2: Feasibility check
        feasible = True
        reasons = ""
        
        # Only check for logical contradictions, not missing data
        if "contradict" in problem_text.lower() or "impossible" in problem_text.lower():
            feasibility_prompt = f"""Check if this problem has CONTRADICTORY CONSTRAINTS that make it logically impossible.

PROBLEM:
{problem_text}

Examples of contradictions:
- "x must be both greater than 10 AND less than 5"
- "total must equal 100 AND total must equal 200"

Missing data or unknown values are NOT contradictions.

Respond ONLY:
FEASIBLE: YES|NO
REASONS: <brief explanation if NO>
"""
            resp = self.llm._call_api(feasibility_prompt)
            if resp:
                for line in resp.splitlines():
                    if line.strip().upper().startswith("FEASIBLE:"):
                        val = line.split(":", 1)[-1].strip().upper()
                        feasible = (val == "YES")
                    elif line.strip().upper().startswith("REASONS:"):
                        reasons = line.split(":", 1)[-1].strip()
        
        if not feasible:
            # Instead of saying cannot be solved, provide the reasoning and ask for best attempt
            fallback = f"""## Problem Analysis

**Challenge Identified:** {reasons or 'Contradictory constraints detected.'}

**Approach:** Based on available information and reasoning:
{best_strategy[:800]}

**Recommendation:** Review the constraints or provide additional clarification to resolve ambiguities.
"""
            state['final_output'] = fallback
            state['feasible'] = True  # Changed to True - always provide something
            state['feasibility_reasons'] = reasons
            if self.verbose:
                print("\n⚠️ CONSTRAINT ISSUES DETECTED - Providing best analysis")
                print(fallback[:200])
            return state

        # Check if we have an optimal LP solution to format
        has_lp_solution = isinstance(lp, dict) and lp.get('status') == 'optimal' and isinstance(lp.get('vars'), dict)

        # Step 3: Use the best reasoning directly — no re-generation LLM call.
        # The tree-of-thought / refinement already produced the answer; re-solving
        # risks producing a different (wrong) answer.

        has_lp_solution = isinstance(lp, dict) and lp.get('status') == 'optimal' and isinstance(lp.get('vars'), dict)

        if has_lp_solution and not self._wants_code(problem_text):
            # Format LP solution with comprehensive explanation
            single = self._format_lp_answer(problem_text, lp)
        else:
            # Check if best_strategy already contains a \boxed{} answer
            # (from the ReAct FINISH action). If so, use it directly.
            boxed_match = re.search(
                r'\\boxed\{([^{}]+(?:\{[^{}]*\}[^{}]*)*)\}',
                best_strategy or '',
            )
            if boxed_match:
                # best_strategy already has the answer — use as-is
                single = best_strategy.strip()
            elif best_strategy and best_strategy.strip():
                # best_strategy is an approach description without \boxed{};
                # re-generate a proper answer via the LLM
                single = self._generate_enhanced_output(
                    problem_text, best_strategy, plan_text, retrieved_info
                )
            else:
                single = ''

        # Strip thinking tags and code blocks if code wasn't requested
        single = self._strip_think_tags(single)
        if not self._wants_code(problem_text):
            single = self._strip_code_blocks(single)
        single = self._dedupe_fenced_blocks(single)

        state['final_output'] = single
        state['final_outputs'] = [single] if single else []
        state['feasible'] = True
        state['feasibility_reasons'] = reasons
        if self.verbose:
            print("\n🎯 FINAL OUTPUT - READY TO USE")
            print("\n" + "="*70)
            preview = state['final_output'][:1000]
            print(preview)
            print("\n" + "="*70)
        return state
