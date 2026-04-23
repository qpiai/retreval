"""Computation Tools for the ReTreVal Agent - Calculator, Linear Programming, and Python REPL."""

import ast
import cmath
import json
import math
import os
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from src.clients.llm_client import LLMClient

if TYPE_CHECKING:
    from src.agents.state import AgentState


class CalculatorTool:
    """Safely evaluates mathematical expressions using AST parsing."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    @staticmethod
    def _format_result(val: Union[float, complex]) -> str:
        """Format a numeric result for display."""
        if isinstance(val, complex):
            real, imag = val.real, val.imag
            # Clean up near-zero parts from floating point noise
            if abs(real) < 1e-10:
                real = 0.0
            if abs(imag) < 1e-10:
                imag = 0.0
            if imag == 0:
                return f"{real:g}"
            if real == 0:
                return f"{imag:g}i" if imag != 1 else "i"
            sign = "+" if imag > 0 else "-"
            abs_imag = abs(imag)
            imag_str = f"{abs_imag:g}i" if abs_imag != 1 else "i"
            return f"{real:g} {sign} {imag_str}"
        return f"{val:g}"

    # ---------- ReAct call() wrapper ----------
    def call(self, expression: str) -> str:
        """String-in/string-out wrapper for the ReAct executor."""
        try:
            result = self._safe_eval(expression)
            if result is not None:
                return f"{expression} = {self._format_result(result)}"
            return f"Could not evaluate expression: {expression}"
        except Exception as exc:
            return f"Calculator error: {exc}"
    
    # Helper: complex-aware versions of math functions
    # These accept both real and complex inputs seamlessly.
    @staticmethod
    def _complex_sqrt(x):
        if isinstance(x, complex) or (isinstance(x, (int, float)) and x < 0):
            return cmath.sqrt(x)
        return math.sqrt(x)

    @staticmethod
    def _complex_exp(x):
        return cmath.exp(x) if isinstance(x, complex) else math.exp(x)

    @staticmethod
    def _complex_log(x, *args):
        if isinstance(x, complex) or (isinstance(x, (int, float)) and x <= 0):
            return cmath.log(x, *args) if args else cmath.log(x)
        return math.log(x, *args) if args else math.log(x)

    @staticmethod
    def _complex_sin(x):
        return cmath.sin(x) if isinstance(x, complex) else math.sin(x)

    @staticmethod
    def _complex_cos(x):
        return cmath.cos(x) if isinstance(x, complex) else math.cos(x)

    @staticmethod
    def _complex_tan(x):
        return cmath.tan(x) if isinstance(x, complex) else math.tan(x)

    @staticmethod
    def _complex_asin(x):
        return cmath.asin(x) if isinstance(x, complex) else math.asin(x)

    @staticmethod
    def _complex_acos(x):
        return cmath.acos(x) if isinstance(x, complex) else math.acos(x)

    @staticmethod
    def _complex_atan(x):
        return cmath.atan(x) if isinstance(x, complex) else math.atan(x)

    @staticmethod
    def _complex_abs(x):
        """Return abs (modulus for complex numbers)."""
        return abs(x)

    @staticmethod
    def _make_complex(real, imag=0):
        """Helper to construct complex numbers: complex(a, b)."""
        return complex(real, imag)

    # Whitelisted math functions and constants available in expressions
    _MATH_NAMESPACE = {
        '__builtins__': None,
        'sqrt': _complex_sqrt.__func__,
        'sin': _complex_sin.__func__,
        'cos': _complex_cos.__func__,
        'tan': _complex_tan.__func__,
        'asin': _complex_asin.__func__,
        'acos': _complex_acos.__func__,
        'atan': _complex_atan.__func__,
        'atan2': math.atan2,
        'log': _complex_log.__func__,
        'log2': math.log2,
        'log10': math.log10,
        'exp': _complex_exp.__func__,
        'abs': _complex_abs.__func__,
        'ceil': math.ceil,
        'floor': math.floor,
        'factorial': math.factorial,
        'gcd': math.gcd,
        'complex': _make_complex.__func__,
        'conjugate': lambda x: x.conjugate() if isinstance(x, complex) else x,
        'phase': cmath.phase,
        'polar': cmath.polar,
        'rect': cmath.rect,
        'pi': math.pi,
        'e': math.e,
        'inf': math.inf,
        'j': 1j,
        'I': 1j,
        'i': 1j,
    }

    # Names allowed in AST validation
    _ALLOWED_NAMES = set(_MATH_NAMESPACE.keys()) - {'__builtins__'}

    # Functions allowed to be called
    _ALLOWED_CALLS = {
        'sqrt', 'sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'atan2',
        'log', 'log2', 'log10', 'exp', 'abs', 'ceil', 'floor',
        'factorial', 'gcd', 'complex', 'conjugate', 'phase', 'polar', 'rect',
    }

    def _safe_eval(self, expr: str) -> Optional[Union[float, complex]]:
        """Safely evaluate a mathematical expression using AST parsing.

        Supports math functions (sqrt, sin, cos, etc.), constants (pi, e),
        complex numbers (j, I, i, complex(a,b)), × and ÷ (converted to * and /),
        ^ as exponent, and decimals.
        Returns float for real results, complex for complex results, None on failure.
        """
        try:
            # Normalize common symbols
            expr = expr.strip()
            expr = expr.replace('×', '*').replace('÷', '/').replace('^', '**')
            # Strip thousands-separator commas (e.g. "1,000") but keep argument commas
            expr = re.sub(r'(\d),(\d)', r'\1\2', expr)
            # Convert n! notation to factorial(n): "7!" -> "factorial(7)",
            # "(4+3)!" -> "factorial((4+3))"
            expr = re.sub(r'(\d+)!', r'factorial(\1)', expr)
            expr = re.sub(r'\(([^)]+)\)!', r'factorial((\1))', expr)
            # Normalize common complex number notations:
            # "3+2i" -> "3+2*i", "3+2j" -> "3+2*j", but not inside function names
            expr = re.sub(r'(\d)([ij])\b', r'\1*\2', expr)

            # Parse AST and validate nodes
            node = ast.parse(expr, mode='eval')

            def _is_safe(n):
                # Allow expression wrapper
                if isinstance(n, ast.Expression):
                    return _is_safe(n.body)
                # Allow binary ops: +, -, *, /, **, %
                if isinstance(n, ast.BinOp):
                    return (_is_safe(n.left) and _is_safe(n.right)
                            and isinstance(n.op, (ast.Add, ast.Sub, ast.Mult,
                                                  ast.Div, ast.Pow, ast.Mod,
                                                  ast.FloorDiv)))
                # Allow unary +/-
                if isinstance(n, ast.UnaryOp):
                    return isinstance(n.op, (ast.UAdd, ast.USub)) and _is_safe(n.operand)
                # Allow numeric literals (int, float, complex)
                if isinstance(n, ast.Num):
                    return True
                if isinstance(n, ast.Constant) and isinstance(n.value, (int, float, complex)):
                    return True
                # Allow whitelisted names (pi, e, j, I, i, etc.)
                if isinstance(n, ast.Name) and n.id in self._ALLOWED_NAMES:
                    return True
                # Allow whitelisted function calls: sqrt(...), sin(...), complex(...), etc.
                if isinstance(n, ast.Call):
                    if (isinstance(n.func, ast.Name)
                            and n.func.id in self._ALLOWED_CALLS
                            and all(_is_safe(a) for a in n.args)
                            and not n.keywords):
                        return True
                    return False
                return False

            if not _is_safe(node):
                return None

            # Evaluate in a restricted namespace with math functions
            val = eval(compile(node, '<ast>', 'eval'), self._MATH_NAMESPACE)
            # Return complex if result has imaginary part, else float
            if isinstance(val, complex):
                if val.imag == 0:
                    return float(val.real)
                return val
            return float(val)
        except Exception:
            return None

    def _extract_and_evaluate(self, text: str) -> Dict[str, Any]:
        """Extract mathematical expressions from text and evaluate them.

        Returns a dict mapping keys to {'expression': expr, 'value': float}
        """
        results: Dict[str, Any] = {}
        # Normalize text
        txt = text.replace('×', '*').replace('÷', '/').replace('^', '**')
        txt = txt.replace('\u2013', '-').replace('\u2014', '-')

        # Collect candidate substrings
        candidates = []

        # 1) After 'Solution', 'Answer', 'Result', '=>', ':'
        for m in re.finditer(r'(?:(?:Solution|Answer|Result)[:\s]*)([^\n;]+)', txt, flags=re.I):
            cand = m.group(1).strip()
            # If contains '=', take left side
            if '=' in cand:
                cand = cand.split('=')[0].strip()
            candidates.append(cand)

        # 2) Parenthesized expressions
        for m in re.finditer(r'\(([^()]{1,120})\)', txt):
            candidates.append(m.group(1).strip())

        # 3) Standalone arithmetic-like substrings
        for m in re.finditer(r'([0-9\.\s\+\-\*\/\(\)\*\*]{3,})', txt):
            s = m.group(1).strip()
            if re.search(r'[0-9]', s) and re.search(r'[+\-*/]', s):
                candidates.append(s)

        # Deduplicate while preserving order
        seen = set()
        ordered = []
        for c in candidates:
            if c and c not in seen:
                seen.add(c)
                ordered.append(c)

        # Evaluate candidates
        idx = 0
        for expr in ordered:
            value = self._safe_eval(expr)
            if value is not None:
                key = f'calc_{idx}'
                results[key] = {'expression': expr, 'value': value}
                idx += 1

        return results

    def invoke(self, state: "AgentState") -> "AgentState":
        thought = state.get('current_thought', '')
        
        if not thought:
            return state
        
        # Check if there are mathematical expressions to evaluate
        calculations = self._extract_and_evaluate(thought)
        
        if calculations:
            # Append calculation results to the thought
            calc_summary = "\n\n[Calculator Results]"
            for var, data in calculations.items():
                calc_summary += f"\n  {data['expression']} = {self._format_result(data['value'])}"
            
            # Update the thought with calculations
            if "[Calculator Results]" not in thought:
                state['current_thought'] = thought + calc_summary
            
            # Update node in tree
            nid = state.get('current_node_id')
            nodes_map = state.get('tree', {}).get('nodes', {})
            if nid in nodes_map:
                node = nodes_map[nid]
                if "[Calculator Results]" not in node['thought']:
                    node['thought'] = node['thought'] + calc_summary
                node.setdefault('metadata', {})['calculations'] = calculations
            
            if self.verbose:
                print("\n🧮 CALCULATOR")
                for var, data in calculations.items():
                    print(f"  {data['expression']} = {data['value']:.6g}")
        else:
            if self.verbose:
                print("\n🧮 CALCULATOR (no expressions found)")
        
        return state


class LinearProgrammingTool:
    """Simplified LP tool using PuLP.

    Differences vs previous version:
    - Supports many variables without imposing any implicit per-variable bounds (variables are free by default).
    - Accepts a concise JSON LP schema if present in the problem/current thought; otherwise does nothing.
    - Emits a plain-text summary (not code) into state without fenced blocks.
    """

    def __init__(self, llm: LLMClient, verbose: bool = True):
        self.llm = llm
        self.verbose = verbose

    # ---------- ReAct call() wrapper ----------
    def call(self, description: str, state: dict = None) -> str:
        """String-in/string-out wrapper for the ReAct executor."""
        try:
            st = dict(state) if state else {'current_thought': description, 'problem': description}
            result_state = self.invoke(st)
            lp = result_state.get('lp_solution')
            if lp and isinstance(lp, dict) and lp.get('status') == 'optimal':
                return f"Optimal solution: {lp.get('vars', {})} with objective = {lp.get('objective', '?')}"
            return "LP solver: no optimal solution found or no LP formulation detected."
        except Exception as exc:
            return f"LP solver error: {exc}"

    def _extract_first_json_object(self, text: str) -> Optional[Dict[str, Any]]:
        """Best-effort: find and parse the first JSON object in text."""
        if not text:
            return None
        # Fast path: entire text is JSON
        try:
            obj = json.loads(text)
            return obj if isinstance(obj, dict) else None
        except Exception:
            pass

        # Scan for a balanced {...} block
        start = text.find('{')
        while start != -1:
            depth = 0
            for i in range(start, len(text)):
                ch = text[i]
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        candidate = text[start:i + 1]
                        try:
                            obj = json.loads(candidate)
                            return obj if isinstance(obj, dict) else None
                        except Exception:
                            break
            start = text.find('{', start + 1)
        return None

    def _parse_linear_expr(self, expr: str) -> Optional[Dict[str, float]]:
        """Parse simple linear expr like '2x + 3 y - z' -> {var: coeff}."""
        if not expr:
            return None
        s = expr.replace(',', ' ')
        tokens = re.finditer(r"([+\-]?)\s*([0-9]*\.?[0-9]*)\s*([A-Za-z_][A-Za-z0-9_]*)", s)
        coeffs: Dict[str, float] = {}
        found = False
        for m in tokens:
            found = True
            sign = -1.0 if (m.group(1) or '').strip() == '-' else 1.0
            coeff_str = (m.group(2) or '').strip()
            coeff = float(coeff_str) if coeff_str not in ('', None) else 1.0
            var = m.group(3)
            coeffs[var] = coeffs.get(var, 0.0) + sign * coeff
        return coeffs if found else None

    def _parse_canonical_lp(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse canonical LP text with an objective and linear constraints."""
        if not text:
            return None
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        sense = None
        objective_map: Dict[str, float] = {}
        constraints: List[Dict[str, Any]] = []
        bounds: Dict[str, List[Optional[float]]] = {}

        obj_re = re.compile(r"^(max(?:imize)?|min(?:imize)?)[\s:]+(.+)$", re.I)
        for ln in lines:
            mo = obj_re.match(ln)
            if mo:
                sense = 'max' if mo.group(1).lower().startswith('max') else 'min'
                parsed = self._parse_linear_expr(mo.group(2))
                if parsed:
                    objective_map = parsed
                break

        cons_re = re.compile(r"^(.+?)\s*(<=|>=|=)\s*([\-]?[0-9]+(?:\.[0-9]+)?)\s*$")
        for ln in lines:
            mo = cons_re.match(ln)
            if not mo:
                continue
            lhs = self._parse_linear_expr(mo.group(1))
            if not lhs:
                continue
            constraints.append({"lhs": lhs, "op": mo.group(2), "rhs": float(mo.group(3))})

        b1 = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*(>=|<=)\s*([\-]?[0-9]+(?:\.[0-9]+)?)$")
        b2 = re.compile(r"^([\-]?[0-9]+(?:\.[0-9]+)?)\s*<=\s*([A-Za-z_][A-Za-z0-9_]*)\s*<=\s*([\-]?[0-9]+(?:\.[0-9]+)?)$")
        for ln in lines:
            mo = b1.match(ln)
            if mo:
                var, op, num = mo.group(1), mo.group(2), float(mo.group(3))
                lb, ub = bounds.get(var, [None, None])
                if op == '>=':
                    lb = num if lb is None else max(lb, num)
                else:
                    ub = num if ub is None else min(ub, num)
                bounds[var] = [lb, ub]
                continue
            mo = b2.match(ln)
            if mo:
                lo, var, hi = float(mo.group(1)), mo.group(2), float(mo.group(3))
                bounds[var] = [lo, hi]

        if not objective_map:
            return None
        return {
            "sense": sense or 'max',
            "objective": objective_map,
            "constraints": constraints,
            "bounds": bounds,
        }

    def _parse_transportation_arrays(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse transportation arrays and build LP spec."""
        try:
            cap_m = re.search(r"capacities\s*=\s*(\[[\s\S]*?\])", text, re.I)
            dem_m = re.search(r"demands\s*=\s*(\[[\s\S]*?\])", text, re.I)
            cost_m = re.search(r"costs\s*=\s*(\[[\s\S]*?\])", text, re.I)
            if not (cap_m and dem_m and cost_m):
                return None
            capacities = ast.literal_eval(cap_m.group(1))
            demands = ast.literal_eval(dem_m.group(1))
            costs = ast.literal_eval(cost_m.group(1))
            if not isinstance(capacities, list) or not isinstance(demands, list) or not isinstance(costs, list):
                return None
            m = len(capacities)
            n = len(demands)
            if m <= 0 or n <= 0:
                return None
            # costs must be m x n
            if len(costs) != m or any((not isinstance(row, list) or len(row) != n) for row in costs):
                return None

            # Build objective and constraints
            objective: Dict[str, float] = {}
            constraints: List[Dict[str, Any]] = []
            bounds: Dict[str, List[Optional[float]]] = {}

            # Variables x_i_j >= 0
            for i in range(m):
                for j in range(n):
                    name = f"x_{i}_{j}"
                    objective[name] = float(costs[i][j])
                    bounds[name] = [0.0, None]

            # Supply constraints: sum_j x_i_j <= capacities[i]
            for i in range(m):
                lhs = {f"x_{i}_{j}": 1.0 for j in range(n)}
                constraints.append({"lhs": lhs, "op": "<=", "rhs": float(capacities[i])})

            # Demand constraints: sum_i x_i_j = demands[j]
            for j in range(n):
                lhs = {f"x_{i}_{j}": 1.0 for i in range(m)}
                constraints.append({"lhs": lhs, "op": "=", "rhs": float(demands[j])})

            return {
                "sense": "min",
                "objective": objective,
                "constraints": constraints,
                "bounds": bounds,
            }
        except Exception:
            return None

    def _is_lp_candidate(self, text: str) -> bool:
        """Heuristic routing for LP detection."""
        if not text:
            return False
        t = text.lower()
        lp_keywords = [
            "minimize", "maximize", "subject to", "constraints", "bounds",
            "factory", "plant", "warehouse", "supply", "demand",
            "cost per unit", "profit", "shipping cost", "production limit",
            "resources", "allocation", "units", "transportation",
        ]

        has_keyword = any(k in t for k in lp_keywords)
        has_number = any(c.isdigit() for c in t)

        # Also treat explicit canonical forms as LPs
        if re.search(r"\b(max(?:imize)?|min(?:imize)?)\b", t):
            return True

        # Trigger LP when keywords and numeric data present
        if has_keyword and has_number:
            return True

        return False

    def _extract_transportation_via_llm(self, text: str) -> Optional[Dict[str, Any]]:
        """Use LLM to extract transportation problem arrays as a fallback."""
        prompt = f"""From the text below, extract the supply capacities, demands, and cost matrix.
Return ONLY a JSON object with keys "capacities", "demands", and "costs".
- "capacities" should be a list of supply numbers.
- "demands" should be a list of demand numbers.
- "costs" should be a list of lists, representing the cost matrix.

If you cannot find these details, return an empty JSON object {{}}.

TEXT:
{text}
"""
        try:
            resp = self.llm._call_api(prompt)
            data = json.loads(resp) if resp else {}
            if not isinstance(data, dict) or not all(k in data for k in ['capacities', 'demands', 'costs']):
                return None

            # Use the data to call the array parser
            array_text = f"capacities = {data['capacities']}\ndemands = {data['demands']}\ncosts = {data['costs']}"
            return self._parse_transportation_arrays(array_text)
        except Exception:
            return None

    def _extract_lp_via_llm(self, text: str) -> Optional[Dict[str, Any]]:
        """Ask the LLM to extract an LP JSON spec as a last resort."""
        prompt = f"""Extract a linear programming problem from the following text.

Return ONLY a JSON object (no markdown, no code fences) with this exact schema:
{{
  "sense": "max" | "min",
  "objective": {{"var": coefficient, ...}},
  "constraints": [
    {{"lhs": {{"var": coefficient, ...}}, "op": "<="|">="|"=", "rhs": number}},
    ...
  ],
  "bounds": {{"var": [lower_or_null, upper_or_null], ...}}
}}

CRITICAL INSTRUCTIONS:
1. If percentage/ratio constraints exist (e.g., "X% must be type A"), convert them to linear form with variables on BOTH sides of inequality
2. Calculate all cost coefficients based on the actual units in the problem (consider time periods, rates, premiums)
3. Ensure all constraints use consistent units
4. Set appropriate bounds for all variables (typically [0, null] for non-negative quantities)

If there is no clear LP, return an empty JSON object {{}}.

TEXT:
{text}
"""
        try:
            resp = self.llm._call_api(prompt)
            data = json.loads(resp) if resp else {}
            if isinstance(data, dict) and data:
                return data
            return None
        except Exception:
            return None

    def _validate_lp_spec(self, spec: Dict[str, Any], text: str) -> Dict[str, Any]:
        """Validate and potentially fix common LP formulation errors."""
        if not isinstance(spec, dict):
            return spec
            
        obj_map = spec.get("objective", {})
        constraints = spec.get("constraints", [])
        bounds = spec.get("bounds", {})
        
        # Collect all variables mentioned
        all_vars = set(obj_map.keys())
        for c in constraints:
            lhs = c.get("lhs", {})
            all_vars.update(lhs.keys())
        
        # Ensure all constraint variables are in bounds
        for var in all_vars:
            if var not in bounds:
                bounds[var] = [0, None]  # Default non-negativity
        
        spec["bounds"] = bounds
        
        if self.verbose:
            print("\n📋 LP PROBLEM FORMULATION:")
            sense = spec.get("sense", "max")
            print(f"   Objective: {sense.upper()}")
            for var, coef in sorted(obj_map.items()):
                print(f"      {coef:+.6g} * {var}")
            print(f"\n   Constraints ({len(constraints)}):")
            for i, c in enumerate(constraints, 1):
                lhs = c.get("lhs", {})
                op = c.get("op", "<=")
                rhs = c.get("rhs", 0)
                lhs_str = " + ".join([f"{coef:+.6g}*{var}" for var, coef in sorted(lhs.items())])
                print(f"      {i}. {lhs_str} {op} {rhs}")
            print(f"\n   Bounds:")
            for var in sorted(all_vars):
                b = bounds.get(var, [0, None])
                lb = b[0] if b[0] is not None else "-inf"
                ub = b[1] if b[1] is not None else "+inf"
                print(f"      {lb} <= {var} <= {ub}")
        
        return spec
    
    def _solve_with_pulp(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        try:
            from pulp import LpProblem, LpVariable, LpMaximize, LpMinimize, lpSum, PULP_CBC_CMD, LpStatusOptimal, value
        except Exception as e:
            return {"status": "error", "error": f"PuLP unavailable: {e}"}

        sense = (spec.get("sense") or "max").strip().lower()
        obj_map: Dict[str, float] = spec.get("objective", {}) or {}
        constraints = spec.get("constraints", []) or []
        bounds = spec.get("bounds", {}) or {}

        if not obj_map:
            return {"status": "no-problem", "error": "Missing objective"}

        # Collect ALL variables from objective and constraints
        all_vars = set(obj_map.keys())
        for c in constraints:
            lhs_map = c.get("lhs", {}) or {}
            all_vars.update(lhs_map.keys())
        
        # Create variables with bounds
        vars: Dict[str, Any] = {}
        for name in sorted(all_vars):
            lb = None
            ub = None
            if name in bounds and isinstance(bounds[name], list) and len(bounds[name]) == 2:
                lb, ub = bounds[name][0], bounds[name][1]
            # Default to non-negative if no bounds specified
            if lb is None and ub is None:
                lb = 0
            vars[name] = LpVariable(name, lowBound=lb, upBound=ub)

        # Build problem
        prob = LpProblem("LinearProgram", LpMaximize if sense.startswith("max") else LpMinimize)

        # Objective - ensure all variables in objective exist
        obj_terms = []
        for name, coeff in obj_map.items():
            if name in vars:
                obj_terms.append(float(coeff) * vars[name])
        
        if not obj_terms:
            return {"status": "no-problem", "error": "No valid objective terms"}
        
        prob += lpSum(obj_terms)

        # Constraints
        for i, c in enumerate(constraints):
            lhs_map = c.get("lhs", {}) or {}
            op = (c.get("op") or "<=").strip()
            rhs = float(c.get("rhs", 0))
            
            # Build constraint expression
            constraint_terms = []
            for name, coeff in lhs_map.items():
                if name in vars:
                    constraint_terms.append(float(coeff) * vars[name])
            
            if not constraint_terms:
                continue  # Skip empty constraints
            
            expr = lpSum(constraint_terms)
            
            # Add constraint with a name for debugging
            constraint_name = f"C{i+1}"
            if op == "<=":
                prob += (expr <= rhs, constraint_name)
            elif op == ">=":
                prob += (expr >= rhs, constraint_name)
            else:  # ==
                prob += (expr == rhs, constraint_name)

        # Solve with PuLP's CBC solver
        solver = PULP_CBC_CMD(msg=False)
        status_code = prob.solve(solver)
        
        # Check if optimal
        is_optimal = (status_code == LpStatusOptimal) or (str(prob.status) == "1")
        
        if self.verbose:
            print(f"   PuLP Status: {prob.status} (code={status_code})")
        
        if not is_optimal:
            # Provide more detailed error information
            status_msg = str(prob.status)
            return {
                "status": "infeasible_or_unbounded", 
                "objective": None, 
                "vars": {}, 
                "spec": spec,
                "solver_status": status_msg
            }

        # Extract solution
        sol = {}
        for name, var in vars.items():
            try:
                sol[name] = float(value(var))
            except:
                sol[name] = 0.0
        
        try:
            obj_val = float(value(prob.objective))
        except:
            obj_val = 0.0
        
        return {"status": "optimal", "objective": obj_val, "vars": sol, "spec": spec}

    def invoke(self, state: "AgentState") -> "AgentState":
        # Prefer problem + current thought; attempt to find a JSON LP spec
        problem_text = state.get("problem") or ""
        thought_text = state.get("current_thought") or ""
        combined = (problem_text + "\n\n" + thought_text).strip() or problem_text

        # Quick heuristic: route to LP only if text looks like an LP problem
        if not self._is_lp_candidate(combined):
            if self.verbose:
                print("LP: input does not appear to describe an LP problem; skipping.")
            return state

        # 1) Embedded JSON object
        spec = self._extract_first_json_object(combined)
        # 2) Canonical maximize/minimize form
        if not spec:
            spec = self._parse_canonical_lp(combined)
        # 3) Transportation-style arrays (capacities, demands, costs)
        if not spec:
            spec = self._parse_transportation_arrays(combined)
        # 4) Fallback to LLM for transportation arrays
        if not spec:
            spec = self._extract_transportation_via_llm(combined)
        # 5) As a last resort, ask the LLM to extract a generic JSON LP
        if not spec:
            spec = self._extract_lp_via_llm(combined)
        if not spec:
            if self.verbose:
                print("LP: No LP specification detected; skipping.")
            return state

        # Validate and potentially fix the spec
        spec = self._validate_lp_spec(spec, combined)
        
        result = self._solve_with_pulp(spec)

        # Keep previous optimal result if the new one isn't optimal
        prev = state.get("lp_solution")
        if isinstance(prev, dict) and prev.get("status") == "optimal" and result.get("status") != "optimal":
            pass
        else:
            state["lp_solution"] = result

        # Attach to node metadata
        nodes_map = state.get("tree", {}).get("nodes", {})
        nid = state.get("current_node_id")
        if nid in nodes_map:
            nodes_map[nid].setdefault("metadata", {})["lp_solution"] = result

        # Debug output
        if self.verbose:
            print("\n📐 LP SOLVER RESULT")
            if result.get("status") == "optimal":
                print(f"   Status: ✓ Optimal")
                print(f"   Objective Value: {result['objective']:.6g}")
                print(f"   Solution:")
                for var, val in sorted(result.get('vars', {}).items()):
                    if abs(val) > 1e-6:
                        print(f"      {var} = {val:.6g}")
            else:
                print(f"   Status: ✗ {result.get('status', 'unknown')}")
                if 'solver_status' in result:
                    print(f"   Solver Message: {result['solver_status']}")
        return state


class PythonREPLTool:
    """Python REPL tool for equation solving and expression simplification using SymPy."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.has_sympy = self._check_sympy()

    # ---------- ReAct call() wrapper ----------
    def call(self, equation: str) -> str:
        """String-in/string-out wrapper for the ReAct executor.

        Supports single equations ('2*x + 3 = 7') and systems of equations
        separated by commas ('x + y = 2, x - y = 0').
        """
        try:
            # Detect system of equations: multiple "= ..." segments separated by commas
            # e.g. "x + y - 2 = 0, z - y - 1 = 0, z - x - 2 = 0"
            parts = [p.strip() for p in equation.split(',') if '=' in p.strip()]
            if len(parts) > 1:
                result = self.solve_system(parts)
                if result['success']:
                    return f"Solutions for system: {result['solutions']}"
                return f"Could not solve system: {result.get('error', 'unknown error')}"

            result = self.solve_equation(equation)
            if result['success']:
                return f"Solutions for '{equation}': {result['solutions']}"
            return f"Could not solve: {result.get('error', 'unknown error')}"
        except Exception as exc:
            return f"Equation solver error: {exc}"
    
    def _check_sympy(self) -> bool:
        """Check if SymPy is available."""
        try:
            import sympy  # type: ignore
            return True
        except ImportError:
            if self.verbose:
                print("⚠️ SymPy not available - equation solving and simplification disabled")
            return False
    
    def solve_equation(self, equation: str) -> Dict[str, Any]:
        """Solve an equation using SymPy.

        Auto-detects the variable to solve for (any single letter that isn't
        a known constant like e). Supports pi and e as constants.
        """
        if not self.has_sympy:
            return {'success': False, 'solutions': [], 'error': 'SymPy not available'}

        try:
            import sympy  # type: ignore
            from sympy import symbols, solve, sympify, pi, E  # type: ignore
            from sympy.parsing.sympy_parser import (  # type: ignore
                parse_expr, standard_transformations,
                implicit_multiplication_application,
                convert_xor,
            )

            equation = equation.strip()

            # Build local dict with common constants
            local_dict: Dict[str, Any] = {'pi': pi, 'e': E}

            # Auto-detect free variables (single letters not in constants)
            # Extract all alphabetic tokens
            tokens = set(re.findall(r'\b([a-zA-Z_]\w*)\b', equation))
            known = {'pi', 'e', 'sin', 'cos', 'tan', 'log', 'sqrt', 'exp',
                     'asin', 'acos', 'atan', 'abs', 'factorial'}
            free_vars = sorted(tokens - known)

            # Create symbols for free variables
            sym_vars = {name: symbols(name) for name in free_vars}
            local_dict.update(sym_vars)

            transformations = standard_transformations + (
                implicit_multiplication_application, convert_xor,
            )

            # Handle equation format: "expression = value" or just "expression"
            if '=' in equation:
                left, right = equation.split('=', 1)
                left_expr = parse_expr(left.strip(), local_dict=local_dict,
                                       transformations=transformations)
                right_expr = parse_expr(right.strip(), local_dict=local_dict,
                                        transformations=transformations)
                expr = left_expr - right_expr
            else:
                expr = parse_expr(equation, local_dict=local_dict,
                                  transformations=transformations)

            # Decide which variable to solve for
            expr_free = expr.free_symbols
            if len(expr_free) == 1:
                solve_for = list(expr_free)[0]
            elif sym_vars:
                # Prefer the first user variable found
                solve_for = list(sym_vars.values())[0]
            else:
                solve_for = symbols('x')

            solutions = solve(expr, solve_for)

            return {
                'success': True,
                'solutions': [str(sol) for sol in solutions],
                'error': None
            }
        except Exception as e:
            return {
                'success': False,
                'solutions': [],
                'error': str(e)
            }
    
    def solve_system(self, equations: List[str]) -> Dict[str, Any]:
        """Solve a system of equations using SymPy.

        Args:
            equations: list of equation strings, e.g. ["x + y = 2", "x - y = 0"]
        """
        if not self.has_sympy:
            return {'success': False, 'solutions': [], 'error': 'SymPy not available'}

        try:
            import sympy
            from sympy import symbols, solve
            from sympy.parsing.sympy_parser import (
                parse_expr, standard_transformations,
                implicit_multiplication_application,
                convert_xor,
            )

            transformations = standard_transformations + (
                implicit_multiplication_application, convert_xor,
            )

            known = {'pi', 'e', 'sin', 'cos', 'tan', 'log', 'sqrt', 'exp',
                      'asin', 'acos', 'atan', 'abs', 'factorial'}
            local_dict: Dict[str, Any] = {'pi': sympy.pi, 'e': sympy.E}

            # Collect all free variables across all equations
            all_tokens: set = set()
            for eq in equations:
                all_tokens |= set(re.findall(r'\b([a-zA-Z_]\w*)\b', eq))
            free_vars = sorted(all_tokens - known)
            sym_vars = {name: symbols(name) for name in free_vars}
            local_dict.update(sym_vars)

            # Parse each equation into a SymPy expression (lhs - rhs)
            exprs = []
            for eq in equations:
                eq = eq.strip()
                if '=' in eq:
                    left, right = eq.split('=', 1)
                    left_expr = parse_expr(left.strip(), local_dict=local_dict,
                                           transformations=transformations)
                    right_expr = parse_expr(right.strip(), local_dict=local_dict,
                                            transformations=transformations)
                    exprs.append(left_expr - right_expr)
                else:
                    exprs.append(parse_expr(eq, local_dict=local_dict,
                                            transformations=transformations))

            solve_for = [sym_vars[v] for v in free_vars if v in sym_vars]
            solutions = solve(exprs, solve_for, dict=True)

            if solutions:
                formatted = []
                for sol in solutions:
                    formatted.append({str(k): str(v) for k, v in sol.items()})
                return {'success': True, 'solutions': formatted, 'error': None}
            return {'success': False, 'solutions': [], 'error': 'No solution found'}
        except Exception as e:
            return {'success': False, 'solutions': [], 'error': str(e)}

    def simplify_expression(self, expression: str) -> Dict[str, Any]:
        """Simplify a mathematical expression using SymPy."""
        if not self.has_sympy:
            return {'success': False, 'simplified': None, 'error': 'SymPy not available'}
        
        try:
            from sympy import sympify, simplify  # type: ignore
            
            expression = expression.strip()
            expr = sympify(expression)
            simplified = simplify(expr)
            
            return {
                'success': True,
                'simplified': str(simplified),
                'original': expression,
                'error': None
            }
        except Exception as e:
            return {
                'success': False,
                'simplified': None,
                'error': str(e)
            }
    
    def evaluate(self) -> Dict[str, Any]:
        """Check if the REPL tool is functional."""
        return {
            'status': 'ok',
            'has_sympy': self.has_sympy,
            'available_operations': ['solve_equation', 'simplify_expression'] if self.has_sympy else []
        }


class CombinatoricsTool:
    """Combinatorics and exact-fraction arithmetic for counting & probability problems."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def call(self, expression: str) -> str:
        """Evaluate a combinatorics / probability expression and return an exact fraction.

        Supports: C(n,k), P(n,k), factorial(n), n!, and standard arithmetic.
        Results are returned as exact fractions (e.g. 11/36) rather than decimals.
        """
        try:
            from fractions import Fraction
            import math

            expr = expression.strip()
            # Normalize notation
            expr = expr.replace('^', '**')
            # Convert n! to factorial(n)
            expr = re.sub(r'(\d+)!', r'factorial(\1)', expr)
            # Convert C(n,k) / nCr(n,k) to comb(n,k)
            expr = re.sub(r'(?:C|nCr|binom)\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)', r'comb(\1,\2)', expr)
            # Convert P(n,k) / nPr(n,k) to perm(n,k)
            expr = re.sub(r'(?:P|nPr)\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)', r'perm(\1,\2)', expr)

            # Wrap int-returning functions to return Fraction
            def _frac_factorial(n):
                return Fraction(math.factorial(int(n)))

            def _frac_comb(n, k):
                return Fraction(math.comb(int(n), int(k)))

            def _frac_perm(n, k):
                return Fraction(math.perm(int(n), int(k)))

            safe_ns = {
                '__builtins__': None,
                'Fraction': Fraction,
                'factorial': _frac_factorial,
                'comb': _frac_comb,
                'perm': _frac_perm,
            }

            # Wrap bare integer literals with Fraction() for exact arithmetic,
            # but skip integers already inside function calls.
            # Strategy: first protect function args, then wrap remaining ints.
            frac_expr = expr
            # Replace bare integer literals not preceded by function-call context
            # We split on known functions and wrap numbers outside of them
            # Simpler: just wrap ALL integers, the functions above accept Fraction args
            frac_expr = re.sub(r'(?<!\w)(\d+)(?!\w)', r'Fraction(\1)', frac_expr)

            result = eval(frac_expr, safe_ns)  # noqa: S307

            if isinstance(result, Fraction):
                if result.denominator == 1:
                    return f"{expression} = {result.numerator}"
                return f"{expression} = {result.numerator}/{result.denominator}"
            return f"{expression} = {result}"
        except Exception as exc:
            return f"Combinatorics error: {exc}"
