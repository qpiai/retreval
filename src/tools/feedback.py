"""
Feedback Loop Tools
Handles validation, failure analysis, and retry control for improved agent performance
"""

import os
import re
from typing import Dict, Any, Optional

from src.clients.llm_client import LLMClient
from src.utils.memory import ReflexionMemory


class ValidationTool:
    """Validate predicted answer against expected answer."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def _strip_think_tags(self, text: str) -> str:
        """Remove <think>...</think> blocks from thinking models."""
        return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

    def _extract_answer_from_output(self, text: str) -> str:
        """Extract the mathematical answer from model output using multiple strategies."""
        if not text:
            return ""

        # Strip thinking model tags first
        text = self._strip_think_tags(text)

        # Strategy 1: \boxed{answer} — MATH dataset standard format
        boxed_patterns = [
            r'\\boxed\{([^{}]+(?:\{[^{}]*\}[^{}]*)*)\}',
            r'\\boxed\s*\{([^}]+)\}',
            r'boxed\s*\{([^}]+)\}',
        ]
        for pattern in boxed_patterns:
            matches = re.findall(pattern, text)
            if matches:
                return matches[-1].strip()

        # Strategy 2: #### format (GSM8K style)
        if "####" in text:
            final_part = text.split("####")[-1].strip()
            numbers = re.findall(r'[-\d\.]+', final_part)
            if numbers:
                return numbers[0].strip()

        # Strategy 3: "the answer is X" / "final answer: X" patterns
        answer_patterns = [
            r'(?:the\s+)?(?:final\s+)?answer\s+is\s*[:\s]*\$?\s*\\?(.+?)(?:\.|$)',
            r'(?:final\s+)?answer\s*[:=]\s*\$?\s*\\?(.+?)(?:\.|$)',
            r'therefore[,\s]+(?:the\s+)?(?:answer\s+is\s+)?\$?\s*\\?(.+?)(?:\.|$)',
        ]
        for pattern in answer_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                answer = match.group(1).strip()
                # Clean up markdown/LaTeX artifacts
                answer = re.sub(r'^[\*\s\$]+|[\*\s\$]+$', '', answer)
                if answer and len(answer) > 0 and answer not in ['.', ',', '-', '*', '**', '$$']:
                    return answer

        # Strategy 4: Look for last equation result  = <value>
        eq_matches = re.findall(r'=\s*([-\d\./\\a-zA-Z{}\(\)\s\^_]+?)\s*(?:\.|$|\n)', text)
        if eq_matches:
            answer = eq_matches[-1].strip()
            answer = re.sub(r'^[\*\s\$]+|[\*\s\$]+$', '', answer)
            if answer and answer not in ['.', ',', '-', '*', '**', '$$']:
                return answer

        return ""

    def _normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison (handle formatting differences)."""
        if not answer:
            return ""
        ans = str(answer).strip()
        # Remove LaTeX formatting
        ans = ans.replace('\\left', '').replace('\\right', '')
        ans = ans.replace('\\text{', '').replace('\\mathrm{', '')
        # Convert \frac{a}{b} to a/b BEFORE stripping braces
        ans = re.sub(r'\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}', r'(\1)/(\2)', ans)
        # Convert \sqrt{x} to sqrt(x)
        ans = re.sub(r'\\sqrt\s*\{([^{}]+)\}', r'sqrt(\1)', ans)
        ans = re.sub(r'\\([a-zA-Z]+)', r'\1', ans)  # \pi -> pi, \sqrt -> sqrt
        ans = ans.replace('{', '').replace('}', '')
        ans = ans.replace('$', '').replace('€', '').replace('£', '')
        ans = ans.replace(' ', '').lower()
        # Remove common suffixes
        for suffix in ['dollars', 'cents', 'apples', 'oranges', 'items', 'units']:
            if ans.endswith(suffix):
                ans = ans[:-len(suffix)].strip()
        return ans

    def _try_numeric_eval(self, expr: str) -> Optional[float]:
        """Try to evaluate a normalized answer string as a number.

        Handles fractions like '(448)/(15625)', 'sqrt(5)', 'pi', etc.
        """
        try:
            return float(expr)
        except (ValueError, TypeError):
            pass
        try:
            import math
            safe_dict = {
                '__builtins__': None,
                'sqrt': math.sqrt, 'pi': math.pi, 'e': math.e,
                'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
                'log': math.log, 'abs': abs,
            }
            val = eval(expr, safe_dict)  # noqa: S307
            return float(val)
        except Exception:
            return None

    def invoke(self, state: dict) -> dict:
        """Check if predicted answer matches expected answer."""
        expected = state.get('expected_answer', '')
        predicted_raw = state.get('final_output', '')

        # Extract answer using robust multi-strategy extraction
        predicted = self._extract_answer_from_output(predicted_raw)

        # Fallback: if extraction failed, try the raw text (truncated)
        if not predicted:
            predicted = predicted_raw.strip()[:200]

        # Normalize both for comparison
        expected_norm = self._normalize_answer(expected)
        predicted_norm = self._normalize_answer(predicted)

        is_correct = expected_norm == predicted_norm

        # Try numerical comparison if direct match fails
        if not is_correct and expected_norm and predicted_norm:
            exp_num = self._try_numeric_eval(expected_norm)
            pred_num = self._try_numeric_eval(predicted_norm)
            if exp_num is not None and pred_num is not None:
                is_correct = abs(pred_num - exp_num) < 0.01

        # Try substring matching for text answers
        if not is_correct and expected_norm and predicted_norm:
            if expected_norm in predicted_norm or predicted_norm in expected_norm:
                is_correct = True

        state['is_correct'] = is_correct
        state['predicted_answer'] = predicted

        if self.verbose:
            status = "✓ CORRECT" if is_correct else "✗ WRONG"
            print(f"\n📊 VALIDATION: {status}")
            print(f"   Expected: {expected}")
            print(f"   Predicted: {predicted}")

        return state


class FailureAnalyzer:
    """Analyze why a solution failed and extract root cause."""
    
    def __init__(self, llm: LLMClient, memory: ReflexionMemory, verbose: bool = True):
        self.llm = llm
        self.memory = memory
        self.verbose = verbose
    
    def invoke(self, state: dict) -> dict:
        """Analyze failure and extract root cause."""
        problem = state.get('problem', '')
        expected = state.get('expected_answer', '')
        predicted = state.get('predicted_answer', '')
        best_thought = state.get('best_thought', '')
        
        if not problem or not predicted or expected == predicted:
            return state
        
        # Use LLM to analyze the failure
        analysis_prompt = f"""Analyze why this solution was incorrect and identify the root cause.

Problem: {problem}

Expected Answer: {expected}
Predicted Answer: {predicted}

Solution Approach: {best_thought[:500]}

Provide analysis in this format:
ROOT CAUSE: [One of: arithmetic_error, logic_error, incomplete_reasoning, wrong_constraint, off_by_one, missing_step, other]
EXPLANATION: [2-3 sentences explaining what went wrong]
LESSON: [How to avoid this in future similar problems]
APPROACH_TYPE: [algebraic, arithmetic, pattern_matching, brute_force, other]
"""
        
        analysis = self.llm._call_api(analysis_prompt)
        
        # Extract structured failure info
        failure_type = "unknown"
        if "ROOT CAUSE:" in analysis:
            for line in analysis.split('\n'):
                if line.startswith("ROOT CAUSE:"):
                    failure_type = line.split("ROOT CAUSE:")[-1].strip().lower()
                    break
        
        # Extract approach type
        approach_type = "unknown"
        if "APPROACH_TYPE:" in analysis:
            for line in analysis.split('\n'):
                if line.startswith("APPROACH_TYPE:"):
                    approach_type = line.split("APPROACH_TYPE:")[-1].strip().lower()
                    break
        
        # Store failure in memory
        failure_detail = {
            'problem_id': state.get('problem_id', ''),
            'expected': expected,
            'predicted': predicted,
            'failure_type': failure_type,
            'approach': approach_type,
            'analysis': analysis
        }
        
        # Normalize failure_type: strip markdown formatting and whitespace
        failure_type = re.sub(r'[*_`]', '', failure_type).strip()

        self.memory.record_failure(failure_type, failure_detail)
        self.memory.update_approach_reliability(approach_type, False)  # Mark as failure

        # TextGrad-inspired: apply failure gradients to active memory entries
        failure_context = f"Problem {state.get('problem_id', '?')}: {failure_type} — expected {expected}, got {predicted}"
        self.memory.apply_gradients_on_failure(failure_context, llm_callable=self.llm._call_api)

        state['error_analysis'] = analysis
        state['failure_type'] = failure_type
        state['approach_type'] = approach_type
        
        if self.verbose:
            print(f"\n🔍 FAILURE ANALYSIS")
            print(f"   Type: {failure_type}")
            print(f"   Approach: {approach_type}")
            print(f"   {analysis[:200]}...")
        
        return state


class FeedbackLoopController:
    """Control retry logic and decide when to continue iterating."""
    
    def __init__(self, memory: ReflexionMemory = None, verbose: bool = True):
        self.memory = memory
        self.verbose = verbose
    
    def invoke(self, state: dict) -> dict:
        """Check if we should retry and update iteration counter."""
        is_correct = state.get('is_correct', False)
        feedback_loops = state.get('feedback_loops', 0)
        max_loops = int(os.getenv('MAX_FEEDBACK_LOOPS', 1))
        
        should_retry = (not is_correct) and (feedback_loops < max_loops)
        
        state['feedback_loops'] = feedback_loops + 1
        state['should_retry'] = should_retry
        
        # Record success in persistent memory
        if is_correct and self.memory:
            problem_id = state.get('problem_id', '')
            predicted = state.get('predicted_answer', '')
            approach_type = state.get('approach_type', 'unknown')
            best_thought = state.get('best_thought', '')[:300]
            self.memory.add_success(
                f"Problem {problem_id}: approach '{approach_type}' produced correct answer '{predicted}'"
            )
            self.memory.record_success(approach_type, {
                'problem_id': problem_id,
                'predicted': predicted,
                'approach': approach_type,
                'thought_summary': best_thought,
            })
            self.memory.update_approach_reliability(approach_type, True)
            # TextGrad-inspired: apply success gradients to active memory entries
            self.memory.apply_gradients_on_success()

        if self.verbose:
            print(f"\n⚙️ FEEDBACK LOOP CONTROL")
            print(f"   Correct: {is_correct}")
            print(f"   Loop {feedback_loops + 1}/{max_loops}")
            print(f"   Should retry: {should_retry}")
        
        return state


class BacktrackingTool:
    """
    Implement tree backtracking when the best node fails to answer correctly.
    
    Backtracking logic:
    1. When best node fails validation, go back to its parent
    2. Find the next best sibling (by combined_score)
    3. Store failure context in failed_nodes list
    4. Continue expanding from the next best sibling
    5. New node uses failure context for better refinement/critique
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
    
    def _get_node_from_tree(self, tree: dict, node_id: str) -> dict:
        """Safely get a node from the tree structure."""
        if node_id == 'root':
            return tree.get('root')
        return tree.get('nodes', {}).get(node_id)
    
    def _get_siblings(self, tree: dict, node_id: str, parent_id: str) -> list:
        """Get all siblings of a node (nodes with same parent)."""
        parent = self._get_node_from_tree(tree, parent_id)
        if not parent:
            return []
        
        sibling_ids = parent.get('children', [])
        return [self._get_node_from_tree(tree, sid) for sid in sibling_ids if sid != node_id]
    
    def _get_best_unexplored_sibling(self, siblings: list, failing_node: dict, explored_nodes: list) -> dict:
        """
        Get the sibling with the best combined_score that hasn't been explored yet.
        
        Returns:
            Best sibling by combined_score, or None if all explored
        """
        unexplored = [s for s in siblings if s and s.get('id') not in explored_nodes]
        if not unexplored:
            return None
        
        # Sort by combined_score (descending)
        best = max(unexplored, key=lambda s: s.get('combined_score', 0.0))
        return best
    
    def _get_parent_after_root(self, tree: dict, node: dict) -> str:
        """
        Get the direct child of root in the path to this node.
        This is needed for backtracking: when root's child fails, we switch to its sibling.
        """
        current = node
        while current:
            parent_id = current.get('parent_id')
            if parent_id == 'root' or parent_id is None:
                return current.get('id')
            
            parent = self._get_node_from_tree(tree, parent_id)
            if not parent:
                return current.get('id')
            current = parent
        
        return node.get('id')
    
    def invoke(self, state: dict) -> dict:
        """
        Attempt backtracking when the best node fails validation.
        
        Returns:
            Updated state with next node to explore, or 'no_alternative' if no backtracking possible
        """
        # Only backtrack if validation failed
        if state.get('is_correct', False):
            state['backtrack_available'] = False
            return state
        
        tree = state.get('tree', {})
        failing_node_id = state.get('best_node_id')
        
        if not failing_node_id or not tree:
            state['backtrack_available'] = False
            return state
        
        failing_node = self._get_node_from_tree(tree, failing_node_id)
        if not failing_node:
            state['backtrack_available'] = False
            return state
        
        # Track failed nodes
        failed_nodes = state.setdefault('failed_nodes', [])
        if failing_node_id not in failed_nodes:
            failed_nodes.append(failing_node_id)
        
        # Get the parent (root's child)
        parent_after_root = self._get_parent_after_root(tree, failing_node)
        parent_id = 'root'
        
        if self.verbose:
            print(f"\n🔙 BACKTRACKING - {failing_node_id} failed validation")
            print(f"   Failed answer: {state.get('predicted_answer', '')}")
            print(f"   Expected: {state.get('expected_answer', '')}")
            print(f"   Parent after root: {parent_after_root}")
        
        # Get parent node and its children (siblings of failing node)
        parent = self._get_node_from_tree(tree, parent_id)
        if not parent:
            state['backtrack_available'] = False
            return state
        
        root_children_ids = parent.get('children', [])
        
        # Get all root children as potential backtrack targets
        root_children = []
        for child_id in root_children_ids:
            child = self._get_node_from_tree(tree, child_id)
            if child:
                root_children.append(child)
        
        # Find the next best unexplored root child
        explored = state.setdefault('explored_root_children', [])
        explored.append(parent_after_root)  # Mark current path as explored
        
        next_best = self._get_best_unexplored_sibling(root_children, failing_node, explored)
        
        if next_best is None:
            if self.verbose:
                print(f"   ❌ No unexplored siblings available")
            state['backtrack_available'] = False
            return state
        
        # SUCCESS: Found next best node to explore
        next_node_id = next_best.get('id')
        
        if self.verbose:
            print(f"\n✅ BACKTRACK SUCCESS")
            print(f"   Next node to explore: {next_node_id}")
            print(f"   Score: {next_best.get('combined_score', 0.0):.3f}")
            print(f"   Depth: {next_best.get('depth', 0)}")
        
        # Store backtracking event
        backtrack_history = state.setdefault('backtrack_history', [])
        backtrack_history.append({
            'from_node': failing_node_id,
            'to_node': next_node_id,
            'from_score': failing_node.get('combined_score', 0.0),
            'to_score': next_best.get('combined_score', 0.0),
            'failure_reason': state.get('failure_type', 'unknown'),
            'iteration': state.get('feedback_loops', 0),
        })
        
        # Store failure context for new node to use
        failure_context = {
            'failed_node_id': failing_node_id,
            'failure_type': state.get('failure_type', ''),
            'failed_answer': state.get('predicted_answer', ''),
            'error_analysis': state.get('error_analysis', ''),
            'failure_thought': failing_node.get('thought', ''),
        }
        state.setdefault('failure_contexts', []).append(failure_context)
        
        # Reset for new exploration from next_best node
        state['best_node_id'] = next_node_id
        state['current_node_id'] = next_node_id
        state['current_thought'] = next_best.get('thought', '')
        state['is_correct'] = False  # Reset validation
        state['predicted_answer'] = ''  # Reset predicted answer
        state['backtrack_available'] = True
        state['backtracking_iteration'] = 1  # Counter for backtracking sub-iterations
        
        return state
