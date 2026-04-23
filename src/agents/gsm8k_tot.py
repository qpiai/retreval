"""
Tree of Thoughts (ToT) solver for GSM8K

Implements the ToT algorithm from "Tree of Thoughts: Deliberate Problem Solving with Large Language Models"
(Yao et al., 2023)

Key differences from ReTreVal:
- ToT: Pure tree search with thought generation, evaluation, and selection
- ReTreVal: ToT + Self-Refinement + Reflexion Memory + Cross-Critique + Adaptive Complexity
"""

import json
import re
from typing import Dict, Any, List, Optional
from src.agents.gsm8k_base import GSM8KBaseSolver


# ToT-specific prompts
TOT_GENERATE_THOUGHTS_PROMPT = """Generate {num_thoughts} different step-by-step solutions for this math problem.
Each solution should take a different approach or make different assumptions.

Problem: {question}

Current partial solution (if any): {partial_solution}

Generate {num_thoughts} distinct next steps or complete solutions.
Format each as:
THOUGHT 1: [solution approach 1]
THOUGHT 2: [solution approach 2]
...

Each thought should be a complete reasoning step that advances toward the answer."""

TOT_EVALUATE_THOUGHT_PROMPT = """Evaluate how promising this solution approach is for solving the problem.

Problem: {question}
Proposed Solution: {thought}

Rate this solution on a scale of 1-10 based on:
- Correctness of reasoning
- Mathematical accuracy
- Completeness toward final answer
- Likelihood of leading to correct solution

Provide your evaluation as:
SCORE: [1-10]
REASONING: [brief explanation]"""

TOT_SOLVE_FROM_THOUGHT_PROMPT = """Complete this solution to find the final answer.

Problem: {question}
Approach: {thought}

Provide a complete step-by-step solution following this approach.
At the end, clearly state the final answer as:
#### [final_answer_number]"""


class GSM8KToTSolver(GSM8KBaseSolver):
    """
    Tree of Thoughts solver for GSM8K.
    
    Algorithm:
    1. Generate K thoughts at each node
    2. Evaluate each thought with value function (LLM scoring)
    3. Keep top B thoughts (beam width) for next level
    4. Repeat until max depth or confident solution
    5. Return best path's answer
    """
    
    def __init__(self, verbose: bool = True, num_thoughts: int = 3, beam_width: int = 2, max_depth: int = 3):
        super().__init__(verbose)
        self.method_name = "ToT"
        self.num_thoughts = num_thoughts  # K: thoughts generated per step
        self.beam_width = beam_width      # B: top thoughts to keep (beam search)
        self.max_depth = max_depth        # Maximum tree depth
    
    def _generate_thoughts(self, question: str, partial_solution: str = "") -> List[str]:
        """Generate multiple thought branches."""
        prompt = TOT_GENERATE_THOUGHTS_PROMPT.format(
            num_thoughts=self.num_thoughts,
            question=question,
            partial_solution=partial_solution or "None - starting fresh"
        )
        
        response = self.llm.generate(prompt, max_tokens=1024)
        
        # Parse thoughts from response
        thoughts = []
        current_thought = ""
        
        for line in response.split('\n'):
            if re.match(r'^THOUGHT\s*\d+:', line, re.IGNORECASE):
                if current_thought:
                    thoughts.append(current_thought.strip())
                current_thought = re.sub(r'^THOUGHT\s*\d+:\s*', '', line, flags=re.IGNORECASE)
            else:
                current_thought += " " + line
        
        if current_thought:
            thoughts.append(current_thought.strip())
        
        # If parsing failed, split by numbered patterns
        if not thoughts:
            thoughts = re.split(r'\d+[\.\)]\s*', response)
            thoughts = [t.strip() for t in thoughts if t.strip()]
        
        # Ensure we have at least some thoughts
        if not thoughts:
            thoughts = [response]
        
        return thoughts[:self.num_thoughts]
    
    def _evaluate_thought(self, question: str, thought: str) -> float:
        """Evaluate a thought using LLM as value function."""
        prompt = TOT_EVALUATE_THOUGHT_PROMPT.format(
            question=question,
            thought=thought
        )
        
        response = self.llm.generate(prompt, max_tokens=256)
        
        # Extract score
        match = re.search(r'SCORE:\s*(\d+(?:\.\d+)?)', response, re.IGNORECASE)
        if match:
            score = float(match.group(1))
            return min(max(score, 0), 10) / 10.0  # Normalize to 0-1
        
        # Fallback: look for any number
        numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', response)
        if numbers:
            score = float(numbers[0])
            if 1 <= score <= 10:
                return score / 10.0
        
        return 0.5  # Default score
    
    def _solve_from_thought(self, question: str, thought: str) -> str:
        """Generate final solution from a thought branch."""
        prompt = TOT_SOLVE_FROM_THOUGHT_PROMPT.format(
            question=question,
            thought=thought
        )
        
        return self.llm.generate(prompt, max_tokens=1024)
    
    def solve(self, problem_id: str, question: str) -> Dict[str, Any]:
        """
        Solve using Tree of Thoughts with beam search.
        
        Args:
            problem_id: Problem identifier
            question: The math word problem
            
        Returns:
            Solution result with answer
        """
        self.log(f"  🌳 ToT: Generating tree with depth={self.max_depth}, "
                 f"thoughts={self.num_thoughts}, beam={self.beam_width}")
        
        # Initialize beam with empty partial solutions
        beam = [{"thought": "", "score": 0.5, "depth": 0}]
        best_solution = None
        best_score = 0.0
        
        for depth in range(self.max_depth):
            self.log(f"  📊 Depth {depth + 1}/{self.max_depth}")
            
            all_candidates = []
            
            for node in beam:
                # Generate thoughts from this node
                thoughts = self._generate_thoughts(question, node["thought"])
                self.log(f"    🔀 Generated {len(thoughts)} thoughts")
                
                for thought in thoughts:
                    # Evaluate thought
                    score = self._evaluate_thought(question, thought)
                    
                    all_candidates.append({
                        "thought": thought,
                        "score": score,
                        "depth": depth + 1,
                        "parent": node["thought"]
                    })
                    
                    self.log(f"    💭 Score: {score:.2f} - {thought[:50]}...")
            
            # Sort by score and keep top beam_width
            all_candidates.sort(key=lambda x: x["score"], reverse=True)
            beam = all_candidates[:self.beam_width]
            
            # Check if any candidate has a high-confidence answer
            for candidate in beam:
                if candidate["score"] > 0.8:
                    # Try to get final answer from this thought
                    solution = self._solve_from_thought(question, candidate["thought"])
                    answer = self.extract_final_answer(solution)
                    
                    if answer and candidate["score"] > best_score:
                        best_solution = solution
                        best_score = candidate["score"]
                        self.log(f"    ✅ High-confidence answer found: {answer}")
        
        # If no high-confidence answer, solve from best thought
        if not best_solution and beam:
            best_thought = beam[0]
            best_solution = self._solve_from_thought(question, best_thought["thought"])
            self.log(f"    📝 Solving from best thought (score: {best_thought['score']:.2f})")
        
        final_answer = self.extract_final_answer(best_solution or "")
        
        return {
            'problem_id': problem_id,
            'solution': best_solution or "",
            'answer': final_answer or "",
            'passed': bool(final_answer),
            'best_score': best_score,
            'tree_depth': self.max_depth
        }
