"""
ReAct solver for GSM8K - Reasoning and Acting with adaptive iterations
"""

import json
import re
from typing import Dict, Any
from src.agents.gsm8k_base import GSM8KBaseSolver
from src.agents.gsm8k_prompts import THOUGHT_PROMPT, ACTION_PROMPT, OBSERVATION_PROMPT


class GSM8KReActSolver(GSM8KBaseSolver):
    """ReAct approach for GSM8K: Thought -> Action -> Observation cycles"""
    
    def __init__(self, verbose: bool = True, max_iterations: int = 3):
        super().__init__(verbose)
        self.method_name = "ReAct"
        self.max_iterations = max_iterations
        self.adaptive_iterations = True  # Determines iterations based on effectiveness
    
    def solve(self, problem_id: str, question: str) -> Dict[str, Any]:
        """
        Solve using ReAct: Thought -> Action -> Observation
        
        Args:
            problem_id: Problem identifier
            question: The math word problem
            
        Returns:
            Solution result
        """
        final_answer = None
        solution_text = ""
        
        for iteration in range(self.max_iterations):
            self.log(f"  🔄 Iteration {iteration + 1}/{self.max_iterations}")
            
            # Step 1: Generate thought
            thought_prompt = THOUGHT_PROMPT.format(question=question)
            thought = self.llm.generate(thought_prompt, max_tokens=256)
            self.log(f"    💭 Thought: {thought[:80]}...")
            
            # Step 2: Use standardized observation prompt with #### format
            # This ensures consistent answer extraction across all methods
            action_prompt = OBSERVATION_PROMPT.format(question=question, approach=thought)
            solution = self.llm.generate(action_prompt, max_tokens=1024)
            self.log(f"    ✍️  Action: Generated solution")
            solution_text = solution
            answer = self.extract_final_answer(solution)
            
            self.log(f"    ✅ Answer extracted: {answer}")
            
            # Check if we have a valid answer
            if answer:
                final_answer = answer
                break
        
        if not final_answer and solution_text:
            final_answer = self.extract_final_answer(solution_text)
        
        return {
            'problem_id': problem_id,
            'solution': solution_text,
            'answer': final_answer or "",
            'passed': bool(final_answer)
        }
