"""
ReAct solver for MATH-500 - Reasoning and Acting with adaptive iterations
"""

from typing import Dict, Any
from src.agents.math500_base import MATH500BaseSolver
from src.agents.math500_prompts import THOUGHT_PROMPT, OBSERVATION_PROMPT


class MATH500ReActSolver(MATH500BaseSolver):
    """ReAct approach for MATH-500: Thought -> Action -> Observation cycles"""
    
    def __init__(self, verbose: bool = True, max_iterations: int = 3):
        super().__init__(verbose)
        self.method_name = "ReAct"
        self.max_iterations = max_iterations
    
    def solve(self, problem_id: str, question: str) -> Dict[str, Any]:
        """
        Solve using ReAct: Thought -> Action -> Observation
        """
        final_answer = None
        solution_text = ""
        
        for iteration in range(self.max_iterations):
            self.log(f"  🔄 Iteration {iteration + 1}/{self.max_iterations}")
            
            # Step 1: Generate thought
            thought_prompt = THOUGHT_PROMPT.format(question=question)
            thought = self.llm.generate(thought_prompt, max_tokens=512)
            self.log(f"    💭 Thought: {thought[:80]}...")
            
            # Step 2: Generate solution based on thought
            action_prompt = OBSERVATION_PROMPT.format(question=question, approach=thought)
            solution = self.llm.generate(action_prompt, max_tokens=2048)
            self.log(f"    ✍️  Generated solution")
            solution_text = solution
            answer = self.extract_final_answer(solution)
            
            self.log(f"    ✅ Answer extracted: {answer}")
            
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
