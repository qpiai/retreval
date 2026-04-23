"""
Self-Refine solver for MATH-500 - Generate, Feedback, Refine with adaptive refinements
"""

from typing import Dict, Any
from src.agents.math500_base import MATH500BaseSolver
from src.agents.math500_prompts import INITIAL_SOLVE_PROMPT, VERIFY_SOLUTION_PROMPT, REFINE_SOLUTION_PROMPT


class MATH500SelfRefineSolver(MATH500BaseSolver):
    """Self-Refine approach for MATH-500: Generate -> Feedback -> Refine -> Repeat"""
    
    def __init__(self, verbose: bool = True, max_refinements: int = 3):
        super().__init__(verbose)
        self.method_name = "Self-Refine"
        self.max_refinements = max_refinements
    
    def solve(self, problem_id: str, question: str) -> Dict[str, Any]:
        """
        Solve using Self-Refine: Generate -> Feedback -> Refine cycles
        """
        # Step 1: Initial generation
        self.log(f"  ✍️  Initial solution generation")
        
        generate_prompt = INITIAL_SOLVE_PROMPT.format(question=question)
        current_solution = self.llm.generate(generate_prompt, max_tokens=2048)
        current_answer = self.extract_final_answer(current_solution)
        self.log(f"    Generated answer: {current_answer}")
        
        best_solution = current_solution
        best_answer = current_answer
        
        # Refinement loops
        for refinement in range(self.max_refinements):
            self.log(f"  🔄 Refinement {refinement + 1}/{self.max_refinements}")
            
            # Step 2: Get feedback
            feedback_prompt = VERIFY_SOLUTION_PROMPT.format(question=question, solution=current_solution)
            feedback = self.llm.generate(feedback_prompt, max_tokens=512)
            self.log(f"    📝 Feedback received")
            
            # Check if solution seems correct
            has_errors = any(word in feedback.lower() for word in ["error", "incorrect", "wrong", "mistake", "unclear"])
            
            if not has_errors:
                self.log(f"    ✅ Feedback: Solution appears correct")
                best_solution = current_solution
                best_answer = current_answer
                break
            
            # Step 3: Refine based on feedback
            refine_prompt = REFINE_SOLUTION_PROMPT.format(
                question=question, 
                solution=current_solution, 
                feedback=feedback
            )
            refined_solution = self.llm.generate(refine_prompt, max_tokens=2048)
            refined_answer = self.extract_final_answer(refined_solution)
            
            self.log(f"    ✍️  Refined answer: {refined_answer}")
            
            current_solution = refined_solution
            current_answer = refined_answer
            
            if refined_answer:
                best_solution = refined_solution
                best_answer = refined_answer
        
        return {
            'problem_id': problem_id,
            'solution': best_solution,
            'answer': best_answer or "",
            'passed': bool(best_answer),
            'refinements_used': refinement + 1 if 'refinement' in dir() else 0
        }
