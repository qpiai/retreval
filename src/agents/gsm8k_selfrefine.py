"""
Self-Refine solver for GSM8K - Generate, Feedback, Refine with adaptive refinements
"""

from typing import Dict, Any
from src.agents.gsm8k_base import GSM8KBaseSolver
from src.agents.gsm8k_prompts import INITIAL_SOLVE_PROMPT, VERIFY_SOLUTION_PROMPT, REFINE_SOLUTION_PROMPT


class GSM8KSelfRefineSolver(GSM8KBaseSolver):
    """Self-Refine approach for GSM8K: Generate -> Feedback -> Refine -> Repeat"""
    
    def __init__(self, verbose: bool = True, max_refinements: int = 3):
        super().__init__(verbose)
        self.method_name = "Self-Refine"
        self.max_refinements = max_refinements
        self.adaptive_refinements = True  # Determines refinements based on feedback quality
    
    def solve(self, problem_id: str, question: str) -> Dict[str, Any]:
        """
        Solve using Self-Refine: Generate -> Feedback -> Refine cycles
        
        Args:
            problem_id: Problem identifier
            question: The math word problem
            
        Returns:
            Solution result
        """
        # Step 1: Initial generation
        self.log(f"  ✍️  Initial solution generation")
        
        generate_prompt = INITIAL_SOLVE_PROMPT.format(question=question)
        current_solution = self.llm.generate(generate_prompt, max_tokens=1024)
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
            
            # Check if solution seems correct based on feedback
            has_errors = any(word in feedback.lower() for word in ["error", "incorrect", "wrong", "mistake", "unclear"])
            
            if not has_errors:
                self.log(f"    ✅ Feedback: Solution appears correct")
                best_solution = current_solution
                best_answer = current_answer
                break
            
            # Step 3: Refine based on feedback
            refine_prompt = REFINE_SOLUTION_PROMPT.format(question=question, solution=current_solution, feedback=feedback)
            refined_solution = self.llm.generate(refine_prompt, max_tokens=1024)
            refined_answer = self.extract_final_answer(refined_solution)
            
            self.log(f"    ✍️  Refined answer: {refined_answer}")
            
            # Update for next iteration
            current_solution = refined_solution
            current_answer = refined_answer
            
            if len(refined_solution) > len(best_solution):
                best_solution = refined_solution
                best_answer = refined_answer
        
        return {
            'problem_id': problem_id,
            'solution': best_solution,
            'answer': best_answer or "",
            'passed': bool(best_answer),
            'refinements_used': refinement + 1
        }
