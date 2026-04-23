"""
Reflexion solver for GSM8K - Try, Fail, Reflect, Learn with adaptive attempts
"""

from typing import Dict, Any
from src.agents.gsm8k_base import GSM8KBaseSolver
from src.agents.gsm8k_prompts import INITIAL_SOLVE_PROMPT, VERIFY_SOLUTION_PROMPT


class GSM8KReflexionSolver(GSM8KBaseSolver):
    """Reflexion approach for GSM8K: Try -> Fail -> Reflect -> Learn -> Retry"""
    
    def __init__(self, verbose: bool = True, max_attempts: int = 3):
        super().__init__(verbose)
        self.method_name = "Reflexion"
        self.max_attempts = max_attempts
        self.adaptive_attempts = True  # Determines attempts based on reflection quality
        self.memory = []  # Store attempts and learnings
    
    def solve(self, problem_id: str, question: str) -> Dict[str, Any]:
        """
        Solve using Reflexion: Try -> Reflect -> Learn -> Retry
        
        Args:
            problem_id: Problem identifier
            question: The math word problem
            
        Returns:
            Solution result
        """
        final_answer = None
        solution_text = ""
        
        for attempt in range(self.max_attempts):
            self.log(f"  🔄 Attempt {attempt + 1}/{self.max_attempts}")
            
            # Build context from previous attempts
            context = ""
            if self.memory:
                context = "\nPrevious attempts and feedback:\n"
                for prev in self.memory:
                    context += f"- Attempt {prev['num']}: {prev['feedback']}\n"
            
            # Step 1: Try to solve
            solve_prompt = INITIAL_SOLVE_PROMPT.format(question=question)
            solution = self.llm.generate(solve_prompt, max_tokens=1024)
            self.log(f"    ✍️  Generated solution")
            solution_text = solution
            answer = self.extract_final_answer(solution)
            
            # Step 2: Reflect on the solution
            if answer:
                reflect_prompt = VERIFY_SOLUTION_PROMPT.format(question=question, solution=solution)
                reflection = self.llm.generate(reflect_prompt, max_tokens=512)
                self.log(f"    🔍 Reflection: Analyzing solution quality")
                
                # Check if solution is likely correct based on reflection
                is_likely_correct = "correct" in reflection.lower() or "yes" in reflection.lower()
                
                if is_likely_correct:
                    self.log(f"    ✅ Solution appears valid: {answer}")
                    final_answer = answer
                    break
                else:
                    # Store feedback for next attempt
                    self.memory.append({
                        'num': attempt + 1,
                        'solution': solution,
                        'answer': answer,
                        'feedback': reflection[:200]
                    })
                    self.log(f"    ⚠️  Reflection suggests improvement needed")
            else:
                self.log(f"    ❌ Could not extract answer")
        
        if not final_answer and solution_text:
            final_answer = self.extract_final_answer(solution_text)
        
        return {
            'problem_id': problem_id,
            'solution': solution_text,
            'answer': final_answer or "",
            'passed': bool(final_answer),
            'attempts_used': len(self.memory) + 1
        }
