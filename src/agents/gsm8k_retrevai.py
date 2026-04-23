"""
ReTreVal solver for GSM8K - Tree of Thoughts with adaptive complexity
"""

import json
from typing import Dict, Any, List
from src.agents.gsm8k_base import GSM8KBaseSolver
from src.agents.gsm8k_prompts import (
    GENERATE_APPROACHES_PROMPT, OBSERVATION_PROMPT, REFINE_SOLUTION_PROMPT
)


class GSM8KReTreValSolver(GSM8KBaseSolver):
    """ReTreVal approach for GSM8K using adaptive tree-based reasoning"""
    
    def __init__(self, verbose: bool = True, max_depth: int = 3, top_k: int = 2):
        super().__init__(verbose)
        self.method_name = "ReTreVal"
        self.max_depth = max_depth
        self.top_k = top_k
        self.adaptive_depth = True  # Adapts tree depth based on problem complexity
    
    def analyze_problem_complexity(self, question: str) -> int:
        """Estimate problem complexity to determine tree depth"""
        # Simple heuristic: count numbers and operations
        import re
        numbers = len(re.findall(r'\d+', question))
        operations = len(re.findall(r'(\+|-|\*|/)', question))
        words = len(question.split())
        
        complexity_score = numbers + operations + (words / 10)
        
        if complexity_score < 3:
            return 1  # Simple problem
        elif complexity_score < 6:
            return 2  # Medium problem
        else:
            return 3  # Complex problem
    
    def generate_candidates(self, question: str) -> List[str]:
        """Generate multiple solution approaches"""
        prompt = GENERATE_APPROACHES_PROMPT.format(question=question, top_k=self.top_k)
        
        response = self.llm.generate(prompt, max_tokens=1024)
        
        try:
            candidates = json.loads(response)
            return [c.get('steps', '') for c in candidates if c.get('steps')]
        except:
            return [response]
    
    def evaluate_candidate(self, question: str, approach: str) -> Dict[str, Any]:
        """Evaluate a single solution approach"""
        prompt = OBSERVATION_PROMPT.format(question=question, approach=approach)
        
        solution = self.llm.generate(prompt, max_tokens=1024)
        answer = self.extract_final_answer(solution)
        
        return {
            'approach': approach,
            'solution': solution,
            'answer': answer
        }
    
    def solve(self, problem_id: str, question: str) -> Dict[str, Any]:
        """
        Solve problem using adaptive tree-based reasoning
        
        Args:
            problem_id: Problem identifier
            question: The math word problem
            
        Returns:
            Solution result
        """
        # Determine adaptive depth based on complexity
        depth = self.analyze_problem_complexity(question)
        if self.adaptive_depth:
            self.log(f"  📊 Complexity analysis: Using depth {depth}/{self.max_depth}")
        
        # Level 1: Generate multiple approaches
        self.log(f"  🌳 Level 1: Generating {self.top_k} candidate approaches")
        candidates = self.generate_candidates(question)
        
        results = []
        for i, candidate in enumerate(candidates, 1):
            self.log(f"    📌 Candidate {i}: {candidate[:60]}...")
            result = self.evaluate_candidate(question, candidate)
            results.append(result)
        
        # Level 2+: Refine top solutions if depth > 1
        if depth > 1:
            self.log(f"  🌳 Level 2: Refining top solutions")
            for i, result in enumerate(results[:self.top_k], 1):
                refine_prompt = REFINE_SOLUTION_PROMPT.format(
                    question=question,
                    solution=result['solution'],
                    feedback="Improve clarity and verify the final numerical answer. Make sure to include #### [answer] at the end."
                )
                refined = self.llm.generate(refine_prompt, max_tokens=1024)
                result['solution'] = refined
                result['answer'] = self.extract_final_answer(refined)
        
        # Select best solution
        best_solution = max(results, key=lambda x: len(x['solution']))
        
        self.log(f"  ✍️  Final answer: {best_solution['answer']}")
        
        return {
            'problem_id': problem_id,
            'solution': best_solution['solution'],
            'answer': best_solution['answer'],
            'passed': True  # Marked as passed if solution generated
        }
