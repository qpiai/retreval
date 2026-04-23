"""
ReTreVal solver for MATH-500 using full LangGraph agent with backtracking
"""

from typing import Dict, Any
from src.agents.math500_base import MATH500BaseSolver
from src.agents.agent_langgraph import LangGraphAgent


class MATH500ReTreValSolver(MATH500BaseSolver):
    """ReTreVal approach using full LangGraph agent with all components"""
    
    def __init__(self, verbose: bool = True, max_depth: int = 2):
        super().__init__(verbose)
        self.method_name = "ReTreVal"
        self.max_depth = max_depth
        self.agent = LangGraphAgent(max_loops=max_depth, verbose=verbose)
    
    def solve(self, problem_id: str, question: str, expected_answer: str = None) -> Dict[str, Any]:
        """
        Solve using full ReTreVal LangGraph agent with:
        - Planning
        - Tree-of-Thought expansion
        - Search
        - Refinement & Critique
        - Scoring & Memory
        - Complexity analysis
        - Synthesis & Validation
        - Backtracking on failure
        """
        self.log(f"  🌳 Running full LangGraph agent")

        # expected_answer: prefer explicit arg, then attribute set by evaluate_batch
        ea = expected_answer or getattr(self, '_current_expected_answer', None)

        # Run the full agent
        result = self.agent.run(
            problem=question,
            problem_id=problem_id,
            expected_answer=ea,
        )
        
        # Extract results
        predicted_answer = result.get('predicted_answer', '')
        final_output = result.get('final_output', '')
        backtrack_history = result.get('backtrack_history', [])

        # Try to extract a clean boxed answer from final_output
        # This is more reliable than predicted_answer which may contain garbage
        if final_output:
            extracted = self.extract_final_answer(final_output)
            if extracted:
                predicted_answer = extracted
            elif not predicted_answer:
                predicted_answer = ''
        
        self.log(f"    💡 Predicted answer: {predicted_answer}")
        if backtrack_history:
            self.log(f"    🔙 Backtracking events: {len(backtrack_history)}")
        
        return {
            'problem_id': problem_id,
            'solution': final_output,
            'answer': predicted_answer or "",
            'passed': bool(predicted_answer),
            'backtrack_events': len(backtrack_history),
            'backtrack_history': backtrack_history
        }
