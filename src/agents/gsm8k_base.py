"""
Base class for GSM8K math problem solving using different agent frameworks
"""

import os
import re
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

load_dotenv()

# LLM Provider detection
provider = os.getenv("LLM_PROVIDER", "openai").lower()
if provider == "openai":
    from src.clients.openai_client import OpenAIClient
    LLMClient = OpenAIClient
else:
    from src.clients.ollama_client import OllamaClient
    LLMClient = OllamaClient


class GSM8KBaseSolver(ABC):
    """Base class for GSM8K problem solving"""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.method_name = "Base"
        self.llm = LLMClient(verbose=verbose)
        self.total_problems = 0
        self.passed_problems = 0
        self.failed_problems = []
        
    @abstractmethod
    def solve(self, problem_id: str, question: str) -> Dict[str, Any]:
        """
        Solve a GSM8K problem
        
        Args:
            problem_id: Unique problem identifier
            question: The math word problem
            
        Returns:
            Dictionary with solution details
        """
        pass
    
    def extract_final_answer(self, text: str) -> str:
        """Extract numerical answer from solution text"""
        if not text:
            return ""
        
        import re
        
        # First, check if there's a "####" separator (GSM8K format for final answer)
        if "####" in text:
            # Extract everything after ####
            final_part = text.split("####")[-1].strip()
            # Extract the first number from the final part
            numbers = re.findall(r'[-\d\.]+', final_part)
            if numbers:
                answer = numbers[0].strip()
                if answer and answer not in ['.', ',', '-']:
                    return answer
        
        # If no ####, try common patterns for final answers
        patterns = [
            r'(?:the\s+)?answer\s+(?:is\s+)?(?:thus\s+)?(?:approximately\s+)?[\$]?([-\d\.,]+)',
            r'therefore[\s,]*[\$]?([-\d\.,]+)',
            r'(?:=|result\s+is)\s*[\$]?([-\d\.,]+)',
            r'total[\s:]*[\$]?([-\d\.,]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                answer = match.group(1).strip().replace(',', '')
                if answer and answer not in ['.', ',', '-']:
                    return answer
        
        # Fallback: extract all numbers from text and take the last substantial one
        numbers = re.findall(r'[-\d\.]+', text)
        if numbers:
            for num in reversed(numbers):
                clean_num = num.strip().replace(',', '')
                if clean_num and clean_num not in ['.', ',', '-']:
                    return clean_num
        
        return ""
    
    def compare_answers(self, predicted: str, expected: str) -> bool:
        """Compare predicted and expected answers (handle numerical comparisons)"""
        try:
            # Remove common currency symbols and clean up
            pred = predicted.strip().replace('$', '').replace(',', '').lower()
            exp = expected.strip().replace('$', '').replace(',', '').lower()
            
            # Direct string comparison
            if pred == exp:
                return True
            
            # Try numerical comparison
            try:
                pred_num = float(pred)
                exp_num = float(exp)
                return abs(pred_num - exp_num) < 0.01
            except (ValueError, TypeError):
                return False
        except:
            return False
    
    def log(self, message: str):
        """Print verbose logging if enabled"""
        if self.verbose:
            print(message)
    
    def evaluate_batch(self, problems: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate a batch of problems
        
        Args:
            problems: List of problems with 'id' and 'question' keys
            
        Returns:
            Results dictionary
        """
        self.total_problems = 0
        self.passed_problems = 0
        self.failed_problems = []
        
        for problem in problems:
            problem_id = problem.get('id', '')
            question = problem.get('question', '')
            expected = problem.get('answer', '')
            
            self.log(f"\n{'='*80}")
            self.log(f"📝 Problem {problem_id}: {question[:100]}...")
            self.log(f"{'='*80}")
            
            result = self.solve(problem_id, question)
            self.total_problems += 1
            
            if result.get('passed', False):
                self.passed_problems += 1
                self.log(f"✅ PASSED")
            else:
                self.failed_problems.append(problem_id)
                self.log(f"❌ FAILED")
        
        return {
            'method': self.method_name,
            'total': self.total_problems,
            'passed': self.passed_problems,
            'failed': len(self.failed_problems),
            'pass_rate': (self.passed_problems / self.total_problems * 100) if self.total_problems > 0 else 0,
            'failed_problems': self.failed_problems
        }
