"""
Base class for MATH-500 problem solving using different agent frameworks
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
if provider == "vllm":
    from src.clients.vllm_client import VLLMClient
    LLMClient = VLLMClient
elif provider == "openai":
    from src.clients.openai_client import OpenAIClient
    LLMClient = OpenAIClient
else:
    from src.clients.ollama_client import OllamaClient
    LLMClient = OllamaClient


class MATH500BaseSolver(ABC):
    """Base class for MATH-500 problem solving"""
    
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
        Solve a MATH-500 problem
        
        Args:
            problem_id: Unique problem identifier
            question: The math problem
            
        Returns:
            Dictionary with solution details
        """
        pass
    
    def extract_final_answer(self, text: str) -> str:
        """Extract answer from solution text - handles MATH dataset format with boxed answers"""
        if not text:
            return ""

        # Strip thinking model tags first
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

        # MATH dataset uses \boxed{answer} format
        boxed_patterns = [
            r'\\boxed\{([^{}]+(?:\{[^{}]*\}[^{}]*)*)\}',  # Handle nested braces
            r'\\boxed\s*\{([^}]+)\}',  # Simple boxed
            r'boxed\s*\{([^}]+)\}',  # Without backslash
        ]
        
        for pattern in boxed_patterns:
            matches = re.findall(pattern, text)
            if matches:
                # Return the last boxed answer (usually the final one)
                return matches[-1].strip()
        
        # Try #### format (GSM8K style)
        if "####" in text:
            final_part = text.split("####")[-1].strip()
            numbers = re.findall(r'[-\d\.]+', final_part)
            if numbers:
                return numbers[0].strip()
        
        # Try "answer is" patterns
        patterns = [
            r'(?:the\s+)?(?:final\s+)?answer\s+(?:is\s+)?(?:thus\s+)?(?:approximately\s+)?[\$\\]?([-\d\.,]+|[A-Za-z]+)',
            r'therefore[\s,]*[\$\\]?([-\d\.,]+)',
            r'(?:=|result\s+is)\s*[\$\\]?([-\d\.,]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                answer = match.group(1).strip().replace(',', '')
                if answer and answer not in ['.', ',', '-']:
                    return answer
        
        # Fallback: look for last equation result
        eq_pattern = r'=\s*([-\d\.]+)\s*(?:\.|$|\n)'
        matches = re.findall(eq_pattern, text)
        if matches:
            return matches[-1].strip()
        
        return ""
    
    def normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison"""
        if not answer:
            return ""
        
        # Remove LaTeX formatting
        answer = answer.replace('\\', '')
        answer = answer.replace('left', '').replace('right', '')
        answer = answer.replace('frac', '').replace('{', '').replace('}', '')
        answer = answer.replace('$', '').replace(' ', '')
        
        # Handle fractions like 3/2
        if '/' in answer and answer.count('/') == 1:
            parts = answer.split('/')
            try:
                num = float(parts[0])
                den = float(parts[1])
                if den != 0:
                    return str(num / den)
            except:
                pass
        
        # Convert to lowercase for text comparison
        answer = answer.lower().strip()
        
        return answer
    
    def compare_answers(self, predicted: str, expected: str) -> bool:
        """Compare predicted and expected answers"""
        try:
            pred_norm = self.normalize_answer(predicted)
            exp_norm = self.normalize_answer(expected)
            
            # Direct string comparison
            if pred_norm == exp_norm:
                return True
            
            # Try numerical comparison
            try:
                pred_num = float(pred_norm)
                exp_num = float(exp_norm)
                # Allow small tolerance for floating point
                return abs(pred_num - exp_num) < 0.01
            except (ValueError, TypeError):
                pass
            
            # Check if one contains the other (for text answers)
            if pred_norm and exp_norm:
                if pred_norm in exp_norm or exp_norm in pred_norm:
                    return True
            
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
            problems: List of problems with 'id', 'question', and 'answer' keys
            
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
            
            self._current_expected_answer = expected
            result = self.solve(problem_id, question)
            self.total_problems += 1
            
            if self.compare_answers(result.get('answer', ''), expected):
                self.passed_problems += 1
                self.log(f"✅ PASSED")
            else:
                self.failed_problems.append(problem_id)
                self.log(f"❌ FAILED (expected: {expected}, got: {result.get('answer', '')})")
        
        return {
            'method': self.method_name,
            'total': self.total_problems,
            'passed': self.passed_problems,
            'failed': len(self.failed_problems),
            'pass_rate': (self.passed_problems / self.total_problems * 100) if self.total_problems > 0 else 0,
            'failed_problems': self.failed_problems
        }
