"""
MATH-500 Evaluation Runner - Test all methods on MATH-500 benchmark
Loads dataset directly from Hugging Face (HuggingFaceH4/MATH-500)
"""

import csv
import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.agents.math500_retrevai import MATH500ReTreValSolver
from src.agents.math500_react import MATH500ReActSolver
from src.agents.math500_reflexion import MATH500ReflexionSolver
from src.agents.math500_selfrefine import MATH500SelfRefineSolver
from src.agents.math500_tot import MATH500ToTSolver


def load_math500_dataset(num_problems: int = 500) -> List[Dict[str, Any]]:
    """
    Load MATH-500 dataset from Hugging Face
    
    Args:
        num_problems: Number of problems to load (max 500)
        
    Returns:
        List of problems with 'id', 'question', 'answer', 'subject', 'level' keys
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("❌ Error: 'datasets' package not found")
        print("Install with: pip install datasets")
        return []
    
    try:
        print(f"📦 Loading MATH-500 dataset from Hugging Face...")
        dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
        
        problems = []
        for i, item in enumerate(dataset):
            if i >= num_problems:
                break
            problems.append({
                'id': item.get('unique_id', f"MATH_{i}"),
                'question': item['problem'],
                'answer': item['answer'],
                'solution': item.get('solution', ''),
                'subject': item.get('subject', ''),
                'level': item.get('level', 0)
            })
        
        print(f"✅ Loaded {len(problems)} problems from MATH-500")
        
        # Print subject distribution
        subjects = {}
        for p in problems:
            subj = p.get('subject', 'Unknown')
            subjects[subj] = subjects.get(subj, 0) + 1
        print(f"   Subjects: {subjects}")
        
        return problems
    
    except Exception as e:
        print(f"❌ Error loading MATH-500: {e}")
        return []


def evaluate_method(method_name: str, problems: List[Dict[str, Any]], verbose: bool = False) -> Dict[str, Any]:
    """
    Evaluate a specific method on MATH-500 problems
    
    Args:
        method_name: Name of method (ReTreVal, ReAct, Reflexion, Self-Refine)
        problems: List of problems to evaluate
        verbose: Enable verbose output per problem
        
    Returns:
        Results dictionary
    """
    print(f"\n{'='*80}")
    print(f"🧪 Evaluating {method_name} on {len(problems)} MATH-500 problems")
    print(f"{'='*80}\n")
    
    if method_name == "ReTreVal":
        solver = MATH500ReTreValSolver(verbose=verbose)
    elif method_name == "ReAct":
        solver = MATH500ReActSolver(verbose=verbose)
    elif method_name == "Reflexion":
        solver = MATH500ReflexionSolver(verbose=verbose)
    elif method_name == "Self-Refine":
        solver = MATH500SelfRefineSolver(verbose=verbose)
    elif method_name == "ToT":
        solver = MATH500ToTSolver(verbose=verbose)
    else:
        print(f"❌ Unknown method: {method_name}")
        return {}
    
    results = []
    passed = 0
    subject_stats = {}  # Track pass rate by subject
    level_stats = {}    # Track pass rate by difficulty level
    
    for i, problem in enumerate(problems, 1):
        # Print progress every 10 problems
        if i % 10 == 0 or i == 1:
            print(f"Progress: {i}/{len(problems)} | Passed: {passed} | Pass Rate: {(passed/i*100):.1f}%", flush=True)
        
        result = solver.solve(problem['id'], problem['question'])
        result['expected_answer'] = problem['answer']
        result['question'] = problem['question']
        result['subject'] = problem.get('subject', '')
        result['level'] = problem.get('level', 0)
        
        # Check if answer is correct
        is_correct = solver.compare_answers(result.get('answer', ''), problem['answer'])
        result['passed'] = is_correct
        
        if is_correct:
            passed += 1
        
        # Track by subject
        subj = problem.get('subject', 'Unknown')
        if subj not in subject_stats:
            subject_stats[subj] = {'total': 0, 'passed': 0}
        subject_stats[subj]['total'] += 1
        if is_correct:
            subject_stats[subj]['passed'] += 1
        
        # Track by level
        level = problem.get('level', 0)
        if level not in level_stats:
            level_stats[level] = {'total': 0, 'passed': 0}
        level_stats[level]['total'] += 1
        if is_correct:
            level_stats[level]['passed'] += 1
        
        results.append(result)
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"📊 Results for {method_name}")
    print(f"{'='*80}")
    print(f"Total: {len(problems)} | Passed: {passed} | Failed: {len(problems) - passed}")
    print(f"Pass@1: {(passed / len(problems) * 100):.1f}%")
    
    # Print by subject
    print(f"\n📚 Results by Subject:")
    for subj, stats in sorted(subject_stats.items()):
        rate = (stats['passed'] / stats['total'] * 100) if stats['total'] > 0 else 0
        print(f"   {subj}: {stats['passed']}/{stats['total']} ({rate:.1f}%)")
    
    # Print by level
    print(f"\n📈 Results by Difficulty Level:")
    for level, stats in sorted(level_stats.items()):
        rate = (stats['passed'] / stats['total'] * 100) if stats['total'] > 0 else 0
        print(f"   Level {level}: {stats['passed']}/{stats['total']} ({rate:.1f}%)")
    
    print(f"{'='*80}\n")
    
    return {
        'method': method_name,
        'total': len(problems),
        'passed': passed,
        'failed': len(problems) - passed,
        'pass_rate': (passed / len(problems) * 100) if len(problems) > 0 else 0,
        'subject_stats': subject_stats,
        'level_stats': level_stats,
        'results': results
    }


def save_results(results: Dict[str, Any], method_name: str) -> Path:
    """Save evaluation results to CSV"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(__file__).parent.parent.parent / "results" / "math500"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    csv_file = results_dir / f"math500_{method_name.lower().replace('-', '_')}_{timestamp}.csv"
    
    try:
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'problem_id', 'subject', 'level', 'question', 
                'expected_answer', 'predicted_answer', 'passed'
            ])
            writer.writeheader()
            
            for result in results['results']:
                writer.writerow({
                    'problem_id': result.get('problem_id', ''),
                    'subject': result.get('subject', ''),
                    'level': result.get('level', ''),
                    'question': result.get('question', '')[:200],  # Truncate for readability
                    'expected_answer': result.get('expected_answer', ''),
                    'predicted_answer': result.get('answer', ''),
                    'passed': result.get('passed', False)
                })
        
        print(f"✅ Results saved to {csv_file}")
        return csv_file
    except Exception as e:
        print(f"❌ Error saving results: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Evaluate methods on MATH-500 benchmark")
    parser.add_argument("--method", type=str, default="ReTreVal", 
                       choices=["ReTreVal", "ReAct", "Reflexion", "Self-Refine", "ToT", "All"],
                       help="Method to evaluate (ToT = Tree of Thoughts, default: ReTreVal)")
    parser.add_argument("--num-problems", type=int, default=500,
                       help="Number of problems to evaluate (max 500, default: 500)")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose output per problem")
    
    args = parser.parse_args()
    
    # Load dataset
    problems = load_math500_dataset(args.num_problems)
    if not problems:
        print("❌ Failed to load MATH-500 dataset")
        sys.exit(1)
    
    # Evaluate methods
    methods = ["ReTreVal", "ToT", "ReAct", "Reflexion", "Self-Refine"] if args.method == "All" else [args.method]
    
    all_results = []
    for method in methods:
        results = evaluate_method(method, problems, verbose=args.verbose)
        if results:
            all_results.append(results)
            save_results(results, method)
    
    # Print comparative summary
    if len(all_results) > 1:
        print(f"\n{'='*80}")
        print("📊 Comparative Results - MATH-500")
        print(f"{'='*80}")
        print(f"{'Method':<15} {'Total':<8} {'Passed':<8} {'Failed':<8} {'Pass@1':<8}")
        print(f"{'-'*50}")
        for result in all_results:
            print(f"{result['method']:<15} {result['total']:<8} {result['passed']:<8} "
                  f"{result['failed']:<8} {result['pass_rate']:<7.1f}%")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
