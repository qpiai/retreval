"""
GSM8K Evaluation Runner - Test all methods on a subset of GSM8K problems
"""

import json
import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.agents.gsm8k_retrevai import GSM8KReTreValSolver
from src.agents.gsm8k_react import GSM8KReActSolver
from src.agents.gsm8k_reflexion import GSM8KReflexionSolver
from src.agents.gsm8k_selfrefine import GSM8KSelfRefineSolver
from src.agents.gsm8k_tot import GSM8KToTSolver


def load_gsm8k_dataset(num_problems: int = 10) -> List[Dict[str, Any]]:
    """
    Load GSM8K dataset from Hugging Face Datasets
    
    Args:
        num_problems: Number of problems to load
        
    Returns:
        List of problems with 'id', 'question', and 'answer' keys
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("❌ Error: 'datasets' package not found")
        print("Install with: pip install datasets")
        return []
    
    try:
        print(f"📦 Loading GSM8K dataset...")
        dataset = load_dataset("openai/gsm8k", "main", split="test")
        
        problems = []
        for i, item in enumerate(dataset.take(num_problems)):
            problems.append({
                'id': f"GSM8K_{i}",
                'question': item['question'],
                'answer': item['answer'].split('####')[-1].strip() if '####' in item['answer'] else item['answer']
            })
        
        print(f"✅ Loaded {len(problems)} problems from GSM8K")
        return problems
    
    except Exception as e:
        print(f"❌ Error loading GSM8K: {e}")
        return []


def evaluate_method(method_name: str, problems: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Evaluate a specific method on GSM8K problems
    
    Args:
        method_name: Name of method (ReTreVal, ReAct, Reflexion, Self-Refine)
        problems: List of problems to evaluate
        
    Returns:
        Results dictionary
    """
    print(f"\n{'='*80}")
    print(f"🧪 Evaluating {method_name} on {len(problems)} GSM8K problems")
    print(f"{'='*80}\n")
    
    if method_name == "ReTreVal":
        solver = GSM8KReTreValSolver(verbose=False)  # Reduced verbosity for large runs
    elif method_name == "ReAct":
        solver = GSM8KReActSolver(verbose=False)
    elif method_name == "Reflexion":
        solver = GSM8KReflexionSolver(verbose=False)
    elif method_name == "Self-Refine":
        solver = GSM8KSelfRefineSolver(verbose=False)
    elif method_name == "ToT":
        solver = GSM8KToTSolver(verbose=False)
    else:
        print(f"❌ Unknown method: {method_name}")
        return {}
    
    results = []
    passed = 0
    
    for i, problem in enumerate(problems, 1):
        # Print progress every 10 problems
        if i % 10 == 0 or i == 1:
            print(f"Progress: {i}/{len(problems)} | Passed: {passed} | Pass Rate: {(passed/i*100):.1f}%", flush=True)
        
        result = solver.solve(problem['id'], problem['question'])
        result['expected_answer'] = problem['answer']
        result['question'] = problem['question']  # Add question to result dict
        
        # Check if answer is correct
        if solver.compare_answers(result.get('answer', ''), problem['answer']):
            result['passed'] = True
            passed += 1
        else:
            result['passed'] = False
        
        results.append(result)
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"📊 Results for {method_name}")
    print(f"{'='*80}")
    print(f"Total: {len(problems)} | Passed: {passed} | Failed: {len(problems) - passed}")
    print(f"Pass@1: {(passed / len(problems) * 100):.1f}%")
    print(f"{'='*80}\n")
    
    return {
        'method': method_name,
        'total': len(problems),
        'passed': passed,
        'failed': len(problems) - passed,
        'pass_rate': (passed / len(problems) * 100) if len(problems) > 0 else 0,
        'results': results
    }


def save_results(results: Dict[str, Any], method_name: str):
    """Save evaluation results to CSV"""
    import csv
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(__file__).parent.parent.parent / "results" / "gsm8k"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    csv_file = results_dir / f"gsm8k_{method_name.lower()}_{timestamp}.csv"
    
    try:
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['problem_id', 'question', 'expected_answer', 'predicted_answer', 'passed'])
            writer.writeheader()
            
            for result in results['results']:
                writer.writerow({
                    'problem_id': result.get('problem_id', ''),
                    'question': result.get('question', '')[:100],
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
    parser = argparse.ArgumentParser(description="Evaluate methods on GSM8K")
    parser.add_argument("--method", type=str, default="All", 
                       choices=["ReTreVal", "ReAct", "Reflexion", "Self-Refine", "ToT", "All"],
                       help="Method to evaluate (ToT = Tree of Thoughts)")
    parser.add_argument("--num-problems", type=int, default=1319,
                       help="Number of problems to evaluate (max 1319)")
    
    args = parser.parse_args()
    
    # Load dataset
    problems = load_gsm8k_dataset(args.num_problems)
    if not problems:
        print("❌ Failed to load GSM8K dataset")
        sys.exit(1)
    
    # Evaluate methods
    methods = ["ReTreVal", "ToT", "ReAct", "Reflexion", "Self-Refine"] if args.method == "All" else [args.method]
    
    all_results = []
    for method in methods:
        results = evaluate_method(method, problems)
        if results:
            all_results.append(results)
            save_results(results, method)
    
    # Print comparative summary
    if len(all_results) > 1:
        print(f"\n{'='*80}")
        print("📊 Comparative Results")
        print(f"{'='*80}")
        print(f"{'Method':<15} {'Total':<8} {'Passed':<8} {'Failed':<8} {'Pass@1':<8}")
        print(f"{'-'*50}")
        for result in all_results:
            print(f"{result['method']:<15} {result['total']:<8} {result['passed']:<8} "
                  f"{result['failed']:<8} {result['pass_rate']:<7.1f}%")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
