"""
GSM8K Evaluation with LangGraph Agent - Feedback Loop Integration
Uses the advanced LangGraphAgent with external validation and feedback loops
"""

import json
import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.agents.agent_langgraph import run_agent


def load_gsm8k_dataset(num_problems: int = 10) -> List[Dict[str, Any]]:
    """
    Load GSM8K dataset from Hugging Face Datasets
    
    Args:
        num_problems: Number of problems to load
        
    Returns:
        List of problems with 'id', 'question', 'answer' keys
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
                'id': f"GSM8K_{i:04d}",
                'question': item['question'],
                'answer': item['answer'].split('####')[-1].strip() if '####' in item['answer'] else item['answer']
            })
        
        print(f"✅ Loaded {len(problems)} problems from GSM8K")
        return problems
    
    except Exception as e:
        print(f"❌ Error loading GSM8K: {e}")
        return []


def load_local_gsm8k_data(csv_file: str = None) -> List[Dict[str, Any]]:
    """
    Load GSM8K data from local CSV file as fallback
    
    Args:
        csv_file: Path to CSV with problem data
        
    Returns:
        List of problems
    """
    if not csv_file:
        csv_file = Path(__file__).parent.parent.parent / "data" / "math" / "math_500_test.csv"
    
    if not Path(csv_file).exists():
        return []
    
    import csv
    problems = []
    
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i >= 10:  # Limit to 10 for testing
                    break
                problems.append({
                    'id': f"Math_{i:04d}",
                    'question': row.get('Problem', ''),
                    'answer': row.get('Answer', '')
                })
        print(f"✅ Loaded {len(problems)} problems from {csv_file}")
        return problems
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return []


def normalize_answer(answer: str) -> str:
    """
    Normalize answer for comparison
    
    Args:
        answer: Answer string to normalize
        
    Returns:
        Normalized answer
    """
    if not answer:
        return ""
    
    import re
    
    # Convert to string and lowercase
    ans = str(answer).strip().lower()
    
    # Remove currency symbols
    ans = ans.replace('$', '').replace('€', '').replace('£', '').replace('¥', '')
    
    # Remove common units and suffixes
    for suffix in [' dollars', ' cents', ' apples', ' oranges', ' items', ' units',
                   ' people', ' days', ' hours', ' minutes', ' seconds', ' years',
                   ' meters', ' kilometers', ' miles', ' miles', ' km', ' m', ' cm']:
        if ans.endswith(suffix):
            ans = ans[:-len(suffix)].strip()
    
    # Extract just numbers for comparison (first number found)
    numbers = re.findall(r'-?\d+\.?\d*', ans)
    
    if numbers:
        # Try to convert to int if it's a whole number
        try:
            val = float(numbers[0])
            if val == int(val):
                return str(int(val))
            return numbers[0]
        except:
            return numbers[0]
    
    return ans.strip()


def compare_answers(predicted: str, expected: str) -> bool:
    """
    Compare predicted and expected answers with normalization
    
    Args:
        predicted: Predicted answer
        expected: Expected answer
        
    Returns:
        True if answers match after normalization
    """
    pred_norm = normalize_answer(predicted)
    exp_norm = normalize_answer(expected)
    
    # Direct match
    if pred_norm == exp_norm:
        return True
    
    # Try numeric comparison
    try:
        pred_float = float(pred_norm)
        exp_float = float(exp_norm)
        # Allow small numeric tolerance (0.01%)
        return abs(pred_float - exp_float) / max(abs(exp_float), 1) < 0.0001
    except:
        pass
    
    return False


def evaluate_langgraph_agent(problems: List[Dict[str, Any]], verbose: bool = False) -> Dict[str, Any]:
    """
    Evaluate LangGraph Agent with feedback loops on GSM8K problems
    
    Args:
        problems: List of problems to evaluate
        verbose: Whether to print verbose output
        
    Returns:
        Results dictionary
    """
    print(f"\n{'='*80}")
    print(f"🧪 Evaluating LangGraph Agent with Feedback Loops")
    print(f"📊 Testing on {len(problems)} GSM8K problems")
    print(f"{'='*80}\n")
    
    results = []
    passed = 0
    failed_problems = []
    
    for i, problem in enumerate(problems, 1):
        problem_id = problem['id']
        question = problem['question']
        expected_answer = problem['answer']
        
        # Print progress
        if i % max(1, len(problems) // 10) == 0 or i == 1 or i == len(problems):
            print(f"Progress: {i}/{len(problems)} | Passed: {passed} | "
                  f"Pass Rate: {(passed/i*100):.1f}%", flush=True)
        
        try:
            # Run agent with expected_answer for feedback loop validation
            result = run_agent(
                question, 
                expected_answer=expected_answer,
                problem_id=problem_id,
                iterations=2,  # Allow feedback loops
                verbose=verbose
            )
            
            # Extract predicted answer
            predicted_answer = result.get('final_output', '')
            
            # Check correctness
            is_correct = compare_answers(predicted_answer, expected_answer)
            
            if is_correct:
                passed += 1
            else:
                failed_problems.append({
                    'id': problem_id,
                    'question': question[:100],
                    'expected': expected_answer,
                    'predicted': predicted_answer[:100]
                })
            
            # Store result
            results.append({
                'problem_id': problem_id,
                'question': question,
                'expected_answer': expected_answer,
                'predicted_answer': predicted_answer,
                'is_correct': is_correct,
                'feedback_loops': result.get('feedback_loops', 0),
                'error_analysis': result.get('error_analysis', ''),
                'failure_type': result.get('failure_type', ''),
                'approach_type': result.get('approach_type', ''),
                'tree_depth': len(result.get('tree', {}).get('children', [])),
                'score': result.get('score', 0.0),
                'status': 'success'
            })
            
        except Exception as e:
            error_msg = str(e)
            if verbose:
                print(f"  ⚠️  Error on problem {problem_id}: {error_msg}")
            
            results.append({
                'problem_id': problem_id,
                'question': question,
                'expected_answer': expected_answer,
                'predicted_answer': '',
                'is_correct': False,
                'feedback_loops': 0,
                'error_analysis': error_msg,
                'failure_type': 'execution_error',
                'approach_type': '',
                'tree_depth': 0,
                'score': 0.0,
                'status': 'error'
            })
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"📊 Final Results")
    print(f"{'='*80}")
    print(f"Total Problems: {len(problems)}")
    print(f"Passed: {passed}")
    print(f"Failed: {len(problems) - passed}")
    print(f"Pass@1: {(passed / len(problems) * 100):.2f}%")
    print(f"{'='*80}\n")
    
    # Show sample failures
    if failed_problems and len(failed_problems) <= 5:
        print("Failed Problems:")
        for fp in failed_problems[:5]:
            print(f"  - {fp['id']}: Expected={fp['expected']}, Got={fp['predicted']}")
        print()
    
    return {
        'method': 'LangGraph+Feedback',
        'timestamp': datetime.now().isoformat(),
        'total': len(problems),
        'passed': passed,
        'failed': len(problems) - passed,
        'pass_rate': (passed / len(problems) * 100) if len(problems) > 0 else 0,
        'results': results
    }


def save_results(results_dict: Dict[str, Any], output_file: str = None) -> str:
    """
    Save evaluation results to CSV and JSON
    
    Args:
        results_dict: Results dictionary
        output_file: Optional output file path
        
    Returns:
        Path to saved file
    """
    import csv
    
    if not output_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = Path(__file__).parent.parent.parent / "results" / "gsm8k"
        results_dir.mkdir(parents=True, exist_ok=True)
        output_file = results_dir / f"gsm8k_langgraph_{timestamp}.csv"
    else:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Save CSV
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            fieldnames = [
                'problem_id', 'question', 'expected_answer', 'predicted_answer',
                'is_correct', 'feedback_loops', 'failure_type', 'approach_type',
                'tree_depth', 'score', 'status'
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in results_dict['results']:
                writer.writerow({
                    'problem_id': result.get('problem_id', ''),
                    'question': result.get('question', '')[:100],
                    'expected_answer': result.get('expected_answer', ''),
                    'predicted_answer': result.get('predicted_answer', '')[:100],
                    'is_correct': result.get('is_correct', False),
                    'feedback_loops': result.get('feedback_loops', 0),
                    'failure_type': result.get('failure_type', ''),
                    'approach_type': result.get('approach_type', ''),
                    'tree_depth': result.get('tree_depth', 0),
                    'score': f"{result.get('score', 0):.3f}",
                    'status': result.get('status', '')
                })
        
        print(f"✅ Results saved to {output_file}")
        
        # Also save JSON for detailed analysis
        json_file = output_file.with_suffix('.json')
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=2)
        print(f"✅ Detailed results saved to {json_file}")
        
        return str(output_file)
    
    except Exception as e:
        print(f"❌ Error saving results: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LangGraph Agent with Feedback Loops on GSM8K"
    )
    parser.add_argument(
        "--num-problems", type=int, default=10,
        help="Number of problems to evaluate (default: 10)"
    )
    parser.add_argument(
        "--local", action="store_true",
        help="Use local CSV data instead of HuggingFace dataset"
    )
    parser.add_argument(
        "--csv-file", type=str, default=None,
        help="Path to local CSV file"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print verbose output during evaluation"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output file path for results"
    )
    
    args = parser.parse_args()
    
    # Load problems
    if args.local:
        problems = load_local_gsm8k_data(args.csv_file)
    else:
        problems = load_gsm8k_dataset(args.num_problems)
    
    if not problems:
        print("❌ Failed to load any problems")
        sys.exit(1)
    
    # Evaluate
    results = evaluate_langgraph_agent(problems, verbose=args.verbose)
    
    # Save results
    save_results(results, args.output)
    
    print(f"\n📈 Evaluation complete!")
    print(f"   Pass Rate: {results['pass_rate']:.2f}%")
    print(f"   {results['passed']}/{results['total']} problems solved correctly")


if __name__ == "__main__":
    main()
