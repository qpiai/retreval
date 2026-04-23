"""
Math Problem Solver using Agent
Reads questions from CSV and generates answers using the agent framework
"""

import csv
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv
import argparse
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.agents.agent_langgraph import run_agent

load_dotenv()

def read_questions_from_csv(csv_file: str, limit: int = None) -> List[Dict[str, Any]]:
    """
    Read math questions from CSV file.
    
    Args:
        csv_file: Path to CSV file with Question column
        limit: Optional limit on number of questions to process
        
    Returns:
        List of dictionaries with question data
    """
    questions = []
    
    if not os.path.exists(csv_file):
        print(f"❌ Error: File '{csv_file}' not found")
        return questions
    
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            if 'Question' not in reader.fieldnames:
                print(f"❌ Error: CSV must have 'Question' column")
                return questions
            
            for idx, row in enumerate(reader, 1):
                if limit and idx > limit:
                    break
                
                question = row['Question'].strip()
                if question:
                    item_dict = {
                        'id': idx,
                        'question': question
                    }
                    # Include answer if available (for feedback loop validation)
                    if 'Answer' in reader.fieldnames and row.get('Answer'):
                        item_dict['answer'] = row['Answer'].strip()
                    questions.append(item_dict)
    
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return questions
    
    print(f"📚 Loaded {len(questions)} questions from {csv_file}")
    return questions


def extract_final_answer(full_answer: str) -> str:
    """
    Keep the full solution with steps, but remove 'Final Answer in Plain English' section.
    
    Args:
        full_answer: Complete answer text from agent
        
    Returns:
        Full answer without plain English summary
    """
    import re
    
    if not full_answer or full_answer == 'NO_ANSWER_GENERATED':
        return full_answer
    
    # Remove "Final Answer in Plain English" section if present
    # Pattern to match the section header and its content
    plain_english_pattern = r'\n#+\s*Final Answer in Plain English\s*\n.*?(?=\n#+|\Z)'
    full_answer = re.sub(plain_english_pattern, '', full_answer, flags=re.IGNORECASE | re.DOTALL)
    
    # Also remove if it appears without markdown headers
    plain_english_pattern2 = r'\*\*Final Answer in Plain English[:\s]*\*\*.*?(?=\n\n|\Z)'
    full_answer = re.sub(plain_english_pattern2, '', full_answer, flags=re.IGNORECASE | re.DOTALL)
    
    # Clean up extra blank lines
    full_answer = re.sub(r'\n{3,}', '\n\n', full_answer)
    
    return full_answer.strip()


def solve_question(question_data: Dict[str, Any], verbose: bool = True) -> Dict[str, Any]:
    """
    Solve a single math question using the agent.
    
    Args:
        question_data: Dictionary with question info
        verbose: Whether to print verbose output
        
    Returns:
        Dictionary with question, answer, and metadata
    """
    question_id = question_data['id']
    question = question_data['question']
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"📝 Question {question_id}")
        print(f"{'='*80}")
        print(f"{question[:200]}..." if len(question) > 200 else question)
        print(f"{'='*80}\n")
    
    try:
        # Run the agent on this question
        iterations = int(os.getenv('MAX_LOOPS', '2'))
        expected_answer = question_data.get('answer')  # From CSV if available
        result = run_agent(
            question, 
            expected_answer=expected_answer,
            problem_id=f"MATH_{question_id}",
            iterations=iterations, 
            verbose=verbose
        )
        
        full_answer = result.get('final_output', 'NO_ANSWER_GENERATED')
        
        # Extract just the final answer
        final_answer = extract_final_answer(full_answer)
        
        return {
            'id': question_id,
            'question': question,
            'expected_answer': expected_answer,
            'answer': final_answer,
            'is_correct': result.get('is_correct', False),
            'feedback_loops': result.get('feedback_loops', 0),
            'error_analysis': result.get('error_analysis', ''),
            'failure_type': result.get('failure_type', ''),
            'status': 'success',
            'iterations': result.get('iterations', 0),
            'feasible': result.get('feasible', True)
        }
    
    except Exception as e:
        error_msg = str(e)
        if verbose:
            print(f"❌ Error solving question {question_id}: {error_msg}")
        
        return {
            'id': question_id,
            'question': question,
            'expected_answer': question_data.get('answer', ''),
            'answer': f"ERROR: {error_msg}",
            'is_correct': False,
            'feedback_loops': 0,
            'error_analysis': error_msg,
            'failure_type': 'runtime_error',
            'status': 'error',
            'iterations': 0,
            'feasible': False
        }


def save_results_to_csv(results: List[Dict[str, Any]], output_file: str):
    """
    Save results to CSV file.
    
    Args:
        results: List of result dictionaries
        output_file: Path to output CSV file
    """
    try:
        with open(output_file, 'w', encoding='utf-8', newline='') as f:
            fieldnames = ['id', 'question', 'expected_answer', 'answer', 'is_correct', 'feedback_loops', 'error_analysis', 'failure_type', 'status', 'iterations', 'feasible']
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')

            writer.writeheader()
            writer.writerows(results)

        print(f"\n✅ Results saved to: {output_file}")
        print(f"   Total questions processed: {len(results)}")

        # Print statistics
        successful = sum(1 for r in results if r['status'] == 'success')
        errors = sum(1 for r in results if r['status'] == 'error')
        correct = sum(1 for r in results if r.get('is_correct', False))

        print(f"\n📊 Statistics:")
        print(f"   ✓ Successful: {successful}")
        print(f"   ✗ Errors: {errors}")
        print(f"   ✅ Correct: {correct}/{len(results)} ({100*correct/len(results):.1f}%)")
        
    except Exception as e:
        print(f"❌ Error saving results: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Solve math problems from CSV using agent'
    )
    parser.add_argument(
        'input_csv',
        help='Input CSV file with Question column'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of questions to process (default: all)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output CSV file (default: auto-generated with timestamp)'
    )
    parser.add_argument(
        '--verbose',
        type=int,
        default=1,
        help='Verbose output (0=quiet, 1=verbose)'
    )
    
    args = parser.parse_args()
    
    # Set verbose mode
    verbose = bool(args.verbose)
    
    # Generate output filename if not provided
    if args.output:
        output_file = args.output
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        input_name = Path(args.input_csv).stem
        output_file = f"{input_name}_solutions_{timestamp}.csv"
    
    print(f"\n🚀 Math Problem Solver")
    print(f"{'='*80}")
    print(f"Input: {args.input_csv}")
    print(f"Output: {output_file}")
    print(f"Limit: {args.limit if args.limit else 'No limit'}")
    print(f"Model: {os.getenv('OLLAMA_MODEL', 'qwen2.5:3b')}")
    print(f"Max Loops: {os.getenv('MAX_LOOPS', '2')}")
    print(f"{'='*80}\n")
    
    # Read questions
    questions = read_questions_from_csv(args.input_csv, limit=args.limit)
    
    if not questions:
        print("❌ No questions to process")
        return 1
    
    # Process each question
    results = []
    
    for idx, question_data in enumerate(questions, 1):
        print(f"\n🔄 Processing {idx}/{len(questions)}...")
        
        result = solve_question(question_data, verbose=verbose)
        results.append(result)
        
        # Save progress after each question (in case of crashes)
        temp_output = output_file.replace('.csv', '_temp.csv')
        save_results_to_csv(results, temp_output)
    
    # Save final results
    save_results_to_csv(results, output_file)
    
    # Remove temp file if it exists
    temp_output = output_file.replace('.csv', '_temp.csv')
    if os.path.exists(temp_output):
        os.remove(temp_output)
    
    print(f"\n✅ All done! Results saved to: {output_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
