"""
Run MATH-500 first 50 problems with vLLM backend (ReTreVal method).
Loads from local CSV, saves results to results/math500/ and logs to logs/.
"""
import csv
import json
import sys
import os
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Ensure project root on path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from src.agents.math500_retrevai import MATH500ReTreValSolver
from src.utils.memory import ReflexionMemory

START_INDEX = int(os.getenv('START_INDEX', '0'))
NUM_PROBLEMS = int(os.getenv('NUM_PROBLEMS', '50'))
CSV_PATH = PROJECT_ROOT / "data" / "math" / "math_500_test_with_answers.csv"
RESULTS_DIR = PROJECT_ROOT / "results" / "math500"
LOGS_DIR = PROJECT_ROOT / "logs"


def load_problems(csv_path: Path, n: int, start: int = 0) -> list:
    problems = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i < start:
                continue
            if i >= start + n:
                break
            problems.append({
                'id': f"MATH_{i}",
                'question': row['Question'],
                'answer': row['Answer'],
                'solution': row.get('Solution', ''),
                'subject': row.get('Subject', ''),
                'level': row.get('Level', ''),
            })
    return problems


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {NUM_PROBLEMS} problems (starting from index {START_INDEX}) from {CSV_PATH}")
    problems = load_problems(CSV_PATH, NUM_PROBLEMS, start=START_INDEX)
    print(f"Loaded {len(problems)} problems")

    # Print subject distribution
    subjects = {}
    for p in problems:
        s = p.get('subject', 'Unknown')
        subjects[s] = subjects.get(s, 0) + 1
    print(f"Subjects: {subjects}")

    # Initialize solver and memory
    solver = MATH500ReTreValSolver(verbose=True, max_depth=2)
    memory = ReflexionMemory(
        memory_file=str(PROJECT_ROOT / "reflexion_memory.md"),
        auto_save=False,
        load_on_init=True,
    )

    results = []
    passed = 0
    subject_stats = {}
    level_stats = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Incremental results file (append after each problem so nothing is lost)
    incremental_csv = RESULTS_DIR / f"math500_retrevai_vllm_{timestamp}.csv"
    with open(incremental_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'problem_id', 'subject', 'level', 'question',
            'expected_answer', 'predicted_answer', 'passed',
        ])
        writer.writeheader()

    print(f"\n{'='*80}")
    print(f"Running ReTreVal on {len(problems)} MATH-500 problems with vLLM (Qwen3.5-9B)")
    print(f"Results: {incremental_csv}")
    print(f"{'='*80}\n")

    for i, problem in enumerate(problems, 1):
        print(f"\n{'='*80}")
        print(f"[{i}/{len(problems)}] Problem {problem['id']} | Subject: {problem['subject']} | Level: {problem['level']}")
        print(f"Q: {problem['question'][:150]}...")
        print(f"Expected: {problem['answer']}")
        print(f"{'='*80}")

        try:
            result = solver.solve(problem['id'], problem['question'], expected_answer=problem['answer'])
        except Exception as e:
            print(f"ERROR solving {problem['id']}: {e}")
            result = {'problem_id': problem['id'], 'answer': '', 'solution': f'ERROR: {e}', 'passed': False}

        result['expected_answer'] = problem['answer']
        result['question'] = problem['question']
        result['subject'] = problem.get('subject', '')
        result['level'] = problem.get('level', '')

        is_correct = solver.compare_answers(result.get('answer', ''), problem['answer'])
        result['passed'] = is_correct

        if is_correct:
            passed += 1
            memory.add_success(f"Problem {problem['id']} ({problem['subject']}): approach worked")
            memory.record_success("correct_answer", {
                'problem_id': problem['id'],
                'subject': problem['subject'],
                'approach': 'retrevai',
            })
        else:
            memory.add_failure(f"Problem {problem['id']} ({problem['subject']}): predicted={result.get('answer','')}, expected={problem['answer']}")
            memory.record_failure("wrong_answer", {
                'problem_id': problem['id'],
                'subject': problem['subject'],
                'approach': 'retrevai',
                'predicted': result.get('answer', ''),
                'expected': problem['answer'],
            })

        memory.record_problem_type_result(problem.get('subject', 'Unknown'), is_correct)
        memory.update_approach_reliability('retrevai', is_correct)
        memory.increment_iteration()

        # Track stats
        subj = problem.get('subject', 'Unknown')
        if subj not in subject_stats:
            subject_stats[subj] = {'total': 0, 'passed': 0}
        subject_stats[subj]['total'] += 1
        if is_correct:
            subject_stats[subj]['passed'] += 1

        level = problem.get('level', '?')
        if level not in level_stats:
            level_stats[level] = {'total': 0, 'passed': 0}
        level_stats[level]['total'] += 1
        if is_correct:
            level_stats[level]['passed'] += 1

        results.append(result)

        # Append to incremental CSV
        with open(incremental_csv, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'problem_id', 'subject', 'level', 'question',
                'expected_answer', 'predicted_answer', 'passed',
            ])
            writer.writerow({
                'problem_id': result.get('problem_id', ''),
                'subject': result.get('subject', ''),
                'level': result.get('level', ''),
                'question': result.get('question', '')[:200],
                'expected_answer': result.get('expected_answer', ''),
                'predicted_answer': result.get('answer', ''),
                'passed': result.get('passed', False),
            })

        # Save memory every 5 problems
        if i % 5 == 0:
            memory.save()
            print(f"\n>>> Memory saved. Pass rate so far: {passed}/{i} ({passed/i*100:.1f}%)")

        status = "PASS" if is_correct else "FAIL"
        print(f"[{status}] Predicted: {result.get('answer', '')} | Expected: {problem['answer']} | Running: {passed}/{i} ({passed/i*100:.1f}%)")

    # Final save
    memory.save()

    # Print summary
    print(f"\n{'='*80}")
    print(f"FINAL RESULTS - ReTreVal + vLLM (Qwen3.5-9B)")
    print(f"{'='*80}")
    print(f"Total: {len(problems)} | Passed: {passed} | Failed: {len(problems) - passed}")
    print(f"Pass@1: {passed / len(problems) * 100:.1f}%")

    print(f"\nBy Subject:")
    for subj, stats in sorted(subject_stats.items()):
        rate = (stats['passed'] / stats['total'] * 100) if stats['total'] > 0 else 0
        print(f"  {subj}: {stats['passed']}/{stats['total']} ({rate:.1f}%)")

    print(f"\nBy Level:")
    for level, stats in sorted(level_stats.items()):
        rate = (stats['passed'] / stats['total'] * 100) if stats['total'] > 0 else 0
        print(f"  Level {level}: {stats['passed']}/{stats['total']} ({rate:.1f}%)")

    # Save full results JSON
    json_path = RESULTS_DIR / f"math500_retrevai_vllm_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump({
            'method': 'ReTreVal',
            'model': 'Qwen/Qwen3.5-9B',
            'backend': 'vllm',
            'total': len(problems),
            'passed': passed,
            'pass_rate': passed / len(problems) * 100,
            'subject_stats': subject_stats,
            'level_stats': level_stats,
            'timestamp': timestamp,
        }, f, indent=2)

    print(f"\nResults saved to:")
    print(f"  CSV: {incremental_csv}")
    print(f"  JSON: {json_path}")
    print(f"  Memory: {PROJECT_ROOT / 'reflexion_memory.md'}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
