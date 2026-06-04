"""Evaluate the simplified ReTreVal agent on the bundled Math500 CSV."""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

from dotenv import load_dotenv

from src.agents.agent_langgraph import extract_answer, run_agent


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = REPO_ROOT / "data" / "math" / "math_500_test_with_answers.csv"


@dataclass
class EvalRow:
    problem_id: str
    subject: str
    level: str
    question: str
    expected_answer: str
    predicted_answer: str
    passed: bool
    final_output: str
    seconds: float


def normalize_answer(value: str) -> str:
    value = extract_answer(value or "")
    value = value.strip()
    value = re.sub(r"\\boxed\{(.+)\}", r"\1", value)
    value = value.replace("\\left", "").replace("\\right", "")
    value = value.replace("\\,", "")
    value = value.replace("$", "")
    value = re.sub(r"\s+", "", value)
    value = value.strip("{}().,")
    return value.lower()


def answers_match(predicted: str, expected: str) -> bool:
    return normalize_answer(predicted) == normalize_answer(expected)


def load_math500(path: Path, limit: int) -> Iterable[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader):
            if index >= limit:
                break
            yield {
                "problem_id": f"math500_{index + 1}",
                "question": row["Question"],
                "answer": row["Answer"],
                "subject": row.get("Subject", ""),
                "level": row.get("Level", ""),
            }


def evaluate(args: argparse.Namespace) -> List[EvalRow]:
    rows: List[EvalRow] = []
    problems = list(load_math500(args.dataset, args.limit))
    for index, problem in enumerate(problems, start=1):
        print(f"[{index}/{len(problems)}] {problem['problem_id']}", flush=True)
        started = time.time()
        result = run_agent(
            problem["question"],
            expected_answer=problem["answer"],
            problem_id=problem["problem_id"],
            iterations=args.iterations,
            verbose=args.verbose,
            provider=args.provider,
            model=args.model,
            memory_file=str(args.memory_file) if args.memory_file else None,
            use_memory=not args.no_memory,
            oracle=args.oracle,
        )
        seconds = time.time() - started
        predicted = result.get("predicted_answer", "")
        expected = problem["answer"]
        passed = answers_match(predicted, expected)
        rows.append(
            EvalRow(
                problem_id=problem["problem_id"],
                subject=problem["subject"],
                level=problem["level"],
                question=problem["question"],
                expected_answer=expected,
                predicted_answer=predicted,
                passed=passed,
                final_output=result.get("final_output", ""),
                seconds=seconds,
            )
        )
        print(f"  predicted={predicted!r} expected={expected!r} passed={passed}")
    return rows


def save_rows(rows: List[EvalRow], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"math500_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(asdict(row), ensure_ascii=True) + "\n")
    return output_path


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Evaluate ReTreVal on the bundled Math500 CSV.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--limit", type=int, default=5, help="Number of problems to evaluate.")
    parser.add_argument("--iterations", type=int, default=1, help="Refinement passes per problem.")
    parser.add_argument("--provider", default=None, help="LLM provider: gemini, openai, ollama, or vllm.")
    parser.add_argument("--model", default=None, help="Provider model name.")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results" / "math500")
    parser.add_argument(
        "--memory-file",
        type=Path,
        default=Path(os.environ["MEMORY_FILE"]) if os.environ.get("MEMORY_FILE") else REPO_ROOT / "memory.md",
        help="Shared cross-problem memory file (default: <repo>/memory.md).",
    )
    parser.add_argument("--no-memory", action="store_true", help="Disable persistent memory.")
    parser.add_argument(
        "--oracle",
        action="store_true",
        help="Show the reference answer to the agent during solving. "
        "Off by default — the agent solves without seeing the reference answer.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    rows = evaluate(args)
    passed = sum(row.passed for row in rows)
    total = len(rows)
    output_path = save_rows(rows, args.output_dir)

    print("\nMath500 summary")
    print(f"Total: {total}")
    print(f"Passed: {passed}")
    print(f"Accuracy: {(passed / total * 100) if total else 0:.1f}%")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
