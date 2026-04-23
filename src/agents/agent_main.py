"""Command-line entry point for the simplified ReTreVal agent."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from src.agents.agent_langgraph import run_agent


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run the ReTreVal agent on one problem.")
    parser.add_argument("--prompt", required=True, help="Problem to solve.")
    parser.add_argument("--expected-answer", default="", help="Optional expected answer.")
    parser.add_argument("--problem-id", default="", help="Optional problem identifier.")
    parser.add_argument("--iterations", type=int, default=1, help="Number of refinement passes.")
    parser.add_argument("--provider", default=None, help="LLM provider: gemini, openai, ollama, or vllm.")
    parser.add_argument("--model", default=None, help="Provider model name.")
    parser.add_argument("--verbose", action="store_true", help="Print intermediate model outputs.")
    parser.add_argument("--save", type=Path, default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    result = run_agent(
        args.prompt,
        expected_answer=args.expected_answer,
        problem_id=args.problem_id,
        iterations=args.iterations,
        verbose=args.verbose,
        provider=args.provider,
        model=args.model,
    )

    print(result.get("final_output", ""))
    print(f"\nPredicted answer: {result.get('predicted_answer', '')}")

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Saved: {args.save}")


if __name__ == "__main__":
    main()
