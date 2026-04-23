"""
HumanEval Benchmark Results Analysis
Compares performance of all 4 reasoning frameworks
"""

import argparse
import pandas as pd
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare HumanEval benchmark result CSVs.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/humaneval"),
        help="Directory containing HumanEval result CSV files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir

    files = {
        "ReTreVal": results_dir / "humaneval_retreval.csv",
        "ReAct": results_dir / "humaneval_react.csv",
        "Reflexion": results_dir / "humaneval_reflexion.csv",
        "Self-Refine": results_dir / "humaneval_self-refine.csv",
    }

    results = {}
    for method, filepath in files.items():
        if filepath.exists():
            df = pd.read_csv(filepath)
            passed = df["passed"].sum()
            total = len(df)
            pass_rate = (passed / total * 100) if total > 0 else 0
            avg_time = df["time_taken"].mean()

            results[method] = {
                "passed": passed,
                "total": total,
                "pass_rate": pass_rate,
                "avg_time": avg_time,
                "total_time": df["time_taken"].sum(),
            }

    print("\n" + "=" * 100)
    print("HUMANEVAL BENCHMARK RESULTS")
    print("=" * 100)
    print(f"{'Method':<15} {'Passed':<10} {'Total':<10} {'Pass@1':<12} {'Avg Time':<12} {'Total Time':<12}")
    print("-" * 100)

    for method in ["ReTreVal", "ReAct", "Reflexion", "Self-Refine"]:
        if method in results:
            r = results[method]
            print(
                f"{method:<15} {r['passed']:<10} {r['total']:<10} "
                f"{r['pass_rate']:>6.2f}%     {r['avg_time']:>8.2f}s     {r['total_time']:>8.2f}s"
            )

    print("=" * 100)

    if results:
        max_pass = max(r["pass_rate"] for r in results.values())
        min_pass = min(r["pass_rate"] for r in results.values())
        pass_diff = max_pass - min_pass

        print("\nKEY FINDINGS:\n")
        print(f"1. Pass Rate Range: {min_pass:.2f}% - {max_pass:.2f}% (Difference: {pass_diff:.2f}%)")

        max_speed = max(r["avg_time"] for r in results.values())
        min_speed = min(r["avg_time"] for r in results.values())
        speed_factor = max_speed / min_speed if min_speed > 0 else 0

        print(f"2. Speed Range: {min_speed:.2f}s - {max_speed:.2f}s avg per problem")
        print(f"   Fastest vs Slowest: {speed_factor:.1f}x difference")

        print("\n3. Trade-offs:")
        for method, r in results.items():
            if r["pass_rate"] >= max_pass - 1:
                print(f"   {method}: Highest pass rate ({r['pass_rate']:.2f}%)")
            if r["avg_time"] <= min_speed + 0.5:
                print(f"   {method}: Fastest execution ({r['avg_time']:.2f}s avg)")
            if r["avg_time"] >= max_speed - 1:
                print(f"   {method}: Most thorough exploration ({r['avg_time']:.2f}s avg)")

    print("\n" + "=" * 100)


if __name__ == "__main__":
    main()
