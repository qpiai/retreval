"""Summarize ReTreVal MATH-500 evaluation runs.

Reads the JSONL files written by ``src.agents.math500_eval`` (one row per
problem) and reports exact-match accuracy, latency, and a per-level / per-subject
breakdown. Pass one or more run files, or a directory (defaults to
``results/math500/``, newest run picked automatically).

    python analyze_results.py                          # newest run in results/math500/
    python analyze_results.py results/math500/run_a.jsonl results/math500/run_b.jsonl
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize ReTreVal MATH-500 eval runs.")
    p.add_argument(
        "runs",
        nargs="*",
        type=Path,
        help="JSONL run files. If omitted, the newest file in --results-dir is used.",
    )
    p.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/math500"),
        help="Directory of MATH-500 result JSONL files (default: results/math500).",
    )
    return p.parse_args()


def load_rows(path: Path) -> List[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def summarize(path: Path) -> None:
    rows = load_rows(path)
    if not rows:
        print(f"{path}: (empty)")
        return

    total = len(rows)
    passed = sum(1 for r in rows if r.get("passed"))
    seconds = [float(r.get("seconds", 0)) for r in rows]
    avg_s = sum(seconds) / total if total else 0.0

    print("=" * 78)
    print(f"RUN: {path.name}   ({total} problems)")
    print("=" * 78)
    print(f"  Accuracy (exact-match) : {passed}/{total}  =  {passed / total * 100:5.1f}%")
    print(f"  Avg time / problem     : {avg_s:6.2f}s   (total {sum(seconds) / 60:.1f} min)")

    by_level: dict = defaultdict(lambda: [0, 0])
    by_subject: dict = defaultdict(lambda: [0, 0])
    for r in rows:
        lvl, sub = str(r.get("level", "?")), str(r.get("subject", "?"))
        by_level[lvl][1] += 1
        by_subject[sub][1] += 1
        if r.get("passed"):
            by_level[lvl][0] += 1
            by_subject[sub][0] += 1

    print("\n  By level:")
    for lvl in sorted(by_level):
        ok, n = by_level[lvl]
        print(f"    level {lvl:<3} {ok:>3}/{n:<3}  {ok / n * 100:5.1f}%")

    print("\n  By subject:")
    for sub in sorted(by_subject):
        ok, n = by_subject[sub]
        print(f"    {sub:<24} {ok:>3}/{n:<3}  {ok / n * 100:5.1f}%")
    print()


def main() -> None:
    args = parse_args()
    runs = list(args.runs)
    if not runs:
        if not args.results_dir.exists():
            raise SystemExit(
                f"No run files given and {args.results_dir} does not exist. "
                "Run an eval first: python -m src.agents.math500_eval --limit 100"
            )
        candidates = sorted(args.results_dir.glob("*.jsonl"))
        if not candidates:
            raise SystemExit(f"No .jsonl runs found in {args.results_dir}.")
        runs = [candidates[-1]]  # newest by timestamped name

    for path in runs:
        summarize(path)


if __name__ == "__main__":
    main()
