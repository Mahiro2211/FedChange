"""
Collect and summarize all experiment results into a comparison table.

Scans results/ directory for results.json files and prints/saves a summary.
"""

import os
import sys
import json
import argparse
from pathlib import Path


def collect_results(results_root="results"):
    """Scan all results.json files and return a list of result dicts."""
    all_results = []
    for json_path in sorted(Path(results_root).rglob("results.json")):
        with open(json_path, "r") as f:
            data = json.load(f)
        data["_path"] = str(json_path)
        all_results.append(data)
    return all_results


def format_table(results, splits=("val", "test", "test2")):
    """Format results into a readable comparison table."""
    if not results:
        print("No results found.")
        return

    header = f"\n{'Experiment':<40s}"
    for s in splits:
        header += f" | {s:^22s}"
    header += "\n"
    header += f"{'':40s}"
    for s in splits:
        header += f" | {'mF1':>7s} {'mIoU':>7s} {'F1_c':>7s}"
    header += "\n" + "=" * len(header)

    print(header)
    for r in results:
        exp = r.get("experiment", r.get("_path", "?"))
        final = r.get("final_results", {})
        row = f"{exp:<40s}"
        for s in splits:
            sc = final.get(s, {})
            mf1 = sc.get("mf1", 0.0) * 100
            miou = sc.get("miou", 0.0) * 100
            f1c = sc.get("f1_1", 0.0) * 100
            row += f" | {mf1:>6.2f}% {miou:>6.2f}% {f1c:>6.2f}%"
        print(row)
    print("=" * len(header) + "\n")


def save_summary(results, output_path="results/summary.json"):
    """Save summary as JSON."""
    summary = []
    for r in results:
        summary.append({
            "experiment": r.get("experiment", ""),
            "args": r.get("args", {}),
            "best_metric": r.get("best_metric", r.get("best_metric", 0)),
            "final_results": r.get("final_results", {}),
        })
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Summarize experiment results")
    parser.add_argument("--results_root", type=str, default="results")
    parser.add_argument("--output", type=str, default="results/summary.json")
    args = parser.parse_args()

    results = collect_results(args.results_root)
    print(f"\nFound {len(results)} experiment(s) in {args.results_root}/")
    format_table(results)
    save_summary(results, args.output)


if __name__ == "__main__":
    main()
