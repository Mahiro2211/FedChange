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
    """Format final-results into a readable comparison table."""
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


def format_best_metrics_table(results, splits=("val", "test", "test2")):
    """Format per-metric best (mF1/mIoU/mPrec/mRec) into a comparison table.

    Reads the 'best_metrics' field written by fed_main.py / centralized_main.py.
    Each split x metric is independently the best across all training rounds.
    """
    has_best = any(r.get("best_metrics") for r in results)
    if not has_best:
        print("(no 'best_metrics' field found; skipping best-metrics table)")
        return

    metrics = (("mf1", "mF1"), ("miou", "mIoU"),
               ("mprecision", "mPrec"), ("mrecall", "mRec"))

    col_w = 9
    line1 = f"{'Experiment':<40s}"
    line2 = f"{'':40s}"
    for s in splits:
        line1 += f" | {s:^{col_w * 4 + 3}s}"
        line2 += " | " + " ".join(f"{lab:>{col_w}s}" for _, lab in metrics)
    sep = "=" * len(line2)
    print("\n" + line1)
    print(line2)
    print(sep)
    for r in results:
        exp = r.get("experiment", r.get("_path", "?"))
        bm = r.get("best_metrics", {})
        first = True
        for s in splits:
            row = (f"{exp:<40s}" if first else f"{'':40s}")
            sc = bm.get(s, {})
            cells = []
            for m_key, _ in metrics:
                entry = sc.get(m_key, {})
                val = entry.get("value", 0.0) * 100
                cells.append(f"{val:{col_w - 1}.2f}%")
            row += f" | " + " ".join(cells) + f"   <- {s}"
            print(row)
            first = False
    print(sep + "\n")


def save_summary(results, output_path="results/summary.json"):
    """Save summary as JSON."""
    summary = []
    for r in results:
        summary.append({
            "experiment": r.get("experiment", ""),
            "args": r.get("args", {}),
            "best_metric": r.get("best_metric", r.get("best_metric", 0)),
            "best_metrics": r.get("best_metrics", {}),
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

    print("[ Final results (best ckpt by val mF1) ]")
    format_table(results)

    print("[ Per-metric best across all rounds ]")
    format_best_metrics_table(results)

    save_summary(results, args.output)


if __name__ == "__main__":
    main()
