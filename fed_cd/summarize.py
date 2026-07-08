"""
Collect and summarize all experiment results into a comparison table.

Scans results/ directory for results.json files and prints/saves a summary.
"""

import os
import sys
import json
import re
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np

from fed_cd.models import BIT_CD_MODELS, TORCHANGE_MODELS


_MODEL_LOSS_FAMILY = {}
for _m in BIT_CD_MODELS:
    _MODEL_LOSS_FAMILY[_m] = 'CE'
for _m in TORCHANGE_MODELS:
    _MODEL_LOSS_FAMILY[_m] = 'BCE+Dice(native)'


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


def aggregate_seeds(results):
    """Group multi-seed runs by base experiment name and compute mean±std.

    A run's project_name is expected to carry a trailing ``_s<seed>`` (e.g.
    ``FedAvg_noniid1_K70_bcd_s42``). The base name strips that suffix so that
    runs sharing the same algo+partition but different seeds aggregate together.

    Args:
        results: list of result dicts (each from a results.json).

    Returns:
        dict: {base_name: {'count': int, 'runs': [results], 'splits': {...}}}
        Only groups with >= 2 seeds are returned (single runs are not aggregated).
        None if no seeded runs are found.
    """
    # Match trailing _s<number>, e.g. _s42, _s2024, _s0
    seed_re = re.compile(r'_s(\d+)$')
    groups = defaultdict(list)
    for r in results:
        exp = r.get("experiment", r.get("_path", "?"))
        base = exp
        m = seed_re.search(exp)
        if m:
            base = exp[:m.start()]
        groups[base].append(r)

    # only keep groups with more than one seed run
    multi = {k: v for k, v in groups.items() if len(v) >= 2}
    return multi if multi else None


def format_seed_summary_table(results, splits=("val", "test", "test2")):
    """Format multi-seed mean±std table.

    Reads final_results from each seed run, groups by base name, and reports
    mean ± std of mF1 / mIoU / F1_change per split.
    """
    groups = aggregate_seeds(results)
    if not groups:
        print("(no multi-seed runs found (_s<seed> suffix); skipping seed table)")
        return

    metrics = (("mf1", "mF1"), ("miou", "mIoU"), ("f1_1", "F1_c"))

    col_w = 18
    header = f"\n{'Experiment (seeds)':<40s}"
    for s in splits:
        header += f" | {s:^{col_w * len(metrics)}}"
    header += "\n" + " " * 40
    for s in splits:
        header += " | " + " ".join(f"{lab:>{col_w - 1}s}" for _, lab in metrics)
    sep = "=" * len(header)
    print("[ Multi-seed mean ± std (final results) ]")
    print(header)
    print(sep)

    for base in sorted(groups.keys()):
        runs = groups[base]
        n = len(runs)
        row = f"{base + f' (n={n})':<40s}"
        for s in splits:
            cells = []
            for m_key, _ in metrics:
                vals = []
                for r in runs:
                    sc = r.get("final_results", {}).get(s, {})
                    v = sc.get(m_key, None)
                    if v is not None:
                        vals.append(v * 100)
                if vals:
                    mean = float(np.mean(vals))
                    std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                    cells.append(f"{mean:>6.2f}±{std:.2f}%")
                else:
                    cells.append(f"{'n/a':>{col_w - 1}s}")
            row += " | " + " ".join(f"{c:>{col_w - 1}s}" for c in cells)
        print(row)
    print(sep + "\n")


def format_best_metrics_table(results, splits=("val", "test", "test2")):
    """Format per-metric best (mF1/mIoU/mPrec/mRec) into a comparison table.

    ⚠️ ORACLE / NOT FOR PAPER REPORTING ⚠️
    Reads the 'best_metrics' field written by fed_main.py, where each split ×
    metric is *independently* the best across all training rounds. This is an
    oracle upper bound that effectively peeks at the test set each round, so it
    overestimates generalization. Use ONLY for convergence/diagnostic analysis.
    For the main paper table, use ``final_results`` (a single checkpoint chosen
    by val mF1) via ``format_table`` / ``format_seed_summary_table``.
    """
    has_best = any(r.get("best_metrics") for r in results)
    if not has_best:
        print("(no 'best_metrics' field found; skipping best-metrics table)")
        return
    print("⚠️  ORACLE upper bound — NOT for paper reporting "
          "(each metric picked independently per round).")
    print("    For the main table use 'final_results' (val-selected ckpt) "
          "or the multi-seed mean±std table above.\n")

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


def format_model_comparison_table(results, splits=("val", "test", "test2")):
    """Format a cross-model comparison table annotated with net_G and loss family.

    Reads net_G from each result's saved args. Loss family is CE for BIT-CD and
    BCE+Dice (native) for torchange baselines, so the fairness of the comparison
    is explicit.
    """
    if not results:
        print("No results found.")
        return

    header = f"\n{'Experiment':<38s}{'net_G':<24s}{'loss':<18s}"
    for s in splits:
        header += f" | {s:^22s}"
    header += "\n" + " " * 80
    for s in splits:
        header += f" | {'mF1':>7s} {'mIoU':>7s} {'F1_c':>7s}"
    header += "\n" + "=" * len(header)
    print(header)

    for r in results:
        exp = r.get("experiment", r.get("_path", "?"))
        net_g = r.get("args", {}).get("net_G", "?")
        loss = _MODEL_LOSS_FAMILY.get(net_g, "?")
        final = r.get("final_results", r.get("results", {}))
        row = f"{exp:<38s}{net_g:<24s}{loss:<18s}"
        for s in splits:
            sc = final.get(s, {})
            mf1 = sc.get("mf1", 0.0) * 100
            miou = sc.get("miou", 0.0) * 100
            f1c = sc.get("f1_1", 0.0) * 100
            row += f" | {mf1:>6.2f}% {miou:>6.2f}% {f1c:>6.2f}%"
        print(row)
    print("=" * len(header) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Summarize experiment results")
    parser.add_argument("--results_root", type=str, default="results")
    parser.add_argument("--output", type=str, default="results/summary.json")
    args = parser.parse_args()

    results = collect_results(args.results_root)
    print(f"\nFound {len(results)} experiment(s) in {args.results_root}/")

    print("[ Cross-model comparison (final results, best ckpt by val mF1) ]")
    format_model_comparison_table(results)

    print("[ Final results (best ckpt by val mF1) ]")
    format_table(results)

    print("[ Per-metric best across all rounds ]")
    format_best_metrics_table(results)

    print("[ Multi-seed mean ± std ]")
    format_seed_summary_table(results)

    save_summary(results, args.output)


if __name__ == "__main__":
    main()
