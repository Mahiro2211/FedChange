"""
Visualization utilities for federated change detection experiments.

Generates publication-style figures from the results.json files written by
fed_main.py / cen_main.py:

    1. Convergence curves  : metric vs. communication round (multi-run overlay,
                             with mean ± std shaded band across seeds)
    2. Train-loss curves   : training loss vs. round
    3. Dirichlet alpha sweep: metric vs. alpha (log-x), with error bars (seed std)
    4. Prediction overlays : TP/FP/FN colored change maps from a checkpoint

Usage (CLI):
    python -m fed_cd.plot --results_root results/alg_comparison \
        --convergence --metric mf1 --split test --out results/figs
    python -m fed_cd.plot --train_loss --results_root results/alg_comparison
    python -m fed_cd.plot --alpha_sweep --results_root results/fed_bcd --split test
    python -m fed_cd.plot --predictions --ckpt results/.../best_ckpt.pt \
        --net_G base_transformer_pos_s4_dd8 --data_root ../WHU-GCD \
        --split test --n 8 --out results/figs
"""

import os
import re
import argparse
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend; safe on headless servers
    import matplotlib.pyplot as plt
except ImportError as _e:  # pragma: no cover
    raise SystemExit(
        "matplotlib is required for plot.py. Install with: pip install matplotlib"
    ) from _e

# Reuse the seed-grouping logic already validated in summarize.py
from fed_cd.summarize import collect_results, aggregate_seeds


# ──────────────────────── shared helpers ────────────────────────

_METRIC_LABELS = {
    "mf1": "mF1 (%)", "miou": "mIoU (%)",
    "f1_change": "F1 (change class, %)", "oa": "Overall Accuracy (%)",
}

# A pleasant, colorblind-friendly palette.
_PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
            "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]


def _sanitize_metric(hist_key, value):
    """Convert a stored metric value to a 0-100 percentage when appropriate."""
    if value is None:
        return None
    # mF1 / mIoU / oa are in [0,1]; scale to percent.
    if hist_key in ("mf1", "miou", "oa", "f1_change", "iou_change"):
        return float(value) * 100.0
    return float(value)


def _group_by_base(results):
    """Group runs by base experiment name (strip trailing _s<seed>).

    Returns: {base_name: [result, ...]} (ALL groups, including singletons).
    """
    seed_re = re.compile(r"_s(\d+)$")
    groups = {}
    for r in results:
        exp = r.get("experiment", r.get("_path", "?"))
        base = exp[:seed_re.search(exp).start()] if seed_re.search(exp) else exp
        groups.setdefault(base, []).append(r)
    return groups


def _series_from_history(runs, split, hist_key):
    """Align per-run eval_history into a mean curve + std band.

    Each run's eval_history[split] is a list of {round, <hist_key>, ...}.
    Different runs may be evaluated at slightly different rounds, so we
    interpolate each onto the union round grid.

    Returns: (rounds, mean, std) or None if no usable data.
    """
    per_run = []  # list of (rounds_array, values_array)
    for r in runs:
        hist = r.get("eval_history", {}).get(split, [])
        if not hist:
            continue
        xs = np.array([rec["round"] for rec in hist], dtype=float)
        ys = np.array(
            [_sanitize_metric(hist_key, rec.get(hist_key)) for rec in hist],
            dtype=float,
        )
        # drop None entries
        mask = ~np.isnan(ys)
        if mask.sum() >= 2:
            per_run.append((xs[mask], ys[mask]))
    if not per_run:
        return None

    grid = sorted(set(np.concatenate([xs for xs, _ in per_run])))
    grid = np.array(grid, dtype=float)
    aligned = []
    for xs, ys in per_run:
        aligned.append(np.interp(grid, xs, ys))
    mat = np.array(aligned)  # (n_runs, n_rounds)
    mean = mat.mean(axis=0)
    std = mat.std(axis=0, ddof=1) if mat.shape[0] > 1 else np.zeros_like(mean)
    return grid, mean, std


def _ensure_out_dir(out):
    os.makedirs(out, exist_ok=True)
    return out


# ──────────────────────── 1. convergence curves ────────────────────────

def plot_convergence(results, split="test", metric="mf1", out="results/figs",
                     title=None):
    """Metric-vs-round convergence curves, one line per base experiment.

    Multi-seed runs (same base name, different _s<seed>) are aggregated into a
    mean line with a shaded ±1 std band.
    """
    groups = _group_by_base(results)
    if not groups:
        print("No results to plot.")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    plotted = 0
    for i, base in enumerate(sorted(groups.keys())):
        series = _series_from_history(groups[base], split, metric)
        if series is None:
            continue
        rounds, mean, std = series
        color = _PALETTE[i % len(_PALETTE)]
        n_seeds = len(groups[base])
        label = f"{base}" + (f" (n={n_seeds})" if n_seeds > 1 else "")
        ax.plot(rounds, mean, color=color, label=label, linewidth=1.8)
        if n_seeds > 1:
            ax.fill_between(rounds, mean - std, mean + std, color=color, alpha=0.15)
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        print(f"No eval_history data for split='{split}' metric='{metric}'.")
        return

    ax.set_xlabel("Communication round")
    ax.set_ylabel(_METRIC_LABELS.get(metric, metric))
    ax.set_title(title or f"Convergence on {split} ({metric})")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    out = _ensure_out_dir(out)
    path = os.path.join(out, f"convergence_{metric}_{split}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


# ──────────────────────── 2. train-loss curves ────────────────────────

def plot_train_loss(results, out="results/figs", title=None):
    """Training-loss-vs-round curves, one line per base experiment.

    train_loss_history is indexed by round (one entry per round), so no
    interpolation is needed; multi-seed runs are averaged into a mean band.
    """
    groups = _group_by_base(results)
    if not groups:
        print("No results to plot.")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    plotted = 0
    for i, base in enumerate(sorted(groups.keys())):
        runs = groups[base]
        per_run = [np.array(r.get("train_loss_history", []), dtype=float)
                   for r in runs if r.get("train_loss_history")]
        if not per_run:
            continue
        # align on the shortest length (rounds should match across seeds)
        min_len = min(len(a) for a in per_run)
        mat = np.array([a[:min_len] for a in per_run])  # (n_runs, min_len)
        rounds = np.arange(1, min_len + 1, dtype=float)
        mean = mat.mean(axis=0)
        color = _PALETTE[i % len(_PALETTE)]
        n_seeds = mat.shape[0]
        label = f"{base}" + (f" (n={n_seeds})" if n_seeds > 1 else "")
        ax.plot(rounds, mean, color=color, label=label, linewidth=1.8)
        if n_seeds > 1:
            std = mat.std(axis=0, ddof=1)
            ax.fill_between(rounds, mean - std, mean + std, color=color, alpha=0.15)
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        print("No train_loss_history data found.")
        return

    ax.set_xlabel("Communication round")
    ax.set_ylabel("Training loss")
    ax.set_title(title or "Training loss")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = _ensure_out_dir(out)
    path = os.path.join(out, "train_loss.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


# ──────────────────────── 3. Dirichlet alpha sweep ────────────────────────

# matches dirichlet_01 / dirichlet_05 / dirichlet_10 / dirichlet_100 in project names
_ALPHA_RE = re.compile(r"dirichlet_?(\d+)")


def _parse_alpha(name):
    """Extract the Dirichlet alpha from an experiment name, else None."""
    m = _ALPHA_RE.search(name)
    if not m:
        return None
    raw = m.group(1)
    # dirichlet_01 -> 0.1, _05 -> 0.5, _10 -> 1.0, _100 -> 100.0
    if raw == "100":
        return 100.0
    return float(raw[0] + "." + raw[1:]) if len(raw) == 2 else float(raw)


def plot_alpha_sweep(results, split="test", metric="mf1", out="results/figs",
                     title=None):
    """Metric vs Dirichlet alpha (log-x), with error bars from seeds.

    Groups runs by the alpha parsed from their name (e.g. dirichlet_01 -> 0.1)
    and by algorithm prefix (FedAvg / FedProx / ...) so each algorithm gets its
    own curve.
    """
    # bucket: (algo, alpha) -> [value, ...]
    # algo prefix = the part of the base name before the dirichlet tag
    buckets = {}
    groups = _group_by_base(results)
    for base, runs in groups.items():
        alpha = _parse_alpha(base)
        if alpha is None:
            continue
        # algorithm prefix = everything before dirichlet tag
        algo = _ALPHA_RE.split(base)[0].rstrip("_")
        if not algo:
            algo = base
        for r in runs:
            sc = r.get("final_results", {}).get(split, {})
            v = _sanitize_metric(metric, sc.get(metric))
            if v is not None:
                buckets.setdefault((algo, alpha), []).append(v)

    if not buckets:
        print(f"No Dirichlet-alpha runs found for split='{split}' metric='{metric}'.")
        print("Expected experiment names like '*dirichlet_01*', '*dirichlet_05*', ...")
        return

    # gather per-algorithm sorted (alphas, means, stds)
    by_algo = {}
    for (algo, alpha), vals in buckets.items():
        by_algo.setdefault(algo, []).append(
            (alpha, float(np.mean(vals)),
             float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
             len(vals)))
    for algo in by_algo:
        by_algo[algo].sort(key=lambda t: t[0])

    fig, ax = plt.subplots(figsize=(7, 5))
    for i, algo in enumerate(sorted(by_algo.keys())):
        rows = by_algo[algo]
        alphas = np.array([r[0] for r in rows])
        means = np.array([r[1] for r in rows])
        stds = np.array([r[2] for r in rows])
        color = _PALETTE[i % len(_PALETTE)]
        ax.errorbar(alphas, means, yerr=stds, color=color, marker="o",
                    capsize=4, linewidth=1.8, label=algo)

    ax.set_xscale("log")
    ax.set_xlabel(r"Dirichlet $\alpha$ (smaller = more heterogeneous)")
    ax.set_ylabel(_METRIC_LABELS.get(metric, metric))
    ax.set_title(title or f"Dirichlet $\\alpha$ sweep on {split} ({metric})")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=9)
    fig.tight_layout()
    out = _ensure_out_dir(out)
    path = os.path.join(out, f"alpha_sweep_{metric}_{split}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


# ──────────────────────── 4. prediction overlays ────────────────────────

def plot_predictions(ckpt, net_g="base_transformer_pos_s4_dd8",
                     data_root="../WHU-GCD", split="test", n=8,
                     img_size=256, out="results/figs", device=None):
    """Render TP/FP/FN-colored change maps for the first n test samples.

    Coloring: TP=green, FP=red, FN=blue, TN=black (overlay on a faint T2 image).
    """
    import torch
    from fed_cd.models import build_cd_model
    from fed_cd.data.cd_dataset import build_eval_dataset
    from fed_cd.data.data_partition import scan_evaluation_set
    from torch.utils.data import DataLoader

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = build_cd_model(net_name=net_g, num_classes=2, pretrained=False).to(device)
    sd = torch.load(ckpt, map_location=device, weights_only=False)
    state = sd["model"] if isinstance(sd, dict) and "model" in sd else sd
    model.load_state_dict(state)
    model.eval()

    samples = scan_evaluation_set(data_root, split)[:n]
    if not samples:
        print(f"No samples found in {data_root}/{split}.")
        return
    ds = build_eval_dataset(samples, img_size=img_size, task="bcd", data_root=data_root)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    cols = n
    fig, axes = plt.subplots(3, cols, figsize=(2.5 * cols, 7.5))
    if cols == 1:
        axes = np.array([[axes[0]], [axes[1]], [axes[2]]]).T  # normalize shape

    for idx, batch in enumerate(loader):
        img1 = batch["A"].to(device)
        img2 = batch["B"].to(device)
        label = batch["L"].to(device).long()
        if label.dim() == 4:
            label = label.squeeze(1)
        with torch.no_grad():
            logits = model(img1, img2)
            if logits.shape[-2:] != label.shape[-2:]:
                logits = torch.nn.functional.interpolate(
                    logits, size=label.shape[-2:], mode="bilinear", align_corners=True)
            pred = torch.argmax(logits, dim=1)

        gt = label[0].cpu().numpy()
        pr = pred[0].cpu().numpy()
        # overlay color image: H x W x 3
        overlay = np.zeros((*gt.shape, 3), dtype=np.float32)
        tp = (gt == 1) & (pr == 1)
        fp = (gt == 0) & (pr == 1)
        fn = (gt == 1) & (pr == 0)
        tn = (gt == 0) & (pr == 0)
        overlay[tp] = [0.0, 0.8, 0.0]   # green
        overlay[fp] = [0.9, 0.0, 0.0]   # red
        overlay[fn] = [0.0, 0.3, 0.9]   # blue
        overlay[tn] = [0.05, 0.05, 0.05]  # near-black

        t1 = img1[0].cpu().permute(1, 2, 0).numpy()
        t2 = img2[0].cpu().permute(1, 2, 0).numpy()
        # de-normalize (mean/std = 0.5/0.5 in CDDataAugmentation)
        t1 = (t1 * 0.5 + 0.5).clip(0, 1)
        t2 = (t2 * 0.5 + 0.5).clip(0, 1)

        axes[0, idx].imshow(t1); axes[0, idx].axis("off")
        axes[0, idx].set_title(batch["name"][0][:14], fontsize=7)
        axes[1, idx].imshow(t2); axes[1, idx].axis("off")
        axes[2, idx].imshow(overlay); axes[2, idx].axis("off")
        if idx == 0:
            axes[0, idx].set_ylabel("T1", fontsize=9)
            axes[1, idx].set_ylabel("T2", fontsize=9)
            axes[2, idx].set_ylabel("Pred\n(green=TP, red=FP, blue=FN)", fontsize=7)

    # hide unused columns
    for j in range(len(samples), cols):
        for r in range(3):
            axes[r, j].axis("off")

    fig.suptitle(f"Prediction overlays on {split} (n={len(samples)})", fontsize=11)
    fig.tight_layout()
    out = _ensure_out_dir(out)
    path = os.path.join(out, f"predictions_{split}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ──────────────────────── CLI ────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Plot federated change-detection experiment figures.")
    p.add_argument("--results_root", type=str, default="results",
                   help="root dir scanned for results.json")
    p.add_argument("--out", type=str, default="results/figs",
                   help="output directory for figures")
    p.add_argument("--split", type=str, default="test",
                   help="eval split for convergence / alpha-sweep plots")
    p.add_argument("--metric", type=str, default="mf1",
                   choices=list(_METRIC_LABELS.keys()),
                   help="metric for convergence / alpha-sweep plots")

    p.add_argument("--convergence", action="store_true",
                   help="plot convergence curves (metric vs round)")
    p.add_argument("--train_loss", action="store_true",
                   help="plot training-loss curves")
    p.add_argument("--alpha_sweep", action="store_true",
                   help="plot Dirichlet-alpha sweep with error bars")

    # prediction overlay (independent of results_root)
    p.add_argument("--predictions", action="store_true",
                   help="render TP/FP/FN prediction overlays from a checkpoint")
    p.add_argument("--ckpt", type=str, default="",
                   help="checkpoint path for --predictions")
    p.add_argument("--net_G", type=str, default="base_transformer_pos_s4_dd8",
                   help="model name for --predictions")
    p.add_argument("--data_root", type=str, default="../WHU-GCD",
                   help="dataset root for --predictions")
    p.add_argument("--n", type=int, default=8,
                   help="number of samples for --predictions")

    args = p.parse_args()

    did_anything = False

    if args.predictions:
        if not args.ckpt:
            print("--predictions requires --ckpt <path>")
        else:
            plot_predictions(args.ckpt, net_g=args.net_G, data_root=args.data_root,
                             split=args.split, n=args.n, out=args.out)
            did_anything = True

    # the other plots all need results.json scanning
    needs_results = args.convergence or args.train_loss or args.alpha_sweep
    if needs_results:
        results = collect_results(args.results_root)
        print(f"Found {len(results)} result(s) in {args.results_root}/")

    if args.convergence:
        plot_convergence(results, split=args.split, metric=args.metric, out=args.out)
        did_anything = True
    if args.train_loss:
        plot_train_loss(results, out=args.out)
        did_anything = True
    if args.alpha_sweep:
        plot_alpha_sweep(results, split=args.split, metric=args.metric, out=args.out)
        did_anything = True

    if not did_anything:
        print("Nothing to do. Pass at least one of "
              "--convergence / --train_loss / --alpha_sweep / --predictions.")


if __name__ == "__main__":
    main()
