"""
Change detection evaluator.

Evaluates a trained model on val/test/test2 splits.
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from fed_cd.evaluation.cd_metrics import ConfuseMatrixMeter


def evaluate_model(model, dataloader, device, n_class=2, verbose=True):
    """Evaluate model on a dataloader and return metrics dict.

    Args:
        model: BIT-CD model with forward(x1, x2) -> logits
        dataloader: yields dicts with keys 'A', 'B', 'L'
        device: torch device
        n_class: number of classes

    Returns:
        dict of metrics from cm2score
    """
    model.eval()
    metric = ConfuseMatrixMeter(n_class=n_class)

    with torch.no_grad():
        for batch in dataloader:
            img1 = batch['A'].to(device)
            img2 = batch['B'].to(device)
            label = batch['L'].to(device).long()
            if label.dim() == 4:
                label = label.squeeze(1)

            logits = model(img1, img2)

            if logits.shape[-2:] != label.shape[-2:]:
                logits = torch.nn.functional.interpolate(
                    logits, size=label.shape[-2:], mode='bilinear', align_corners=True)

            pred = torch.argmax(logits, dim=1)
            metric.update_cm(pred.cpu().numpy(), label.cpu().numpy())

    scores = metric.get_scores()
    if verbose:
        print(f"  OA={scores['acc']:.4f}  mF1={scores['mf1']:.4f}  mIoU={scores['miou']:.4f}  "
              f"F1_change={scores.get('f1_1', 0):.4f}  IoU_change={scores.get('iou_1', 0):.4f}")
    return scores


def evaluate_on_splits(model, eval_loaders, device, n_class=2, verbose=True):
    """Evaluate model on multiple splits (val/test/test2).

    Args:
        model: trained model
        eval_loaders: dict of {split_name: DataLoader}
        device: torch device

    Returns:
        dict of {split_name: scores_dict}
    """
    results = {}
    for split_name, loader in eval_loaders.items():
        if verbose:
            print(f"\nEvaluating on {split_name} ({len(loader.dataset)} samples)...")
        scores = evaluate_model(model, loader, device, n_class=n_class, verbose=verbose)
        results[split_name] = scores
    return results


def print_results_table(all_results, splits=None):
    """Print a formatted results table.

    Args:
        all_results: dict of {exp_name: {split_name: scores_dict}}
    """
    if splits is None:
        first_exp = list(all_results.values())[0]
        splits = list(first_exp.keys())

    header = f"{'Experiment':<35s}"
    for split in splits:
        header += f" | {split:>12s} mF1  {split:>12s} IoU_c"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))

    for exp_name, split_results in all_results.items():
        row = f"{exp_name:<35s}"
        for split in splits:
            sc = split_results.get(split, {})
            mf1 = sc.get('mf1', 0.0)
            iou_c = sc.get('iou_1', 0.0)
            row += f" | {mf1:>17.4f}  {iou_c:>12.4f}"
        print(row)
    print("=" * len(header))
