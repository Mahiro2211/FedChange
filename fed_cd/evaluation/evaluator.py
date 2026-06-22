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

