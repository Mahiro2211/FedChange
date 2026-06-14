"""
Change detection evaluation metrics.

Computes pixel-level metrics from a confusion matrix:
  - Overall Accuracy (OA)
  - Mean F1 (mF1)
  - Mean IoU (mIoU)
  - Per-class Precision, Recall, F1, IoU
"""

import numpy as np


class ConfuseMatrixMeter:
    """Accumulates confusion matrix and computes change detection metrics."""

    def __init__(self, n_class=2):
        self.n_class = n_class
        self.mat = None

    def update_cm(self, pr, gt):
        """Update confusion matrix with predictions and ground truths.

        Args:
            pr: predicted labels (numpy int array)
            gt: ground truth labels (numpy int array)
        """
        val = get_confuse_matrix(self.n_class, gt, pr)
        if self.mat is None:
            self.mat = val
        else:
            self.mat += val
        return cm2f1(self.mat)

    def get_scores(self):
        return cm2score(self.mat)

    def clear(self):
        self.mat = None


def get_confuse_matrix(num_classes, label_gts, label_preds):
    def __fast_hist(label_gt, label_pred):
        mask = (label_gt >= 0) & (label_gt < num_classes)
        hist = np.bincount(
            num_classes * label_gt[mask].astype(int) + label_pred[mask],
            minlength=num_classes ** 2
        ).reshape(num_classes, num_classes)
        return hist

    cm = np.zeros((num_classes, num_classes))
    for lt, lp in zip(label_gts, label_preds):
        cm += __fast_hist(lt.flatten(), lp.flatten())
    return cm


def cm2f1(confusion_matrix):
    hist = confusion_matrix
    tp = np.diag(hist)
    sum_a1 = hist.sum(axis=1)
    sum_a0 = hist.sum(axis=0)
    recall = tp / (sum_a1 + np.finfo(np.float32).eps)
    precision = tp / (sum_a0 + np.finfo(np.float32).eps)
    f1 = 2 * recall * precision / (recall + precision + np.finfo(np.float32).eps)
    return float(np.nanmean(f1))


def cm2score(confusion_matrix):
    hist = confusion_matrix
    n_class = hist.shape[0]
    tp = np.diag(hist)
    sum_a1 = hist.sum(axis=1)
    sum_a0 = hist.sum(axis=0)

    acc = tp.sum() / (hist.sum() + np.finfo(np.float32).eps)
    recall = tp / (sum_a1 + np.finfo(np.float32).eps)
    precision = tp / (sum_a0 + np.finfo(np.float32).eps)
    f1 = 2 * recall * precision / (recall + precision + np.finfo(np.float32).eps)
    mean_f1 = float(np.nanmean(f1))

    iu = tp / (sum_a1 + sum_a0 - tp + np.finfo(np.float32).eps)
    mean_iu = float(np.nanmean(iu))

    freq = sum_a1 / (hist.sum() + np.finfo(np.float32).eps)
    fwavacc = float((freq[freq > 0] * iu[freq > 0]).sum())

    scores = {
        'acc': float(acc),
        'mf1': mean_f1,
        'miou': mean_iu,
        'fwavacc': fwavacc,
    }
    for i in range(n_class):
        scores[f'iou_{i}'] = float(iu[i]) if not np.isnan(iu[i]) else 0.0
        scores[f'f1_{i}'] = float(f1[i]) if not np.isnan(f1[i]) else 0.0
        scores[f'precision_{i}'] = float(precision[i]) if not np.isnan(precision[i]) else 0.0
        scores[f'recall_{i}'] = float(recall[i]) if not np.isnan(recall[i]) else 0.0

    return scores
