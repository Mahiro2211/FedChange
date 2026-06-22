"""
Zero-shot evaluation of the Changen2 ChangeStar1x256 model on WHU-GCD.

Changen2 is a change-detection foundation model (TPAMI 2024) pretrained on
Changen2-S1-15k. This script evaluates it WITHOUT any training (zero-shot) to
provide a reference point alongside the supervised baselines.

Two notes vs. the supervised pipeline:
  1. The pretrained ViT-B backbone expects ImageNet normalization, but CDDataset
     normalizes with mean=std=0.5. We undo the 0.5 normalization and re-normalize
     to ImageNet stats (an exact linear transform).
  2. No training loop is involved; we only run the eval splits.

Run from FedChange/ directory:
    python -m scripts.run_changen2_zeroshot --data_root ../WHU-GCD
or:
    python scripts/run_changen2_zeroshot.py --data_root ../WHU-GCD
"""

import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from fed_cd.models import build_cd_model
from fed_cd.data.cd_dataset import CDDataset
from fed_cd.data.data_partition import scan_evaluation_set
from fed_cd.evaluation.evaluator import evaluate_model


class _ImageNetRenorm(nn.Module):
    """Re-normalize inputs from 0.5/0.5 stats to ImageNet stats, then run model.

    CDDataset outputs x_norm = (x/255 - 0.5)/0.5 = 2*(x/255) - 1, so
    x/255 = (x_norm + 1) / 2 and the ImageNet-normalized input is
    (x/255 - imagenet_mean) / imagenet_std.
    """

    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x1, x2):
        x1n = ((x1 + 1.0) / 2.0 - self.mean) / self.std
        x2n = ((x2 + 1.0) / 2.0 - self.mean) / self.std
        return self.inner(x1n, x2n)


def main():
    parser = argparse.ArgumentParser(description='Changen2 zero-shot change detection on WHU-GCD')
    parser.add_argument('--data_root', type=str, default='../WHU-GCD')
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--eval_splits', type=str, default='val,test,test2')
    parser.add_argument('--checkpoint_root', type=str, default='results/torchange')
    parser.add_argument('--project_name', type=str, default='Changen2_zeroshot')
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(os.path.join(args.checkpoint_root, args.project_name), exist_ok=True)

    print('Building Changen2 ChangeStar1x256 (ViT-B, s1c1) ...')
    print('  (downloads pretrained weights from HuggingFace EVER-Z/Changen2-ChangeStar1x256)')
    model = build_cd_model('changen2_zeroshot', num_classes=2, pretrained=True)
    n_params = sum(p.numel() for p in model.parameters())
    model = _ImageNetRenorm(model).to(device).eval()
    print(f'  params: {n_params/1e6:.2f}M')

    eval_splits = [s.strip() for s in args.eval_splits.split(',') if s.strip()]
    all_results = {}
    for split in eval_splits:
        samples = scan_evaluation_set(args.data_root, split)
        if not samples:
            print(f'  split {split}: no samples found, skipping')
            continue
        ds = CDDataset(samples, img_size=args.img_size, is_train=False,
                       task='bcd', data_root=args.data_root)
        loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.num_workers)
        print(f'\nEvaluating on {split} ({len(ds)} samples) ...')
        scores = evaluate_model(model, loader, device, n_class=2, verbose=True)
        all_results[split] = scores

    out_path = os.path.join(args.checkpoint_root, args.project_name, 'results.json')
    with open(out_path, 'w') as f:
        json.dump({'model': 'changen2_zeroshot', 'mode': 'zero_shot',
                   'params_M': round(n_params / 1e6, 2),
                   'results': all_results}, f, indent=2)
    print(f'\nResults saved to {out_path}')

    print('\n========== Changen2 zero-shot summary ==========')
    print(f"{'split':<8s} {'OA':>8s} {'mF1':>8s} {'mIoU':>8s} {'F1_c':>8s} {'IoU_c':>8s}")
    for split, sc in all_results.items():
        print(f"{split:<8s} {sc['acc']:>8.4f} {sc['mf1']:>8.4f} {sc['miou']:>8.4f} "
              f"{sc.get('f1_1', 0):>8.4f} {sc.get('iou_1', 0):>8.4f}")


if __name__ == '__main__':
    main()
