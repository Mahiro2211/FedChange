"""
Centralized training for change detection (non-federated baseline).

Trains BIT-CD on all WHU-GCD training data without federated partitioning.
This establishes the performance upper bound for federated experiments.

Usage:
    python -m fed_cd.centralized_main \
        --train_sources gcd,ugcd_full \
        --net_G base_transformer_pos_s4_dd8 \
        --epochs 200 \
        --project_name Centralized_bcd
"""

import os
import sys
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fed_cd.options import parse_args, print_args
from fed_cd.models.bit_cd import build_bit_cd_model
from fed_cd.data.cd_dataset import CDDataset
from fed_cd.data.data_partition import scan_evaluation_set
from fed_cd.evaluation.evaluator import evaluate_model
from fed_cd.logging_config import setup_logger

import partition_utils


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_centralized_dataset(data_root, sources, img_size, task):
    """Build centralized training dataset from specified sources."""
    all_data = partition_utils.scan_all_sources(data_root)

    if sources == "all" or sources == "":
        selected = ["gcd", "ugcd_full", "ucd", "ugcd"]
    else:
        selected = [s.strip() for s in sources.split(",")]

    samples = []
    for src in selected:
        n = len(all_data.get(src, []))
        samples.extend(all_data.get(src, []))
        print(f"  {src}: {n} samples")
    print(f"  Total: {len(samples)} samples")

    return CDDataset(samples, img_size=img_size, is_train=True, task=task, data_root=data_root)


def run_centralized(args):
    set_seed(args.seed)

    # ─── Logger ───
    logger = setup_logger(
        args.project_name, args.checkpoint_root,
        log_to_file=args.log_to_file, level=args.log_level,
    )
    print_args(args)
    logger.info(f"=== Centralized experiment: {args.project_name} ===")
    logger.info(f"Logs -> {os.path.join(args.checkpoint_dir, 'train.log')}")

    device = torch.device(args.device)

    # ─── Build datasets ───
    logger.info("Building centralized training dataset...")
    train_dataset = build_centralized_dataset(
        args.data_root, args.train_sources, args.img_size, args.task)

    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = random_split(
        train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(
        train_subset, batch_size=args.local_bs, shuffle=True,
        num_workers=args.num_workers, drop_last=True, pin_memory=True)
    val_loader = DataLoader(
        val_subset, batch_size=1, shuffle=False, num_workers=args.num_workers)

    # ─── Build eval loaders ───
    eval_splits = [s.strip() for s in args.eval_splits.split(",")]
    eval_loaders = {}
    for split in eval_splits:
        samples = scan_evaluation_set(args.data_root, split)
        if samples:
            ds = CDDataset(samples, img_size=args.img_size, is_train=False,
                           task=args.task, data_root=args.data_root)
            eval_loaders[split] = DataLoader(ds, batch_size=1, shuffle=False,
                                             num_workers=args.num_workers)

    # ─── Build model ───
    model = build_bit_cd_model(
        net_name=args.net_G, num_classes=args.num_classes,
        pretrained=args.pretrained).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {args.net_G} | params: {n_params/1e6:.2f}M")

    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

    if args.lr_policy == 'linear':
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda e: 1.0 - e / float(args.epochs + 1))
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.epochs // 3, gamma=0.1)

    # ─── Training loop ───
    best_metric = 0.0
    best_epoch = 0
    train_losses = []
    eval_history = {split: [] for split in eval_loaders}

    logger.info(f"Training for {args.epochs} epochs...")

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = []
        for batch_idx, batch in enumerate(train_loader):
            img1 = batch['A'].to(device)
            img2 = batch['B'].to(device)
            labels = batch['L'].to(device).long()
            if labels.dim() == 4:
                labels = labels.squeeze(1)

            logits = model(img1, img2)
            if logits.shape[-2:] != labels.shape[-2:]:
                logits = F.interpolate(logits, size=labels.shape[-2:],
                                       mode='bilinear', align_corners=True)

            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())

        avg_loss = sum(epoch_loss) / max(len(epoch_loss), 1)
        train_losses.append(avg_loss)
        scheduler.step()

        # 每轮记录训练损失（仅 loguru，控制台节流）
        logger.info(
            f"Epoch {epoch+1}/{args.epochs} | loss={avg_loss:.6f} | "
            f"lr={optimizer.param_groups[0]['lr']:.6f}"
        )

        # ─── Evaluate ───
        if (epoch + 1) % args.global_test_frequency == 0 or epoch == args.epochs - 1:
            model.eval()
            current_results = {}
            for split_name, loader in eval_loaders.items():
                scores = evaluate_model(model, loader, device,
                                        n_class=args.num_classes, verbose=False)
                current_results[split_name] = scores

                # 计算所有类平均的 Precision / Recall
                n_cls = args.num_classes
                mprec = sum(scores.get(f'precision_{i}', 0.0) for i in range(n_cls)) / n_cls
                mrec = sum(scores.get(f'recall_{i}', 0.0) for i in range(n_cls)) / n_cls

                eval_history[split_name].append({
                    'epoch': epoch + 1,
                    'mf1': scores['mf1'],
                    'miou': scores['miou'],
                    'mprecision': mprec,
                    'mrecall': mrec,
                    'f1_change': scores.get(f'f1_{n_cls - 1}', 0.0),
                    'iou_change': scores.get(f'iou_{n_cls - 1}', 0.0),
                    'precision_change': scores.get(f'precision_{n_cls - 1}', 0.0),
                    'recall_change': scores.get(f'recall_{n_cls - 1}', 0.0),
                    'oa': scores.get('acc', 0.0),
                })

                # loguru 每轮记录：Precision / Recall / mIoU / F1（mean + change-class）
                logger.info(
                    f"  [{split_name:>5s}] mean  : "
                    f"F1={scores['mf1']:.4f}  IoU={scores['miou']:.4f}  "
                    f"Prec={mprec:.4f}  Rec={mrec:.4f}"
                )
                logger.info(
                    f"  [{split_name:>5s}] change: "
                    f"F1={scores.get(f'f1_{n_cls - 1}', 0.0):.4f}  "
                    f"IoU={scores.get(f'iou_{n_cls - 1}', 0.0):.4f}  "
                    f"Prec={scores.get(f'precision_{n_cls - 1}', 0.0):.4f}  "
                    f"Rec={scores.get(f'recall_{n_cls - 1}', 0.0):.4f}"
                )

            val_mf1 = current_results.get('val', {}).get('mf1', 0)
            if val_mf1 > best_metric:
                best_metric = val_mf1
                best_epoch = epoch + 1
                torch.save({
                    'model': model.state_dict(),
                    'epoch': epoch,
                    'best_metric': best_metric,
                }, os.path.join(args.checkpoint_dir, 'best_ckpt.pt'))
                logger.info(f"  ★ New best (val mF1={best_metric:.4f})")

        if (epoch + 1) % args.save_frequency == 0:
            torch.save({
                'model': model.state_dict(),
                'epoch': epoch,
            }, os.path.join(args.checkpoint_dir, 'last_ckpt.pt'))

    # ─── Final evaluation ───
    logger.info(f"{'=' * 60}")
    logger.info(f"Training complete. Best val mF1 = {best_metric:.4f} (epoch {best_epoch})")
    logger.info(f"{'=' * 60}")

    if os.path.exists(os.path.join(args.checkpoint_dir, 'best_ckpt.pt')):
        ckpt = torch.load(os.path.join(args.checkpoint_dir, 'best_ckpt.pt'),
                          map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])

    model.eval()
    final_results = {}
    for split_name, loader in eval_loaders.items():
        logger.info(f"Final eval on {split_name}...")
        scores = evaluate_model(model, loader, device,
                                n_class=args.num_classes, verbose=False)
        final_results[split_name] = scores
        n_cls = args.num_classes
        mprec = sum(scores.get(f'precision_{i}', 0.0) for i in range(n_cls)) / n_cls
        mrec = sum(scores.get(f'recall_{i}', 0.0) for i in range(n_cls)) / n_cls
        logger.info(
            f"  [{split_name:>5s}] FINAL mean  : "
            f"F1={scores['mf1']:.4f}  IoU={scores['miou']:.4f}  "
            f"Prec={mprec:.4f}  Rec={mrec:.4f}"
        )
        logger.info(
            f"  [{split_name:>5s}] FINAL change: "
            f"F1={scores.get(f'f1_{n_cls - 1}', 0.0):.4f}  "
            f"IoU={scores.get(f'iou_{n_cls - 1}', 0.0):.4f}  "
            f"Prec={scores.get(f'precision_{n_cls - 1}', 0.0):.4f}  "
            f"Rec={scores.get(f'recall_{n_cls - 1}', 0.0):.4f}"
        )

    if args.save_results:
        results = {
            'experiment': args.project_name,
            'args': {k: str(v) for k, v in vars(args).items()},
            'best_metric': best_metric,
            'best_epoch': best_epoch,
            'train_losses': train_losses,
            'eval_history': eval_history,
            'final_results': final_results,
        }
        results_path = os.path.join(args.checkpoint_dir, 'results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {results_path}")

    return final_results


if __name__ == '__main__':
    args = parse_args()
    args.mode = 'centralized'
    run_centralized(args)
