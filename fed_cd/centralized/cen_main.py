"""
Centralized change detection training on WHU-GCD.

Trains a single model on the union of all training samples (gcd + ugcd_full +
ucd + ugd) — the same pool that the federated partitions draw from — to provide
the performance upper bound for the federated experiments.

The training set, hyperparameters, evaluation splits and ``results.json`` schema
mirror :mod:`fed_cd.federated.fed_main` so that the two are directly comparable
(the only difference is centralized vs. federated optimization). The loss
computation (BIT-CD cross-entropy vs. torchange built-in ``compute_loss``) is
shared with :class:`fed_cd.federated.local_update.LocalUpdate`.

Usage:
    python -m fed_cd.centralized.cen_main \
        --net_G base_transformer_pos_s4_dd8 \
        --epochs 200 --batch_size 8 --lr 0.01 \
        --project_name Centr_base_bcd
"""

import os
import sys
import json
import time
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Ensure project root is on path (for partitions.generate.scan_all_sources)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fed_cd.options import parse_centralized_args, print_args
from fed_cd.models import build_cd_model
from fed_cd.data.cd_dataset import CDDataset, build_eval_dataset
from fed_cd.data.data_partition import scan_evaluation_set
from fed_cd.evaluation.evaluator import evaluate_model
from fed_cd.logging_config import setup_logger
from partitions.generate import scan_all_sources


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_train_dataset(data_root, img_size, task):
    """Build a single CDDataset from ALL training samples across the four sources.

    This is the union of every client's data — i.e. exactly the sample pool the
    federated partitions draw from — so the centralized run is a fair upper bound.
    """
    source_data = scan_all_sources(data_root)
    all_samples = []
    for src in ["gcd", "ugcd_full", "ucd", "ugcd"]:
        all_samples.extend(source_data.get(src, []))
    return CDDataset(all_samples, img_size=img_size, is_train=True,
                     task=task, data_root=data_root)


def build_eval_loaders(data_root, img_size, task, eval_splits, num_workers=4):
    """Build evaluation dataloaders for val/test/test2 (mirrors fed_main)."""
    loaders = {}
    for split in eval_splits:
        samples = scan_evaluation_set(data_root, split)
        if not samples:
            continue
        ds = build_eval_dataset(samples, img_size=img_size, task=task, data_root=data_root)
        loaders[split] = DataLoader(ds, batch_size=1, shuffle=False, num_workers=num_workers)
    return loaders


def run_centralized(args):
    set_seed(args.seed)
    if not hasattr(args, 'checkpoint_dir') or not args.checkpoint_dir:
        args.checkpoint_dir = os.path.join(args.checkpoint_root, args.project_name)
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    # ─── Logger ───
    logger = setup_logger(
        args.project_name, args.checkpoint_root,
        log_to_file=args.log_to_file, level=args.log_level,
    )
    print_args(args)
    logger.info(f"=== Centralized experiment: {args.project_name} ===")
    logger.info(f"Logs -> {os.path.join(args.checkpoint_dir, 'train.log')}")

    device = torch.device(args.device)

    # ─── Build training dataset (union of all sources) ───
    logger.info("Loading centralized training set (union of all sources)...")
    train_dataset = build_train_dataset(args.data_root, args.img_size, args.task)
    logger.info(f"Training samples: {len(train_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True, pin_memory=True,
    )

    # ─── Build eval loaders ───
    eval_splits = [s.strip() for s in args.eval_splits.split(",")]
    eval_loaders = build_eval_loaders(
        args.data_root, args.img_size, args.task, eval_splits, args.num_workers)
    logger.info(f"Eval splits: {list(eval_loaders.keys())}")

    # ─── Build model ───
    model = build_cd_model(
        net_name=args.net_G, num_classes=args.num_classes,
        pretrained=args.pretrained).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {args.net_G} | params: {n_params/1e6:.2f}M")

    # ─── Optimizer ───
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.lr,
            momentum=0.9, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr,
            weight_decay=args.weight_decay)

    # CE loss is only used for BIT-CD models; torchange models use compute_loss.
    criterion = nn.CrossEntropyLoss(ignore_index=255)

    # ─── Training loop ───
    train_loss_history = []
    eval_history = {split: [] for split in eval_loaders}
    best_metric = 0.0
    best_round = 0
    total_elapsed = 0.0

    # 每个测试集 × 每个 mean 指标 独立追踪历史最优及其轮次（与 fed_main 一致）
    _BEST_METRICS = ('mf1', 'miou', 'mprecision', 'mrecall')
    bests = {
        split: {m: {'value': 0.0, 'round': 0} for m in _BEST_METRICS}
        for split in eval_loaders
    }

    logger.info(f"Starting centralized training: {args.epochs} epochs")

    for epoch in range(args.epochs):
        epoch_start = time.time()
        model.train()

        # lr linear decay by epoch (fed_main decays by global round)
        if args.lr_policy == 'linear':
            lr_factor = max(0.0, 1.0 - epoch / float(args.epochs + 1))
        elif args.lr_policy == 'step':
            lr_factor = 1.0 if epoch < args.epochs * 0.7 else 0.1
        else:
            lr_factor = 1.0
        for pg in optimizer.param_groups:
            pg['lr'] = args.lr * lr_factor

        logger.info(f"{'─' * 60}")
        logger.info(f"Epoch {epoch+1}/{args.epochs} | lr={args.lr * lr_factor:.6f}")

        batch_loss = []
        for batch in train_loader:
            img1 = batch['A'].to(device)
            img2 = batch['B'].to(device)
            labels = batch['L'].to(device).long()
            if labels.dim() == 4:
                labels = labels.squeeze(1)

            if hasattr(model, 'compute_loss'):
                loss = model.compute_loss(img1, img2, labels)
            else:
                logits = model(img1, img2)
                if logits.shape[-2:] != labels.shape[-2:]:
                    logits = F.interpolate(
                        logits, size=labels.shape[-2:],
                        mode='bilinear', align_corners=True)
                loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())

        avg_loss = sum(batch_loss) / max(len(batch_loss), 1)
        train_loss_history.append(avg_loss)
        logger.info(f"  [train] avg_loss={avg_loss:.6f}")

        # ─── Evaluate ───
        if (epoch + 1) % args.global_test_frequency == 0 or epoch == args.epochs - 1:
            model.eval()

            current_results = {}
            for split_name, loader in eval_loaders.items():
                scores = evaluate_model(model, loader, device,
                                        n_class=args.num_classes, verbose=False)
                current_results[split_name] = scores

                n_cls = args.num_classes
                mprec = sum(scores.get(f'precision_{i}', 0.0) for i in range(n_cls)) / n_cls
                mrec = sum(scores.get(f'recall_{i}', 0.0) for i in range(n_cls)) / n_cls

                eval_history[split_name].append({
                    'round': epoch + 1,
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

                rec = eval_history[split_name][-1]
                for m in _BEST_METRICS:
                    cur = rec[m]
                    if bests[split_name][m]['round'] == 0 or cur > bests[split_name][m]['value']:
                        bests[split_name][m]['value'] = cur
                        bests[split_name][m]['round'] = epoch + 1

            logger.info(f"  ── Best-so-far (per-metric, up to epoch {epoch+1}) ──")
            for split_name in eval_loaders:
                b = bests[split_name]
                logger.info(
                    f"  [{split_name:>5s}] BEST: "
                    f"mF1={b['mf1']['value']:.4f}(e{b['mf1']['round']})  "
                    f"mIoU={b['miou']['value']:.4f}(e{b['miou']['round']})  "
                    f"mPrec={b['mprecision']['value']:.4f}(e{b['mprecision']['round']})  "
                    f"mRec={b['mrecall']['value']:.4f}(e{b['mrecall']['round']})"
                )

            # 标准做法：val mF1 选 best_ckpt（与 fed_main 一致，呼应弱点 5）
            val_mf1 = current_results.get('val', {}).get('mf1', 0)
            if val_mf1 > best_metric:
                best_metric = val_mf1
                best_round = epoch + 1
                torch.save({
                    'model': model.state_dict(),
                    'epoch': epoch,
                    'best_metric': best_metric,
                }, os.path.join(args.checkpoint_dir, 'best_ckpt.pt'))
                logger.info(f"  ★ New best (val mF1={best_metric:.4f})")

            epoch_time = time.time() - epoch_start
            total_elapsed += epoch_time
            logger.info(f"  Epoch time: {epoch_time:.1f}s | Total elapsed: {total_elapsed/60:.1f}min")

        # ─── Save checkpoint ───
        if (epoch + 1) % args.save_frequency == 0 or epoch == args.epochs - 1:
            torch.save({
                'model': model.state_dict(),
                'epoch': epoch,
            }, os.path.join(args.checkpoint_dir, 'last_ckpt.pt'))

    # ─── Final evaluation with best model ───
    logger.info(f"{'=' * 60}")
    logger.info(f"Training complete. Best val mF1 = {best_metric:.4f} (epoch {best_round})")
    logger.info(f"{'=' * 60}")

    logger.info(f"{'─' * 60}")
    logger.info("Best metrics per split (per-metric independent):")
    for split_name in eval_loaders:
        b = bests[split_name]
        logger.info(f"  [{split_name:>5s}]")
        for m_name, label in (('mf1', 'mF1 '), ('miou', 'mIoU'),
                              ('mprecision', 'mPrec'), ('mrecall', 'mRec ')):
            logger.info(f"    {label}: {b[m_name]['value']:.4f}  (epoch {b[m_name]['round']})")
    logger.info(f"{'─' * 60}")

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

    # ─── Save results JSON (schema identical to fed_main for summarize.py) ───
    if args.save_results:
        results = {
            'experiment': args.project_name,
            'args': {k: str(v) for k, v in vars(args).items()},
            'best_metric': best_metric,
            'best_round': best_round,
            'best_metrics': bests,
            'train_loss_history': train_loss_history,
            'eval_history': eval_history,
            'final_results': final_results,
        }
        results_path = os.path.join(args.checkpoint_dir, 'results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {results_path}")

    return final_results


if __name__ == '__main__':
    args = parse_centralized_args()
    run_centralized(args)
