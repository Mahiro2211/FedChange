"""
Federated learning main loop for change detection.

Orchestrates: client selection -> local training -> weight aggregation -> global evaluation.

Usage:
    python -m fed_cd.federated.fed_main \
        --partition_json partitions/partition_source.json \
        --net_G base_transformer_pos_s4_dd8 \
        --epochs 200 --frac_num 5 --local_ep 2 \
        --project_name FedAvg_source_bcd
"""

import os
import sys
import json
import time
import copy
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fed_cd.options import parse_args, print_args
from fed_cd.models import build_cd_model
from fed_cd.data.cd_dataset import CDDataset, build_eval_dataset
from fed_cd.data.data_partition import load_partition, scan_evaluation_set
from fed_cd.federated.local_update import LocalUpdate
from fed_cd.federated.aggregation import average_weights, weighted_average_weights, EMA
from fed_cd.evaluation.evaluator import evaluate_model
from fed_cd.logging_config import setup_logger


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_client_datasets(partition_path, img_size, task, data_root):
    """Build per-client CDDatasets from partition JSON."""
    partition = load_partition(partition_path)
    client_ids = sorted(partition["clients"].keys(), key=lambda x: int(x.split("_")[1]))
    datasets = {}
    for cid in client_ids:
        samples = partition["clients"][cid]["samples"]
        datasets[cid] = CDDataset(samples, img_size=img_size, is_train=True,
                                  task=task, data_root=data_root)
    return datasets, client_ids


def build_eval_loaders(data_root, img_size, task, eval_splits, num_workers=4):
    """Build evaluation dataloaders for val/test/test2."""
    loaders = {}
    for split in eval_splits:
        samples = scan_evaluation_set(data_root, split)
        if not samples:
            continue
        ds = build_eval_dataset(samples, img_size=img_size, task=task, data_root=data_root)
        loaders[split] = DataLoader(ds, batch_size=1, shuffle=False, num_workers=num_workers)
    return loaders


def run_federated(args):
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
    logger.info(f"=== Federated experiment: {args.project_name} ===")
    logger.info(f"Logs -> {os.path.join(args.checkpoint_dir, 'train.log')}")

    device = torch.device(args.device)

    # ─── Build client datasets ───
    logger.info("Loading partition...")
    client_datasets, client_ids = build_client_datasets(
        args.partition_json, args.img_size, args.task, args.data_root)
    logger.info(f"Clients: {len(client_ids)}")
    for cid in client_ids:
        logger.info(f"  {cid}: {len(client_datasets[cid])} samples")

    # ─── Build eval loaders ───
    eval_splits = [s.strip() for s in args.eval_splits.split(",")]
    eval_loaders = build_eval_loaders(
        args.data_root, args.img_size, args.task, eval_splits, args.num_workers)
    logger.info(f"Eval splits: {list(eval_loaders.keys())}")

    # ─── Build global model ───
    global_model = build_cd_model(
        net_name=args.net_G, num_classes=args.num_classes,
        pretrained=args.pretrained).to(device)
    global_weights = global_model.state_dict()
    n_params = sum(p.numel() for p in global_model.parameters())
    logger.info(f"Model: {args.net_G} | params: {n_params/1e6:.2f}M")

    # ─── EMA (optional) ───
    ema = None
    if args.globalema:
        ema = EMA(global_model, decay=0.999)

    # ─── Training loop ───
    train_loss_history = []
    eval_history = {split: [] for split in eval_loaders}
    best_metric = 0.0
    best_round = 0
    total_elapsed = 0.0

    # 每个测试集 × 每个 mean 指标 独立追踪历史最优及其轮次
    _BEST_METRICS = ('mf1', 'miou', 'mprecision', 'mrecall')
    bests = {
        split: {m: {'value': 0.0, 'round': 0} for m in _BEST_METRICS}
        for split in eval_loaders
    }

    num_clients = len(client_ids)
    frac_num = min(args.frac_num, num_clients)

    logger.info(f"Starting federated training: {args.epochs} rounds, "
                f"{frac_num}/{num_clients} clients per round")

    for epoch in range(args.epochs):
        round_start = time.time()

        # Select clients
        selected = np.random.choice(num_clients, frac_num, replace=False)
        selected_ids = [client_ids[i] for i in selected]

        logger.info(f"{'─' * 60}")
        logger.info(f"Round {epoch+1}/{args.epochs} | Clients: {selected_ids}")

        local_weights = []
        local_losses = []
        client_lens = []

        for cid in selected_ids:
            local_update = LocalUpdate(args, client_datasets[cid], device)
            w, loss = local_update.update_weights(
                model=global_model, global_round=epoch)
            local_weights.append(w)
            local_losses.append(loss)
            client_lens.append(len(client_datasets[cid]))

        avg_loss = sum(local_losses) / len(local_losses)
        train_loss_history.append(avg_loss)
        logger.info(f"  [train] avg_loss={avg_loss:.6f}")

        # Aggregate weights
        if args.iid:
            global_weights = average_weights(local_weights)
        else:
            global_weights = weighted_average_weights(local_weights, client_lens)

        global_model.load_state_dict(global_weights)

        # Update EMA shadow with the freshly aggregated global weights
        if ema is not None:
            ema.update(global_model)

        # ─── Evaluate ───
        if (epoch + 1) % args.global_test_frequency == 0 or epoch == args.epochs - 1:
            global_model.eval()

            # Temporarily swap in EMA weights for evaluation
            if ema is not None:
                ema.apply_shadow(global_model)

            current_results = {}
            for split_name, loader in eval_loaders.items():
                scores = evaluate_model(global_model, loader, device,
                                        n_class=args.num_classes, verbose=False)
                current_results[split_name] = scores

                # 计算所有类平均的 Precision / Recall（与 mF1/mIoU 口径一致）
                n_cls = args.num_classes
                mprec = sum(scores.get(f'precision_{i}', 0.0) for i in range(n_cls)) / n_cls
                mrec = sum(scores.get(f'recall_{i}', 0.0) for i in range(n_cls)) / n_cls

                # 全量指标记录到 eval_history（供 results.json 分析）
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

                # 每指标独立更新历史最优（首次评估无条件写入）
                rec = eval_history[split_name][-1]
                for m in _BEST_METRICS:
                    cur = rec[m]
                    if bests[split_name][m]['round'] == 0 or cur > bests[split_name][m]['value']:
                        bests[split_name][m]['value'] = cur
                        bests[split_name][m]['round'] = epoch + 1

            # 每轮评估后打印各测试集 best-so-far（每指标独立）
            logger.info(f"  ── Best-so-far (per-metric, up to round {epoch+1}) ──")
            for split_name in eval_loaders:
                b = bests[split_name]
                logger.info(
                    f"  [{split_name:>5s}] BEST: "
                    f"mF1={b['mf1']['value']:.4f}(r{b['mf1']['round']})  "
                    f"mIoU={b['miou']['value']:.4f}(r{b['miou']['round']})  "
                    f"mPrec={b['mprecision']['value']:.4f}(r{b['mprecision']['round']})  "
                    f"mRec={b['mrecall']['value']:.4f}(r{b['mrecall']['round']})"
                )

            val_mf1 = current_results.get('val', {}).get('mf1', 0)
            if val_mf1 > best_metric:
                best_metric = val_mf1
                best_round = epoch + 1
                torch.save({
                    'model': global_model.state_dict(),
                    'epoch': epoch,
                    'best_metric': best_metric,
                }, os.path.join(args.checkpoint_dir, 'best_ckpt.pt'))
                logger.info(f"  ★ New best (val mF1={best_metric:.4f})")

            # Restore current aggregated weights for next round's broadcast
            if ema is not None:
                ema.restore(global_model)

            round_time = time.time() - round_start
            total_elapsed += round_time
            logger.info(f"  Round time: {round_time:.1f}s | Total elapsed: {total_elapsed/60:.1f}min")

        # ─── Save checkpoint ───
        if (epoch + 1) % args.save_frequency == 0 or epoch == args.epochs - 1:
            torch.save({
                'model': global_model.state_dict(),
                'epoch': epoch,
            }, os.path.join(args.checkpoint_dir, 'last_ckpt.pt'))

    # ─── Final evaluation with best model ───
    logger.info(f"{'=' * 60}")
    logger.info(f"Training complete. Best val mF1 = {best_metric:.4f} (round {best_round})")
    logger.info(f"{'=' * 60}")

    # 各测试集每个 mean 指标的历史最优汇总（每指标独立追踪）
    logger.info(f"{'─' * 60}")
    logger.info("Best metrics per split (per-metric independent):")
    for split_name in eval_loaders:
        b = bests[split_name]
        logger.info(f"  [{split_name:>5s}]")
        for m_name, label in (('mf1', 'mF1 '), ('miou', 'mIoU'),
                              ('mprecision', 'mPrec'), ('mrecall', 'mRec ')):
            logger.info(f"    {label}: {b[m_name]['value']:.4f}  (round {b[m_name]['round']})")
    logger.info(f"{'─' * 60}")

    if os.path.exists(os.path.join(args.checkpoint_dir, 'best_ckpt.pt')):
        ckpt = torch.load(os.path.join(args.checkpoint_dir, 'best_ckpt.pt'),
                          map_location=device, weights_only=False)
        global_model.load_state_dict(ckpt['model'])

    global_model.eval()
    final_results = {}
    for split_name, loader in eval_loaders.items():
        logger.info(f"Final eval on {split_name}...")
        scores = evaluate_model(global_model, loader, device,
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

    # ─── Save results JSON ───
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
    args = parse_args()
    run_federated(args)
