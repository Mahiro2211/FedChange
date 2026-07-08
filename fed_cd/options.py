"""
Unified argument parser for federated change detection experiments.
"""

import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_parser():
    parser = argparse.ArgumentParser(description='Federated Change Detection on WHU-GCD')

    # ─── Data ───
    parser.add_argument('--data_root', type=str, default='../WHU-GCD',
                        help='path to WHU-GCD dataset root')
    parser.add_argument('--partition_json', type=str, default='',
                        help='path to partition JSON (required for federated training)')
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--task', type=str, default='bcd', choices=['bcd'],
                        help='task type (this framework is binary change '
                             'detection only)')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='2 for binary change detection')

    # ─── Model ───
    from fed_cd.models import BIT_CD_MODELS, TORCHANGE_MODELS
    _all_models = sorted(BIT_CD_MODELS | TORCHANGE_MODELS)
    parser.add_argument('--net_G', type=str, default='base_transformer_pos_s4_dd8',
                        choices=_all_models,
                        help='BIT-CD variants (framework own) or torchange baselines '
                             '(changesparse_bcd, changestar_1xd, changestar_1xd_r18, '
                             'changestar_2_5, changen2_zeroshot). torchange baselines '
                             'are binary change detection only.')
    parser.add_argument('--pretrained', type=str2bool, default=True,
                        help='use ImageNet-pretrained backbone weights')

    # ─── Federated ───
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of global communication rounds')
    parser.add_argument('--num_users', type=int, default=7,
                        help='number of federated clients (K)')
    parser.add_argument('--frac_num', type=int, default=5,
                        help='number of clients selected per round')
    parser.add_argument('--local_ep', type=int, default=2,
                        help='local training epochs (E)')
    parser.add_argument('--local_bs', type=int, default=8,
                        help='local batch size (B)')
    parser.add_argument('--iid', type=str2bool, default=False,
                        help='IID uses simple average, non-IID uses weighted average')

    # ─── FL Algorithm ───
    # --fed_alg 是算法路由主开关；--fedprox_mu 仅在 fed_alg=fedprox 时生效。
    # 向后兼容：旧脚本传 --fedprox_mu>0 而 fed_alg 保持默认 fedavg 时，仍会
    # 触发 proximal 项（见 local_update._should_apply_fedprox），行为与旧版一致。
    parser.add_argument('--fed_alg', type=str, default='fedavg',
                        choices=['fedavg', 'fedprox', 'fednova', 'scaffold'],
                        help='federated optimization algorithm '
                             '(fedavg/fedprox/fednova/scaffold)')
    parser.add_argument('--fedprox_mu', type=float, default=0.0,
                        help='FedProx proximal term weight (0=pure FedAvg). '
                             'Active when --fed_alg fedprox.')
    parser.add_argument('--scaffold_lr', type=float, default=1.0,
                        help='SCAFFOLD global learning rate (server-side control '
                             'variate update step). Active when --fed_alg scaffold.')
    parser.add_argument('--globalema', type=str2bool, default=False,
                        help='use global EMA')

    # ─── Optimization ───
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adam'])
    parser.add_argument('--lr_policy', type=str, default='linear', choices=['linear', 'step'])
    parser.add_argument('--num_workers', type=int, default=4)

    # ─── Evaluation ───
    parser.add_argument('--eval_splits', type=str, default='val,test,test2',
                        help='comma-separated splits to evaluate')
    parser.add_argument('--global_test_frequency', type=int, default=20)
    parser.add_argument('--save_frequency', type=int, default=20)
    parser.add_argument('--local_test_frequency', type=int, default=9999)

    # ─── Logging ───
    parser.add_argument('--project_name', type=str, default='fed_cd_exp')
    parser.add_argument('--checkpoint_root', type=str, default='results')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--save_results', type=str2bool, default=True,
                        help='save results JSON after training')
    parser.add_argument('--log_to_file', type=str2bool, default=True,
                        help='write loguru logs to results/<project_name>/train.log')
    parser.add_argument('--log_level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='loguru log level')

    return parser


def parse_args():
    parser = get_parser()
    args = parser.parse_args()

    args.device = 'cuda' if _check_cuda() else 'cpu'

    import os
    args.checkpoint_dir = os.path.join(args.checkpoint_root, args.project_name)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    return args


def get_centralized_parser():
    """Argument parser for centralized (non-federated) change detection.

    Shares the data / model / optimization / evaluation / logging flags with the
    federated parser, but drops federated-only knobs (frac_num, local_ep,
    fedprox_mu, globalema, iid, num_users) and renames ``local_bs`` to
    ``batch_size`` since centralized training has no per-client notion.
    """
    parser = argparse.ArgumentParser(
        description='Centralized Change Detection on WHU-GCD (federated upper bound)')

    # ─── Data ───
    parser.add_argument('--data_root', type=str, default='../WHU-GCD',
                        help='path to WHU-GCD dataset root')
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--task', type=str, default='bcd', choices=['bcd'],
                        help='task type (this framework is binary change '
                             'detection only)')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='2 for binary change detection')

    # ─── Model ───
    from fed_cd.models import BIT_CD_MODELS, TORCHANGE_MODELS
    _all_models = sorted(BIT_CD_MODELS | TORCHANGE_MODELS)
    parser.add_argument('--net_G', type=str, default='base_transformer_pos_s4_dd8',
                        choices=_all_models,
                        help='BIT-CD variants (framework own) or torchange baselines '
                             '(torchange baselines are binary change detection only).')
    parser.add_argument('--pretrained', type=str2bool, default=True,
                        help='use ImageNet-pretrained backbone weights')

    # ─── Optimization ───
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='training batch size (B)')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adam'])
    parser.add_argument('--lr_policy', type=str, default='linear', choices=['linear', 'step'])
    parser.add_argument('--num_workers', type=int, default=4)

    # ─── Evaluation ───
    parser.add_argument('--eval_splits', type=str, default='val,test,test2',
                        help='comma-separated splits to evaluate')
    parser.add_argument('--global_test_frequency', type=int, default=20)
    parser.add_argument('--save_frequency', type=int, default=20)

    # ─── Logging ───
    parser.add_argument('--project_name', type=str, default='centr_exp')
    parser.add_argument('--checkpoint_root', type=str, default='results')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--save_results', type=str2bool, default=True,
                        help='save results JSON after training')
    parser.add_argument('--log_to_file', type=str2bool, default=True,
                        help='write loguru logs to results/<project_name>/train.log')
    parser.add_argument('--log_level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='loguru log level')

    return parser


def parse_centralized_args():
    """Parse args for centralized training (mirrors :func:`parse_args` minus FL knobs)."""
    parser = get_centralized_parser()
    args = parser.parse_args()

    args.device = 'cuda' if _check_cuda() else 'cpu'

    import os
    args.checkpoint_dir = os.path.join(args.checkpoint_root, args.project_name)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    return args


def _check_cuda():
    import torch
    return torch.cuda.is_available()


def print_args(args):
    print('\n' + '=' * 60)
    print('Experiment Configuration')
    print('=' * 60)
    for k, v in sorted(vars(args).items()):
        print(f'  {k:<25s}: {v}')
    print('=' * 60 + '\n')
