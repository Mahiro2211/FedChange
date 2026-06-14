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
                        help='path to partition JSON (empty for centralized)')
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--task', type=str, default='bcd', choices=['bcd', 'scd'])
    parser.add_argument('--num_classes', type=int, default=2,
                        help='2 for BCD, 8 for SCD')
    parser.add_argument('--train_sources', type=str, default='all',
                        help='comma-separated sources for centralized (all/gcd,gcd+ugcd_full)')

    # ─── Model ───
    parser.add_argument('--net_G', type=str, default='base_transformer_pos_s4_dd8',
                        choices=['base_resnet18', 'base_transformer_pos_s4',
                                 'base_transformer_pos_s4_dd8',
                                 'base_transformer_pos_s4_dd8_dedim8'])
    parser.add_argument('--pretrained', type=str2bool, default=True,
                        help='use ImageNet-pretrained ResNet weights')

    # ─── Federated ───
    parser.add_argument('--mode', type=str, default='federated',
                        choices=['centralized', 'federated'])
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of global rounds (federated) or epochs (centralized)')
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
    parser.add_argument('--fedprox_mu', type=float, default=0.0,
                        help='FedProx proximal term weight (0=pure FedAvg)')
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
