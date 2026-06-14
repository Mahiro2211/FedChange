"""
Federated weight aggregation strategies.

All functions operate on lists of state_dicts and are model-agnostic.
"""

import copy
import torch


def average_weights(w):
    """FedAvg: simple uniform average of weights.

    Args:
        w: list of state_dicts

    Returns:
        averaged state_dict
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def weighted_average_weights(w, client_dataset_len):
    """Weighted FedAvg: average weighted by client dataset size.

    Args:
        w: list of state_dicts
        client_dataset_len: list of int, dataset size per client

    Returns:
        weighted-averaged state_dict
    """
    w_avg = copy.deepcopy(w[0])
    total = sum(client_dataset_len)
    for key in w_avg.keys():
        w_avg[key] = torch.mul(w_avg[key], float(client_dataset_len[0]))
        for i in range(1, len(w)):
            w_avg[key] += torch.mul(w[i][key], float(client_dataset_len[i]))
        w_avg[key] = torch.div(w_avg[key], float(total))
    return w_avg


class EMA:
    """Exponential Moving Average for model parameters.

    Usage:
        ema = EMA(global_model, decay=0.999)
        # after each aggregation:
        ema.update(global_model)          # first call initializes shadow,
                                          # subsequent calls blend.
        # for evaluation only:
        ema.apply_shadow(global_model)    # swap in EMA weights
        evaluate(global_model)
        ema.restore(global_model)         # swap back current weights
    """

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.initialized = False
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model):
        for name, param in model.named_parameters():
            if name not in self.shadow:
                continue
            if not self.initialized:
                self.shadow[name] = param.data.clone()
            else:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)
        self.initialized = True

    @torch.no_grad()
    def apply_shadow(self, model):
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    @torch.no_grad()
    def restore(self, model):
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}
