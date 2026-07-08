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


# ──────────────────────── FedNova ────────────────────────

def fednova_aggregate(local_weights, norm_coeffs, client_dataset_len):
    """FedNova: aggregate by *normalized* local update magnitudes.

    Reference: Wang et al., "Tackling the Objective Inconsistency Problem in
    Heterogeneous Federated Optimization", NeurIPS 2020.

    Each client reports its averaged per-step *normalized update* over the local
    epochs, i.e. ``a_i = (1/tau_i) * (w_i^0 - w_i^K)`` normalized to unit l2 norm.
    The server forms a weighted average of these normalized directions, weighted
    by relative dataset size ``n_k / sum(n_k)``. This keeps the global update
    objective-consistent under heterogeneous local epoch counts.

    Args:
        local_weights: list of state_dicts of *locally trained* models (w_i^K).
        norm_coeffs: list of 1-D tensors, each client's normalized cumulative
            update direction (a_i), same key set as the state_dicts. May be None
            per-client -> that client contributes a zero update.
        client_dataset_len: list of int, dataset size per client (for weighting).

    Returns:
        averaged state_dict (in place over a copy of the first client's weights).
    """
    w_avg = copy.deepcopy(local_weights[0])
    total = sum(client_dataset_len)
    # zero-out every *floating-point* tensor so we can accumulate weighted
    # normalized directions. Integer buffers (num_batches_tracked etc.) are
    # left untouched and not part of the update direction.
    for key in w_avg.keys():
        if torch.is_floating_point(w_avg[key]):
            w_avg[key] = torch.zeros_like(w_avg[key])

    for ci, nc in enumerate(norm_coeffs):
        if nc is None:
            continue
        weight = float(client_dataset_len[ci]) / float(total)
        for key in w_avg.keys():
            if key in nc:
                w_avg[key] = w_avg[key] + weight * nc[key]

    # The returned dict is the *update direction* to add to the global model.
    # fed_main adds it to the previous global weights: w_{t+1} = w_t + sum_k(w_k * a_k).
    # To keep the contract "returned dict is the new global state_dict", we fold
    # the addition here: callers pass prev global weights via the first slot of
    # local_weights only conceptually; we instead return the delta and let
    # fed_main apply it. For uniformity with average_weights (which returns a
    # full state_dict), we return the delta here and document it.
    return w_avg


# ──────────────────────── SCAFFOLD ────────────────────────

def zero_state_dict_like(model):
    """Create a zero-valued state_dict over a model's requires_grad parameters.

    Used to initialize SCAFFOLD control variates (c_global, per-client c_local).
    Keyed by parameter name (matches named_parameters()).
    """
    sd = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            sd[name] = torch.zeros_like(p.data)
    return sd


def scaffold_aggregate_control(local_delta_c, client_dataset_len, lr_g=1.0):
    """SCAFFOLD server-side control variate update.

    Reference: Karimireddy et al., "SCAFFOLD: Stochastic Controlled Averaging
    for Federated Learning", ICML 2020, Eq. (5) server step.

    c^{t+1} = c^t + (|S|/K) * (1/K) * sum_k (c_i^new - c_i^old)
            ≈ c^t + lr_g * mean_k(delta_c_k)

    where delta_c_k = c_i^new - c_i^old is what each client reports.

    Args:
        local_delta_c: list of state_dicts (per-client c_i^new - c_i^old). Each
            entry may be None (client did not update c, e.g. torchange fallback).
        client_dataset_len: list of int (kept for API symmetry; SCAFFOLD's
            standard server update is an unweighted mean over participants).
        lr_g: global/server learning rate for the control update.

    Returns:
        the aggregated delta_c (a state_dict) to be added to c_global by the
        caller. Entries where all clients reported None are zero.
    """
    # collect only valid (non-None) deltas
    valid = [dc for dc in local_delta_c if dc is not None]
    if not valid:
        return None

    agg = {}
    for key in valid[0].keys():
        agg[key] = torch.zeros_like(valid[0][key])
        for dc in valid:
            agg[key] += dc[key]
        agg[key] = torch.div(agg[key], float(len(valid)))
        agg[key] = torch.mul(agg[key], float(lr_g))
    return agg


def add_state_dicts(a, b):
    """Element-wise add two same-keyed state_dicts (a + b), returning a new dict.

    Used by fed_main to apply the SCAFFOLD control-viate delta to c_global.
    Missing keys in b are skipped (kept from a unchanged).
    """
    out = {k: v.clone() for k, v in a.items()}
    for key in out:
        if key in b:
            out[key] += b[key]
    return out
