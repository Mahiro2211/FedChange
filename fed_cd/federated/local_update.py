"""
Local update for federated change detection.

Supports four federated optimization algorithms via ``args.fed_alg``:
    - fedavg   : classical local SGD + (simple/weighted) averaging
    - fedprox  : FedAvg + proximal regularization to the global model
    - fednova  : reports the normalized cumulative local update direction
    - scaffold : local SGD corrected by a control variate (c_i - c_global)

Adapted from FedSeg/update.py for bitemporal image pair inputs (BIT-CD model).
The training loop branches on ``hasattr(model, 'compute_loss')`` to support both
BIT-CD (external CrossEntropyLoss) and torchange baselines (built-in BCE+Dice).
SCAFFOLD falls back to plain FedAvg for torchange baselines because their loss
is a black box and the controlled update rule is unreliable there.

update_weights() always returns ``(state_dict, avg_loss, extras)`` where
``extras`` carries algorithm-specific data consumed by the server in fed_main:
    - fedavg / fedprox : {}
    - fednova          : {'norm_dir': {key: tensor}}  (normalized update a_i)
    - scaffold         : {'delta_c': {key: tensor}}    (c_i^new - c_i^old)
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class LocalUpdate:
    """Performs local training on a single client's change detection data.

    Args:
        args: parsed arguments
        dataset: CDDataset for this client
        device: torch device
    """

    def __init__(self, args, dataset, device):
        self.args = args
        self.dataset = dataset
        self.device = device
        self.trainloader = DataLoader(
            dataset,
            batch_size=args.local_bs,
            shuffle=True,
            num_workers=args.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        # SCAFFOLD per-client control variate; lazily initialized on first use.
        # Persisted across rounds for THIS client instance. Because fed_main
        # builds a fresh LocalUpdate per round, we keep a class-level registry
        # keyed by the memory address is NOT enough; fed_main must pass the
        # persistent c_local in via update_weights(c_local=...). We still keep a
        # fallback attribute here for self-containment.
        self.c_local = None

    # ─────────────── algorithm routing helpers ───────────────

    @property
    def fed_alg(self):
        return getattr(self.args, 'fed_alg', 'fedavg')

    @property
    def apply_fedprox(self):
        """Whether to add the FedProx proximal term this round.

        True when ``--fed_alg fedprox`` OR (backwards-compat) when an explicit
        ``--fedprox_mu > 0`` is passed while fed_alg defaulted to fedavg. This
        keeps old shell scripts (which only set --fedprox_mu) working unchanged.
        """
        if self.fed_alg == 'fedprox':
            return self.args.fedprox_mu > 0
        return self.fed_alg == 'fedavg' and self.args.fedprox_mu > 0

    @property
    def apply_scaffold(self):
        """SCAFFOLD only for models whose loss is external (BIT-CD family).

        torchange baselines (hasattr compute_loss) have a black-box loss; the
        controlled update rule is unreliable there, so we gracefully degrade to
        FedAvg and the caller logs a warning.
        """
        if self.fed_alg != 'scaffold':
            return False
        # torchange models expose compute_loss -> skip SCAFFOLD correction.
        # (The actual model object is passed into update_weights; the caller is
        # responsible for passing the global model. We detect it there.)
        return True

    # ─────────────── main entry ───────────────

    def update_weights(self, model, global_round, c_global=None,
                       scaffold_enabled=None):
        """Train model locally for ``args.local_ep`` epochs.

        Args:
            model: global model (deep-copied here for local training).
            global_round: current global round index.
            c_global: optional SCAFFOLD global control variate state_dict
                (keyed by named_parameters). Used only for scaffold.
            scaffold_enabled: optional bool to override SCAFFOLD activation
                (the caller decides based on the *concrete* model type, since
                this instance cannot inspect the model before being called).

        Returns:
            (state_dict, avg_loss, extras) where extras is an algorithm dict.
        """
        args = self.args
        w_global = copy.deepcopy(model.state_dict())  # w^0 for this round
        model = copy.deepcopy(model)
        model.train()
        model.to(self.device)

        # decide SCAFFOLD activation (caller may disable for torchange models)
        do_scaffold = (self.fed_alg == 'scaffold') and bool(scaffold_enabled)

        # ─── FedProx reference model (proximal term) ───
        global_ref = None
        if self.apply_fedprox:
            global_ref = copy.deepcopy(model)
            global_ref.eval()
            for p in global_ref.parameters():
                p.requires_grad = False

        # ─── SCAFFOLD control variates ───
        c_local = self.c_local
        c_global_dev = None
        if do_scaffold:
            # lazily init per-client c_local (zeros) on first invocation
            if c_local is None:
                c_local = {n: torch.zeros_like(p.data).to(self.device)
                           for n, p in model.named_parameters() if p.requires_grad}
                self.c_local = c_local
            # bring c_global onto device, aligned to named parameters
            c_global_dev = {}
            for n, p in model.named_parameters():
                if p.requires_grad and c_global is not None and n in c_global:
                    c_global_dev[n] = c_global[n].to(self.device)

        criterion = nn.CrossEntropyLoss(ignore_index=255)

        if args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(
                model.parameters(), lr=args.lr,
                momentum=0.9, weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.Adam(
                model.parameters(), lr=args.lr,
                weight_decay=args.weight_decay)

        scheduler = self._get_scheduler(optimizer, global_round)

        epoch_loss = []
        n_local_steps = 0  # FedNova counts optimizer steps (batches)
        for ep in range(args.local_ep):
            batch_loss = []
            for batch_idx, batch in enumerate(self.trainloader):
                img1 = batch['A'].to(self.device)
                img2 = batch['B'].to(self.device)
                labels = batch['L'].to(self.device).long()
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

                if self.apply_fedprox:
                    proximal = 0.0
                    for w, w_t in zip(model.parameters(), global_ref.parameters()):
                        proximal += (w - w_t).pow(2).sum()
                    loss = loss + (args.fedprox_mu / 2) * proximal

                optimizer.zero_grad()
                loss.backward()

                # ─── SCAFFOLD: correct gradients before step ───
                # g_corrected = g + c_i - c_global
                if do_scaffold:
                    with torch.no_grad():
                        for n, p in model.named_parameters():
                            if not p.requires_grad or p.grad is None:
                                continue
                            if n in c_local and n in c_global_dev:
                                p.grad.add_(c_local[n] - c_global_dev[n])

                optimizer.step()

                batch_loss.append(loss.item())
                scheduler.step()
                n_local_steps += 1

            ep_loss = sum(batch_loss) / max(len(batch_loss), 1)
            epoch_loss.append(ep_loss)

        avg_loss = sum(epoch_loss) / max(len(epoch_loss), 1)

        # ─── assemble extras per algorithm ───
        extras = {}
        local_sd = model.state_dict()

        if self.fed_alg == 'fednova':
            # normalized cumulative update direction a_i = (w^0 - w^K) / tau_i,
            # then l2-normalized per-tensor as in the original paper. tau_i is
            # the number of local optimizer steps. Only floating-point parameters
            # are normalized; integer buffers (e.g. num_batches_tracked) are
            # skipped, and the aggregated model still carries the local copies.
            tau = max(n_local_steps, 1)
            norm_dir = {}
            for key in w_global:
                if key not in local_sd:
                    continue
                if not torch.is_floating_point(w_global[key]):
                    continue  # skip int buffers (num_batches_tracked, etc.)
                delta = (w_global[key].to(self.device).float()
                         - local_sd[key].to(self.device).float())
                delta = delta / float(tau)
                # l2-normalize each tensor to unit norm (avoid division by zero)
                flat = delta.flatten()
                norm = flat.norm()
                if float(norm) > 0:
                    delta = delta / norm
                norm_dir[key] = delta.cpu()
            extras['norm_dir'] = norm_dir

        elif do_scaffold:
            # update c_local (server-side equivalent): the SCAFFOLD paper uses
            # c_i^{new} = c_i - eta_g * (c_global - c_i) + eta_g * (w^0 - w^K)/tau.
            # We report delta_c = c_i^{new} - c_i^{old} so the server aggregates it.
            tau = max(n_local_steps, 1)
            lr = args.lr
            delta_c = {}
            with torch.no_grad():
                for n, p in model.named_parameters():
                    if not p.requires_grad or n not in c_local:
                        continue
                    ci = c_local[n]
                    cg = c_global_dev[n]
                    w0 = w_global[n].to(self.device)
                    wk = p.data
                    # standard SCAFFOLD Option II update for c_i
                    ci_new = ci - cg + (w0 - wk) / (lr * tau)
                    delta_c[n] = (ci_new - ci).cpu()
                    c_local[n].copy_(ci_new)
            extras['delta_c'] = delta_c

        return local_sd, avg_loss, extras

    def _get_scheduler(self, optimizer, global_round):
        args = self.args
        if args.lr_policy == 'linear':
            lr_factor = max(0.0, 1.0 - global_round / float(args.epochs + 1))
            return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x: lr_factor)
        elif args.lr_policy == 'step':
            factor = 1.0 if global_round < args.epochs * 0.7 else 0.1
            return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x: factor)
        else:
            return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x: 1.0)
