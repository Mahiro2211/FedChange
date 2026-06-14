"""
Local update for federated change detection.

Adapted from FedSeg/update.py for bitemporal image pair inputs (BIT-CD model).
"""

import time
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from fed_cd.evaluation.cd_metrics import ConfuseMatrixMeter


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

    def update_weights(self, model, global_round):
        """Train model locally for local_ep epochs.

        Args:
            model: global model (will be copied for local training)
            global_round: current global round index

        Returns:
            (state_dict, avg_loss)
        """
        args = self.args
        model = copy.deepcopy(model)
        model.train()
        model.to(self.device)

        if args.fedprox_mu > 0:
            global_model = copy.deepcopy(model)
            global_model.eval()
            for p in global_model.parameters():
                p.requires_grad = False

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
        for ep in range(args.local_ep):
            batch_loss = []
            for batch_idx, batch in enumerate(self.trainloader):
                img1 = batch['A'].to(self.device)
                img2 = batch['B'].to(self.device)
                labels = batch['L'].to(self.device).long()
                if labels.dim() == 4:
                    labels = labels.squeeze(1)

                logits = model(img1, img2)

                if logits.shape[-2:] != labels.shape[-2:]:
                    logits = F.interpolate(
                        logits, size=labels.shape[-2:],
                        mode='bilinear', align_corners=True)

                loss = criterion(logits, labels)

                if args.fedprox_mu > 0:
                    proximal = 0.0
                    for w, w_t in zip(model.parameters(), global_model.parameters()):
                        proximal += (w - w_t).pow(2).sum()
                    loss += (args.fedprox_mu / 2) * proximal

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())
                scheduler.step()

            ep_loss = sum(batch_loss) / max(len(batch_loss), 1)
            epoch_loss.append(ep_loss)

        avg_loss = sum(epoch_loss) / max(len(epoch_loss), 1)
        return model.state_dict(), avg_loss

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

    def inference(self, model):
        """Evaluate model on this client's local data.

        Returns:
            (f1, iou_mean, scores_dict)
        """
        model.eval()
        metric = ConfuseMatrixMeter(n_class=self.args.num_classes)
        testloader = DataLoader(
            self.dataset, batch_size=1, shuffle=False,
            num_workers=self.args.num_workers)

        with torch.no_grad():
            for batch in testloader:
                img1 = batch['A'].to(self.device)
                img2 = batch['B'].to(self.device)
                labels = batch['L'].to(self.device)

                logits = model(img1, img2)
                if logits.shape[-2:] != labels.shape[-2:]:
                    logits = F.interpolate(
                        logits, size=labels.shape[-2:],
                        mode='bilinear', align_corners=True)

                pred = torch.argmax(logits, dim=1)
                metric.update_cm(pred.cpu().numpy(), labels.cpu().numpy())

        scores = metric.get_scores()
        return scores.get('mf1', 0.0), scores.get('miou', 0.0), scores
