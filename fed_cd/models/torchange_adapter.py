"""
Adapter that wraps torchange change-detection models into the FedChange contract.

FedChange expects every model to expose:
    forward(x1, x2) -> logits of shape [B, num_classes, H, W]
so that the existing training loops can apply CrossEntropyLoss and the existing
evaluator can do argmax over the class dimension.

torchange models instead follow the `ever` paradigm:
    - in train() mode: forward(cat([x1, x2], 1), y) returns a *loss dict*
      (BCE + Dice are baked into the model and need the ground truth `y`).
    - in eval() mode: forward(cat([x1, x2], 1)) returns predictions
      (sigmoid probabilities for binary change detection).

Because torchange bakes its own loss into the train-time forward, a single
unified CrossEntropyLoss across all models is not possible at training time.
This adapter therefore exposes two methods:
    - forward(x1, x2)      : inference only, returns 2-class logits whose
                             softmax recovers the model's change probability,
                             so FedChange's evaluator/argmax work unchanged.
    - compute_loss(x1, x2, label): training only, returns the model's native
                             scalar loss (sum of its BCE+Dice components).

The training loops branch on `hasattr(model, 'compute_loss')`.

Supported binary change-detection baselines (all bi-temporal supervised):
    changesparse_bcd        ChangeSparseBCD, er.R18 backbone (ISPRS 2024, PCM)
    changestar_1xd          ChangeStar1xd, FarSeg ResNet50 (IJCV 2024 default)
    changestar_1xd_r18      ChangeStar1xd, FarSeg ResNet18 (controlled ablation)
    changestar_2_5          ChangeStar2_5, FarSeg ResNet50 (IJCV 2024, modern)
    changen2_zeroshot       Changen2 ChangeStar1x256 ViT-B (TPAMI 2024, eval only)

ChangeStar2 is intentionally NOT included: its native training pipeline is the
G-STAR single-temporal-supervised protocol (needs y[XIMG1]/y[XMASK1] semantic
masks and synthesizes a pseudo T2), which is incompatible with the bi-temporal
binary supervised setting of WHU-GCD.
"""

import torch
import torch.nn as nn


def _apply_timm_compat_shim():
    """Make torchange's windowed attention work with timm >= 1.0.

    torchange.models.changesparse calls timm's window_partition / window_reverse
    with an integer window_size, but timm>=1.0 requires a (h, w) tuple. We patch
    the names bound inside the changesparse module namespace.
    """
    try:
        import timm.models.swin_transformer as _swin
        import torchange.models.changesparse as _cs
    except Exception:
        return

    if getattr(_cs, '_fedchange_timm_shim', False):
        return

    _orig_wp = _swin.window_partition
    _orig_wr = _swin.window_reverse

    def _wp_compat(x, window_size):
        if isinstance(window_size, int):
            window_size = (window_size, window_size)
        return _orig_wp(x, window_size)

    def _wr_compat(windows, window_size, H, W):
        if isinstance(window_size, int):
            window_size = (window_size, window_size)
        return _orig_wr(windows, window_size, H, W)

    _cs.window_partition = _wp_compat
    _cs.window_reverse = _wr_compat
    _cs._fedchange_timm_shim = True


_apply_timm_compat_shim()


def _extract_change_prob(out):
    """Return the binary change probability map as a [B, 1, H, W] tensor."""
    if isinstance(out, dict):
        prob = out['change_prediction']
    else:
        prob = out.change_prediction
    if prob.dim() == 3:
        prob = prob.unsqueeze(1)
    return prob.float()


def _prob_to_2class_logits(prob):
    """Convert a [B, 1, H, W] change probability into 2-class logits.

    softmax([0, logit(p)])[1] == sigmoid(logit(p)) == p, so the FedChange
    evaluator's argmax(dim=1) and CrossEntropyLoss stay consistent with the
    model's own probability output.
    """
    p = prob.clamp(1e-7, 1 - 1e-7)
    logit_change = torch.log(p / (1.0 - p))
    zero = torch.zeros_like(logit_change)
    return torch.cat([zero, logit_change], dim=1)


def _sum_loss_dict(loss_dict):
    """Sum all scalar, gradient-bearing tensors in a torchange loss dict.

    Metric/buffer entries (max_mem, IoU, ECR, ...) are detached or constant and
    therefore excluded automatically because they do not require gradients.
    """
    total = None
    for v in loss_dict.values():
        if isinstance(v, torch.Tensor) and v.requires_grad and v.dim() == 0:
            total = v if total is None else total + v
    if total is None:
        raise RuntimeError("torchange model returned no gradient-bearing loss. "
                           "Check that it is in train() mode and received a valid y.")
    return total


class TorchangeCDAdapter(nn.Module):
    """Wrap a torchange binary change-detection model as a FedChange model.

    Args:
        torchange_model: a torchange model with forward(x, y) -> loss_dict (train)
                         and forward(x) -> predictions (eval).
        model_name: name string used for logging / checkpointing.
    """

    def __init__(self, torchange_model, model_name='torchange'):
        super().__init__()
        self.model = torchange_model
        self.model_name = model_name

    def forward(self, x1, x2):
        """Inference path: return 2-class logits [B, 2, H, W].

        Always evaluated in inference mode so the wrapped model returns
        predictions regardless of the adapter's outer train/eval state.
        """
        x = torch.cat([x1, x2], dim=1)
        was_training = self.model.training
        self.model.eval()
        try:
            out = self.model(x)
        finally:
            if was_training:
                self.model.train()
        prob = _extract_change_prob(out)
        return _prob_to_2class_logits(prob)

    def compute_loss(self, x1, x2, label):
        """Training path: return the model's native scalar loss.

        Args:
            x1, x2: [B, 3, H, W] bitemporal images
            label: [B, H, W] long tensor with values in {0, 1} (255 ignored)
        """
        x = torch.cat([x1, x2], dim=1)
        mask = label if label.dim() == 3 else label.squeeze(1)
        y = {'masks': [mask]}
        self.model.train()
        loss_dict = self.model(x, y)
        return _sum_loss_dict(loss_dict)


def _build_changesparse_bcd(pretrained=True):
    from torchange.models.changesparse import ChangeSparseBCD
    return ChangeSparseBCD(dict(
        backbone=dict(name='er.R18', pretrained=pretrained, drop_path_rate=0.),
    ))


def _build_changestar_1xd(resnet_type='resnet50', pretrained=True):
    from torchange.models.changestar_1xd import ChangeStar1xd, TrainMode
    return ChangeStar1xd(dict(
        encoder=dict(
            type='FarSegEncoder',
            params=dict(resnet_type=resnet_type, pretrained=pretrained),
            bitemporal_forward=True,
        ),
        head=dict(num_semantic_classes=None),
        loss=dict(change=dict(bce=dict(), dice=dict())),
        train_mode=TrainMode.BSL,
    ))


def _build_changestar_2_5(pretrained=True):
    from torchange.models.changestar2_5 import ChangeStar2_5
    from torchange.models.changestar_1xd import TrainMode
    return ChangeStar2_5(dict(
        image_dense_encoder=dict(
            type='FarSegEncoder',
            params=dict(resnet_type='resnet50', pretrained=pretrained, out_channels=256),
        ),
        mixin=dict(c=1, s=-1, temporal_symmetric=True,
                   t1_on=True, t2_on=True, upsample_scale=4),
        loss=dict(change=dict(bce=dict(), dice=dict())),
        train_mode=TrainMode.BSL,
    ))


def _build_changen2_zeroshot():
    from torchange.models.changen2 import s1_init_s1c1_changestar_vitb_1x256
    return s1_init_s1c1_changestar_vitb_1x256()


TORCHANGE_MODELS = {
    'changesparse_bcd',
    'changestar_1xd',
    'changestar_1xd_r18',
    'changestar_2_5',
    'changen2_zeroshot',
}


def is_torchange_model(net_name):
    return net_name in TORCHANGE_MODELS


def build_torchange_model(net_name, num_classes=2, pretrained=True):
    """Build an adapter-wrapped torchange model.

    Args:
        net_name: one of TORCHANGE_MODELS
        num_classes: accepted for API compatibility; torchange baselines are
                     binary (2 classes), so any value is treated as binary.
        pretrained: ImageNet-pretrained backbone (ignored for changen2_zeroshot,
                    which always downloads its own Changen2 weights).
    """
    if net_name == 'changesparse_bcd':
        inner = _build_changesparse_bcd(pretrained=pretrained)
    elif net_name == 'changestar_1xd':
        inner = _build_changestar_1xd(resnet_type='resnet50', pretrained=pretrained)
    elif net_name == 'changestar_1xd_r18':
        inner = _build_changestar_1xd(resnet_type='resnet18', pretrained=pretrained)
    elif net_name == 'changestar_2_5':
        inner = _build_changestar_2_5(pretrained=pretrained)
    elif net_name == 'changen2_zeroshot':
        inner = _build_changen2_zeroshot()
    else:
        raise NotImplementedError(f'Unknown torchange model: {net_name}')

    return TorchangeCDAdapter(inner, model_name=net_name)
