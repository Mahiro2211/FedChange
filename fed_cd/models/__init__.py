"""Model registry and unified builder for federated change detection."""

from fed_cd.models.bit_cd import build_bit_cd_model
from fed_cd.models.torchange_adapter import (
    build_torchange_model,
    is_torchange_model,
    TORCHANGE_MODELS,
)

BIT_CD_MODELS = {
    'base_resnet18',
    'base_transformer_pos_s4',
    'base_transformer_pos_s4_dd8',
    'base_transformer_pos_s4_dd8_dedim8',
}


def build_cd_model(net_name='base_transformer_pos_s4_dd8', num_classes=2, pretrained=True):
    """Build a change-detection model by name.

    Routes BIT-CD variants (the framework's own model) to build_bit_cd_model and
    torchange baselines to build_torchange_model (returned as a TorchangeCDAdapter).

    Args:
        net_name: model name (BIT-CD variant or a torchange baseline name)
        num_classes: 2 for BCD, 8 for SCD (torchange baselines are binary only)
        pretrained: ImageNet-pretrained backbone

    Returns:
        nn.Module with forward(x1, x2) -> logits [B, num_classes, H, W] for BIT-CD,
        or a TorchangeCDAdapter (forward(x1,x2) for inference, compute_loss for
        training) for torchange baselines.
    """
    if net_name in BIT_CD_MODELS:
        return build_bit_cd_model(net_name, num_classes=num_classes, pretrained=pretrained)
    if is_torchange_model(net_name):
        return build_torchange_model(net_name, num_classes=num_classes, pretrained=pretrained)
    raise NotImplementedError(f'Unknown model: {net_name}')


__all__ = [
    'build_bit_cd_model',
    'build_torchange_model',
    'build_cd_model',
    'is_torchange_model',
    'TORCHANGE_MODELS',
    'BIT_CD_MODELS',
]
