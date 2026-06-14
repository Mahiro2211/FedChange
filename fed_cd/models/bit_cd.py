"""
Bitemporal Image Transformer (BIT-CD) for change detection.

Adapted from: Chen et al., "Remote Sensing Image Change Detection with Transformers",
IEEE TGRS, 2021. Original code: https://github.com/justchenhao/BIT_CD

Architecture:
  ResNet18 backbone (1/8 downsampling) -> Tokenizer -> Transformer Encoder
  -> Transformer Decoder -> Feature Differencing -> Classifier

forward(x1, x2) -> logits of shape (B, num_classes, H, W)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from einops import rearrange

import functools
from torchvision import models as tv_models


class TwoLayerConv2d(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                      padding=kernel_size // 2, stride=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                      padding=kernel_size // 2, stride=1)
        )


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class Residual2(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, x2, **kwargs):
        return self.fn(x, x2, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class PreNorm2(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, x2, **kwargs):
        return self.fn(self.norm(x), self.norm(x2), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Cross_Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., softmax=True):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim ** -0.5
        self.softmax = softmax
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x, m, mask=None):
        b, n, _, h = *x.shape, self.heads
        q = self.to_q(x)
        k = self.to_k(m)
        v = self.to_v(m)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), [q, k, v])
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        if self.softmax:
            attn = dots.softmax(dim=-1)
        else:
            attn = dots
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = dots.softmax(dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))
            ]))

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout, softmax=True):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual2(PreNorm2(dim, Cross_Attention(dim, heads=heads,
                                                         dim_head=dim_head, dropout=dropout,
                                                         softmax=softmax))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))
            ]))

    def forward(self, x, m, mask=None):
        for attn, ff in self.layers:
            x = attn(x, m, mask=mask)
            x = ff(x)
        return x


class ResNetBackbone(nn.Module):
    """ResNet backbone with configurable output stage, using torchvision pretrained weights.

    Replicates the original BIT-CD behavior of replace_stride_with_dilation=[False, True, True]
    by manually setting stride=1 for layer3 and layer4 (BasicBlock doesn't support dilation>1
    in torchvision, so the original code effectively used stride=1 without actual dilation).
    """

    def __init__(self, backbone='resnet18', resnet_stages_num=4, pretrained=True):
        super().__init__()
        expand = 1
        if backbone == 'resnet18':
            weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
            self.resnet = tv_models.resnet18(weights=weights)
        elif backbone == 'resnet34':
            weights = tv_models.ResNet34_Weights.DEFAULT if pretrained else None
            self.resnet = tv_models.resnet34(weights=weights)
        elif backbone == 'resnet50':
            weights = tv_models.ResNet50_Weights.DEFAULT if pretrained else None
            self.resnet = tv_models.resnet50(weights=weights)
            expand = 4
        else:
            raise NotImplementedError(f'Unknown backbone: {backbone}')

        self._modify_layer_strides()

        self.relu = nn.ReLU()
        self.upsamplex2 = nn.Upsample(scale_factor=2)
        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.resnet_stages_num = resnet_stages_num
        self.if_upsample_2x = True

        if resnet_stages_num == 5:
            layers = 512 * expand
        elif resnet_stages_num == 4:
            layers = 256 * expand
        elif resnet_stages_num == 3:
            layers = 128 * expand
        else:
            raise NotImplementedError
        self.conv_pred = nn.Conv2d(layers, 32, kernel_size=3, padding=1)

    def _modify_layer_strides(self):
        """Set stride=1 in layer3 and layer4 downsampling to replicate
        replace_stride_with_dilation=[False, True, True] from original BIT-CD.

        This keeps spatial resolution at 1/8 for layer3 output instead of 1/16.
        """
        for layer_name in ['layer3', 'layer4']:
            layer = getattr(self.resnet, layer_name, None)
            if layer is None:
                continue
            block = layer[0]
            if hasattr(block, 'conv1'):
                block.conv1.stride = (1, 1)
            if hasattr(block, 'downsample'):
                ds = block.downsample
                if ds is not None and len(ds) > 0 and isinstance(ds[0], nn.Conv2d):
                    ds[0].stride = (1, 1)

    def forward_single(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x_4 = self.resnet.layer1(x)
        x_8 = self.resnet.layer2(x_4)

        if self.resnet_stages_num > 3:
            x_8 = self.resnet.layer3(x_8)

        if self.resnet_stages_num == 5:
            x_8 = self.resnet.layer4(x_8)
        elif self.resnet_stages_num > 5:
            raise NotImplementedError

        if self.if_upsample_2x:
            x = self.upsamplex2(x_8)
        else:
            x = x_8
        x = self.conv_pred(x)
        return x


class BASE_Transformer(ResNetBackbone):
    """BIT-CD model: ResNet backbone + BIT (tokenizer + transformer encoder/decoder)
    + bitemporal feature differencing + small CNN classifier."""

    def __init__(self, input_nc=3, output_nc=2, with_pos='learned', resnet_stages_num=4,
                 token_len=4, token_trans=True,
                 enc_depth=1, dec_depth=1,
                 dim_head=64, decoder_dim_head=64,
                 tokenizer=True, if_upsample_2x=True,
                 pool_mode='max', pool_size=2,
                 backbone='resnet18',
                 decoder_softmax=True, with_decoder_pos=None,
                 with_decoder=True, pretrained=True):
        super().__init__(backbone=backbone, resnet_stages_num=resnet_stages_num,
                         pretrained=pretrained)
        self.token_len = token_len
        self.conv_a = nn.Conv2d(32, self.token_len, kernel_size=1, padding=0, bias=False)
        self.tokenizer = tokenizer
        if not self.tokenizer:
            self.pooling_size = pool_size
            self.pool_mode = pool_mode
            self.token_len = self.pooling_size * self.pooling_size

        self.token_trans = token_trans
        self.with_decoder = with_decoder
        dim = 32
        mlp_dim = 2 * dim

        self.with_pos = with_pos
        if with_pos == 'learned':
            self.pos_embedding = nn.Parameter(torch.randn(1, self.token_len * 2, 32))

        decoder_pos_size = 256 // 4
        self.with_decoder_pos = with_decoder_pos
        if self.with_decoder_pos == 'learned':
            self.pos_embedding_decoder = nn.Parameter(
                torch.randn(1, 32, decoder_pos_size, decoder_pos_size))

        self.enc_depth = enc_depth
        self.dec_depth = dec_depth
        self.dim_head = dim_head
        self.decoder_dim_head = decoder_dim_head
        self.transformer = Transformer(dim=dim, depth=self.enc_depth, heads=8,
                                       dim_head=self.dim_head, mlp_dim=mlp_dim, dropout=0)
        self.transformer_decoder = TransformerDecoder(
            dim=dim, depth=self.dec_depth, heads=8,
            dim_head=self.decoder_dim_head, mlp_dim=mlp_dim, dropout=0,
            softmax=decoder_softmax)

        self.classifier = TwoLayerConv2d(in_channels=32, out_channels=output_nc)
        self.output_sigmoid = False
        self.sigmoid = nn.Sigmoid()

    def _forward_semantic_tokens(self, x):
        b, c, h, w = x.shape
        spatial_attention = self.conv_a(x)
        spatial_attention = spatial_attention.view([b, self.token_len, -1]).contiguous()
        spatial_attention = torch.softmax(spatial_attention, dim=-1)
        x = x.view([b, c, -1]).contiguous()
        tokens = torch.einsum('bln,bcn->blc', spatial_attention, x)
        return tokens

    def _forward_reshape_tokens(self, x):
        if self.pool_mode == 'max':
            x = F.adaptive_max_pool2d(x, [self.pooling_size, self.pooling_size])
        elif self.pool_mode == 'ave':
            x = F.adaptive_avg_pool2d(x, [self.pooling_size, self.pooling_size])
        tokens = rearrange(x, 'b c h w -> b (h w) c')
        return tokens

    def _forward_transformer(self, x):
        if self.with_pos:
            x += self.pos_embedding
        x = self.transformer(x)
        return x

    def _forward_transformer_decoder(self, x, m):
        b, c, h, w = x.shape
        if self.with_decoder_pos == 'learned':
            x = x + self.pos_embedding_decoder
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.transformer_decoder(x, m)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h)
        return x

    def _forward_simple_decoder(self, x, m):
        b, c, h, w = x.shape
        b, l, c = m.shape
        m = m.expand([h, w, b, l, c])
        m = rearrange(m, 'h w b l c -> l b c h w')
        m = m.sum(0)
        x = x + m
        return x

    def forward(self, x1, x2):
        x1 = self.forward_single(x1)
        x2 = self.forward_single(x2)

        if self.tokenizer:
            token1 = self._forward_semantic_tokens(x1)
            token2 = self._forward_semantic_tokens(x2)
        else:
            token1 = self._forward_reshape_tokens(x1)
            token2 = self._forward_reshape_tokens(x2)

        if self.token_trans:
            self.tokens_ = torch.cat([token1, token2], dim=1)
            self.tokens = self._forward_transformer(self.tokens_)
            token1, token2 = self.tokens.chunk(2, dim=1)

        if self.with_decoder:
            x1 = self._forward_transformer_decoder(x1, token1)
            x2 = self._forward_transformer_decoder(x2, token2)
        else:
            x1 = self._forward_simple_decoder(x1, token1)
            x2 = self._forward_simple_decoder(x2, token2)

        x = torch.abs(x1 - x2)
        if not self.if_upsample_2x:
            x = self.upsamplex2(x)
        x = self.upsamplex4(x)
        x = self.classifier(x)
        if self.output_sigmoid:
            x = self.sigmoid(x)
        return x


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize only the newly-added layers, preserving pretrained backbone.

    The ``self.resnet`` submodule (ImageNet-pretrained backbone) is explicitly
    skipped so that ``--pretrained True`` actually takes effect.
    """
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    # Apply init only to non-backbone modules; keep pretrained ResNet intact.
    for name, module in net.named_children():
        if name == 'resnet':
            continue
        module.apply(init_func)


def build_bit_cd_model(net_name='base_transformer_pos_s4_dd8', num_classes=2,
                       pretrained=True):
    """Build a BIT-CD model.

    Args:
        net_name: model variant name
            'base_resnet18' — ResNet18 only
            'base_transformer_pos_s4' — BIT with 1 decoder layer
            'base_transformer_pos_s4_dd8' — BIT with 8 decoder layers (default, best)
            'base_transformer_pos_s4_dd8_dedim8' — BIT with 8 decoder layers, dim_head=8
        num_classes: 2 for BCD, 8 for SCD
        pretrained: whether to use ImageNet-pretrained ResNet weights
    """
    if net_name == 'base_resnet18':
        net = BASE_Transformer(
            input_nc=3, output_nc=num_classes, resnet_stages_num=4,
            with_pos=None, pretrained=pretrained)
    elif net_name == 'base_transformer_pos_s4':
        net = BASE_Transformer(
            input_nc=3, output_nc=num_classes, token_len=4, resnet_stages_num=4,
            with_pos='learned', pretrained=pretrained)
    elif net_name == 'base_transformer_pos_s4_dd8':
        net = BASE_Transformer(
            input_nc=3, output_nc=num_classes, token_len=4, resnet_stages_num=4,
            with_pos='learned', enc_depth=1, dec_depth=8, pretrained=pretrained)
    elif net_name == 'base_transformer_pos_s4_dd8_dedim8':
        net = BASE_Transformer(
            input_nc=3, output_nc=num_classes, token_len=4, resnet_stages_num=4,
            with_pos='learned', enc_depth=1, dec_depth=8, decoder_dim_head=8,
            pretrained=pretrained)
    else:
        raise NotImplementedError(f'Unknown model: {net_name}')

    init_weights(net)
    return net
