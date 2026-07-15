"""Building blocks for the Weight-Space Planner (WSP).

The generic components deliberately follow LeWM's PyTorch implementation.  The
weight-space encoder, slot pooling, functional INR renderer, and anchor are the
WSP-specific pieces ported from the supplied Equinox model.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    """AdaLN modulation used by DiT and LeWM."""
    return x * (1.0 + scale) + shift


class FeedForward(nn.Module):
    """LeWM-style Transformer feed-forward network."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        # Normalisation is performed explicitly by ConditionalBlock.
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim), nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class Attention(nn.Module):
    """LeWM-style scaled dot-product attention with native causal masking.

    PyTorch's SDPA causal mode constructs the smooth, sequence-length-dependent
    triangular mask at call time.  Thus no rigid JAX mask or maximum-length mask
    is stored in the model.
    """

    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64,
                 dropout: float = 0.0):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head**-0.5
        self.dropout = dropout
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x: Tensor, causal: bool = True) -> Tensor:
        b, t, _ = x.shape
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q = q.view(b, t, self.heads, self.dim_head).transpose(1, 2)
        k = k.view(b, t, self.heads, self.dim_head).transpose(1, 2)
        v = v.view(b, t, self.heads, self.dim_head).transpose(1, 2)
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=causal,
            scale=self.scale,
        )
        return self.to_out(out.transpose(1, 2).reshape(b, t, -1))


class ConditionalBlock(nn.Module):
    """LeWM/DiT AdaLN-zero block conditioned by an action embedding."""

    def __init__(self, dim: int, heads: int, dim_head: int, mlp_dim: int,
                 dropout: float = 0.0):
        super().__init__()
        self.attn = Attention(dim, heads, dim_head, dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True)
        )
        # This exact zero initialisation makes every block an identity at init.
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        parts = self.adaLN_modulation(c).chunk(6, dim=-1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = parts
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class Transformer(nn.Module):
    """Conditional Transformer retaining the public structure used by LeWM."""

    def __init__(self, input_dim: int, condition_dim: int, hidden_dim: int,
                 output_dim: int, depth: int, heads: int, dim_head: int,
                 mlp_dim: int, dropout: float = 0.0):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.cond_proj = (
            nn.Linear(condition_dim, hidden_dim)
            if condition_dim != hidden_dim else nn.Identity()
        )
        self.layers = nn.ModuleList([
            ConditionalBlock(hidden_dim, heads, dim_head, mlp_dim, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        x, c = self.input_proj(x), self.cond_proj(c)
        for block in self.layers:
            x = block(x, c)
        return self.output_proj(self.norm(x))


class Embedder(nn.Module):
    """Action embedder matching LeWM's named component and tensor contract."""

    def __init__(self, input_dim: int = 10, smoothed_dim: int | None = None,
                 emb_dim: int = 384, mlp_scale: int = 4):
        super().__init__()
        smoothed_dim = smoothed_dim or input_dim
        self.patch_embed = nn.Conv1d(input_dim, smoothed_dim, 1)
        # mlp_scale=0 exposes the source WSP's single action projection while
        # retaining LeWM's dedicated Embedder abstraction and Conv1d smoother.
        self.embed = (
            nn.Linear(smoothed_dim, emb_dim)
            if mlp_scale == 0 else nn.Sequential(
                nn.Linear(smoothed_dim, mlp_scale * emb_dim), nn.SiLU(),
                nn.Linear(mlp_scale * emb_dim, emb_dim),
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.patch_embed(x.float().transpose(1, 2)).transpose(1, 2)
        return self.embed(x)


class SlotAttention(nn.Module):
    """Single-pass competitive learned-query pooling from the JAX WSP.

    This is intentionally not the recurrent Slot Attention variant: learned
    queries cross-attend once, tokens compete across slots, and each slot is
    subsequently normalised over spatial tokens exactly as in the source model.
    """

    def __init__(self, in_channels: int, slot_dim: int, num_slots: int,
                 num_heads: int = 8):
        super().__init__()
        if slot_dim % num_heads:
            raise ValueError("slot_dim must be divisible by num_heads")
        self.slot_dim = slot_dim
        self.num_heads = num_heads  # Kept in state/config parity with JAX.
        self.queries = nn.Parameter(torch.empty(num_slots, slot_dim))
        self.to_kv = nn.Linear(in_channels, 2 * slot_dim)
        self.to_q = nn.Linear(slot_dim, slot_dim)
        self.pos_proj = nn.Linear(2, in_channels)
        nn.init.normal_(self.queries, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        ys = torch.linspace(-1, 1, h, device=x.device, dtype=x.dtype)
        xs = torch.linspace(-1, 1, w, device=x.device, dtype=x.dtype)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        coords = torch.stack((gy, gx), dim=-1).reshape(h * w, 2)
        tokens = tokens + self.pos_proj(coords).unsqueeze(0)
        k, v = self.to_kv(tokens).chunk(2, dim=-1)
        q = self.to_q(self.queries).unsqueeze(0).expand(b, -1, -1)
        logits = torch.matmul(q, k.transpose(-1, -2)) * self.slot_dim**-0.5
        attention = logits.softmax(dim=1)
        attention = attention / (attention.sum(dim=-1, keepdim=True) + 1e-8)
        return torch.matmul(attention, v).flatten(1)


def _coord_channels(x: Tensor) -> Tensor:
    b, _, h, w = x.shape
    ys = torch.linspace(-1, 1, h, device=x.device, dtype=x.dtype)
    xs = torch.linspace(-1, 1, w, device=x.device, dtype=x.dtype)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    coords = torch.stack((gy, gx)).unsqueeze(0).expand(b, -1, -1, -1)
    return torch.cat((x, coords), dim=1)


class WeightCNN(nn.Module):
    """The original CoordConv + Slot Attention WSP encoder."""

    def __init__(self, in_channels: int, out_dim: int, hidden_width: int = 48,
                 depth: int = 5, use_coordinate_chans: bool = True,
                 num_slots: int = 6, slot_dim: int = 128, slot_heads: int = 8):
        super().__init__()
        self.use_coordinate_chans = use_coordinate_chans
        effective_in = in_channels + (2 if use_coordinate_chans else 0)
        channels = [effective_in] + [hidden_width * 2**i for i in range(depth)]
        self.convs = nn.ModuleList([
            nn.Conv2d(channels[i], channels[i + 1], 3, stride=2, padding=1)
            for i in range(depth)
        ])
        self.slot_pool = SlotAttention(
            channels[-1], slot_dim, num_slots, slot_heads
        )
        self.proj = nn.Linear(num_slots * slot_dim, out_dim, bias=True)

    def forward(self, x: Tensor, **_: object) -> Tensor:
        if self.use_coordinate_chans:
            x = _coord_channels(x)
        for conv in self.convs:
            x = F.relu(conv(x))
        return self.proj(self.slot_pool(x))


class WeightViT(nn.Module):
    """Configurable ViT alternative that returns an INR weight offset."""

    def __init__(self, in_channels: int, out_dim: int, image_size: int = 224,
                 patch_size: int = 16, hidden_dim: int = 384, depth: int = 6,
                 heads: int = 6, mlp_dim: int = 1536, dropout: float = 0.1):
        super().__init__()
        if image_size % patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        n = (image_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(in_channels, hidden_dim, patch_size, patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, n + 1, hidden_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            hidden_dim, heads, mlp_dim, dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(hidden_dim)
        self.proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: Tensor, **_: object) -> Tensor:
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls, x), dim=1) + self.pos_embedding[:, :x.shape[1] + 1]
        return self.proj(self.norm(self.transformer(x)[:, 0]))


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.skip = nn.Identity() if in_channels == out_channels and stride == 1 else nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        return F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x))))) + self.skip(x))


class WeightResNet(nn.Module):
    """Compact ResNet alternative whose pooled output is an INR offset."""

    def __init__(self, in_channels: int, out_dim: int,
                 channels: Sequence[int] = (64, 128, 256, 512),
                 blocks_per_stage: int = 2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], 7, 2, 3, bias=False),
            nn.BatchNorm2d(channels[0]), nn.ReLU(), nn.MaxPool2d(3, 2, 1),
        )
        stages, previous = [], channels[0]
        for stage_idx, width in enumerate(channels):
            blocks = [ResidualBlock(previous, width, 1 if stage_idx == 0 else 2)]
            blocks += [ResidualBlock(width, width) for _ in range(blocks_per_stage - 1)]
            stages.append(nn.Sequential(*blocks))
            previous = width
        self.stages = nn.Sequential(*stages)
        self.proj = nn.Linear(channels[-1], out_dim)

    def forward(self, x: Tensor, **_: object) -> Tensor:
        x = self.stages(self.stem(x)).mean(dim=(-2, -1))
        return self.proj(x)


class Encoder(nn.Module):
    """YAML-switchable wrapper for CNN, ViT, and ResNet weight encoders."""

    def __init__(self, kind: str, in_channels: int, out_dim: int,
                 cnn: Mapping | None = None, vit: Mapping | None = None,
                 resnet: Mapping | None = None):
        super().__init__()
        options = {"cnn": WeightCNN, "vit": WeightViT, "resnet": WeightResNet}
        if kind not in options:
            raise ValueError(f"unknown encoder kind {kind!r}; choose {tuple(options)}")
        settings = dict({"cnn": cnn, "vit": vit, "resnet": resnet}[kind] or {})
        self.kind = kind
        self.encoder = options[kind](in_channels=in_channels, out_dim=out_dim, **settings)

    def forward(self, x: Tensor, **kwargs: object) -> Tensor:
        return self.encoder(x, **kwargs)


class Predictor(nn.Module):
    """Causal AdaLN predictor for the next INR weight offset."""

    def __init__(self, num_frames: int, input_dim: int, condition_dim: int,
                 hidden_dim: int, depth: int, heads: int, mlp_dim: int,
                 output_dim: int | None = None, dim_head: int | None = None,
                 dropout: float = 0.0, emb_dropout: float = 0.0):
        super().__init__()
        self.num_frames = num_frames  # stable-world-model calls this history_size.
        output_dim = output_dim or input_dim
        dim_head = dim_head or hidden_dim // heads
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, input_dim) * 0.02)
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            input_dim, condition_dim, hidden_dim, output_dim, depth, heads,
            dim_head, mlp_dim, dropout,
        )

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        t = x.shape[1]
        if t > self.num_frames:
            raise ValueError(f"sequence length {t} exceeds history_size {self.num_frames}")
        return self.transformer(self.dropout(x + self.pos_embedding[:, :t]), c)


class FunctionalINR(nn.Module):
    """Tiny Fourier-feature INR evaluated from a batched flat weight vector."""

    def __init__(self, in_dim: int, out_dim: int, width: int, depth: int):
        super().__init__()
        if depth < 2:
            raise ValueError("inr_depth must contain at least two linear layers")
        self.dims = (in_dim, *([width] * (depth - 1)), out_dim)

    @property
    def num_parameters(self) -> int:
        return sum(o * i + o for i, o in zip(self.dims[:-1], self.dims[1:]))

    def initial_flat_weights(self, device: torch.device | None = None) -> Tensor:
        """Initialise exactly as ordinary PyTorch Linear layers, then flatten."""
        pieces = []
        for in_features, out_features in zip(self.dims[:-1], self.dims[1:]):
            layer = nn.Linear(in_features, out_features, device=device)
            pieces.extend((layer.weight.detach().flatten(), layer.bias.detach().flatten()))
        return torch.cat(pieces)

    def forward(self, features: Tensor, flat_weights: Tensor) -> Tensor:
        """features: (N,F), weights: (B,Z) -> pixels: (B,N,C)."""
        x = features.unsqueeze(0).expand(flat_weights.shape[0], -1, -1)
        cursor = 0
        for index, (in_features, out_features) in enumerate(zip(self.dims[:-1], self.dims[1:])):
            count = in_features * out_features
            weight = flat_weights[:, cursor:cursor + count].view(-1, out_features, in_features)
            cursor += count
            bias = flat_weights[:, cursor:cursor + out_features]
            cursor += out_features
            x = torch.bmm(x, weight.transpose(1, 2)) + bias[:, None]
            if index < len(self.dims) - 2:
                x = F.gelu(x)
        return x


def fourier_encode(coords: Tensor, num_frequencies: int) -> Tensor:
    """JAX-compatible [sin, cos] Fourier encoding of (y, x) coordinates."""
    frequencies = 2.0 ** torch.arange(num_frequencies, device=coords.device, dtype=coords.dtype)
    arguments = coords[..., None] * frequencies * torch.pi
    return torch.cat((arguments.sin(), arguments.cos()), dim=-1).flatten(-2)


__all__ = [
    "Attention", "ConditionalBlock", "Embedder", "FunctionalINR", "Predictor",
    "SlotAttention", "Transformer", "WeightCNN", "WeightEncoder", "WeightResNet", "WeightViT",
    "fourier_encode", "modulate",
]
