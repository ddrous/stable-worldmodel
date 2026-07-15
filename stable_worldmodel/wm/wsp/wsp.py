"""Stable-world-model compatible definition of the Weight-Space Planner."""

from __future__ import annotations

from typing import Any, Mapping

import hydra
import torch
from einops import rearrange
from omegaconf import DictConfig, OmegaConf
from torch import Tensor, nn

from stable_worldmodel.wm.wsp.module import FunctionalINR, fourier_encode


def _instantiate(config: Any, **kwargs: Any) -> nn.Module:
    """Instantiate a Hydra node while allowing derived dimensions to override YAML."""
    if isinstance(config, nn.Module):
        return config
    return hydra.utils.instantiate(config, **kwargs)


class WSP(nn.Module):
    """End-to-end world model whose state is a residual INR weight vector.

    Public methods intentionally implement LeWM's ``encode``/``predict``/
    ``rollout`` contract so the standard stable-world-model planners can use WSP
    without a WSP-specific CEM implementation.
    """

    def __init__(
        self,
        encoder: Mapping[str, Any] | DictConfig | nn.Module,
        predictor: Mapping[str, Any] | DictConfig | nn.Module,
        action_encoder: Mapping[str, Any] | DictConfig | nn.Module,
        inr: Mapping[str, Any] | DictConfig,
        frame_channels: int = 3,
        image_size: int = 224,
        **_: Any,
    ):
        super().__init__()
        inr_cfg = OmegaConf.to_container(inr, resolve=True) if isinstance(inr, DictConfig) else dict(inr)
        self.num_fourier_frequencies = int(inr_cfg["num_fourier_frequencies"])
        fourier_dim = 4 * self.num_fourier_frequencies
        self.inr = FunctionalINR(
            fourier_dim, frame_channels, int(inr_cfg["width"]), int(inr_cfg["depth"])
        )
        self.z_dim = self.inr.num_parameters
        self.image_size = image_size

        # The derived z_dim is the encoder output and predictor input/output.
        self.encoder = _instantiate(encoder, out_dim=self.z_dim)
        self.predictor = _instantiate(
            predictor, input_dim=self.z_dim, output_dim=self.z_dim
        )
        self.action_encoder = _instantiate(action_encoder)

        # WSP's stabilising contribution: this is not a random latent.  It begins
        # as the complete flat parameter vector of a valid template INR and is
        # learned jointly with the encoder and predictor.
        self.anchor = nn.Parameter(self.inr.initial_flat_weights())

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def encode_pixels(self, pixels: Tensor) -> Tensor:
        """Encode BCHW pixels in [0,1] into residual INR weights."""
        pixels = pixels.to(device=self.anchor.device, dtype=self.dtype)
        # Preserve the JAX encoder's sole input normalisation.
        return self.encoder(pixels.mul(2.0).sub(1.0))

    def encode(self, info: dict[str, Any]) -> dict[str, Any]:
        """LeWM-compatible encoding of a batch dictionary.

        ``pixels`` may be (B,T,C,H,W) or (B,C,H,W).  The returned ``emb`` is
        always temporal, (B,T,z_dim), which is what SWM objectives expect.
        """
        pixels = info["pixels"]
        if pixels.ndim == 4:
            pixels = pixels[:, None]
        if pixels.ndim != 5:
            raise ValueError(f"expected pixels shaped B,T,C,H,W; got {tuple(pixels.shape)}")
        b, t = pixels.shape[:2]
        emb = self.encode_pixels(rearrange(pixels, "b t c h w -> (b t) c h w"))
        info["emb"] = rearrange(emb, "(b t) d -> b t d", b=b, t=t)
        if "action" in info:
            info["act_emb"] = self.action_encoder(torch.nan_to_num(info["action"], nan=0.0))
        return info

    def predict(self, emb: Tensor, act_emb: Tensor) -> Tensor:
        """Predict each token's next residual weight vector causally."""
        return self.predictor(emb, act_emb)

    def coordinate_grid(self, height: int, width: int, *, device=None,
                        dtype=None) -> Tensor:
        """Return an HxWx2 grid ordered (y,x), exactly like the JAX model."""
        device = device or self.anchor.device
        dtype = dtype or self.anchor.dtype
        ys = torch.linspace(-1, 1, height, device=device, dtype=dtype)
        xs = torch.linspace(-1, 1, width, device=device, dtype=dtype)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack((gy, gx), dim=-1)

    def render(self, z_offset: Tensor, coords: Tensor | None = None,
               image_size: int | None = None) -> Tensor:
        """Render residual weights as images, retaining gradients to all weights.

        Args:
            z_offset: (..., z_dim) residual INR weights.
            coords: optional (...,2) coordinate grid or sampled coordinates.
            image_size: square render size when ``coords`` is omitted.
        Returns:
            (..., spatial..., C), matching the JAX renderer's channel-last form.
        """
        if coords is None:
            size = image_size or self.image_size
            coords = self.coordinate_grid(size, size)
        coords = coords.to(device=self.anchor.device, dtype=self.anchor.dtype)
        spatial_shape = coords.shape[:-1]
        features = fourier_encode(coords.reshape(-1, 2), self.num_fourier_frequencies)
        leading_shape = z_offset.shape[:-1]
        flat_offsets = z_offset.reshape(-1, self.z_dim).to(self.anchor.dtype)
        pixels = self.inr(features, flat_offsets + self.anchor)
        return pixels.reshape(*leading_shape, *spatial_shape, -1)

    def rollout(self, info: dict[str, Any], action_sequence: Tensor,
                history_size: int | None = None) -> dict[str, Any]:
        """SWM planner rollout, matching LeWM's B,S,T action interface."""
        history_size = history_size or self.predictor.num_frames
        if "pixels" not in info:
            raise KeyError("pixels not in info")
        h = info["pixels"].size(2)
        b, samples, horizon = action_sequence.shape[:3]
        initial_actions, future_actions = torch.split(
            action_sequence, [h, horizon - h], dim=2
        )
        info["action"] = initial_actions

        if "emb" not in info:
            initial = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
            initial = self.encode(initial)
            info["emb"] = initial["emb"].detach().unsqueeze(1).expand(
                b, samples, -1, -1
            )

        emb_init = rearrange(info["emb"], "b s ... -> (b s) ...")
        actions = rearrange(action_sequence, "b s ... -> (b s) ...")
        action_emb = self.action_encoder(actions)
        states = list(emb_init.unbind(dim=1))
        n_steps = horizon - h
        for step in range(n_steps + 1):
            end = h + step
            start = max(0, end - history_size)
            state_window = torch.stack(states[start:end], dim=1)
            action_window = action_emb[:, start:end]
            states.append(self.predict(state_window, action_window)[:, -1])
        trajectory = torch.stack(states, dim=1)
        info["predicted_emb"] = rearrange(
            trajectory, "(b s) ... -> b s ...", b=b, s=samples
        )
        return info


# A descriptive alias is useful outside Hydra while WSP preserves checkpoint names.
WeightSpacePlanner = WSP

__all__ = ["WSP", "WeightSpacePlanner"]
