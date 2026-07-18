"""Train WSP end-to-end inside stable-world-model's Lightning/Hydra stack."""

from __future__ import annotations

import os
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import numpy as np
import stable_pretraining as spt
from stable_pretraining import data as dt
import stable_worldmodel as swm
import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict
import wandb

from stable_worldmodel.data import column_normalizer as get_column_normalizer
from stable_worldmodel.wm.utils import save_pretrained


def get_wsp_img_preprocessor(source: str, target: str, img_size: int = 224):
    """Convert uint8 images to [0,1] CHW tensors; WSP does not use ImageNet stats."""
    to_image = dt.transforms.ToImage(
        mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0),
        source=source, target=target,
    )
    return dt.transforms.Compose(
        to_image, dt.transforms.Resize(img_size, source=source, target=target)
    )


class SaveWSPCallback(Callback):
    """Save portable model objects through SWM's standard checkpoint helper."""

    def __init__(self, run_name, cfg, epoch_interval: int = 5):
        super().__init__()
        self.run_name, self.cfg, self.epoch_interval = run_name, cfg, epoch_interval

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        if trainer.is_global_zero and (
            epoch % self.epoch_interval == 0 or epoch == trainer.max_epochs
        ):
            save_pretrained(
                pl_module.model, run_name=self.run_name, config=self.cfg,
                filename=f"weights_epoch_{epoch}.pt",
            )


def _predict_next_offsets(model, emb, act_emb, history_size: int):
    """Predict every adjacent target using at most ``history_size`` states. (Single forward pass)."""
    predictions = []
    for target_index in range(1, emb.shape[1]):
        start = max(0, target_index - history_size)
        prediction = model.predict(
            emb[:, start:target_index], act_emb[:, start:target_index]
        )[:, -1]
        predictions.append(prediction)
    return torch.stack(predictions, dim=1)


def _pad_or_truncate_frames(frames: torch.Tensor, length: int) -> tuple[torch.Tensor, int]:
    """Pad or trim a time sequence to a fixed length with blank frames."""
    frames = frames.detach().cpu()
    if frames.ndim < 4:
        raise ValueError(f'Expected a frame sequence, got shape {tuple(frames.shape)}')

    # Ensure missing frames are initialized as pure zeros (black boxes)
    padded = torch.zeros((length, *frames.shape[1:]), dtype=frames.dtype)
    used = min(length, frames.shape[0])
    padded[:used] = frames[:used]
    return padded, used


def _predict_rollout_pixels(model, batch, rollout_steps: int):
    """Roll out WSP on a single validation episode and render predicted pixels."""
    steps = max(2, int(rollout_steps))

    # Initial frame (t=0) to start the rollout shared between Truth and Prediction
    init_state = {}
    for key, value in batch.items():
        if torch.is_tensor(value) and value.ndim >= 3:
            init_state[key] = value[:1, :1].unsqueeze(0)

    action_sequence = batch['action'][:1, : steps - 1].unsqueeze(0)
    if action_sequence.shape[2] < steps - 1:
        pad = torch.zeros(
            (1, 1, steps - 1 - action_sequence.shape[2], action_sequence.shape[-1]),
            dtype=action_sequence.dtype,
            device=action_sequence.device,
        )
        action_sequence = torch.cat([action_sequence, pad], dim=2)

    rolled = model.rollout(init_state, action_sequence)
    predicted_emb = rolled.get('predicted_emb', rolled.get('predicted_embedding'))
    if predicted_emb is None:
        raise KeyError('rollout did not return predicted embeddings')

    # Pad truth to visually indicate missing frames if rollout_steps > available truth frames
    truth_pixels, _ = _pad_or_truncate_frames(batch['pixels'][0], steps)
    predicted_pixels, _ = _pad_or_truncate_frames(model.render(predicted_emb[0, 0]), steps)
    return truth_pixels, predicted_pixels


def _pixel_reconstruction_loss(model, predicted, target_pixels):
    """Compute full-image reconstruction loss for forward-pass predicted dynamics (last T-1 steps)."""
    predicted_pixels = model.render(predicted).float()
    target_pixels = target_pixels[:, : predicted_pixels.shape[1]].permute(0, 1, 3, 4, 2).float()
    predicted_pixels = predicted_pixels[:, : target_pixels.shape[1]]
    return (predicted_pixels - target_pixels).pow(2).mean()


def _prepare_frames_for_media(frames: torch.Tensor) -> np.ndarray:
    """Format and safely normalize frame tensors for visual rendering [T, 3, H, W].

    Normalizes ONCE across the whole sequence (not per-frame) so that
    brightness/contrast stays stable and the video doesn't flicker between
    frames purely because of independent min/max rescaling.
    """
    if frames.ndim != 4:
        raise ValueError(f"Expected 4D tensor, got {frames.shape}")

    if frames.shape[1] not in (1, 3):
        frames = frames.permute(0, 3, 1, 2)
    if frames.shape[1] == 1:
        frames = frames.repeat(1, 3, 1, 1)

    frames = frames.float()
    f_min, f_max = frames.min(), frames.max()
    if f_max > f_min:
        frames = (frames - f_min) / (f_max - f_min)
    else:
        frames = torch.zeros_like(frames)

    return frames.clamp(0.0, 1.0).contiguous().cpu().numpy()


def _upsample_frames(frames: np.ndarray, min_size: int = 128) -> np.ndarray:
    """Nearest-neighbor upsample [T, C, H, W] frames so small renders are still legible in wandb."""
    h, w = frames.shape[-2], frames.shape[-1]
    scale = max(1, int(np.ceil(min_size / max(h, w))))
    if scale == 1:
        return frames
    return np.repeat(np.repeat(frames, scale, axis=-2), scale, axis=-1)


def _make_pixel_perfect_media(truth_pixels, pred_pixels, upsample_to: int = 128):
    """Constructs pure numpy arrays for image/video logging bypassing matplotlib completely."""
    truth_frames = _prepare_frames_for_media(truth_pixels)
    pred_frames = _prepare_frames_for_media(pred_pixels)

    truth_frames = _upsample_frames(truth_frames, upsample_to)
    pred_frames = _upsample_frames(pred_frames, upsample_to)

    n_frames = min(truth_frames.shape[0], pred_frames.shape[0])
    if n_frames == 0:
        raise ValueError('No frames available for media rendering.')

    # 1. Construct Image Array (Top: Truth Row, Bottom: Predicted Row)
    truth_strip = np.concatenate([truth_frames[i] for i in range(n_frames)], axis=2)  # [3, H, T*W]
    pred_strip = np.concatenate([pred_frames[i] for i in range(n_frames)], axis=2)    # [3, H, T*W]

    # 4-pixel white gap between rows for clean visual separation
    img_gap = np.ones((3, 4, truth_strip.shape[2]), dtype=np.float32)
    full_image = np.concatenate([truth_strip, img_gap, pred_strip], axis=1)           # [3, 2H+4, T*W]

    # Convert [C, H, W] -> [H, W, C] for wandb.Image compatibility
    image_array_hwc = np.transpose(full_image, (1, 2, 0))
    image_array_hwc = (image_array_hwc * 255).astype(np.uint8)

    # 2. Construct Video Array (Left: Truth Box, Right: Predicted Box)
    video_frames = []
    for i in range(n_frames):
        t_frame = truth_frames[i]
        p_frame = pred_frames[i]
        vid_gap = np.ones((3, t_frame.shape[1], 4), dtype=np.float32)
        combined = np.concatenate([t_frame, vid_gap, p_frame], axis=2)                # [3, H, 2W+4]
        video_frames.append(combined)

    video_array = np.stack(video_frames, axis=0)                                      # [T, 3, H, 2W+4]

    # Convert to standard 8-bit unsigned integer limits for flawless video encoding
    video_array = np.ascontiguousarray((video_array * 255).astype(np.uint8))

    return image_array_hwc, video_array


def _log_val_episode_visualization(self, batch, cfg):
    """Log validation episode image + video panels, throttled to avoid slowing training.

    The throttle gates the *entire* rollout/render computation, not just the
    wandb.Video packaging step -- that rollout is the expensive part, so on
    non-logging steps we bail out before doing any of that work at all.

    We never call wandb.log() with an explicit `step=` here. Mixing an explicit
    step with Lightning's own auto-stepped self.log_dict() calls causes wandb to
    silently drop any subsequent scalar logs whose auto-assigned step is <= the
    explicit step we used -- which is why the loss panels vanished. We use
    wandb.run.log(..., commit=False) instead, so the media folds into
    Lightning's own next commit at the correct step.
    """
    if not cfg.wandb.enabled or getattr(self, 'logger', None) is None:
        return
    if wandb.run is None:
        return

    trainer = getattr(self, 'trainer', None)
    if trainer is not None and not getattr(trainer, 'is_global_zero', True):
        return

    current_step = int(getattr(self, 'global_step', 0))

    # Single interval gates the whole expensive block (rollout + render + image +
    # video). This is what fixes the slowdown: validation can fire very often
    # depending on val_check_interval, but we only pay for a rollout every
    # `log_every_n_steps` global steps.
    log_every_n_steps = int(cfg.get('val_check_interval', 250))
    last_log_step = getattr(self, '_wsp_last_val_log_step', -log_every_n_steps)
    if (current_step - last_log_step) < log_every_n_steps:
        return

    # Independent, coarser cadence just for the video (encoding is the priciest
    # part of this block), expressed as a multiple of log_every_n_steps.
    video_every_n_logs = int(cfg.get('rollout_val_video_every_n_logs', 1))
    log_count = getattr(self, '_wsp_val_log_count', 0)
    should_log_video = (log_count % max(1, video_every_n_logs)) == 0

    # Commit the throttle state BEFORE doing any expensive work or logging.
    # If we waited until after a successful log to update this, any failure
    # (like the logger-attribute bug below) would leave the throttle stuck at
    # its initial state forever, causing every single step to redo the full
    # rollout + video encode -- which is exactly what was happening here and
    # was the real cause of the training slowdown, not the logging itself.
    self._wsp_last_val_log_step = current_step
    self._wsp_val_log_count = log_count + 1

    try:
        # Default to the training sequence length if not explicitly specified.
        rollout_steps = int(cfg.get('rollout_val_steps', batch['pixels'].shape[1]))

        truth_pixels, pred_pixels = _predict_rollout_pixels(
            self.model, batch, rollout_steps
        )

        image_array, video_array = _make_pixel_perfect_media(truth_pixels, pred_pixels)

        log_payload = {
            # Panel 8
            'val/rollout_truth_vs_pred_image': wandb.Image(
                image_array,
                caption=f'Top: Ground Truth | Bottom: Predicted | Columns: Timeline (step={current_step})'
            ),
        }

        if should_log_video:
            # Panel 9
            log_payload['val/rollout_truth_vs_pred_video'] = wandb.Video(
                video_array,
                fps=int(cfg.get('rollout_val_fps', 4)),
                format='mp4',
                caption="Left: Ground Truth | Right: Predicted"
            )

        # wandb.run is the actual run object regardless of which Lightning
        # logger wrapper is in front of it (self.logger.experiment is not
        # reliably that object across Lightning/stable_pretraining versions,
        # which is what raised the AttributeError). commit=False folds this
        # into Lightning's own next step commit instead of forcing a step.
        wandb.run.log(log_payload, commit=False)

    except Exception as exc:  # noqa: BLE001 - never let visualization kill training
        print(f"[wsp] WARNING: validation visualization failed at step {current_step}: {exc}")


def wsp_forward(self, batch, stage, cfg):
    """Original single-phase WSP objective in stable-pretraining form."""

    # print(
    #     f"stage={stage}, global_step={self.global_step}"
    # )

    # print(
    #     f"stage={stage}, "
    #     f"module.training={self.training}, "
    #     f"trainer.training={self.trainer.training}, "
    #     f"trainer.validating={self.trainer.validating}, "
    #     f"global_step={self.global_step}"
    # )

    batch["action"] = torch.nan_to_num(batch["action"], nan=0.0)
    output = self.model.encode(batch)
    emb, act_emb = output["emb"], output["act_emb"]

    predicted = _predict_next_offsets(
        self.model, emb, act_emb, int(cfg.wm.history_size)
    )
    loss_dyn = (predicted - emb[:, 1:]).pow(2).mean()

    pixels = batch["pixels"].to(device=emb.device, dtype=emb.dtype)
    height, width = pixels.shape[-2:]
    coords = self.model.coordinate_grid(height, width)
    sample_count = cfg.loss.pixel_subsample_size
    if sample_count is not None and sample_count < height * width:
        indices = torch.randperm(height * width, device=emb.device)[:sample_count]
        sampled_coords = coords.reshape(-1, 2)[indices]
        targets = pixels.flatten(-2).transpose(-1, -2)[:, :, indices]
    else:
        sampled_coords = coords
        targets = pixels.permute(0, 1, 3, 4, 2)
    rendered = self.model.render(emb, sampled_coords)
    loss_enc = (rendered - targets).pow(2).mean()

    loss = loss_dyn + float(cfg.loss.reconstruction_weight) * loss_enc
    output.update(loss=loss, loss_dyn=loss_dyn, loss_enc=loss_enc)

    loss_recon_pred = None
    if stage.startswith('val'):
        loss_recon_pred = _pixel_reconstruction_loss(
            self.model, predicted, batch['pixels'][:, 1:]
        )
        output['loss_recon_pred'] = loss_recon_pred

    # --------------------------------------------------------------------------
    # Forced logging architecture to ensure exactly 7 scalar panels + 2 media
    # --------------------------------------------------------------------------
    if stage == "validate":
        val_logs = {
            "validation/loss": loss.detach(),
            "validation/loss_dyn": loss_dyn.detach(),
            "validation/loss_enc": loss_enc.detach(),
        }

        if loss_recon_pred is not None:
            val_logs["validation/loss_recon_pred"] = loss_recon_pred.detach()

        self.log_dict(
            val_logs,
            on_step=True,
            on_epoch=False,
            sync_dist=True,
            batch_size=batch["pixels"].shape[0],
        )

        _log_val_episode_visualization(self, batch, cfg)

    else:
        self.log_dict(
            {
                "train/loss": loss.detach(),
                "train/loss_dyn": loss_dyn.detach(),
                "train/loss_enc": loss_enc.detach(),
            },
            on_step=True,
            on_epoch=False,
            sync_dist=True,
            batch_size=batch["pixels"].shape[0],
        )

    return output


@hydra.main(version_base=None, config_path="./config", config_name="wsp")
def run(cfg):
    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop("name")
    cache_dir = os.environ.get("LOCAL_DATASET_DIR")
    dataset = swm.data.load_dataset(
        dataset_name, transform=None, cache_dir=cache_dir, **dataset_cfg
    )

    transforms = [get_wsp_img_preprocessor("pixels", "pixels", cfg.img_size)]
    with open_dict(cfg):
        for column in cfg.data.dataset.keys_to_load:
            if column.startswith("pixels"):
                continue
            transforms.append(get_column_normalizer(dataset, column, column))
        cfg.model.action_encoder.input_dim = (
            cfg.data.dataset.frameskip * dataset.get_dim("action")
        )
    dataset.transform = spt.data.transforms.Compose(*transforms)

    generator = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=generator
    )
    train_loader = torch.utils.data.DataLoader(train_set, **cfg.loader, generator=generator)
    val_options = {**cfg.loader, "shuffle": False, "drop_last": False}
    val_loader = torch.utils.data.DataLoader(val_set, **val_options)

    model = hydra.utils.instantiate(cfg.model)
    parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"WSP trainable parameters: {parameter_count:,} ({parameter_count / 1e6:.3f}M)")

    total_steps = cfg.trainer.max_epochs * len(train_loader)
    optimizers = {
        "model_opt": {
            "modules": "model", "optimizer": dict(cfg.optimizer),
            "scheduler": {
                "type": "LinearWarmupCosineAnnealingLR",
                "warmup_steps": max(1, int(0.01 * total_steps)),
                "max_steps": total_steps,
            },
            "interval": "epoch",
        }
    }
    module = spt.Module(
        model=model, forward=partial(wsp_forward, cfg=cfg), optim=optimizers
    )
    data_module = spt.data.DataModule(train=train_loader, val=val_loader)

    run_dir = Path(swm.data.utils.get_cache_dir(sub_folder="checkpoints"), cfg.get("subdir") or "")
    run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, run_dir / "config.yaml")
    logger = WandbLogger(**cfg.wandb.config) if cfg.wandb.enabled else None
    if logger:
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    callback = SaveWSPCallback(cfg.output_model_name, cfg.model, cfg.checkpoint_interval)

    trainer = pl.Trainer(
        **cfg.trainer, callbacks=[callback], num_sanity_val_steps=1, logger=logger,
        enable_checkpointing=True,
    )
    ckpt_path = run_dir / f"{cfg.output_model_name}_weights.ckpt"
    spt.Manager(
        trainer=trainer, module=module, data=data_module,
        ckpt_path=ckpt_path if ckpt_path.exists() else None,
    )()


if __name__ == "__main__":
    run()