"""Train WSP end-to-end inside stable-world-model's Lightning/Hydra stack."""

from __future__ import annotations

import os
import tempfile
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import matplotlib.pyplot as plt
import numpy as np
import imageio.v2 as imageio
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


class LogWSPValidationCallback(Callback):
    """Aggregate validation-only rollout losses and log a single epoch chart."""

    def on_validation_epoch_start(self, trainer, pl_module):
        pl_module._wsp_val_recon_losses = []

    def on_validation_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
        dataloader_idx=0,
    ):
        if isinstance(outputs, dict) and 'loss_recon_pred' in outputs:
            pl_module._wsp_val_recon_losses.append(
                outputs['loss_recon_pred'].detach().float().cpu()
            )

    def on_validation_epoch_end(self, trainer, pl_module):
        if not getattr(trainer, 'is_global_zero', True):
            return
        if wandb.run is None or getattr(trainer, 'logger', None) is None:
            return
        losses = getattr(pl_module, '_wsp_val_recon_losses', None) or []
        if not losses:
            return
        recon_loss = torch.stack(losses).mean().item()
        wandb.log(
            {'validate/loss_recon_pred_epoch': recon_loss},
            step=int(getattr(pl_module, 'global_step', 0)),
        )


def _predict_next_offsets(model, emb, act_emb, history_size: int):
    """Predict every adjacent target using at most ``history_size`` states.

    With the original WSP default history_size=1 this is exactly the JAX
    context-one computation, while larger values enable ordinary SWM causal
    history without introducing a fixed global mask.
    """
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

    padded = torch.zeros((length, *frames.shape[1:]), dtype=frames.dtype)
    used = min(length, frames.shape[0])
    padded[:used] = frames[:used]
    return padded, used


def _predict_rollout_pixels(model, batch, rollout_steps: int):
    """Roll out WSP on a single validation episode and render predicted pixels."""
    steps = max(2, int(rollout_steps))

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

    truth_pixels, _ = _pad_or_truncate_frames(batch['pixels'][0], steps)
    predicted_pixels, _ = _pad_or_truncate_frames(
        model.render(predicted_emb[0, 0]), steps
    )
    return truth_pixels, predicted_pixels


def _stage_name(stage) -> str:
    return str(stage).lower()


def _is_validation_stage(stage) -> bool:
    return _stage_name(stage).startswith('val')


def _pixel_reconstruction_loss(model, predicted, target_pixels):
    """Compute full-image reconstruction loss for predicted rollouts."""
    predicted_pixels = model.render(predicted).float()
    target_pixels = target_pixels[:, : predicted_pixels.shape[1]].permute(0, 1, 3, 4, 2).float()
    predicted_pixels = predicted_pixels[:, : target_pixels.shape[1]]
    return (predicted_pixels - target_pixels).pow(2).mean()


def _make_episode_figure(truth_pixels, predicted_pixels):
    """Build a rollout grid image with truth and prediction rows."""
    truth_pixels = truth_pixels.detach().float().cpu()
    predicted_pixels = predicted_pixels.detach().float().cpu()
    if truth_pixels.ndim != 4 or predicted_pixels.ndim != 4:
        raise ValueError('Expected truth/predicted pixels shaped (T, C, H, W).')

    n_frames = min(truth_pixels.shape[0], predicted_pixels.shape[0])
    truth_pixels = truth_pixels[:n_frames]
    predicted_pixels = predicted_pixels[:n_frames]

    fig, axes = plt.subplots(
        2, n_frames, figsize=(max(12, 0.55 * n_frames), 4.8), squeeze=False
    )

    def _to_hwc(frame):
        if frame.ndim != 3:
            raise ValueError(f'Expected a 3D frame, got shape {tuple(frame.shape)}')
        if frame.shape[0] in (1, 3) and frame.shape[-1] not in (1, 3):
            return frame.permute(1, 2, 0).numpy()
        return frame.numpy()

    for idx in range(n_frames):
        truth_frame = _to_hwc(truth_pixels[idx])
        pred_frame = _to_hwc(predicted_pixels[idx])

        truth_frame = (truth_frame - truth_frame.min()) / (
            truth_frame.max() - truth_frame.min() + 1e-6
        )
        pred_frame = (pred_frame - pred_frame.min()) / (
            pred_frame.max() - pred_frame.min() + 1e-6
        )

        axes[0, idx].imshow(truth_frame)
        axes[1, idx].imshow(pred_frame)
        axes[0, idx].axis('off')
        axes[1, idx].axis('off')
        if idx == 0:
            axes[0, idx].set_ylabel('Ground truth', rotation=0, labelpad=30, va='center')
            axes[1, idx].set_ylabel('Predicted', rotation=0, labelpad=30, va='center')

        axes[0, idx].set_title(f'Ground Truth t={idx}')
        axes[1, idx].set_title(f'Predicted t={idx}')

    fig.tight_layout()
    return fig


def _figure_to_rgb_array(fig):
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    return rgba.reshape(height, width, 4)[..., :3]


def _make_episode_video(truth_pixels, predicted_pixels):
    """Animate the rollout as a two-box truth/prediction comparison."""
    truth_pixels = truth_pixels.detach().float().cpu()
    predicted_pixels = predicted_pixels.detach().float().cpu()
    n_frames = min(truth_pixels.shape[0], predicted_pixels.shape[0])
    if n_frames == 0:
        raise ValueError('No frames available for rollout video.')

    def _frame_or_blank(frames, index):
        if index >= frames.shape[0]:
            return np.zeros((*frames.shape[1:],), dtype=np.float32)
        frame = frames[index]
        if frame.ndim != 3:
            raise ValueError(f'Expected a 3D frame, got shape {tuple(frame.shape)}')
        if frame.shape[0] in (1, 3) and frame.shape[-1] not in (1, 3):
            frame = frame.permute(1, 2, 0)
        frame = frame.numpy()
        frame = (frame - frame.min()) / (frame.max() - frame.min() + 1e-6)
        return frame

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), squeeze=False)
    axes = axes[0]
    truth_im = axes[0].imshow(np.zeros((*truth_pixels.shape[2:], 3), dtype=np.float32))
    pred_im = axes[1].imshow(np.zeros((*predicted_pixels.shape[2:], 3), dtype=np.float32))
    axes[0].axis('off')
    axes[1].axis('off')
    axes[0].set_title('Ground truth')
    axes[1].set_title('Predicted')

    frames = []
    for frame_index in range(n_frames):
        truth_im.set_data(_frame_or_blank(truth_pixels, frame_index))
        pred_im.set_data(_frame_or_blank(predicted_pixels, frame_index))
        axes[0].set_title(f'Ground Truth t={frame_index}')
        axes[1].set_title(f'Predicted t={frame_index}')
        frames.append(_figure_to_rgb_array(fig))
    plt.close(fig)
    return np.stack(frames, axis=0)


def _log_val_episode_visualization(self, batch, predicted, cfg):
    """Log one validation episode image panel once per validation phase."""
    if not cfg.wandb.enabled or getattr(self, 'logger', None) is None:
        return
    if wandb.run is None:
        return

    trainer = getattr(self, 'trainer', None)
    if trainer is not None and not getattr(trainer, 'is_global_zero', True):
        return

    current_step = int(getattr(self, 'global_step', 0))
    last_logged_step = getattr(self, '_wsp_last_val_image_step', None)
    if last_logged_step == current_step:
        return

    rollout_steps = int(cfg.get('rollout_val_steps', batch['pixels'].shape[1]))
    truth_pixels, pred_pixels = _predict_rollout_pixels(
        self.model, batch, rollout_steps
    )

    fig = _make_episode_figure(truth_pixels, pred_pixels)
    video = _make_episode_video(truth_pixels, pred_pixels)

    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
        video_path = tmp.name
    try:
        imageio.mimsave(
            video_path,
            video,
            fps=int(cfg.get('rollout_val_fps', 2)),
            macro_block_size=1,
        )
        wandb.log(
            {
                'val/rollout_truth_vs_pred_image': wandb.Image(
                    fig, caption=f'step={current_step}'
                ),
                'val/rollout_truth_vs_pred_video': wandb.Video(
                    video_path, format='mp4'
                ),
            },
            step=current_step,
            commit=False,
        )
    finally:
        try:
            os.remove(video_path)
        except OSError:
            pass
    plt.close(fig)
    self._wsp_last_val_image_step = current_step


def wsp_forward(self, batch, stage, cfg):
    """Original single-phase WSP objective in stable-pretraining form.

    The source implementation computes ``loss_dyn + beta * loss_enc`` (despite
    one stale docstring claiming the inverse); this function follows executable
    source code.  Dynamics targets remain differentiable because the supplied
    JAX stop-gradient line is commented out.
    """
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
        # One shared random subset across frames, exactly as in the JAX step.
        indices = torch.randperm(height * width, device=emb.device)[:sample_count]
        sampled_coords = coords.reshape(-1, 2)[indices]
        targets = pixels.flatten(-2).transpose(-1, -2)[:, :, indices]
    else:
        sampled_coords = coords
        targets = pixels.permute(0, 1, 3, 4, 2)
    rendered = self.model.render(emb, sampled_coords)
    loss_enc = (rendered - targets).pow(2).mean()

    loss_recon_pred = None
    if _is_validation_stage(stage):
        loss_recon_pred = _pixel_reconstruction_loss(
            self.model, predicted, batch['pixels'][:, 1:]
        )

    loss = loss_dyn + float(cfg.loss.reconstruction_weight) * loss_enc
    output.update(loss=loss, loss_dyn=loss_dyn, loss_enc=loss_enc)
    if loss_recon_pred is not None:
        output['loss_recon_pred'] = loss_recon_pred
    self.log_dict(
        {f"{stage}/{name}": value.detach() for name, value in output.items() if name in {'loss', 'loss_dyn', 'loss_enc'}},
        on_step=True, on_epoch=True, sync_dist=True,
        batch_size=batch['pixels'].shape[0],
    )
    if _is_validation_stage(stage):
        _log_val_episode_visualization(self, batch, predicted, cfg)

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
        # frameskip actions are concatenated by the SWM dataset pipeline.
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
    val_callback = LogWSPValidationCallback()
    trainer = pl.Trainer(
        **cfg.trainer, callbacks=[callback, val_callback], num_sanity_val_steps=1, logger=logger,
        enable_checkpointing=True,
    )
    ckpt_path = run_dir / f"{cfg.output_model_name}_weights.ckpt"
    spt.Manager(
        trainer=trainer, module=module, data=data_module,
        ckpt_path=ckpt_path if ckpt_path.exists() else None,
    )()


if __name__ == "__main__":
    run()
