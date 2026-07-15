"""Train WSP end-to-end inside stable-world-model's Lightning/Hydra stack."""

from __future__ import annotations

import os
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
from stable_pretraining import data as dt
import stable_worldmodel as swm
import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

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

    loss = loss_dyn + float(cfg.loss.reconstruction_weight) * loss_enc
    output.update(loss=loss, loss_dyn=loss_dyn, loss_enc=loss_enc)
    self.log_dict(
        {f"{stage}/{name}": value.detach() for name, value in output.items() if "loss" in name},
        on_step=True, sync_dist=True,
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
