"""Evaluate a trained WSP with stable-world-model's standard MPC pipeline."""

import os
os.environ["MUJOCO_GL"] = "egl"

import time
from pathlib import Path

import hydra
import numpy as np
import stable_worldmodel as swm
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms


def img_transform(cfg, dtype=torch.float32):
    """WSP input transform: [0,255] HWC -> [0,1] CHW, with no ImageNet stats."""
    return transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(dtype, scale=True),
        transforms.Resize(size=cfg.eval.img_size),
    ])


def get_episodes_length(dataset, episodes):
    column = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    episode_idx = dataset.get_col_data(column)
    step_idx = dataset.get_col_data("step_idx")
    return np.asarray([np.max(step_idx[episode_idx == episode]) + 1 for episode in episodes])


def get_dataset(cfg, dataset_name):
    return swm.data.load_dataset(
        dataset_name, cache_dir=cfg.get("cache_dir"),
        keys_to_cache=list(cfg.dataset.keys_to_cache),
    )


@hydra.main(version_base=None, config_path="./config", config_name="tworoom")
def run(cfg: DictConfig):
    """Run the unchanged SWM shooting planner and environment evaluation."""
    assert cfg.plan_config.horizon * cfg.plan_config.action_block <= cfg.eval.eval_budget
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = swm.World(**cfg.world, image_shape=(cfg.eval.img_size, cfg.eval.img_size))

    dtype = torch.bfloat16 if cfg.get("bf16", False) else torch.float32
    transform = {"pixels": img_transform(cfg, dtype), "goal": img_transform(cfg, dtype)}
    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    episode_column = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    episode_ids = np.unique(dataset.get_col_data(episode_column))

    # This is the library's action/proprio normalisation used during planning.
    process = {}
    for column in cfg.dataset.keys_to_cache:
        if column == "pixels":
            continue
        values = dataset.get_col_data(column)
        values = values[~np.isnan(values).any(axis=1)]
        process[column] = preprocessing.StandardScaler().fit(values)
        if column != "action":
            process[f"goal_{column}"] = process[column]

    policy_name = cfg.get("policy", "random")
    if policy_name == "random":
        policy = swm.policy.RandomPolicy()
    else:
        model = swm.wm.utils.load_pretrained(policy_name)
        model = model.to(device="cuda", dtype=dtype).eval().requires_grad_(False)
        if cfg.get("compile", False):
            model.encoder = torch.compile(model.encoder)
            model.predictor = torch.compile(model.predictor)
        plan_config = swm.PlanConfig(**cfg.plan_config)
        objective = hydra.utils.instantiate(cfg.objective)
        cost = swm.planning.ShootingCostEvaluator(model, objective)
        solver = hydra.utils.instantiate(cfg.solver, cost=cost)
        policy = swm.policy.WorldModelPolicy(
            solver=solver, config=plan_config, process=process, transform=transform
        )
    world.set_policy(policy)

    lengths = get_episodes_length(dataset, episode_ids)
    max_starts = lengths - cfg.eval.goal_offset_steps - 1
    maximum_by_episode = dict(zip(episode_ids, max_starts))
    row_maximum = np.asarray([
        maximum_by_episode[e] for e in dataset.get_col_data(episode_column)
    ])
    valid = np.nonzero(dataset.get_col_data("step_idx") <= row_maximum)[0]
    if len(valid) < cfg.eval.num_eval:
        raise ValueError("Not enough valid starting points for the requested evaluation")
    rng = np.random.default_rng(cfg.seed)
    rows = np.sort(rng.choice(valid, size=cfg.eval.num_eval, replace=False))
    row_data = dataset.get_row_data(rows)
    eval_episodes = row_data[episode_column]
    eval_starts = row_data["step_idx"]

    results_dir = (
        Path(swm.data.utils.get_cache_dir(sub_folder="checkpoints"), policy_name).parent
        if policy_name != "random" else Path(__file__).parent
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    autocast = torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.get("bf16", False))
    start = time.time()
    with autocast:
        metrics = world.evaluate(
            dataset=dataset, start_steps=eval_starts.tolist(),
            goal_offset=cfg.eval.goal_offset_steps, eval_budget=cfg.eval.eval_budget,
            episodes_idx=eval_episodes.tolist(),
            callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
            video=results_dir,
        )
    elapsed = time.time() - start
    print(metrics)

    results_file = results_dir / cfg.output.filename
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with results_file.open("a") as stream:
        stream.write("\n==== CONFIG ====\n")
        stream.write(OmegaConf.to_yaml(cfg))
        stream.write(f"\n==== RESULTS ====\nmetrics: {metrics}\nevaluation_time: {elapsed} seconds\n")


if __name__ == "__main__":
    run()
