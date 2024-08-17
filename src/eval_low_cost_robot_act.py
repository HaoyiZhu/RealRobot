from __future__ import annotations

import os
import time
from collections import defaultdict
from typing import Any, Dict, Optional, Tuple

import hydra
import imageio.v3 as iio
import lightning as L
import numpy as np
import rootutils
import torch
from einops import rearrange
from lightning import LightningModule
from omegaconf import DictConfig, OmegaConf

OmegaConf.register_new_resolver("eval", eval)

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from src.data.components.transformpcd import ComposePCD
from src.utils import RankedLogger, extras, point_collate_fn
from src.utils.pointcloud_utils import rgbd_to_pointcloud

log = RankedLogger(__name__, rank_zero_only=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
@torch.inference_mode()
def eval(cfg: DictConfig):
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    use_pcd = "pcd" in cfg.data.train._target_.lower()
    if use_pcd:
        calib_dict = np.load(cfg.data.train.calib_file, allow_pickle=True).item()
        pcd_range = np.array(cfg.data.train.pcd_range).reshape(3, 2)
        transform_pcd = ComposePCD(
            hydra.utils.instantiate(cfg.data.train.transform_pcd)
        )

    model: LightningModule = hydra.utils.instantiate(cfg.model)
    policy = model.policy

    ckpt = torch.load(cfg.ckpt_path, map_location="cpu")["state_dict"]
    ckpt = {k.replace("policy.", ""): v for k, v in ckpt.items() if "policy" in k}
    policy.load_state_dict(ckpt, strict=True)

    policy.eval()
    policy = policy.to(device)

    del model
    torch.cuda.empty_cache()

    if os.path.exists(os.path.join(cfg.norm_stats_file)):
        norm_stats = np.load(cfg.norm_stats_file, allow_pickle=True).item()
    else:
        log.error(f"Stats file not found: {cfg.norm_stats_file}")
        exit(1)

    log.info(f"Instantiating cameras <{cfg.camera._target_}>")
    camera = hydra.utils.instantiate(cfg.camera)

    log.info(f"Instantiating robot <{cfg.robot._target_}>")
    robot = hydra.utils.instantiate(cfg.robot)
    robot.connect()

    for _ in range(10):
        robot.read_position()
        camera.get_state()

    time.sleep(2)

    # load preprocessing and postprocessing functions
    pre_process = (
        lambda s_qpos: ((s_qpos / 2048 - 1) - norm_stats["qpos_mean"])
        / norm_stats["qpos_std"]
    )
    post_process = lambda a: (
        ((a * norm_stats["action_std"] + norm_stats["action_mean"]) + 1) * 2048
    ).astype(int)

    # load eval constants
    max_timesteps = cfg.max_timesteps
    query_frequency = cfg.model.policy.num_queries
    temporal_agg = cfg.temporal_agg
    if temporal_agg:
        query_frequency = 1
        num_queries = cfg.model.policy.num_queries
        all_time_actions = torch.zeros(
            [max_timesteps, max_timesteps + num_queries, cfg.model.policy.qpos_dim]
        ).to(device)
    step_time = 1.0 / cfg.frame_rate
    crop_info = cfg.data.train.get("crop_info")
    num_rollouts = cfg.num_rollouts

    for rollout_idx in range(num_rollouts):
        if cfg.save_video:
            videos = defaultdict(list)
            save_foler = os.path.join(cfg.paths.output_dir, "videos")
            os.makedirs(save_foler, exist_ok=True)

        robot.reset()
        log.info(f"Rollout: {rollout_idx} will start in 5 seconds")
        time.sleep(5)
        try:
            for t in range(max_timesteps):
                start_time = time.time()
                imgs = camera.get_state(
                    depth=cfg.data.train.get("include_depth") or use_pcd
                )
                if not use_pcd:
                    image_dict = dict()
                    for serial in cfg.data.train.camera_serials:
                        if not cfg.data.train.include_depth:
                            img = imgs[f"{serial}"].astype(float)[
                                crop_info[serial][0] : crop_info[serial][2],
                                crop_info[serial][1] : crop_info[serial][3],
                            ]
                            if cfg.save_video:
                                videos[f"{serial}_color"].append(img)
                            image_dict[serial] = img / 255.0
                        else:
                            img = imgs[f"{serial}"][0].astype(float)[
                                crop_info[serial][0] : crop_info[serial][2],
                                crop_info[serial][1] : crop_info[serial][3],
                            ]
                            depth = imgs[f"{serial}"][1].astype(float)[..., None][
                                crop_info[serial][0] : crop_info[serial][2],
                                crop_info[serial][1] : crop_info[serial][3],
                            ]
                            if cfg.save_video:
                                videos[f"{serial}_color"].append(img.astype(np.uint8))
                                videos[f"{serial}_depth"].append(
                                    (
                                        (
                                            (depth - depth.min())
                                            / (depth.max() - depth.min())
                                        )
                                        * 255
                                    ).astype(np.uint8)
                                )

                            image_dict[serial] = np.concatenate(
                                [img / 255.0, depth], axis=-1
                            )

                    all_cam_images = []
                    for serial in cfg.data.train.camera_serials:
                        all_cam_images.append(image_dict[serial])
                    img = torch.from_numpy(np.stack(all_cam_images)).float().to(device)
                    qpos = (
                        torch.from_numpy(pre_process(np.array(robot.read_position())))
                        .float()
                        .to(device)
                    )
                    img = rearrange(img, "k h w c -> 1 k c h w")
                    qpos = rearrange(qpos, "c -> 1 c")
                    input_dict = dict(image=img, qpos=qpos)
                else:
                    points, colors = [], []
                    for serial in cfg.data.train.camera_serials:
                        calib = calib_dict[serial]
                        color_image = imgs[f"{serial}"][0].astype(float)
                        depth_image = imgs[f"{serial}"][1].astype(float)
                        if cfg.save_video:
                            videos[f"{serial}_color"].append(
                                color_image.astype(np.uint8)
                            )
                            videos[f"{serial}_depth"].append(
                                (
                                    (
                                        (depth_image - depth_image.min())
                                        / (depth_image.max() - depth_image.min())
                                    )
                                    * 255
                                ).astype(np.uint8)[..., None]
                            )
                        _points, _colors = rgbd_to_pointcloud(
                            color_image,
                            depth_image,
                            calib["camera_matrix"],
                            calib["extrinsic_matrix"],
                            min_depth=0.5,
                            max_depth=1.0,
                            depth_scale=1.0,
                        )
                        mask = (
                            (_points[:, 0] > pcd_range[0][0])
                            & (_points[:, 0] < pcd_range[0][1])
                            & (_points[:, 1] > pcd_range[1][0])
                            & (_points[:, 1] < pcd_range[1][1])
                            & (_points[:, 2] > pcd_range[2][0])
                            & (_points[:, 2] < pcd_range[2][1])
                        )
                        points.append(_points[mask])
                        colors.append(_colors[mask])

                    points = np.concatenate(points, axis=0)
                    colors = np.concatenate(colors, axis=0)

                    pcds = [transform_pcd(dict(coord=points, color=colors))]
                    pcds = point_collate_fn(pcds)

                    for k in pcds:
                        pcds[k] = pcds[k].to(device)

                    qpos = (
                        torch.from_numpy(pre_process(np.array(robot.read_position())))
                        .float()
                        .to(device)
                    )
                    qpos = rearrange(qpos, "c -> 1 c")
                    input_dict = dict(pcds=pcds, qpos=qpos)

                if t % query_frequency == 0:
                    all_actions = policy(input_dict)["a_hat"]

                if temporal_agg:
                    all_time_actions[[t], t : t + num_queries] = all_actions
                    actions_for_curr_step = all_time_actions[:, t]
                    actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                    actions_for_curr_step = actions_for_curr_step[actions_populated]
                    k = 0.01
                    exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                    exp_weights = exp_weights / exp_weights.sum()
                    exp_weights = (
                        torch.from_numpy(exp_weights.astype(np.float32))
                        .unsqueeze(dim=1)
                        .to(device)
                    )
                    raw_action = (actions_for_curr_step * exp_weights).sum(
                        dim=0, keepdim=True
                    )
                else:
                    raw_action = all_actions[:, t % query_frequency]

                # post-process actions
                raw_action = raw_action.squeeze(0).cpu().numpy()
                action = post_process(raw_action)

                log.info(t, action)
                robot.set_goal_pos(action)

                torch.cuda.empty_cache()

                duration = time.time() - start_time
                if duration < step_time:
                    time.sleep(step_time - duration)

        except KeyboardInterrupt:
            log.info("KeyboardInterrupt")

        if cfg.save_video:
            for key, value in videos.items():
                if "depth" in key:
                    value = np.stack(value)
                    value = np.repeat(value, 3, axis=-1)
                else:
                    value = np.stack(value)
                iio.imwrite(
                    os.path.join(save_foler, f"{rollout_idx}_{key}.mp4"),
                    value,
                    fps=cfg.frame_rate,
                )
            log.info(f"Videos saved to {save_foler}")

        robot.reset()
        time.sleep(2)

    robot.disconnect()


@hydra.main(
    version_base="1.3",
    config_path="../configs",
    config_name="eval_low_cost_robot_act.yaml",
)
def main(cfg: DictConfig) -> None:
    """Main entry point for evaluation.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    OmegaConf.set_struct(cfg, False)
    extras(cfg)

    eval(cfg)


if __name__ == "__main__":
    main()
