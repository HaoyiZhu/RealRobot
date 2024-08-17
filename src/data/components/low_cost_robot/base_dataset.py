import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

import src.utils as U

log = U.RankedLogger(__name__, rank_zero_only=True)


class LowCostRobotSingleArmRGBDDataset(Dataset):
    def __init__(
        self,
        split="train",
        root="data/teleop/",
        task_name="reach_cube",
        user_name="default",
        leader_key="leader",
        follower_key="follower",
        camera_serials=["023322060111", "013422060628"],
        include_depth=False,  # `True` for RGB-D
        chunk_size=20,
        eps=list(range(40)),
        crop_info={
            "013422060628": [100, 600, 600, 1200],  # h1, w1, h2, w2
            "023322060111": [0, 600, 500, 1200],
        },
        loop=1,
        cahce_traj=True,
    ):
        self.split = split
        self.root = os.path.join(root, task_name, user_name)
        self.leader_key = leader_key
        self.follower_key = follower_key
        self.camera_serials = camera_serials
        self.include_depth = include_depth
        self.chunk_size = chunk_size
        self.eps = eps
        self.crop_info = crop_info
        self.loop = loop
        self.cahce_traj = cahce_traj

        self.cache = []
        if self.cahce_traj:
            for ep in tqdm(self.eps):
                traj = dict()
                meta = json.load(open(os.path.join(self.root, f"ep_{ep}", "meta.json")))
                meta["timestamps"].sort()
                traj["meta"] = meta
                for ts in meta["timestamps"]:
                    ts_data = np.load(
                        os.path.join(self.root, f"ep_{ep}", f"{ts}.npy"),
                        allow_pickle=True,
                    ).item()
                    all_keys = list(ts_data.keys())
                    for k in all_keys:
                        if "depth" in k and not self.include_depth:
                            ts_data.pop(k)
                        if ("color" in k or "depth" in k) and k.split("_")[
                            0
                        ] not in self.camera_serials:
                            ts_data.pop(k)
                    ts_data = self.normalize_timestamp_data(ts_data)
                    traj[ts] = ts_data
                self.cache.append(traj)

        if self.split == "train" and (
            not os.path.exists(os.path.join(self.root, "stats.npy"))
        ):
            self.compute_stats()

        self.stats = np.load(
            os.path.join(self.root, "stats.npy"), allow_pickle=True
        ).item()

    def normalize_timestamp_data(self, ts_data):
        for key in [self.leader_key, self.follower_key]:
            ts_data[key] = ts_data[key] / 2048.0 - 1.0
        return ts_data

    def get_qpos(self, traj, ts):
        return traj[ts][self.follower_key]

    def get_action(self, traj, ts):
        return traj[ts][self.leader_key]

    def __len__(self):
        return len(self.eps) * self.loop

    def __getitem__(self, idx):
        if self.cahce_traj:
            traj = self.cache[idx % len(self.cache)]
        else:
            traj = dict()
            ep = self.eps[idx % len(self.eps)]
            meta = json.load(open(os.path.join(self.root, f"ep_{ep}", "meta.json")))
            meta["timestamps"].sort()
            traj["meta"] = meta
            for ts in meta["timestamps"]:
                ts_data = np.load(
                    os.path.join(self.root, f"ep_{ep}", f"{ts}.npy"), allow_pickle=True
                ).item()
                ts_data = self.normalize_timestamp_data(ts_data)
                traj[ts] = ts_data
        episode_len = len(traj["meta"]["timestamps"])
        start_ts_idx = np.random.choice(episode_len)
        start_ts = traj["meta"]["timestamps"][start_ts_idx]
        qpos = self.get_qpos(traj, start_ts)
        image_dict = dict()
        for serial in self.camera_serials:
            image = (
                traj[start_ts][f"{serial}_color"].astype(float)[
                    self.crop_info[serial][0] : self.crop_info[serial][2],
                    self.crop_info[serial][1] : self.crop_info[serial][3],
                ]
                / 255.0
            )
            if self.include_depth:
                depth = traj[start_ts][f"{serial}_depth"].astype(float)[..., None][
                    self.crop_info[serial][0] : self.crop_info[serial][2],
                    self.crop_info[serial][1] : self.crop_info[serial][3],
                ]
                image = np.concatenate([image, depth], axis=-1)
            image_dict[serial] = image

        chunk_ts = traj["meta"]["timestamps"][
            start_ts_idx : start_ts_idx + self.chunk_size
        ]
        action = []
        for ts in chunk_ts:
            action.append(self.get_action(traj, ts))
        action = np.array(action)
        action_len = action.shape[0]
        padded_action = np.zeros((self.chunk_size, action.shape[1]), dtype=np.float32)

        padded_action[:action_len] = action
        is_pad = np.zeros(self.chunk_size)
        is_pad[action_len:] = 1

        # new axis for different cameras
        all_cam_images = []
        for serial in self.camera_serials:
            all_cam_images.append(image_dict[serial])

        all_cam_images = np.stack(all_cam_images, axis=0)

        qpos = (qpos - self.stats["qpos_mean"]) / self.stats["qpos_std"]
        padded_action = (padded_action - self.stats["action_mean"]) / self.stats[
            "action_std"
        ]

        # construct observations
        image_data = torch.from_numpy(all_cam_images).float()
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # channel last
        image_data = torch.einsum("k h w c -> k c h w", image_data)

        # image_data[:, :3] = image_data[:, :3] / 255.0
        # image_data[:, 3:] = image_data[:, 3:] / 1000.

        data_dict = dict(
            image=image_data,
            qpos=qpos_data,
            actions=action_data,
            is_pad=is_pad,
        )

        return data_dict

    def compute_stats(self):
        log.info("Computing stats for the dataset ...")
        all_qpos_data = []
        all_action_data = []
        for traj in tqdm(self.cache):
            for ts in traj["meta"]["timestamps"]:
                all_qpos_data.append(self.get_qpos(traj, ts))
                all_action_data.append(self.get_action(traj, ts))
        all_qpos_data = np.array(all_qpos_data)
        all_action_data = np.array(all_action_data)
        stats = dict()
        stats["qpos_mean"] = np.mean(all_qpos_data, axis=0)
        stats["qpos_std"] = np.std(all_qpos_data, axis=0)
        stats["action_mean"] = np.mean(all_action_data, axis=0)
        stats["action_std"] = np.std(all_action_data, axis=0)

        stats["qpos_min"] = np.min(all_qpos_data, axis=0)
        stats["qpos_max"] = np.max(all_qpos_data, axis=0)
        stats["action_min"] = np.min(all_action_data, axis=0)
        stats["action_max"] = np.max(all_action_data, axis=0)

        np.save(os.path.join(self.root, "stats.npy"), stats)


class LowCostRobotDualArmRGBDDataset(LowCostRobotSingleArmRGBDDataset):
    def __init__(
        self,
        split="train",
        root="data/teleop/",
        task_name="reach_cube",
        user_name="default",
        left_leader_key="left_leader",
        left_follower_key="left_follower",
        right_leader_key="right_leader",
        right_follower_key="right_follower",
        camera_serials=["023322060111", "013422060628"],
        include_depth=False,  # `True` for RGB-D
        chunk_size=20,
        eps=list(range(40)),
        crop_info={
            "013422060628": [100, 400, 900, 1200],  # h1, w1, h2, w2
            "023322060111": [0, 400, 800, 1200],
        },
        loop=1,
        cahce_traj=True,
    ):
        self.left_leader_key = left_leader_key
        self.left_follower_key = left_follower_key
        self.right_leader_key = right_leader_key
        self.right_follower_key = right_follower_key

        super().__init__(
            split=split,
            root=root,
            task_name=task_name,
            user_name=user_name,
            leader_key=None,
            follower_key=None,
            camera_serials=camera_serials,
            include_depth=include_depth,
            chunk_size=chunk_size,
            eps=eps,
            crop_info=crop_info,
            loop=loop,
            cahce_traj=cahce_traj,
        )

    def normalize_timestamp_data(self, ts_data):
        for key in [
            self.left_leader_key,
            self.left_follower_key,
            self.right_leader_key,
            self.right_follower_key,
        ]:
            ts_data[key] = ts_data[key] / 2048.0 - 1.0
        return ts_data

    def get_qpos(self, traj, ts):
        return np.concatenate(
            [traj[ts][self.left_follower_key], traj[ts][self.right_follower_key]],
            axis=-1,
        )

    def get_action(self, traj, ts):
        return np.concatenate(
            [traj[ts][self.left_leader_key], traj[ts][self.right_leader_key]], axis=-1
        )
