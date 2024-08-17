import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

import src.utils as U
from src.data.components.transformpcd import ComposePCD
from src.utils.pointcloud_utils import rgbd_to_pointcloud

log = U.RankedLogger(__name__, rank_zero_only=True)


class LowCostRobotSingleArmPCDDataset(Dataset):
    def __init__(
        self,
        calib_file,
        transform_pcd,
        split="train",
        root="data/teleop/",
        task_name="reach_cube",
        user_name="default",
        leader_key="leader",
        follower_key="follower",
        camera_serials=["023322060111", "013422060628"],
        chunk_size=20,
        eps=list(range(40)),
        pcd_range=[
            [-0.15, 0.27],  # xmin, xmax
            [-0.5, 0.1],  # ymin, ymax
            [-0.5, -0.0075],  # zmin, zmax
        ],
        random_range_factor=[0.01, 0.01, 0.005],
        loop=1,
        cahce_traj=True,
    ):
        self.calib_dict = np.load(calib_file, allow_pickle=True).item()
        self.split = split
        self.root = os.path.join(root, task_name, user_name)
        self.leader_key = leader_key
        self.follower_key = follower_key
        self.camera_serials = camera_serials
        self.chunk_size = chunk_size
        self.eps = eps
        self.pcd_range = np.array(pcd_range)
        self.random_range_factor = np.array(random_range_factor).reshape(3, 1)
        self.loop = loop
        self.cahce_traj = cahce_traj

        self.transform_pcd = ComposePCD(transform_pcd)
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
        return len(self.cache) * self.loop

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

        pcd_range = (
            np.array(self.pcd_range).reshape(3, 2)
            + np.random.uniform(-1, 1, (3, 2)) * self.random_range_factor
        )

        points, colors = [], []
        for serial in self.camera_serials:
            calib = self.calib_dict[serial]
            color_image = traj[start_ts][f"{serial}_color"].astype(float)
            depth_image = traj[start_ts][f"{serial}_depth"].astype(float)
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
        pcds = [self.transform_pcd(dict(coord=points, color=colors))]

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

        qpos = (qpos - self.stats["qpos_mean"]) / self.stats["qpos_std"]
        padded_action = (padded_action - self.stats["action_mean"]) / self.stats[
            "action_std"
        ]

        # construct observations
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        data_dict = dict(
            pcds=pcds,
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


class LowCostRobotDualArmPCDDataset(LowCostRobotSingleArmPCDDataset):
    def __init__(
        self,
        calib_file,
        transform_pcd,
        split="train",
        root="data/teleop/",
        task_name="reach_cube",
        user_name="default",
        left_leader_key="left_leader",
        left_follower_key="left_follower",
        right_leader_key="right_leader",
        right_follower_key="right_follower",
        camera_serials=["023322060111", "013422060628"],
        chunk_size=20,
        eps=list(range(40)),
        pcd_range=[
            [-0.15, 0.35],  # xmin, xmax
            [-0.5, 0.3],  # ymin, ymax
            [-0.5, 0.005],  # zmin, zmax
        ],
        random_range_factor=[0.01, 0.01, 0.005],
        loop=1,
    ):
        self.left_leader_key = left_leader_key
        self.left_follower_key = left_follower_key
        self.right_leader_key = right_leader_key
        self.right_follower_key = right_follower_key

        super().__init__(
            calib_file=calib_file,
            transform_pcd=transform_pcd,
            split=split,
            root=root,
            task_name=task_name,
            user_name=user_name,
            leader_key=None,
            follower_key=None,
            camera_serials=camera_serials,
            chunk_size=chunk_size,
            eps=eps,
            pcd_range=pcd_range,
            random_range_factor=random_range_factor,
            loop=loop,
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
