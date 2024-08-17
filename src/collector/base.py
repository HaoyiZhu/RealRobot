import json
import os
import threading
import time

import numpy as np
from omegaconf import OmegaConf

from src.utils import RankedLogger, SharedMemoryManager


class CollectorBase:
    def __init__(self, data_path, shm_manager_cfg, **meta):
        self.data_path = data_path
        os.makedirs(self.data_path, exist_ok=True)

        self.logger = RankedLogger(__name__, rank_zero_only=True)
        self.shm_manager_cfg = shm_manager_cfg

        self.meta = {
            "start_time": None,
            "stop_time": None,
            "timestamps": [],
            "shm_manager_cfg": OmegaConf.to_container(shm_manager_cfg, resolve=True),
            **meta,
        }

        self.is_collecting = False
        self.collection_thread = None

    def _init_shm_managers(self, cfgs):
        managers = dict()
        for cfg in cfgs:
            names = cfg["names"]
            shapes = cfg["shapes"]
            dtypes = cfg["dtypes"]
            assert len(names) == len(shapes) == len(dtypes)
            for name, shape, dtype in zip(names, shapes, dtypes):
                managers[name] = SharedMemoryManager(name, 1, shape, dtype)
        return managers

    def receive(self):
        """
        Receive data from shared memory.
        """
        data = {"time": int(time.time() * 1000)}
        for shm_name, manager in self.shm_managers.items():
            data[shm_name] = manager.execute()
        return data

    def save(self):
        """
        Save collected data to the data path, return the timestamp at collection.
        """
        data = self.receive()
        timestamp = data["time"]
        np.save(
            os.path.join(self.data_path, f"{timestamp}.npy"), data, allow_pickle=True
        )
        return timestamp

    def start(self, delay_time=0.0):
        """
        Start collecting data.

        Parameters:
        - delay_time: float, optional, default: 0.0, the delay time before collecting data.
        """
        self.shm_managers = self._init_shm_managers(self.shm_manager_cfg)

        self.collection_thread = threading.Thread(
            target=self._collecting_thread, args=(delay_time,)
        )
        self.collection_thread.setDaemon(True)
        self.collection_thread.start()
        self.meta["start_time"] = int(time.time() * 1000)

    def _collecting_thread(self, delay_time):
        time.sleep(delay_time)
        self.is_collecting = True
        self.logger.info("Start collecting data ...")
        while self.is_collecting:
            timestamp = self.save()
            self.meta["timestamps"].append(timestamp)

    def stop(self):
        """
        Stop collecting data.
        """
        self.meta["stop_time"] = int(time.time() * 1000)
        self.is_collecting = False
        if self.collection_thread:
            self.collection_thread.join()

        self.logger.info("Stop collecting data.")
        self.save_meta()

    def save_meta(self):
        """
        Save metadata.
        """
        meta_path = os.path.join(self.data_path, "meta.json")

        with open(meta_path, "w") as f:
            json.dump(self.meta, f)
