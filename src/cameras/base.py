"""
Camera Base Interface.

Author: Haoyi Zhu.
"""

import threading
import time

import numpy as np

from src.utils import RankedLogger, SharedMemoryManager


class CameraBase:
    def __init__(
        self,
        shm_name_rgb: str = None,
        shm_name_depth: str = None,
        frame_rate: int = 30,
        **kwargs,
    ):
        self.logger = RankedLogger(__name__, rank_zero_only=True)
        self.frame_rate = frame_rate
        self.shm_name_rgb = shm_name_rgb
        self.shm_name_depth = shm_name_depth
        self.is_streaming = False
        self.stream_thread = None
        self.use_shared_memory = shm_name_rgb is not None or shm_name_depth is not None
        if self.use_shared_memory:
            self._setup_shared_memory()

    @property
    def name(self):
        return self.shm_name_rgb

    def _setup_shared_memory(self):
        rgb, depth = self.get_state()
        if rgb is not None:
            info = np.array(rgb).astype(np.uint8)
            self.shm_manager_rgb = SharedMemoryManager(
                self.shm_name_rgb, 0, info.shape, info.dtype
            )
            self.shm_manager_rgb.execute(info)
        if depth is not None:
            info = np.array(depth).astype(float)
            self.shm_manager_depth = SharedMemoryManager(
                self.shm_name_depth, 0, info.shape, info.dtype
            )
            self.shm_manager_depth.execute(info)

    def start(self):
        self.start_streaming()

    def start_streaming(self, delay: float = 0.0):
        if not self.use_shared_memory:
            raise ValueError("Shared memory name must be set to use streaming.")

        self.stream_thread = threading.Thread(target=self._stream, args=(delay,))
        self.stream_thread.setDaemon(True)
        self.stream_thread.start()

    def _stream(self, delay: float):
        time.sleep(delay)
        self.is_streaming = True
        self.logger.info("Streaming started.")
        while self.is_streaming:
            start_time = time.time()
            rgb, depth = self.get_state()
            if rgb is not None:
                rgb = rgb.astype(np.uint8)
                self.shm_manager_rgb.execute(rgb)
            if depth is not None:
                depth = depth.astype(float)
                self.shm_manager_depth.execute(depth)
            elapsed_time = time.time() - start_time
            sleep_time = max(0, 1 / self.frame_rate - elapsed_time)
            time.sleep(sleep_time)

    def stop_streaming(self, permanent: bool = True):
        self.is_streaming = False
        if self.stream_thread:
            self.stream_thread.join()
        self.logger.info("Streaming stopped.")
        if permanent and self.use_shared_memory:
            self._shutdown_shared_memory()
            self.use_shared_memory = False

    def stop(self):
        self.stop_streaming(permanent=True)

    def _shutdown_shared_memory(self):
        if self.use_shared_memory:
            if hasattr(self, "shm_manager_rgb"):
                self.shm_manager_rgb.close()
            if hasattr(self, "shm_manager_depth"):
                self.shm_manager_depth.close()

    def get_state(self):
        raise NotImplementedError

    def calibrate(self, *args, **kwargs):
        raise NotImplementedError


class MultiCameraBase:
    def __init__(self, cameras, **kwargs):
        self.cameras = cameras

    def __len__(self):
        return len(self.cameras)

    def start(self):
        for cam in self.cameras:
            cam.start()

    def start_streaming(self, delay: float = 0.0):
        for cam in self.cameras:
            cam.start_streaming(delay=delay)

    def stop_streaming(self, permanent: bool = True):
        for cam in self.cameras:
            cam.stop_streaming(permanent=permanent)

    def stop(self):
        for cam in self.cameras:
            cam.stop()

    def calibrate(self, *args, **kwargs):
        results = []
        for cam in self.cameras:
            results.append(cam.calibrate(*args, **kwargs))
        return results

    def get_state(self, *args, **kwargs):
        return {cam.name: cam.get_state(*args, **kwargs) for cam in self.cameras}
