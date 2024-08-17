"""
Robot Base Interface.

Author: Haoyi Zhu.
"""

import threading
import time

import numpy as np

from src.utils import RankedLogger, SharedMemoryManager


class RobotBase:
    def __init__(
        self,
        gripper=None,
        shm_name: str = None,
        frame_rate: int = 30,
        **kwargs,
    ):
        """
        Initializes the RobotBase class.

        Args:
            gripper: Optional gripper object.
            shm_name (str): Name for the shared memory.
            frame_rate (int): Frame rate for streaming data.
            **kwargs: Additional arguments.
        """
        self.logger = RankedLogger(__name__, rank_zero_only=True)
        self.gripper = gripper
        self.frame_rate = frame_rate
        self.shm_name = shm_name
        self.is_streaming = False
        self.stream_thread = None
        self.use_shared_memory = shm_name is not None

    def _setup_shared_memory(self):
        state = np.array(self.get_state()).astype(np.float64)
        self.shm_manager = SharedMemoryManager(
            self.shm_name, 0, state.shape, state.dtype
        )
        self.shm_manager.execute(state)

    def connect(self):
        pass

    def disconnect(self):
        pass

    def start(self):
        """Starts the robot."""
        self.connect()
        if self.use_shared_memory:
            self._setup_shared_memory()
            self.start_streaming()

    def start_streaming(self, delay: float = 0.0):
        """
        Starts the data streaming in a separate thread.

        Args:
            delay (float): Delay before starting the stream.
        """
        if not self.use_shared_memory:
            raise ValueError("Shared memory name must be set to use streaming.")

        self.stream_thread = threading.Thread(target=self._stream, args=(delay,))
        self.stream_thread.setDaemon(True)
        self.stream_thread.start()

        if self.gripper is not None:
            self.gripper.start_streaming(delay)

    def _stream(self, delay: float):
        time.sleep(delay)
        self.is_streaming = True
        self.logger.info("Streaming started.")
        while self.is_streaming:
            start_time = time.time()
            state = np.array(self.get_state()).astype(np.float64)
            self.shm_manager.execute(state)

            elapsed_time = time.time() - start_time
            sleep_time = max(0, 1 / self.frame_rate - elapsed_time)
            time.sleep(sleep_time)

    def stop_streaming(self, permanent: bool = True):
        """Stops the data streaming."""
        self.is_streaming = False
        if self.stream_thread:
            self.stream_thread.join()
        self.logger.info("Streaming stopped.")
        if permanent and self.use_shared_memory:
            self._shutdown_shared_memory()
            self.use_shared_memory = False

        if self.gripper is not None:
            self.gripper.stop_streaming(permanent)

    def stop(self):
        """Stops the robot."""
        self.disconnect()
        if self.use_shared_memory:
            self.stop_streaming()

    def _shutdown_shared_memory(self):
        if self.use_shared_memory:
            self.shm_manager.close()

    def get_state(self):
        """Returns the current state of the robot."""
        raise NotImplementedError

    def reset(self, *args, **kwargs):
        """Resets the robot."""
        raise NotImplementedError

    def action(self, *args, **kwargs):
        """Performs an action with the robot."""
        raise NotImplementedError
