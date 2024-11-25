"""
Low cost robot tele operation.

Author: Haoyi Zhu.
"""

import threading
import time

from src.utils import RankedLogger

from .low_cost_robot import LowCostRobot


class SingleArmTeleOperator:
    def __init__(
        self,
        leader: LowCostRobot,
        follower: LowCostRobot,
        frame_rate: int = 30,
        **kwargs,
    ):
        self.logger = RankedLogger(__name__, rank_zero_only=True)

        self.frame_rate = frame_rate
        self.is_teleop = False
        self.leader = leader
        self.leader.connect()
        self.follower = follower
        self.follower.connect()
        time.sleep(1)

    def start(self):
        self.leader.start(is_leader=True)
        self.follower.start()

        self.start_streaming()

    def start_streaming(self, delay_time: float = 0.0):
        self.thread = threading.Thread(target=self.teleop_thread, args=(delay_time,))
        self.thread.setDaemon(True)
        self.thread.start()

    def teleop_thread(self, delay_time: float = 0.0):
        time.sleep(delay_time)
        self.is_teleop = True
        self.logger.info("Start tele-operation ...")
        while self.is_teleop:
            state = self.leader.cur_state
            self.follower.action(list(state))

    def stop(self):
        self.stop_streaming(permanent=True)

    def stop_streaming(self, permanent: bool = True):
        self.is_teleop = False
        if self.thread:
            self.thread.join()
        self.logger.info("Stop tele-operation.")
        self.leader.stop()
        self.follower.stop()


class DualArmTeleOperator:
    def __init__(
        self, op_left: SingleArmTeleOperator, op_right: SingleArmTeleOperator, **kwargs
    ):
        self.op_left = op_left
        self.op_right = op_right

    def start(self):
        self.op_left.start()
        self.op_right.start()

    def start_streaming(self, delay_time: float = 0.0):
        self.op_left.start_streaming(delay_time)
        self.op_right.start_streaming(delay_time)

    def stop(self):
        self.op_left.stop()
        self.op_right.stop()

    def stop_streaming(self, permanent: bool = True):
        self.op_left.stop_streaming(permanent)
        self.op_right.stop_streaming(permanent)
