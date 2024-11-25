import threading
from enum import Enum, auto
from typing import Dict, List, Union

import numpy as np
from dynamixel_sdk import (
    DXL_HIBYTE,
    DXL_HIWORD,
    DXL_LOBYTE,
    DXL_LOWORD,
    GroupSyncRead,
    GroupSyncWrite,
)

from ..base import RobotBase
from .dynamixel import DynamixelConfig, OperatingMode, ReadAttribute


class MotorControlType(Enum):
    PWM = auto()
    POSITION_CONTROL = auto()
    DISABLED = auto()
    UNKNOWN = auto()


class LowCostRobot(RobotBase):
    def __init__(
        self,
        shm_name: str = None,
        frame_rate: int = 30,
        device_name: str = None,
        baudrate: int = 1_000_000,
        servo_ids: List[int] = [1, 2, 3, 4, 5, 6],
        use_pwm_reader: bool = False,
        **kwargs,
    ):
        """
        Initializes the LowCostRobot class.

        Args:
            shm_name (str): Name for the shared memory.
            frame_rate (int): Frame rate for streaming data.
            device_name (str): Name of the device.
            baudrate (int): Baudrate for the device.
            servo_ids (list): List of servo IDs.
            **kwargs: Additional arguments.
        """
        super().__init__(shm_name=shm_name, frame_rate=frame_rate, **kwargs)
        self.servo_ids = servo_ids
        self.device_name = device_name
        self.baudrate = baudrate
        self.use_pwm_reader = use_pwm_reader

        self.dynamixel = None
        self.position_reader = None
        self.velocity_reader = None
        self.pwm_reader = None
        self.pos_writer = None
        self.pwm_writer = None
        self.control_mode = MotorControlType.DISABLED  # Current control mode

    @property
    def name(self):
        return self.shm_name

    def connect(self):
        """Connects to the Dynamixel device."""
        self.dynamixel = DynamixelConfig(
            baudrate=self.baudrate, device_name=self.device_name
        ).instantiate()
        self._init_motors()
        self.reset()

    def start(self, is_leader=False):
        if self.use_shared_memory:
            self._setup_shared_memory()
            self.start_streaming()

        if is_leader:
            self._disable_torque()
            self.set_trigger_torque()

        self.is_read_state = True
        self.read_state_thread = threading.Thread(target=self._read_state, args=())
        self.read_state_thread.setDaemon(True)
        self.read_state_thread.start()

    def _init_motors(self):
        self.position_reader = GroupSyncRead(
            self.dynamixel.portHandler,
            self.dynamixel.packetHandler,
            ReadAttribute.POSITION.value,
            4,
        )
        for id in self.servo_ids:
            self.position_reader.addParam(id)

        self.velocity_reader = GroupSyncRead(
            self.dynamixel.portHandler,
            self.dynamixel.packetHandler,
            ReadAttribute.VELOCITY.value,
            4,
        )
        for id in self.servo_ids:
            self.velocity_reader.addParam(id)

        if self.use_pwm_reader:
            self.pwm_reader = GroupSyncRead(
                self.dynamixel.portHandler,
                self.dynamixel.packetHandler,
                ReadAttribute.PWM.value,
                2,
            )
            for id in self.servo_ids:
                self.pwm_reader.addParam(id)

        self.pos_writer = GroupSyncWrite(
            self.dynamixel.portHandler,
            self.dynamixel.packetHandler,
            self.dynamixel.ADDR_GOAL_POSITION,
            4,
        )
        for id in self.servo_ids:
            self.pos_writer.addParam(id, [2048])

        self.pwm_writer = GroupSyncWrite(
            self.dynamixel.portHandler,
            self.dynamixel.packetHandler,
            self.dynamixel.ADDR_GOAL_PWM,
            2,
        )
        for id in self.servo_ids:
            self.pwm_writer.addParam(id, [2048])

        self.cur_state = self._get_state()
        self._disable_torque()
        self.motor_control_state = MotorControlType.DISABLED

    def disconnect(self):
        """Disconnects from the Dynamixel device."""
        self.is_read_state = False
        if hasattr(self, "read_state_thread"):
            self.read_state_thread.join()
        self._disable_torque()
        self.dynamixel.disconnect()

    def read_position(self, tries=2) -> List[int]:
        """
        Reads the joint positions of the robot. 2048 is the center position. 0 and 4096 are 180 degrees in each direction.

        Args:
            tries (int): Maximum number of tries to read the position.

        Returns:
            list: List of joint positions in range [0, 4096].
        """
        # result = 1
        result = self.position_reader.txRxPacket()
        # while result != 0:
        #     result = self.position_reader.txRxPacket()

        if result != 0:
            if tries > 0:
                return self.read_position(tries=tries - 1)
            else:
                # self.logger.error("Failed to read position")
                # breakpoint()
                # return []
                return self.cur_state
        # self.logger.info("Succeed to read position")
        positions = []
        for id in self.servo_ids:
            position = self.position_reader.getData(id, ReadAttribute.POSITION.value, 4)
            if position > 2**31:
                position -= 2**32
            positions.append(position)
        return positions

    def read_velocity(self) -> List[int]:
        """
        Reads the joint velocities of the robot.

        Returns:
            list: List of joint velocities.
        """
        self.velocity_reader.txRxPacket()
        velocities = []
        for id in self.servo_ids:
            velocity = self.velocity_reader.getData(id, ReadAttribute.VELOCITY.value, 4)
            if velocity > 2**31:
                velocity -= 2**32
            velocities.append(velocity)
        return velocities

    def read_pwm(self) -> List[int]:
        self.pwm_reader.txRxPacket()
        pwms = []
        for id in self.servo_ids:
            pwm = self.pwm_reader.getData(id, ReadAttribute.PWM.value, 2)
            pwms.append(pwm)
        return pwms

    def set_goal_pos(self, action: Union[List, np.ndarray]):
        """
        Sets the goal positions for the servos.

        Args:
            action (Union[List, np.ndarray]): List or numpy array of target joint positions in range [0, 4096].
        """
        if self.motor_control_state is not MotorControlType.POSITION_CONTROL:
            self._set_position_control()
        for i, motor_id in enumerate(self.servo_ids):
            data_write = [
                DXL_LOBYTE(DXL_LOWORD(action[i])),
                DXL_HIBYTE(DXL_LOWORD(action[i])),
                DXL_LOBYTE(DXL_HIWORD(action[i])),
                DXL_HIBYTE(DXL_HIWORD(action[i])),
            ]
            self.pos_writer.changeParam(motor_id, data_write)
        self.pos_writer.txPacket()

    def set_pwm(self, action: Union[List, np.ndarray]):
        """
        Sets the PWM values for the servos.

        Args:
            action (Union[List, np.ndarray]): List or numpy array of PWM values in range [0, 885].
        """
        if self.motor_control_state is not MotorControlType.PWM:
            self._set_pwm_control()
        for i, motor_id in enumerate(self.servo_ids):
            data_write = [
                DXL_LOBYTE(DXL_LOWORD(action[i])),
                DXL_HIBYTE(DXL_LOWORD(action[i])),
            ]
            self.pwm_writer.changeParam(motor_id, data_write)
        self.pwm_writer.txPacket()

    def set_trigger_torque(self):
        """
        Sets a constant torque torque for the last servo in the chain.
        This is useful for the trigger of the leader arm.
        """
        self.dynamixel._enable_torque(self.servo_ids[-1])
        self.dynamixel.set_pwm_value(self.servo_ids[-1], 200)

    def limit_pwm(self, limit: Union[int, list, np.ndarray]):
        """
        Limits the PWM values for the servos in position control mode.

        Args:
            limit (Union[int, list, np.ndarray]): PWM limit (0 ~ 885).
        """
        if isinstance(limit, int):
            limits = [limit] * len(self.servo_ids)
        else:
            limits = limit
        self._disable_torque()
        for motor_id, limit_value in zip(self.servo_ids, limits):
            self.dynamixel.set_pwm_limit(motor_id, limit_value)
        self._enable_torque()

    def _set_pwm_control(self):
        self._disable_torque()
        for motor_id in self.servo_ids:
            self.dynamixel.set_operating_mode(motor_id, OperatingMode.PWM)
        self._enable_torque()
        self.motor_control_state = MotorControlType.PWM

    def _set_position_control(self):
        self._disable_torque()
        for motor_id in self.servo_ids:
            self.dynamixel.set_operating_mode(motor_id, OperatingMode.POSITION)
        self._enable_torque()
        self.motor_control_state = MotorControlType.POSITION_CONTROL

    def _disable_torque(self):
        for motor_id in self.servo_ids:
            self.dynamixel._disable_torque(motor_id)

    def _enable_torque(self):
        for motor_id in self.servo_ids:
            self.dynamixel._enable_torque(motor_id)

    def action(
        self,
        action: Union[List, np.ndarray],
        control_mode: MotorControlType = MotorControlType.POSITION_CONTROL,
    ):
        """
        Executes an action with the robot.

        Args:
            action (Union[List, np.ndarray]): List or numpy array of target values.
            control_mode (MotorControlType, optional): Control mode for the action. Defaults to current mode.
        """

        if control_mode == MotorControlType.POSITION_CONTROL:
            self.set_goal_pos(action)
        elif control_mode == MotorControlType.PWM:
            self.set_pwm(action)
        else:
            raise ValueError(f"Unsupported control mode: {control_mode}")

    def reset(self):
        """
        Resets all motors of the robot.
        """
        self.set_goal_pos([2048, 2048, 2048, 1024, 4096, 2048])

    def _get_state(self, include_pwm=False, include_velocity=False) -> Dict:
        """
        Returns the current state of the robot.

        Args:
            include_pwm (bool): Whether to include PWM values in the state.

        Returns:
            dict: Dictionary containing joint positions, velocities, and optionally PWM values.
        """
        state = np.array(self.read_position())
        assert state.shape[0] == len(
            self.servo_ids
        ), f"Invalid state shape: {state.shape}"

        if include_velocity:
            velocities = np.array(self.read_velocity())
            state = np.concatenate([state, velocities])

        if include_pwm:
            if not self.use_pwm_reader:
                self.logger.warning("PWM reader is not enabled.")
            else:
                pwms = self.read_pwm()
                state = np.concatenate([state, pwms])

        self.cur_state = state
        return state

    def _read_state(self):
        while self.is_read_state:
            self.cur_state = self._get_state()

    def get_state(self):
        return self.cur_state


class DualArmLowCostRobot:
    def __init__(self, robot1: LowCostRobot, robot2: LowCostRobot):
        self.robot1 = robot1
        self.robot2 = robot2

    def connect(self):
        self.robot1.connect()
        self.robot2.connect()

    def start(self):
        self.robot1.start()
        self.robot2.start()

    def disconnect(self):
        self.robot1.disconnect()
        self.robot2.disconnect()

    def stop(self):
        self.robot1.stop()
        self.robot2.stop()

    def get_state(self):
        return np.concatenate([self.robot1.get_state(), self.robot2.get_state()])

    def action(self, action):
        left_action = action[:6]
        right_action = action[6:]
        self.robot1.action(left_action)
        self.robot2.action(right_action)

    def reset(self):
        self.robot1.reset()
        self.robot2.reset()

    def read_position(self):
        return self.robot1.read_position() + self.robot2.read_position()

    def set_goal_pos(self, action):
        left_action = action[:6]
        right_action = action[6:]
        self.robot1.set_goal_pos(left_action)
        self.robot2.set_goal_pos(right_action)


# Example usage
if __name__ == "__main__":
    robot = LowCostRobot(device_name="/dev/ttyUSB0")
    robot.connect()
    print(robot.get_state(include_pwm=True))
    robot.disconnect()
