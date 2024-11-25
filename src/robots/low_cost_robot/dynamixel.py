from __future__ import annotations

import enum
import math
import os
import time
from dataclasses import dataclass

from dynamixel_sdk import Config, PortHandler, PacketHandler, COMM_SUCCESS


class ReadAttribute(enum.Enum):
    TEMPERATURE = 146
    VOLTAGE = 145
    VELOCITY = 128
    POSITION = 132
    CURRENT = 126
    PWM = 124
    HARDWARE_ERROR_STATUS = 70
    HOMING_OFFSET = 20
    BAUDRATE = 8


class OperatingMode(enum.Enum):
    VELOCITY = 1
    POSITION = 3
    CURRENT_CONTROLLED_POSITION = 5
    PWM = 16
    UNKNOWN = -1


@dataclass
class DynamixelConfig:
    """Configuration dataclass for Dynamixel."""

    baudrate: int = 57600
    protocol_version: float = 2.0
    device_name: str = ""  # Example: '/dev/tty.usbserial-1120'
    dynamixel_id: int = 1

    def instantiate(self) -> Dynamixel:
        """Instantiate the Dynamixel class with the current configuration."""
        return Dynamixel(self)


class Dynamixel:
    ADDR_TORQUE_ENABLE = 64
    ADDR_GOAL_POSITION = 116
    ADDR_VELOCITY_LIMIT = 44
    ADDR_GOAL_PWM = 100
    OPERATING_MODE_ADDR = 11
    POSITION_I = 82
    POSITION_P = 84
    ADDR_ID = 7

    def __init__(self, config: Config):
        self.config = config
        self.connect()

    def connect(self):
        """Connect to the Dynamixel device."""
        if self.config.device_name == "":
            for port_name in os.listdir("/dev"):
                if "ttyUSB" in port_name or "ttyACM" in port_name:
                    self.config.device_name = "/dev/" + port_name
                    print(f"using device {self.config.device_name}")
        self.portHandler = PortHandler(self.config.device_name)
        # self.portHandler.LA
        self.packetHandler = PacketHandler(self.config.protocol_version)
        if not self.portHandler.openPort():
            raise Exception(f"Failed to open port {self.config.device_name}")

        if not self.portHandler.setBaudRate(self.config.baudrate):
            raise Exception(f"failed to set baudrate to {self.config.baudrate}")

        # self.operating_mode = OperatingMode.UNKNOWN
        # self.torque_enabled = False
        # self._disable_torque()

        self.operating_modes = [None for _ in range(32)]
        self.torque_enabled = [None for _ in range(32)]
        return True

    def disconnect(self):
        """Disconnect from the Dynamixel device."""
        self.portHandler.closePort()

    def set_goal_position(self, motor_id, goal_position):
        """Set the goal position of the motor."""
        dxl_comm_result, dxl_error = self.packetHandler.write4ByteTxRx(
            self.portHandler, motor_id, self.ADDR_GOAL_POSITION, goal_position
        )

    def set_pwm_value(self, motor_id: int, pwm_value, tries=3):
        """Set the PWM value of the motor."""
        if self.operating_modes[motor_id] is not OperatingMode.PWM:
            self._disable_torque(motor_id)
            self.set_operating_mode(motor_id, OperatingMode.PWM)

        if not self.torque_enabled[motor_id]:
            self._enable_torque(motor_id)

        dxl_comm_result, dxl_error = self.packetHandler.write2ByteTxRx(
            self.portHandler, motor_id, self.ADDR_GOAL_PWM, pwm_value
        )

        if dxl_comm_result != COMM_SUCCESS:
            if tries <= 1:
                raise ConnectionError(
                    f"dxl_comm_result: {self.packetHandler.getTxRxResult(dxl_comm_result)}"
                )
            else:
                print(
                    f"dynamixel pwm setting failure trying again with {tries - 1} tries"
                )
                self.set_pwm_value(motor_id, pwm_value, tries=tries - 1)
        elif dxl_error != 0:
            print(f"dxl error {dxl_error}")
            raise ConnectionError(
                f"dynamixel error: {self.packetHandler.getTxRxResult(dxl_error)}"
            )

    def read_temperature(self, motor_id: int):
        """Read the temperature of the motor."""
        return self._read_value(motor_id, ReadAttribute.TEMPERATURE, 1)

    def read_velocity(self, motor_id: int):
        """Read the velocity of the motor."""
        pos = self._read_value(motor_id, ReadAttribute.VELOCITY, 4)
        if pos > 2**31:
            pos -= 2**32
        return pos

    def read_position(self, motor_id: int):
        """Read the position of the motor."""
        pos = self._read_value(motor_id, ReadAttribute.POSITION, 4)
        if pos > 2**31:
            pos -= 2**32
        return pos

    def read_position_degrees(self, motor_id: int) -> float:
        """Read the position of the motor in degrees."""
        return (self.read_position(motor_id) / 4096) * 360

    def read_position_radians(self, motor_id: int) -> float:
        """Read the position of the motor in radians."""
        return (self.read_position(motor_id) / 4096) * 2 * math.pi

    def read_current(self, motor_id: int):
        """Read the current of the motor."""
        current = self._read_value(motor_id, ReadAttribute.CURRENT, 2)
        if current > 2**15:
            current -= 2**16
        return current

    def read_pwm(self, motor_id: int) -> int:
        """Read the present PWM value of the motor."""
        return self._read_value(motor_id, ReadAttribute.PWM, 2)

    def read_hardware_error_status(self, motor_id: int):
        """Read the hardware error status of the motor."""
        return self._read_value(motor_id, ReadAttribute.HARDWARE_ERROR_STATUS, 1)

    def set_id(self, old_id, new_id, use_broadcast_id: bool = False):
        """
        sets the id of the dynamixel servo
        @param old_id: current id of the servo
        @param new_id: new id
        @param use_broadcast_id: set ids of all connected dynamixels if True.
         If False, change only servo with self.config.id
        @return:
        """
        if use_broadcast_id:
            current_id = 254
        else:
            current_id = old_id
        dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(
            self.portHandler, current_id, self.ADDR_ID, new_id
        )
        self._process_response(dxl_comm_result, dxl_error, old_id)
        self.config.id = id

    def _enable_torque(self, motor_id):
        dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(
            self.portHandler, motor_id, self.ADDR_TORQUE_ENABLE, 1
        )
        self._process_response(dxl_comm_result, dxl_error, motor_id)
        self.torque_enabled[motor_id] = True

    def _disable_torque(self, motor_id):
        dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(
            self.portHandler, motor_id, self.ADDR_TORQUE_ENABLE, 0
        )
        self._process_response(dxl_comm_result, dxl_error, motor_id)
        self.torque_enabled[motor_id] = False

    def _process_response(self, dxl_comm_result: int, dxl_error: int, motor_id: int):
        if dxl_comm_result != COMM_SUCCESS:
            raise ConnectionError(
                f"dxl_comm_result for motor {motor_id}: {self.packetHandler.getTxRxResult(dxl_comm_result)}"
            )
        elif dxl_error != 0:
            print(f"dxl error {dxl_error}")
            raise ConnectionError(
                f"dynamixel error for motor {motor_id}: {self.packetHandler.getTxRxResult(dxl_error)}"
            )

    def set_operating_mode(self, motor_id: int, operating_mode: OperatingMode):
        """Set the operating mode of the motor."""
        dxl_comm_result, dxl_error = self.packetHandler.write2ByteTxRx(
            self.portHandler, motor_id, self.OPERATING_MODE_ADDR, operating_mode.value
        )
        self._process_response(dxl_comm_result, dxl_error, motor_id)
        self.operating_modes[motor_id] = operating_mode

    def set_pwm_limit(self, motor_id: int, limit: int):
        """Set the PWM limit of the motor."""
        dxl_comm_result, dxl_error = self.packetHandler.write2ByteTxRx(
            self.portHandler, motor_id, 36, limit
        )
        self._process_response(dxl_comm_result, dxl_error, motor_id)

    def set_velocity_limit(self, motor_id: int, velocity_limit):
        """Set the velocity limit of the motor."""
        dxl_comm_result, dxl_error = self.packetHandler.write4ByteTxRx(
            self.portHandler, motor_id, self.ADDR_VELOCITY_LIMIT, velocity_limit
        )
        self._process_response(dxl_comm_result, dxl_error, motor_id)

    def set_P(self, motor_id: int, P: int):
        """Set the position P gain of the motor."""
        dxl_comm_result, dxl_error = self.packetHandler.write2ByteTxRx(
            self.portHandler, motor_id, self.POSITION_P, P
        )
        self._process_response(dxl_comm_result, dxl_error, motor_id)

    def set_I(self, motor_id: int, I_gain: int):
        """Set the position I gain of the motor."""
        dxl_comm_result, dxl_error = self.packetHandler.write2ByteTxRx(
            self.portHandler, motor_id, self.POSITION_I, I_gain
        )
        self._process_response(dxl_comm_result, dxl_error, motor_id)

    def read_home_offset(self, motor_id: int):
        """Read the home offset of the motor."""
        self._disable_torque(motor_id)
        home_offset = self._read_value(motor_id, ReadAttribute.HOMING_OFFSET, 4)
        self._enable_torque(motor_id)
        return home_offset

    def set_home_offset(self, motor_id: int, home_position: int):
        """Set the home offset of the motor."""
        self._disable_torque(motor_id)
        dxl_comm_result, dxl_error = self.packetHandler.write4ByteTxRx(
            self.portHandler, motor_id, ReadAttribute.HOMING_OFFSET.value, home_position
        )
        self._process_response(dxl_comm_result, dxl_error, motor_id)
        self._enable_torque(motor_id)

    def set_baudrate(self, motor_id: int, baudrate):
        """Set the baudrate of the motor."""
        # translate baudrate into dynamixel baudrate setting id
        if baudrate == 57600:
            baudrate_id = 1
        elif baudrate == 1_000_000:
            baudrate_id = 3
        elif baudrate == 2_000_000:
            baudrate_id = 4
        elif baudrate == 3_000_000:
            baudrate_id = 5
        elif baudrate == 4_000_000:
            baudrate_id = 6
        else:
            raise Exception("baudrate not implemented")

        self._disable_torque(motor_id)
        dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(
            self.portHandler, motor_id, ReadAttribute.BAUDRATE.value, baudrate_id
        )
        self._process_response(dxl_comm_result, dxl_error, motor_id)

    def _read_value(self, motor_id, attribute: ReadAttribute, num_bytes: int, tries=10):
        """Read a value from the motor."""
        try:
            if num_bytes == 1:
                value, dxl_comm_result, dxl_error = self.packetHandler.read1ByteTxRx(
                    self.portHandler, motor_id, attribute.value
                )
            elif num_bytes == 2:
                value, dxl_comm_result, dxl_error = self.packetHandler.read2ByteTxRx(
                    self.portHandler, motor_id, attribute.value
                )
            elif num_bytes == 4:
                value, dxl_comm_result, dxl_error = self.packetHandler.read4ByteTxRx(
                    self.portHandler, motor_id, attribute.value
                )
        except Exception:
            if tries == 0:
                raise Exception
            else:
                return self._read_value(motor_id, attribute, num_bytes, tries=tries - 1)
        if dxl_comm_result != COMM_SUCCESS:
            if tries <= 1:
                raise ConnectionError(
                    f"dxl_comm_result {dxl_comm_result} for servo {motor_id} value {value}"
                )
            else:
                print(
                    f"dynamixel read failure for servo {motor_id} trying again with {tries - 1} tries"
                )
                time.sleep(0.02)
                return self._read_value(motor_id, attribute, num_bytes, tries=tries - 1)
        elif dxl_error != 0:
            if tries == 0 and dxl_error != 128:
                raise Exception(
                    f"Failed to read value from motor {motor_id} error is {dxl_error}"
                )
            else:
                return self._read_value(motor_id, attribute, num_bytes, tries=tries - 1)
        return value

    def set_home_position(self, motor_id: int):
        self.set_home_offset(motor_id, 0)
        current_position = self.read_position(motor_id)
        self.set_home_offset(motor_id, -current_position)


if __name__ == "__main__":
    dynamixel = Dynamixel.Config(
        baudrate=1_000_000, device_name="/dev/tty.usbmodem57380045631"
    ).instantiate()
    motor_id = 1
    pos = dynamixel.read_position(motor_id)
    for i in range(10):
        s = time.monotonic()
        pos = dynamixel.read_position(motor_id)
        delta = time.monotonic() - s
        print(f"read position took {delta}")
        print(f"position {pos}")
