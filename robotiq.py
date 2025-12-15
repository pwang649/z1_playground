from pymodbus.client.sync import ModbusSerialClient
import time

class RobotiqGripper:

    def __init__(self, port="/dev/ttyUSB0", slave_id=9):
        self.client = ModbusSerialClient(
            method="rtu",
            port=port,
            baudrate=115200,
            stopbits=1,
            bytesize=8,
            parity="N",
            timeout=0.2
        )
        self.slave = slave_id

    def connect(self):
        if not self.client.connect():
            raise RuntimeError("Failed to connect to gripper")

    def _write_cmd(self, rACT, rGTO, rPR, rSP, rFR):
        cmd = (rGTO << 3) | (rACT << 0)

        # Register 0x03E8 (Byte 0: cmd, Byte 1: 0x00) -> (cmd << 8) | 0
        reg_03E8 = cmd << 8

        # Register 0x03E9 (Byte 2: 0x00, Byte 3: rPR) -> (0 << 8) | rPR
        reg_03E9 = rPR

        # Register 0x03EA (Byte 4: rSP, Byte 5: rFR) -> (rSP << 8) | rFR.
        reg_03EA = (rSP << 8) | rFR

        regs = [
            reg_03E8,  # 0x03E8 (Action Request)
            reg_03E9,  # 0x03E9 (Position Request)
            reg_03EA  # 0x03EA (Speed/Force)
        ]

        self.client.write_registers(0x03E8, regs,
            unit=0x0009)

    def activate(self):
        # Clear rACT
        self.client.write_registers(
            0x03E8,
            [0x0000, 0x0000, 0x0000],
            unit=0x0009
        )
        time.sleep(0.5)

        # Set rACT
        self.client.write_registers(
            0x03E8,
            [0x0100, 0x0000, 0x0000],
            unit=0x0009
        )
        time.sleep(1.0)

    def set_grip(self, position, speed=128, force=128):
        position = max(0, min(255, position))
        speed = max(0, min(255, speed))
        force = max(0, min(255, force))

        self._write_cmd(1, 1, position, speed, force)

    def open(self):
        self._write_cmd(1, 1, 0, 255, 255)

    def close(self):
        self._write_cmd(1, 1, 255, 255, 255)

    def disconnect(self):
        self.client.close()


if __name__ == "__main__":
    gripper = RobotiqGripper("/dev/ttyUSB1")
    try:
        gripper.connect()
        gripper.activate()
        gripper.close()
        time.sleep(1)
        gripper.open()
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        gripper.disconnect()