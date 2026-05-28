import logging
import time
from dataclasses import dataclass
from typing import Optional

import serial
import serial.tools.list_ports


PROTO_BAUDRATE        = 115200
PROTO_OPEN_DELAY_S    = 2.0
MSG_VELOCITY          = 1
MSG_WHEEL_SPEEDS      = 2


@dataclass(frozen=True)
class JetbotGeometry:
    wheel_base: float   = 0.117
    wheel_radius: float = 0.0325
    max_linear: float   = 0.51


class RobotController:

    geometry = JetbotGeometry()

    def __init__(self,
                 port: str = '/dev/ttyUSB0',
                 baudrate: int = PROTO_BAUDRATE,
                 read_timeout: float = 1.0):
        self._port = port
        self._baud = baudrate
        self._read_timeout = read_timeout
        self._link: Optional[serial.Serial] = None
        self._log = logging.getLogger('esp32')

    @property
    def connected(self) -> bool:
        return self._link is not None and self._link.is_open

    def connect(self) -> bool:
        try:
            self._link = serial.Serial(
                port=self._port,
                baudrate=self._baud,
                timeout=self._read_timeout,
            )
        except serial.SerialException as exc:
            self._log.error(f'не удалось открыть {self._port}: {exc}')
            self._link = None
            return False

        time.sleep(PROTO_OPEN_DELAY_S)
        try:
            self._link.reset_input_buffer()
        except serial.SerialException:
            pass

        self._log.info(f'serial открыт: {self._port} @ {self._baud}')
        return True

    def disconnect(self):
        self.stop()
        time.sleep(0.1)
        if self.connected:
            try:
                self._link.close()
            except serial.SerialException:
                pass
        self._link = None

    def move(self, linear: float, angular: float):
        self._send_frame(MSG_VELOCITY, (linear, angular))

    def set_wheels(self, left: float, right: float):
        self._send_frame(MSG_WHEEL_SPEEDS, (left, right))

    def stop(self):
        self.move(0.0, 0.0)

    def _send_frame(self, msg_type: int, values):
        if not self.connected:
            return
        payload = ';'.join(f'{v:.4f}' for v in values)
        frame = f'${msg_type};{payload};#\n'.encode('utf-8')
        try:
            self._link.write(frame)
            self._link.flush()
        except serial.SerialException as exc:
            self._log.error(f'serial write: {exc}')
            self._link = None

    @staticmethod
    def find_esp32_port() -> Optional[str]:
        ports = list(serial.tools.list_ports.comports())

        text_markers = ('cp210', 'ch340', 'esp32', 'usb-serial', 'usb serial')
        for entry in ports:
            descr = ((entry.description or '') + ' ' +
                     (entry.manufacturer or '')).lower()
            if any(m in descr for m in text_markers):
                return entry.device

        for entry in ports:
            if entry.vid == 0x1A86 and entry.pid == 0x7523:
                return entry.device

        for entry in ports:
            dev = entry.device or ''
            if 'USB' in dev or 'ACM' in dev:
                return dev

        return None
