import argparse
import asyncio
import logging
import signal
from typing import Optional
from robot_controller import RobotController
LISTEN_HOST = '0.0.0.0'
LISTEN_PORT = 9000
V_LIMIT = 0.51
W_LIMIT = 3.5
WATCHDOG_S = 1.5
COMMAND_MAP = {'FORWARD': (0.2, 0.0), 'BACKWARD': (-0.15, 0.0), 'LEFT': (0.0, 1.5), 'RIGHT': (0.0, -1.5), 'FWD_LEFT': (0.2, 0.5), 'FWD_RIGHT': (0.2, -0.5), 'SEARCH': (0.0, -0.6), 'STOP': (0.0, 0.0)}
log = logging.getLogger('jetserver')

class CommandRouter:

    def __init__(self, controller: Optional[RobotController]):
        self.controller = controller

    def dispatch(self, action: str) -> str:
        action = action.strip()
        if not action:
            return 'ERR:EMPTY'
        if action.upper().startswith('MOVE'):
            parts = action.split()
            if len(parts) != 3:
                return f'ERR:{action}'
            try:
                v = float(parts[1])
                w = float(parts[2])
            except ValueError:
                return f'ERR:{action}'
            v = max(-V_LIMIT, min(V_LIMIT, v))
            w = max(-W_LIMIT, min(W_LIMIT, w))
            if self.controller is not None:
                self.controller.move(v, w)
            return f'OK:MOVE {v:.3f} {w:.3f}'
        action = action.upper()
        params = COMMAND_MAP.get(action)
        if params is None:
            return f'ERR:{action}'
        if self.controller is not None:
            v, w = params
            self.controller.move(v, w)
        return f'OK:{action}'

    def halt(self):
        if self.controller is not None:
            self.controller.stop()

async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter, router: CommandRouter):
    peer = writer.get_extra_info('peername')
    log.info(f'клиент подключён: {peer}')
    try:
        while not reader.at_eof():
            try:
                raw = await asyncio.wait_for(reader.readline(), timeout=WATCHDOG_S)
            except asyncio.TimeoutError:
                router.halt()
                continue
            if not raw:
                break
            line = raw.decode('utf-8', errors='ignore').strip()
            if not line:
                continue
            reply = router.dispatch(line)
            log.info(f'{peer} → {line} → {reply}')
            writer.write((reply + '\n').encode('utf-8'))
            try:
                await writer.drain()
            except ConnectionError:
                break
    finally:
        log.info(f'клиент отключился: {peer}')
        router.halt()
        try:
            writer.close()
            await writer.wait_closed()
        except OSError:
            pass

async def start_server(port: int, router: CommandRouter):
    server = await asyncio.start_server(lambda r, w: handle_client(r, w, router), host=LISTEN_HOST, port=port, reuse_address=True)
    addrs = ', '.join((str(s.getsockname()) for s in server.sockets))
    log.info(f'сервер запущен на {addrs}')
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop_event.set)
        except NotImplementedError:
            pass
    async with server:
        serve_task = asyncio.create_task(server.serve_forever())
        stop_task = asyncio.create_task(stop_event.wait())
        await asyncio.wait({serve_task, stop_task}, return_when=asyncio.FIRST_COMPLETED)
        log.info('останов сервера…')
        serve_task.cancel()
        try:
            await serve_task
        except (asyncio.CancelledError, Exception):
            pass
    router.halt()
    log.info('сервер остановлен')

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=LISTEN_PORT, help=f'TCP-порт прослушивания (по умолчанию {LISTEN_PORT})')
    ap.add_argument('--serial', default=None, help='последовательный порт ESP32, например /dev/ttyUSB0')
    ap.add_argument('--dry-run', action='store_true', help='без обращения к ESP32, только логирование команд')
    return ap.parse_args()

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname).1s | %(message)s', datefmt='%H:%M:%S')
    args = parse_args()
    controller: Optional[RobotController] = None
    if not args.dry_run:
        port = args.serial or RobotController.find_esp32_port()
        if port is None:
            log.warning('ESP32 не найден, перехожу в режим --dry-run')
        else:
            controller = RobotController(port=port)
            if controller.connect():
                log.info(f'ESP32: {port}')
            else:
                log.warning('подключение к ESP32 не удалось — режим --dry-run')
                controller = None
    router = CommandRouter(controller)
    try:
        asyncio.run(start_server(args.port, router))
    except KeyboardInterrupt:
        pass
    finally:
        if controller is not None:
            controller.stop()
            controller.disconnect()
if __name__ == '__main__':
    main()
