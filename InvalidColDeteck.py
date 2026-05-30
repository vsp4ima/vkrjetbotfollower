# -*- coding: utf-8 -*-
"""InvalidColDeteck.py — клиент-следователь робота за инвалидной коляской.

Архитектурно строится по образцу follower.py. Отличия:
  * Целевой объект — инвалидная коляска. Используется собственный
    детектор YOLO с весами best.pt, классы модели:
    0 — crutches, 1 — prams, 2 — wheelchair. Целевой класс
    TARGET_CLS = 2 (wheelchair), порог CONF_THRESH = 0.40.
  * «Близко / далеко» определяется не по высоте bbox, а по касанию
    верхней границы bbox к верхнему краю кадра. Если y1 <=
    TOP_CLIP_MARGIN — коляска признаётся близкой и робот
    останавливается (линейная = 0). Иначе робот едет вперёд
    со скоростью V_FORWARD.
  * Чтобы переиспользовать неизменённый decide() из follower.py,
    функция estimate_distance_m возвращает фиктивную дистанцию:
    STOP_DIST (попадает в зону «остановка») при top_clipped и
    DRIVE_DIST (попадает в зону «ехать вперёд») в остальных случаях.
  * Боковая коррекция по dx — пропорциональный регулятор angular_p,
    как в follower.py.

Источник видео по умолчанию — MJPEG-стрим с робота по адресу
http://<ip>:5000/video, как в follower.py; через --source можно
подать локальный видеофайл. Команды роботу — те же текстовые
команды, что у follower.py, через TCP к jetserver.py.
"""

import argparse
import socket
import time

import cv2
from ultralytics import YOLO


DEFAULT_IP   = '192.168.1.112'
STREAM_PORT  = 5000
ROBOT_PORT   = 9000

MODEL_PATH   = 'best.pt'
TARGET_CLS   = 2           # 0 — crutches, 1 — prams, 2 — wheelchair
CONF_THRESH  = 0.40

TOP_CLIP_MARGIN = 5        # допуск в пикселях на касание верхнего края кадра
STOP_DIST    = 0.5         # фиктивная дистанция, попадает в STOP в decide()
DRIVE_DIST   = 5.0         # фиктивная дистанция, попадает в FORWARD в decide()
DIST_MIN_M   = 1.0
DIST_MAX_M   = 1.7
FOCAL_FACTOR = 1288        # калибровка по опорному кадру: коляска ≈ 2.0 м, bbox_h = 644 px,
                           # запись 1274×958. Для рабочего разрешения 640×480 значение
                           # масштабируется пропорционально (≈ 644).

V_FORWARD       = 0.20
K_P_TURN        = 1.5     # рад/с при dx = W/2
K_P_DRIVE       = 0.8     # рад/с при dx = W/2 в режиме DRIVE
W_MAX           = 1.8     # ограничение угловой
FRAME_HALF_W    = 320

ALIGN_INTERVAL      = 3.0
ALIGN_TOLERANCE_PX  = 25

SEND_PERIOD       = 0.10
LOST_FRAMES_STOP  = 5
SEARCH_TIMEOUT_S  = 15.0


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


class RobotClient:

    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.sock = None
        self.last_cmd = None
        self.last_send = 0.0

    def connect(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(2.0)
            self.sock.connect((self.host, self.port))
            print(f'[net] connected {self.host}:{self.port}')
            return True
        except OSError as e:
            print(f'[net] connect failed: {e}')
            self.sock = None
            return False

    def send(self, cmd):
        if not self.sock:
            return
        now = time.time()
        prefix = cmd.split()[0] if cmd else cmd
        last_prefix = self.last_cmd.split()[0] if self.last_cmd else None
        if prefix == last_prefix and now - self.last_send < SEND_PERIOD:
            return
        try:
            self.sock.sendall((cmd + '\n').encode('utf-8'))
            self.last_cmd = cmd
            self.last_send = now
        except OSError as e:
            print(f'[net] send error: {e}')

    def close(self):
        if not self.sock:
            return
        try:
            self.sock.sendall(b'STOP\n')
        except OSError:
            pass
        self.sock.close()
        self.sock = None


class TargetTracker:

    def __init__(self, model, conf):
        self.model = model
        self.conf = conf
        self.target_id = None

    def update(self, frame):
        results = self.model.track(
            frame, persist=True, verbose=False, classes=[TARGET_CLS]
        )[0]
        if results.boxes is None or results.boxes.id is None:
            self.target_id = None
            return None, None
        boxes = results.boxes.xyxy.cpu().numpy()
        ids = results.boxes.id.cpu().numpy().astype(int)
        confs = results.boxes.conf.cpu().numpy()
        mask = confs >= self.conf
        boxes = boxes[mask]
        ids = ids[mask]
        if len(ids) == 0:
            self.target_id = None
            return None, None
        if self.target_id is not None and self.target_id in ids:
            idx = list(ids).index(self.target_id)
            return boxes[idx], int(self.target_id)
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        idx = int(areas.argmax())
        self.target_id = int(ids[idx])
        return boxes[idx], self.target_id


def estimate_distance_m(bbox, frame_shape):
    if bbox is None:
        return 0.0
    x1, y1, x2, y2 = map(int, bbox)
    top_clipped = y1 <= TOP_CLIP_MARGIN
    if top_clipped:
        return STOP_DIST
    return DRIVE_DIST


def angular_p(dx, k_p):
    # знак: dx>0 → цель справа → нужен поворот вправо → ω<0
    w = -k_p * dx / FRAME_HALF_W
    return clamp(w, -W_MAX, W_MAX)


def decide(state, dx, dist_m, now, last_align_ts):
    """Возвращает (новое_state, command_str, new_last_align_ts)."""
    if state == 'ALIGN':
        if abs(dx) <= ALIGN_TOLERANCE_PX:
            return 'DRIVE', 'STOP', now
        w = angular_p(dx, K_P_TURN)
        return 'ALIGN', f'MOVE 0.0 {w:.3f}', last_align_ts

    # DRIVE
    if (now - last_align_ts) >= ALIGN_INTERVAL:
        return 'ALIGN', 'STOP', last_align_ts

    if dist_m <= 0.0 or dist_m < DIST_MIN_M:
        v = 0.0
    elif dist_m <= DIST_MAX_M:
        v = 0.0
    else:
        v = V_FORWARD

    w = angular_p(dx, K_P_DRIVE)
    return 'DRIVE', f'MOVE {v:.3f} {w:.3f}', last_align_ts


def draw_hud(frame, *, state, cmd, dx=None, dist=None, bbox=None, tid=None):
    h, w = frame.shape[:2]
    cx_frame = w // 2
    cv2.line(frame, (cx_frame, 0), (cx_frame, h), (255, 255, 255), 1)
    for off in (ALIGN_TOLERANCE_PX, -ALIGN_TOLERANCE_PX):
        cv2.line(frame, (cx_frame + off, 0), (cx_frame + off, h), (180, 180, 180), 1)
    if bbox is not None:
        x1, y1, x2, y2 = map(int, bbox)
        bx = (x1 + x2) // 2
        by = (y1 + y2) // 2
        bbox_h_px = y2 - y1
        top_clipped_local = y1 <= TOP_CLIP_MARGIN
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, (bx, by), 5, (0, 0, 255), -1)
        cv2.line(frame, (cx_frame, by), (bx, by), (0, 255, 0), 2)
        if tid is not None:
            label = f'wheelchair ID:{tid}'
            if not top_clipped_local and bbox_h_px > 0:
                D_metric = FOCAL_FACTOR / bbox_h_px
                label += f'  D={D_metric:.2f} m'
            cv2.putText(frame, label,
                        (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f'STATE: {state}', (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f'CMD:   {cmd}', (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    if dx is not None and dist is not None:
        zone_label = '-'
        bottom_text = f'dx={dx:+d}px  zone={zone_label}'
        if bbox is not None:
            x1b, y1b, x2b, y2b = map(int, bbox)
            bbox_h_b = y2b - y1b
            if y1b <= TOP_CLIP_MARGIN:
                zone_label = 'CLOSE'
                bottom_text = f'dx={dx:+d}px  zone={zone_label}'
            else:
                zone_label = 'FAR'
                if bbox_h_b > 0:
                    D_metric = FOCAL_FACTOR / bbox_h_b
                    bottom_text = f'dx={dx:+d}px  zone={zone_label}  D={D_metric:.2f}m'
                else:
                    bottom_text = f'dx={dx:+d}px  zone={zone_label}'
        cv2.putText(frame, bottom_text,
                    (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ip', default=DEFAULT_IP)
    ap.add_argument('--no-robot', action='store_true')
    ap.add_argument('--source', default=None)
    args = ap.parse_args()

    src = args.source or f'http://{args.ip}:{STREAM_PORT}/video'
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f'[cam] cannot open {src}')
        return

    model = YOLO(MODEL_PATH)
    print('Классы модели:', model.names)
    tracker = TargetTracker(model, CONF_THRESH)

    robot = None
    if not args.no_robot:
        robot = RobotClient(args.ip, ROBOT_PORT)
        if not robot.connect():
            robot = None

    state = 'ALIGN'
    last_align_ts = time.time()
    lost_frames = 0
    search_start_ts = 0.0

    print("[ui] start, 'q' to quit")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            bbox, tid = tracker.update(frame)
            now = time.time()
            cmd = 'STOP'
            dx = 0
            dist = 0.0

            if bbox is None:
                lost_frames += 1
                if state != 'SEARCH' and lost_frames >= LOST_FRAMES_STOP:
                    state = 'SEARCH'
                    search_start_ts = now
                if state == 'SEARCH':
                    if now - search_start_ts < SEARCH_TIMEOUT_S:
                        cmd = 'SEARCH'
                    else:
                        cmd = 'STOP'
                else:
                    cmd = 'STOP'
                draw_hud(frame, state=state, cmd=cmd)
            else:
                lost_frames = 0
                if state == 'SEARCH':
                    state = 'ALIGN'
                x1, y1, x2, y2 = map(int, bbox)
                bx = (x1 + x2) // 2
                dx = bx - frame.shape[1] // 2
                dist = estimate_distance_m(bbox, frame.shape)
                state, cmd, last_align_ts = decide(
                    state, dx, dist, now, last_align_ts
                )
                draw_hud(frame, state=state, cmd=cmd,
                         dx=dx, dist=dist, bbox=bbox, tid=tid)

            if robot:
                robot.send(cmd)

            cv2.imshow('Follow wheelchair (q to quit)', frame)
            if cv2.waitKey(1) & 255 == ord('q'):
                break
    finally:
        if robot:
            robot.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
