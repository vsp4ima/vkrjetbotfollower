# -*- coding: utf-8 -*-
"""follower.py — клиент-следователь на MacBook M1.

Получает MJPEG-стрим с JetBot, детектит и трекает человека (YOLOv8n +
встроенный BoT-SORT), формирует управляющие команды и шлёт их по TCP
на jetserver.

Управление:
  * пропорциональный регулятор по горизонтальному отклонению dx:
        ω = −K_p · dx / (W / 2),      W = 640 px
    знак ω: положительная угловая = поворот влево.
  * линейная скорость = 0.2 м/с (по ТЗ) когда цель дальше DIST_MAX_M;
    в комфортной зоне или ближе DIST_MIN_M — линейная = 0.
  * конечный автомат состояний:
        ALIGN  — стоим, крутимся пропорционально dx, пока |dx| > ALIGN_TOLERANCE_PX
        DRIVE  — едем вперёд + плавная коррекция на ходу
        SEARCH — цель потеряна > LOST_FRAMES_STOP кадров: вращение по
                 часовой стрелке до повторного обнаружения; через
                 SEARCH_TIMEOUT_S без цели — STOP.
  * каждые ALIGN_INTERVAL секунд из DRIVE возвращаемся в ALIGN.

Протокол с jetserver:
  MOVE <v> <w>    — линейная м/с, угловая рад/с
  STOP            — остановка
  SEARCH          — режим поиска (трактуется сервером как медленное вращение)
"""

import argparse
import socket
import time

import cv2
from ultralytics import YOLO


DEFAULT_IP   = '192.168.1.112'
STREAM_PORT  = 5000
ROBOT_PORT   = 9000

MODEL_PATH   = 'yolov8n.pt'
PERSON_CLS   = 0
CONF_THRESH  = 0.5

FOCAL_FACTOR = 2117.0
STOP_WIDTH_RATIO      = 0.45
WIDTH_ZONE_STOP_DIST  = 0.5
WIDTH_ZONE_DRIVE_DIST = 5.0
TOP_CLIP_MARGIN       = 5
BOT_CLIP_MARGIN       = 5
DIST_MIN_M   = 1.0
DIST_MAX_M   = 1.7

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


class PersonTracker:

    def __init__(self, model, conf):
        self.model = model
        self.conf = conf
        self.target_id = None

    def update(self, frame):
        results = self.model.track(
            frame, persist=True, verbose=False, classes=[PERSON_CLS]
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
    H, W = frame_shape[0], frame_shape[1]
    x1, y1, x2, y2 = map(int, bbox)
    bbox_h = y2 - y1
    bbox_w = x2 - x1
    top_clipped    = y1 <= TOP_CLIP_MARGIN
    bottom_clipped = y2 >= H - 1 - BOT_CLIP_MARGIN
    if top_clipped or bottom_clipped:
        if bbox_w >= W * STOP_WIDTH_RATIO:
            return WIDTH_ZONE_STOP_DIST
        return WIDTH_ZONE_DRIVE_DIST
    if bbox_h <= 0:
        return 0.0
    return FOCAL_FACTOR / bbox_h


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
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, (bx, by), 5, (0, 0, 255), -1)
        cv2.line(frame, (cx_frame, by), (bx, by), (0, 255, 0), 2)
        if tid is not None and dist is not None:
            cv2.putText(frame, f'ID:{tid}  D={dist:.2f} m',
                        (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f'STATE: {state}', (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f'CMD:   {cmd}', (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    if dx is not None and dist is not None:
        zone_label = '-'
        if bbox is not None:
            x1b, y1b, x2b, y2b = map(int, bbox)
            if y1b <= TOP_CLIP_MARGIN or y2b >= h - 1 - BOT_CLIP_MARGIN:
                ratio = (x2b - x1b) / w
                zone_label = f'WIDTH({ratio:.2f})'
            else:
                zone_label = 'HEIGHT'
        cv2.putText(frame,
                    f'dx={dx:+d}px  D={dist:.2f}m  zone={zone_label}',
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
    tracker = PersonTracker(model, CONF_THRESH)

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

            cv2.imshow('Follow person (q to quit)', frame)
            if cv2.waitKey(1) & 255 == ord('q'):
                break
    finally:
        if robot:
            robot.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()