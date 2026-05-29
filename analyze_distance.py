import csv
import os
import cv2
from ultralytics import YOLO


INPUT_VIDEO = '/Users/imamramazanov/Desktop/диплом/Sledovanie.mov'
OUTPUT_CSV = '/Users/imamramazanov/Desktop/диплом/distance_analysis.csv'
SNAPSHOTS_DIR = '/Users/imamramazanov/Desktop/диплом/snapshots'
MODEL_PATH = 'yolov8n.pt'
PERSON_CLS = 0
CONF_THRESH = 0.5
FOCAL_FACTOR          = 2117.0
STOP_WIDTH_RATIO      = 0.45
WIDTH_ZONE_STOP_DIST  = 0.5
WIDTH_ZONE_DRIVE_DIST = 5.0
TOP_CLIP_MARGIN       = 5
BOT_CLIP_MARGIN       = 5
SNAPSHOT_INTERVAL_S = 2.0


def main():
    os.makedirs(SNAPSHOTS_DIR, exist_ok=True)

    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        print(f'cannot open video: {INPUT_VIDEO}')
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    snapshot_step = max(1, round(fps * SNAPSHOT_INTERVAL_S))

    print(f'video: {INPUT_VIDEO}')
    print(f'fps={fps:.2f}, frames={total_frames}, height={frame_height}')
    print(f'snapshots every {snapshot_step} frames -> {SNAPSHOTS_DIR}')
    print(f'csv -> {OUTPUT_CSV}')

    frames_processed = 0
    frames_with_target = 0
    frames_top_clipped = 0
    frames_bottom_clipped = 0
    bbox_h_list = []
    D_list = []
    all_rows = []

    csv_file = open(OUTPUT_CSV, 'w', newline='', encoding='utf-8')
    writer = csv.writer(csv_file, delimiter=',')
    writer.writerow([
        'frame_idx', 'time_s', 'target_id',
        'x1', 'y1', 'x2', 'y2',
        'bbox_w', 'bbox_h', 'D',
        'top_clipped', 'bottom_clipped', 'zone',
    ])

    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            time_s = frame_idx / fps if fps > 0 else 0.0

            results = model.track(
                frame, persist=True, verbose=False, classes=[PERSON_CLS]
            )[0]

            bbox = None
            target_id = None

            if results.boxes is not None and results.boxes.id is not None:
                boxes = results.boxes.xyxy.cpu().numpy()
                ids = results.boxes.id.cpu().numpy().astype(int)
                confs = results.boxes.conf.cpu().numpy()
                mask = confs >= CONF_THRESH
                boxes = boxes[mask]
                ids = ids[mask]
                if len(ids) > 0:
                    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                    idx = int(areas.argmax())
                    bbox = boxes[idx]
                    target_id = int(ids[idx])

            top_clipped = False
            bottom_clipped = False

            if bbox is not None:
                H, W = frame.shape[0], frame.shape[1]
                x1, y1, x2, y2 = (int(v) for v in bbox)
                bbox_w = x2 - x1
                bbox_h = y2 - y1
                top_clipped = (y1 <= TOP_CLIP_MARGIN)
                bottom_clipped = (y2 >= H - 1 - BOT_CLIP_MARGIN)

                if top_clipped or bottom_clipped:
                    if bbox_w >= W * STOP_WIDTH_RATIO:
                        D = WIDTH_ZONE_STOP_DIST
                        zone = 'WIDTH-STOP'
                    else:
                        D = WIDTH_ZONE_DRIVE_DIST
                        zone = 'WIDTH-DRIVE'
                elif bbox_h > 0:
                    D = FOCAL_FACTOR / bbox_h
                    zone = 'HEIGHT'
                else:
                    D = 0.0
                    zone = 'NONE'

                writer.writerow([
                    frame_idx,
                    f'{time_s:.3f}',
                    target_id,
                    x1, y1, x2, y2,
                    bbox_w, bbox_h,
                    f'{D:.3f}',
                    str(top_clipped), str(bottom_clipped),
                    zone,
                ])
                all_rows.append({'zone': zone})

                frames_with_target += 1
                if top_clipped:
                    frames_top_clipped += 1
                if bottom_clipped:
                    frames_bottom_clipped += 1
                if bbox_h > 0:
                    bbox_h_list.append(bbox_h)
                    D_list.append(D)
            else:
                zone = 'NONE'
                writer.writerow([
                    frame_idx,
                    f'{time_s:.3f}',
                    '', '', '', '', '', '', '',
                    f'{0.0:.3f}',
                    str(top_clipped), str(bottom_clipped),
                    zone,
                ])
                all_rows.append({'zone': zone})

            if frame_idx % snapshot_step == 0:
                snap = frame.copy()
                if bbox is not None:
                    cv2.rectangle(snap, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = (
                        f'frame={frame_idx} t={time_s:.1f}s '
                        f'D={D:.2f}m h={bbox_h}px w={bbox_w}px [{zone}]'
                    )
                else:
                    label = (
                        f'frame={frame_idx} t={time_s:.1f}s '
                        f'D=-m h=-px w=-px [{zone}]'
                    )
                cv2.putText(
                    snap, label, (15, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
                    cv2.LINE_AA,
                )
                if top_clipped or bottom_clipped:
                    h, w = snap.shape[:2]
                    cv2.rectangle(snap, (0, 0), (w - 1, h - 1), (0, 0, 255), 3)
                snap_name = f'snap_{frame_idx:06d}_t{time_s:.1f}s.png'
                cv2.imwrite(os.path.join(SNAPSHOTS_DIR, snap_name), snap)

            frames_processed += 1
            frame_idx += 1

            if frame_idx % 100 == 0:
                pct = (frame_idx / total_frames * 100.0) if total_frames > 0 else 0.0
                print(f'frame {frame_idx}/{total_frames} ({pct:.1f}%)')

    finally:
        csv_file.close()
        cap.release()

    print('')
    print('=== SUMMARY ===')
    print(f'всего кадров обработано: {frames_processed}')
    if frames_processed > 0:
        pct_target = frames_with_target / frames_processed * 100.0
    else:
        pct_target = 0.0
    print(f'кадров с target: {frames_with_target} ({pct_target:.1f}%)')
    print(f'кадров с обрезкой сверху: {frames_top_clipped}')
    print(f'кадров с обрезкой снизу: {frames_bottom_clipped}')
    if bbox_h_list:
        min_h = min(bbox_h_list)
        max_h = max(bbox_h_list)
        mean_h = sum(bbox_h_list) / len(bbox_h_list)
        print(f'bbox_h по кадрам с target: min={min_h} px, max={max_h} px, mean={mean_h:.1f} px')
    else:
        print('bbox_h: нет данных')
    if D_list:
        min_d = min(D_list)
        max_d = max(D_list)
        mean_d = sum(D_list) / len(D_list)
        print(f'D по кадрам с target: min={min_d:.3f} m, max={max_d:.3f} m, mean={mean_d:.3f} m')
    else:
        print('D: нет данных')

    print()
    print("=== ПО ЗОНАМ ===")
    zone_counts = {}
    for row in all_rows:
        z = row.get('zone', 'NONE')
        zone_counts[z] = zone_counts.get(z, 0) + 1
    for z, c in sorted(zone_counts.items()):
        pct = 100 * c / len(all_rows) if all_rows else 0
        print(f"  {z}: {c} кадров ({pct:.1f}%)")


if __name__ == '__main__':
    main()
