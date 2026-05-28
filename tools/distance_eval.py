import cv2
import time
import logging
from ultralytics import YOLO
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)
INPUT_VIDEO = '/Users/imamramazanov/Desktop/диплом/Sledovanie.mov'
OUTPUT_VIDEO = '/Users/imamramazanov/Desktop/result_distance2.mp4'
MODEL_PATH = 'yolov8n.pt'
CONF_THRESH = 0.5
FOCAL_FACTOR = 1600.0
TARGET_DIST = 1.35

class VideoTracker:

    def __init__(self, model_path, focal_factor=600.0, conf=0.5):
        self.model = YOLO(model_path)
        self.focal_length_factor = focal_factor
        self.conf_threshold = conf
        self.frame_center_x = None
        self.target_id = None

    def estimate_distance(self, bbox_h):
        if bbox_h <= 0:
            return 0.0
        return self.focal_length_factor / bbox_h

    def get_tracked_target(self, frame):
        results = self.model.track(frame, persist=True, verbose=False, classes=[0])[0]
        if results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            ids = results.boxes.id.cpu().numpy().astype(int)
            confs = results.boxes.conf.cpu().numpy()
            mask = confs >= self.conf_threshold
            boxes = boxes[mask]
            ids = ids[mask]
            if len(ids) == 0:
                self.target_id = None
                return (None, None)
            if self.target_id is not None and self.target_id in ids:
                idx = list(ids).index(self.target_id)
                return (boxes[idx], self.target_id)
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            idx = areas.argmax()
            self.target_id = ids[idx]
            return (boxes[idx], self.target_id)
        self.target_id = None
        return (None, None)

    def annotate_frame(self, frame, bbox, track_id):
        h, w = frame.shape[:2]
        if self.frame_center_x is None:
            self.frame_center_x = w // 2
        cv2.line(frame, (self.frame_center_x, 0), (self.frame_center_x, h), (0, 255, 255), 2)
        if bbox is not None:
            x1, y1, x2, y2 = map(int, bbox)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            bbox_h = y2 - y1
            distance = self.estimate_distance(bbox_h)
            deviation = cx - self.frame_center_x
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f'ID:{track_id} {distance:.2f}m'
            cv2.putText(frame, label, (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            h_label = f'H_bbox:{bbox_h}px'
            cv2.putText(frame, h_label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            cv2.line(frame, (self.frame_center_x, h // 2), (cx, cy), (0, 255, 0), 2)
            dev_text = f'dx={deviation:+d}px'
            cv2.putText(frame, dev_text, (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return frame

def main():
    tracker = VideoTracker(MODEL_PATH, FOCAL_FACTOR, CONF_THRESH)
    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        logger.error('Не удалось открыть видео: ' + INPUT_VIDEO)
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info(f'Видео: {w}x{h}, {fps:.2f} fps')
    writer = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'avc1'), fps, (w, h))
    logger.info('Обработка... (q для выхода)')
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        bbox, tid = tracker.get_tracked_target(frame)
        annotated = tracker.annotate_frame(frame, bbox, tid)
        cv2.imshow('Tracking + Distance', annotated)
        writer.write(annotated)
        if cv2.waitKey(1) & 255 == ord('q'):
            break
    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    logger.info('✅ Результат сохранён в ' + OUTPUT_VIDEO)
if __name__ == '__main__':
    main()
