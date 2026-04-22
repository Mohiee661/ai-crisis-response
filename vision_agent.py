import cv2
import numpy as np
from ultralytics import YOLO


FIRE_MODEL_PATH = "fire_model.pt"
PERSON_MODEL_PATH = "yolov8n.pt"
DEFAULT_VIDEO_PATH = "videos/fire.mp4"

FIRE_CONFIDENCE_THRESHOLD = 0.30
PERSON_CONFIDENCE_THRESHOLD = 0.30
MIN_BOX_AREA = 80
MIN_VISUAL_FIRE_AREA = 45
MAX_VISUAL_FIRE_AREA = 3500
INFERENCE_IMAGE_SIZE = 960

PERSON_CLASS_ID = 0
FIRE_COLOR = (0, 0, 255)
SMOKE_COLOR = (0, 165, 255)
PERSON_COLOR = (255, 0, 0)
FALL_COLOR = (0, 0, 255)
WINDOW_NAME = "Vision Agent"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_model(path: str) -> YOLO:
    try:
        return YOLO(path)
    except Exception as exc:
        raise RuntimeError(f"Failed to load model '{path}': {exc}") from exc


fire_model = _load_model(FIRE_MODEL_PATH)
person_model = _load_model(PERSON_MODEL_PATH)

# ---------------------------------------------------------------------------
# Dynamic class mapping — no hardcoded IDs
# ---------------------------------------------------------------------------

class_names: dict[int, str] = fire_model.model.names
print(f"[MODEL CLASSES] {class_names}")

fire_classes: list[int] = [cid for cid, name in class_names.items() if "fire" in name.lower()]
smoke_classes: list[int] = [cid for cid, name in class_names.items() if "smoke" in name.lower()]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def event_to_crisis(event: dict) -> str:
    if event["fire"] or event["smoke"]:
        return "FIRE"
    if event["fall_detected"]:
        return "MEDICAL"
    return "SAFE"


def create_empty_event() -> dict:
    return {
        "fire": False,
        "smoke": False,
        "person": False,
        "fall_detected": False,
        "confidence": 0.0,
    }


def box_area(box_coordinates) -> int:
    x1, y1, x2, y2 = map(int, box_coordinates)
    return max(0, x2 - x1) * max(0, y2 - y1)


def draw_box(frame, box_coordinates, label: str, color: tuple[int, int, int]) -> None:
    x1, y1, x2, y2 = map(int, box_coordinates)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        frame,
        label,
        (x1, max(y1 - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        cv2.LINE_AA,
    )


def rectangle_iou(
    first_box: tuple[int, int, int, int],
    second_box: tuple[int, int, int, int],
) -> float:
    ax1, ay1, ax2, ay2 = first_box
    bx1, by1, bx2, by2 = second_box

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def overlaps_person(box_coordinates: tuple[int, int, int, int], person_boxes: list) -> bool:
    return any(
        rectangle_iou(box_coordinates, person_box) > 0.10
        for person_box in person_boxes
    )


# ---------------------------------------------------------------------------
# Detection — fire / smoke
# ---------------------------------------------------------------------------

def detect_fire_and_smoke(frame, event: dict) -> float:
    results = fire_model.predict(
        frame,
        conf=FIRE_CONFIDENCE_THRESHOLD,
        imgsz=INFERENCE_IMAGE_SIZE,
        verbose=False,
    )

    max_conf = 0.0
    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            coords = box.xyxy[0]

            if conf <= FIRE_CONFIDENCE_THRESHOLD or box_area(coords) < MIN_BOX_AREA:
                continue

            max_conf = max(max_conf, conf)
            if cls in fire_classes:
                event["fire"] = True
                draw_box(frame, coords, f"fire {conf:.2f}", FIRE_COLOR)

            elif cls in smoke_classes:
                event["smoke"] = True
                draw_box(frame, coords, f"smoke {conf:.2f}", SMOKE_COLOR)
    
    return max_conf


# ---------------------------------------------------------------------------
# Detection — visual flame fallback (only when model finds nothing)
# ---------------------------------------------------------------------------

def detect_visual_fire(frame, person_boxes: list) -> float:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_orange = np.array([0, 130, 170])
    upper_orange = np.array([35, 255, 255])
    lower_red = np.array([160, 130, 170])
    upper_red = np.array([179, 255, 255])

    warm_mask = cv2.bitwise_or(
        cv2.inRange(hsv, lower_orange, upper_orange),
        cv2.inRange(hsv, lower_red, upper_red),
    )

    b, g, r = cv2.split(frame)
    dominance = (
        (r > g * 0.95) & (r > b * 1.15) & (r > 120)
    ).astype(np.uint8) * 255

    fire_mask = cv2.bitwise_and(warm_mask, dominance)
    fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    fire_mask = cv2.dilate(fire_mask, np.ones((5, 5), np.uint8), iterations=1)

    contours, _ = cv2.findContours(fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total_area = sum(cv2.contourArea(c) for c in contours)
    if total_area < MIN_VISUAL_FIRE_AREA or total_area > MAX_VISUAL_FIRE_AREA:
        return

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_VISUAL_FIRE_AREA:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        coords = (x, y, x + w, y + h)
        if overlaps_person(coords, person_boxes):
            continue

        conf = min(0.95, 0.35 + (area / max(1, frame.shape[0] * frame.shape[1])) * 12)
        draw_box(frame, coords, f"fire(fallback) {conf:.2f}", FIRE_COLOR)
        return conf
    return 0.0


# ---------------------------------------------------------------------------
# Detection — person / fall
# ---------------------------------------------------------------------------

def detect_person_and_fall(frame, event: dict) -> tuple[float, list]:
    results = person_model.predict(
        frame,
        classes=[PERSON_CLASS_ID],
        conf=PERSON_CONFIDENCE_THRESHOLD,
        imgsz=INFERENCE_IMAGE_SIZE,
        verbose=False,
    )

    max_conf = 0.0
    person_boxes = []
    for result in results:
        for box in result.boxes:
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w = x2 - x1
            h = y2 - y1

            if conf <= PERSON_CONFIDENCE_THRESHOLD or w * h < MIN_BOX_AREA:
                continue

            max_conf = max(max_conf, conf)
            event["person"] = True
            person_boxes.append((x1, y1, x2, y2))

            is_fall = w > h
            if is_fall:
                event["fall_detected"] = True

            label = "fall" if is_fall else "person"
            color = FALL_COLOR if is_fall else PERSON_COLOR
            draw_box(frame, (x1, y1, x2, y2), f"{label} {conf:.2f}", color)
    
    return max_conf, person_boxes


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def process_frame(frame) -> dict:
    event = create_empty_event()
    
    p_conf, person_boxes = detect_person_and_fall(frame, event)
    f_conf = detect_fire_and_smoke(frame, event)
    
    # Fallback only when the model found no fire
    if not event["fire"]:
        fb_conf = detect_visual_fire(frame, person_boxes)
        if fb_conf > 0:
            event["fire"] = True
            f_conf = max(f_conf, fb_conf)
    
    event["confidence"] = max(p_conf, f_conf)
    
    print(f"[VISION EVENT] {event}")
    return event


# Alias kept for any code that imports vision_agent directly
vision_agent = process_frame


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

def run_video(video_path: str = DEFAULT_VIDEO_PATH) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video source: {video_path}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            process_frame(frame)
            cv2.imshow(WINDOW_NAME, frame)
            if cv2.waitKey(1) == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_video(DEFAULT_VIDEO_PATH)
