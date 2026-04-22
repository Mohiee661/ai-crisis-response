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


def load_models() -> tuple[YOLO, YOLO]:
    fire_detector = YOLO(FIRE_MODEL_PATH)
    person_detector = YOLO(PERSON_MODEL_PATH)
    return fire_detector, person_detector


fire_model, person_model = load_models()


def create_empty_event() -> dict:
    return {
        "fire": False,
        "smoke": False,
        "person": False,
        "fall_detected": False,
        "confidence": 0.0,
        "_person_boxes": [],
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


def rectangle_iou(first_box: tuple[int, int, int, int], second_box: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = first_box
    bx1, by1, bx2, by2 = second_box

    intersection_x1 = max(ax1, bx1)
    intersection_y1 = max(ay1, by1)
    intersection_x2 = min(ax2, bx2)
    intersection_y2 = min(ay2, by2)

    intersection_area = max(0, intersection_x2 - intersection_x1) * max(0, intersection_y2 - intersection_y1)
    first_area = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    second_area = max(0, bx2 - bx1) * max(0, by2 - by1)
    union_area = first_area + second_area - intersection_area

    if union_area == 0:
        return 0.0
    return intersection_area / union_area


def overlaps_person(box_coordinates: tuple[int, int, int, int], event: dict) -> bool:
    return any(rectangle_iou(box_coordinates, person_box) > 0.10 for person_box in event["_person_boxes"])


def get_class_name(model: YOLO, class_id: int) -> str:
    names = getattr(model, "names", {})
    if isinstance(names, dict):
        return str(names.get(class_id, class_id)).lower()
    if isinstance(names, list) and 0 <= class_id < len(names):
        return str(names[class_id]).lower()
    return str(class_id)


def classify_fire_event(class_id: int, class_name: str) -> str | None:
    if class_name in {"fire", "flame", "flames"}:
        return "fire"
    if class_name in {"smoke", "smoky"}:
        return "smoke"
    if class_name == str(class_id):
        if class_id == 0:
            return "fire"
        if class_id == 1:
            return "smoke"
    return None


def detect_fire_and_smoke(frame, event: dict) -> None:
    results = fire_model.predict(
        frame,
        conf=FIRE_CONFIDENCE_THRESHOLD,
        imgsz=INFERENCE_IMAGE_SIZE,
        verbose=False,
    )

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            coordinates = box.xyxy[0]
            class_name = get_class_name(fire_model, class_id)
            event_type = classify_fire_event(class_id, class_name)

            if event_type is None:
                continue
            if confidence <= FIRE_CONFIDENCE_THRESHOLD or box_area(coordinates) < MIN_BOX_AREA:
                continue

            event[event_type] = True
            event["confidence"] = max(event["confidence"], confidence)

            color = FIRE_COLOR if event_type == "fire" else SMOKE_COLOR
            label = f"{event_type} {confidence:.2f}"
            draw_box(frame, coordinates, label, color)


def detect_visual_fire(frame, event: dict) -> None:
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_orange = np.array([0, 130, 170])
    upper_orange = np.array([35, 255, 255])
    lower_red = np.array([160, 130, 170])
    upper_red = np.array([179, 255, 255])

    warm_mask = cv2.bitwise_or(
        cv2.inRange(hsv_frame, lower_orange, upper_orange),
        cv2.inRange(hsv_frame, lower_red, upper_red),
    )

    blue_channel, green_channel, red_channel = cv2.split(frame)
    dominance_mask = (
        (red_channel > green_channel * 0.95)
        & (red_channel > blue_channel * 1.15)
        & (red_channel > 120)
    ).astype(np.uint8) * 255

    fire_mask = cv2.bitwise_and(warm_mask, dominance_mask)
    fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    fire_mask = cv2.dilate(fire_mask, np.ones((5, 5), np.uint8), iterations=1)

    contours, _ = cv2.findContours(fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total_fire_area = sum(cv2.contourArea(contour) for contour in contours)
    if total_fire_area < MIN_VISUAL_FIRE_AREA or total_fire_area > MAX_VISUAL_FIRE_AREA:
        return

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_VISUAL_FIRE_AREA:
            continue

        x, y, width, height = cv2.boundingRect(contour)
        box_coordinates = (x, y, x + width, y + height)
        if overlaps_person(box_coordinates, event):
            continue

        confidence = min(0.95, 0.35 + (area / max(1, frame.shape[0] * frame.shape[1])) * 12)
        event["fire"] = True
        event["confidence"] = max(event["confidence"], confidence)
        draw_box(frame, box_coordinates, f"fire {confidence:.2f}", FIRE_COLOR)


def detect_person_and_fall(frame, event: dict) -> None:
    results = person_model.predict(
        frame,
        classes=[PERSON_CLASS_ID],
        conf=PERSON_CONFIDENCE_THRESHOLD,
        imgsz=INFERENCE_IMAGE_SIZE,
        verbose=False,
    )

    for result in results:
        for box in result.boxes:
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            width = x2 - x1
            height = y2 - y1
            area = width * height

            if confidence <= PERSON_CONFIDENCE_THRESHOLD or area < MIN_BOX_AREA:
                continue

            event["person"] = True
            event["confidence"] = max(event["confidence"], confidence)
            event["_person_boxes"].append((x1, y1, x2, y2))

            is_fall_posture = width > height
            if is_fall_posture:
                event["fall_detected"] = True

            label_name = "fall" if is_fall_posture else "person"
            color = FALL_COLOR if is_fall_posture else PERSON_COLOR
            draw_box(frame, (x1, y1, x2, y2), f"{label_name} {confidence:.2f}", color)


def process_frame(frame) -> dict:
    event = create_empty_event()
    detect_person_and_fall(frame, event)
    detect_fire_and_smoke(frame, event)
    if not event["fire"] and not event["smoke"]:
        detect_visual_fire(frame, event)
    event.pop("_person_boxes", None)
    return event


def vision_agent(frame) -> dict:
    return process_frame(frame)


def run_video(video_path: str = DEFAULT_VIDEO_PATH) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video source: {video_path}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            event = process_frame(frame)
            cv2.imshow(WINDOW_NAME, frame)

            if cv2.waitKey(1) == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_video(DEFAULT_VIDEO_PATH)
