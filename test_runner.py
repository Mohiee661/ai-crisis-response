import argparse

import cv2

import vision_agent
from alert_system import send_alert
from decision_engine import decision_engine, detect_danger


DEFAULT_VIDEO_PATH = "videos/fire.mp4"
STABILITY_FRAMES = 3
WINDOW_NAME = "AI Crisis Response"


def update_stability_counter(active: bool, current_count: int) -> int:
    if active:
        return current_count + 1
    return 0


def apply_temporal_stability(event: dict, fire_frames: int, fall_frames: int) -> dict:
    stable_event = event.copy()
    stable_event["fire"] = event["fire"] and fire_frames >= STABILITY_FRAMES
    stable_event["smoke"] = event["smoke"] and fire_frames >= STABILITY_FRAMES
    stable_event["fall_detected"] = event["fall_detected"] and fall_frames >= STABILITY_FRAMES
    return stable_event


def run_pipeline(video_path: str) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video source: {video_path}")

    fire_frames = 0
    fall_frames = 0
    last_action = "NO_ACTION"

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            event = vision_agent.process_frame(frame)
            fire_frames = update_stability_counter(event["fire"] or event["smoke"], fire_frames)
            fall_frames = update_stability_counter(event["fall_detected"], fall_frames)
            stable_event = apply_temporal_stability(event, fire_frames, fall_frames)

            danger = detect_danger(stable_event)
            action = decision_engine(danger)

            print(f"[VISION] {stable_event}")
            print(f"[DANGER] {danger}")
            print(f"[ACTION] {action}")

            if action != "NO_ACTION" and action != last_action:
                send_alert(action)
            last_action = action

            cv2.imshow(WINDOW_NAME, frame)

            if cv2.waitKey(1) == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the AI crisis response pipeline.")
    parser.add_argument(
        "--video",
        default=DEFAULT_VIDEO_PATH,
        help="Path to the input video file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args.video)
