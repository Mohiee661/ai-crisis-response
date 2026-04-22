import argparse

import cv2

import vision_agent
from alert_system import send_alert
from decision_engine import decision_engine, detect_danger


DEFAULT_VIDEO_PATH = "videos/fire.mp4"
STABILITY_FRAMES = 3
WINDOW_NAME = "AI Crisis Response"


def run_pipeline(video_path: str) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video source: {video_path}")

    fire_count = 0
    fall_count = 0
    last_action = "NO_ACTION"

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 1. Vision Detection
            event = vision_agent.process_frame(frame)
            
            # 2. Temporal Stability (Smoothed)
            fire_count = max(0, fire_count + 1 if (event["fire"] or event["smoke"]) else fire_count - 1)
            fall_count = max(0, fall_count + 1 if event["fall_detected"] else fall_count - 1)
            
            fire_confirmed = fire_count >= STABILITY_FRAMES
            fall_confirmed = fall_count >= STABILITY_FRAMES
            
            # Create stable event for decision
            stable_event = event.copy()
            stable_event["fire"] = fire_confirmed
            stable_event["fall_detected"] = fall_confirmed
            
            # 3. Decision Logic
            crisis = vision_agent.event_to_crisis(stable_event)
            danger = detect_danger(stable_event)
            action = decision_engine(danger)

            # 4. Clean Logging
            print(f"[VISION EVENT] {event}")
            print(f"[CRISIS] {crisis}")
            print(f"[ACTION] {action}")

            # 5. Alert Triggering
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
