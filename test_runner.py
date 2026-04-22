from __future__ import annotations

import argparse
import logging
import os

import cv2

from alert_system import send_alert
from decision_engine import decision_engine, detect_danger
from pipeline_support import (
    DEFAULT_CAMERA_ID,
    TemporalEventTracker,
    configure_logging,
    log_payload,
    normalize_video_source,
)
import vision_agent


DEFAULT_VIDEO_SOURCE = os.getenv("CRISIS_VIDEO_SOURCE", "0")
STABILITY_FRAMES = 3
WINDOW_NAME = "AI Crisis Response"
DEFAULT_LOG_LEVEL = os.getenv("CRISIS_LOG_LEVEL", "INFO")

logger = logging.getLogger(__name__)


def run_pipeline(
    video_source: str,
    camera_id: str = DEFAULT_CAMERA_ID,
    display: bool = False,
) -> None:
    cap = cv2.VideoCapture(normalize_video_source(video_source))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video source: {video_source}")

    tracker = TemporalEventTracker(stability_frames=STABILITY_FRAMES)
    frame_index = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            raw_event = vision_agent.process_frame(
                frame,
                camera_id=camera_id,
                frame_index=frame_index,
            )
            stable_event = tracker.stabilize(raw_event, frame)
            danger = detect_danger(stable_event)
            action = decision_engine(danger)
            attention_state = danger if danger != "SAFE" else stable_event.get("validation_state", "SAFE")

            log_payload(
                logger,
                logging.INFO if attention_state != "SAFE" else logging.DEBUG,
                "pipeline_event",
                {
                    "camera_id": camera_id,
                    "frame_index": frame_index,
                    "raw_event": raw_event,
                    "stable_event": stable_event,
                    "danger": danger,
                    "attention_state": attention_state,
                    "action": action,
                },
            )

            if tracker.should_send_alert(camera_id, action):
                send_alert(action, camera_id=camera_id)

            if display:
                cv2.imshow(WINDOW_NAME, frame)
                if cv2.waitKey(1) == 27:
                    break

            frame_index += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the deterministic crisis response pipeline.")
    parser.add_argument(
        "--video",
        default=DEFAULT_VIDEO_SOURCE,
        help="Path to a video file or webcam index such as 0.",
    )
    parser.add_argument(
        "--camera-id",
        default=DEFAULT_CAMERA_ID,
        help="Logical camera identifier used for temporal tracking.",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Display the annotated video stream while processing.",
    )
    parser.add_argument(
        "--log-level",
        default=DEFAULT_LOG_LEVEL,
        help="Logging level (DEBUG, INFO, WARNING, ERROR).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    configure_logging(args.log_level)
    run_pipeline(args.video, camera_id=args.camera_id, display=args.display)
