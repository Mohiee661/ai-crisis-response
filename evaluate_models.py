from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import cv2

from decision_engine import detect_danger
from pipeline_support import DEFAULT_CAMERA_ID, TemporalEventTracker
from vision_agent import process_frame, set_active_fire_model


def discover_videos(pattern: str) -> list[str]:
    return sorted(glob.glob(pattern))


def evaluate_video(
    video_path: str,
    *,
    label: str,
    camera_id: str,
    max_frames: int | None = None,
) -> dict:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video source: {video_path}")

    tracker = TemporalEventTracker()
    total_frames = 0
    fire_alert_frames = 0
    fire_monitor_frames = 0
    fall_frames = 0

    try:
        while True:
            if max_frames is not None and total_frames >= max_frames:
                break

            ok, frame = cap.read()
            if not ok:
                break

            raw_event = process_frame(frame, camera_id=camera_id, frame_index=total_frames)
            stable_event = tracker.stabilize(raw_event, frame)
            danger = detect_danger(stable_event)

            if danger in {"FIRE", "BOTH"}:
                fire_alert_frames += 1
            elif stable_event["validation_state"] == "MONITOR":
                fire_monitor_frames += 1

            if stable_event["fall_detected"]:
                fall_frames += 1

            total_frames += 1
    finally:
        cap.release()

    return {
        "video": video_path,
        "label": label,
        "frames": total_frames,
        "fire_alert_frames": fire_alert_frames,
        "fire_monitor_frames": fire_monitor_frames,
        "fall_frames": fall_frames,
        "false_fire_frames": fire_alert_frames if label == "fall" else 0,
    }


def summarize_model(model_reference: str, fire_videos: list[str], fall_videos: list[str], max_frames: int | None) -> dict:
    set_active_fire_model(model_reference)

    video_results = []
    for index, video_path in enumerate(fire_videos, start=1):
        video_results.append(
            evaluate_video(
                video_path,
                label="fire",
                camera_id=f"{DEFAULT_CAMERA_ID}-fire-{index}",
                max_frames=max_frames,
            )
        )

    for index, video_path in enumerate(fall_videos, start=1):
        video_results.append(
            evaluate_video(
                video_path,
                label="fall",
                camera_id=f"{DEFAULT_CAMERA_ID}-fall-{index}",
                max_frames=max_frames,
            )
        )

    return {
        "model": model_reference,
        "videos": video_results,
        "totals": {
            "fire_alert_frames": sum(item["fire_alert_frames"] for item in video_results if item["label"] == "fire"),
            "false_fire_frames": sum(item["false_fire_frames"] for item in video_results if item["label"] == "fall"),
            "fall_frames": sum(item["fall_frames"] for item in video_results),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate fire/smoke models on fire and fall videos.")
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Fire model paths or ACTIVE_FIRE_MODEL aliases to evaluate.",
    )
    parser.add_argument(
        "--fire-videos",
        nargs="*",
        default=None,
        help="Fire video paths. Defaults to videos/*fire*.mp4.",
    )
    parser.add_argument(
        "--fall-videos",
        nargs="*",
        default=None,
        help="Fall video paths. Defaults to videos/*fall*.mp4.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap on frames processed per video.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    fire_videos = args.fire_videos or discover_videos("videos/*fire*.mp4")
    fall_videos = args.fall_videos or discover_videos("videos/*fall*.mp4")
    models = args.models or [str(Path("fire_smoke_yolov8n.pt"))]

    summary = [
        summarize_model(model_reference, fire_videos, fall_videos, args.max_frames)
        for model_reference in models
    ]
    print(json.dumps(summary, indent=2))
