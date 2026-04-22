import numpy as np

from pipeline_support import TemporalEventTracker, create_vision_event


def make_fire_event(frame_index: int, fire_confidence: float, smoke_confidence: float = 0.0):
    event = create_vision_event(camera_id="cam-1", frame_index=frame_index)
    event["fire"] = fire_confidence > 0.0
    event["smoke"] = smoke_confidence > 0.0
    event["fire_confidence"] = fire_confidence
    event["smoke_confidence"] = smoke_confidence
    event["raw_fire_detected"] = event["fire"]
    event["raw_smoke_detected"] = event["smoke"]
    event["confidence"] = max(fire_confidence, smoke_confidence)
    return event


def test_temporal_tracker_promotes_persistent_fire_to_alert(monkeypatch) -> None:
    monkeypatch.setenv("FIRE_THRESHOLD", "0.40")
    monkeypatch.setenv("SMOKE_THRESHOLD", "0.30")
    monkeypatch.setenv("MOTION_THRESHOLD", "1.00")
    monkeypatch.setenv("FIRE_TEMPORAL_WINDOW", "4")
    monkeypatch.setenv("FIRE_TEMPORAL_MIN_FRAMES", "3")

    tracker = TemporalEventTracker()
    frames = [
        np.zeros((32, 32, 3), dtype=np.uint8),
        np.full((32, 32, 3), 30, dtype=np.uint8),
        np.full((32, 32, 3), 90, dtype=np.uint8),
        np.full((32, 32, 3), 150, dtype=np.uint8),
    ]

    first = tracker.stabilize(make_fire_event(0, 0.55), frames[0])
    second = tracker.stabilize(make_fire_event(1, 0.55), frames[1])
    third = tracker.stabilize(make_fire_event(2, 0.55), frames[2])

    assert first["validation_state"] == "SAFE"
    assert second["validation_state"] == "MONITOR"
    assert third["validation_state"] == "ALERT"
    assert third["fire"] is True


def test_temporal_tracker_rejects_low_motion_fire(monkeypatch) -> None:
    monkeypatch.setenv("FIRE_THRESHOLD", "0.40")
    monkeypatch.setenv("SMOKE_THRESHOLD", "0.25")
    monkeypatch.setenv("MOTION_THRESHOLD", "1.00")

    tracker = TemporalEventTracker()
    frame = np.zeros((32, 32, 3), dtype=np.uint8)

    tracker.stabilize(make_fire_event(0, 0.80, 0.40), frame)
    stable_event = tracker.stabilize(make_fire_event(1, 0.80, 0.40), frame.copy())

    assert stable_event["validation_state"] == "SAFE"
    assert stable_event["fire"] is False
    assert "low motion" in stable_event["validation_reason"]
