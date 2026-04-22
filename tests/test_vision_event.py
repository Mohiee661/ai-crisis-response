import numpy as np

import vision_agent


def test_process_frame_returns_normalized_event(monkeypatch) -> None:
    def fake_detect_person_and_fall(frame, event):
        event["person"] = True
        event["fall_detected"] = True
        return 0.72, [(1, 1, 10, 4)]

    def fake_detect_fire_and_smoke(frame, event):
        event["smoke"] = True
        event["smoke_confidence"] = 0.61
        event["raw_smoke_detected"] = True
        return 0.61

    monkeypatch.setattr(vision_agent, "detect_person_and_fall", fake_detect_person_and_fall)
    monkeypatch.setattr(vision_agent, "detect_fire_and_smoke", fake_detect_fire_and_smoke)

    frame = np.zeros((24, 24, 3), dtype=np.uint8)
    event = vision_agent.process_frame(frame, camera_id="cam-2", frame_index=8)

    assert event == {
        "camera_id": "cam-2",
        "frame_index": 8,
        "fire": False,
        "smoke": True,
        "person": True,
        "fall_detected": True,
        "confidence": 0.72,
        "fire_confidence": 0.0,
        "smoke_confidence": 0.61,
        "raw_fire_detected": False,
        "raw_smoke_detected": True,
        "motion_score": 0.0,
        "temporal_fire_average": 0.0,
        "temporal_smoke_average": 0.0,
        "temporal_fire_support": 0,
        "validation_state": "SAFE",
        "validation_reason": "no_signal",
    }


def test_event_to_crisis_prefers_combined_state() -> None:
    event = {
        "fire": True,
        "smoke": False,
        "fall_detected": True,
    }

    assert vision_agent.event_to_crisis(event) == "BOTH"
