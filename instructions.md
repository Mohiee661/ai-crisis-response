# AI Crisis Response Instructions

## Pipeline

```text
Video Input -> Vision Detection -> Event Structuring -> Danger Detection -> Decision Engine -> Alert System
```

## Vision Agent

`vision_agent.py` loads YOLO models once and exposes:

```python
from vision_agent import process_frame

event = process_frame(frame)
```

Event format:

```python
{
    "fire": False,
    "smoke": False,
    "person": False,
    "fall_detected": False,
    "confidence": 0.0,
}
```

The core function returns structured data and annotates the input frame. It does not print logs.

## Detection Notes

- Fire/smoke first uses `fire_model.pt`.
- If the model does not expose fire/smoke classes, the agent uses a visual flame-color fallback.
- Person detection uses `yolov8n.pt`.
- Fall detection uses a simple posture heuristic: bounding box width greater than height.
- `test_runner.py` applies 3-frame temporal stability before making decisions.

## Decision Engine

- `FIRE` -> `ALERT_FIRE_STATION`
- `MEDICAL` -> `ALERT_AMBULANCE`
- `BOTH` -> `ALERT_BOTH`
- `SAFE` -> `NO_ACTION`

## Run Commands

```bash
python test_runner.py --video videos/fire.mp4
python test_runner.py --video videos/fall.mp4
```

## Local Assets

These are intentionally ignored by Git:

- `.env`
- `fire_model.pt`
- `yolov8n.pt`
- `videos/fire.mp4`
- `videos/fall.mp4`

Use `.env.example` as the template for local LLM credentials.
