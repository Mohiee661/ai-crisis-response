# AI Crisis Response Instructions

## Pipeline

The project is organized as:

```text
Video Input -> Vision Detection -> Event Structuring -> Danger Detection -> Decision Engine -> Alert System
```

## Vision Agent

`vision_agent.py` loads the YOLO models once and exposes:

```python
from vision_agent import process_frame

event = process_frame(frame)
```

The event format is:

```python
{
    "fire": False,
    "smoke": False,
    "person": False,
    "fall_detected": False,
    "confidence": 0.0,
}
```

The function does not print logs. It only returns structured data and annotates the frame for display.

## Decision Engine

`decision_engine.py` maps events to danger categories and actions:

- `FIRE` -> `ALERT_FIRE_STATION`
- `MEDICAL` -> `ALERT_AMBULANCE`
- `BOTH` -> `ALERT_BOTH`
- `SAFE` -> `NO_ACTION`

## Alert System

`alert_system.py` simulates emergency alerts with terminal output.

## Test Runner

Run the full pipeline:

```bash
python test_runner.py --video videos/fire.mp4
python test_runner.py --video videos/fall.mp4
```

The runner applies 3-frame temporal stability to fire and fall events before sending them to the decision engine.

## Local Assets

Model and video files are ignored by Git:

- `fire_model.pt`
- `yolov8n.pt`
- `videos/fire.mp4`
- `videos/fall.mp4`
