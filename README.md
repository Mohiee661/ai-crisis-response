# Vision Agent

Python-based AI Crisis Response pipeline:

```text
Video Input -> Vision Detection -> Event Structuring -> Danger Detection -> Decision Engine -> Alert System
```

## Project Structure

```text
ai-crisis-response/
├── vision_agent.py
├── decision_engine.py
├── alert_system.py
├── test_runner.py
├── fire_model.pt
├── yolov8n.pt
└── videos/
    ├── fire.mp4
    └── fall.mp4
```

## Usage

```python
from vision_agent import process_frame

event = process_frame(frame)
```

## Output

```python
{
    "fire": bool,
    "smoke": bool,
    "person": bool,
    "fall_detected": bool,
    "confidence": float,
}
```

## Run Pipeline

```bash
python test_runner.py --video videos/fire.mp4
python test_runner.py --video videos/fall.mp4
```

## Requirements

```bash
pip install ultralytics opencv-python
```
