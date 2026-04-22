# AI Crisis Response

Python-based crisis response pipeline with a real vision stage and a LangGraph orchestration stage:

```text
detection -> fusion -> risk -> decision -> action
```

## What Is In The Repo

```text
ai-crisis-response/
|-- vision_agent.py
|-- decision_engine.py
|-- alert_system.py
|-- pipeline_support.py
|-- test_runner.py
|-- main.py
|-- tests/
|-- .env.example
`-- requirements.txt
```

Local assets such as `.pt` model files and sample videos are intentionally not tracked.

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Copy `.env.example` to `.env`.
3. Set `GROQ_API_KEY` if you want the Groq-backed fusion and risk agents.
4. Make sure the YOLO model paths are valid:

```env
FIRE_MODEL_PATH=fire_model.pt
PERSON_MODEL_PATH=yolov8n.pt
```

You can also set `MODEL_DIR` and keep both model files there.

## Vision Event Schema

`vision_agent.process_frame(frame, camera_id="camera-0", frame_index=0)` returns:

```python
{
    "camera_id": "camera-0",
    "frame_index": 0,
    "fire": False,
    "smoke": False,
    "person": False,
    "fall_detected": False,
    "confidence": 0.0,
}
```

The frame is annotated in place, and event details are logged through `logging`.
Fire and smoke detection now comes only from the configured fire/smoke model; there is no color-based fallback.

## Run The Deterministic Pipeline

Use `test_runner.py` when you want the vision + decision + alert pipeline without the LangGraph layer:

```bash
python test_runner.py --video 0 --display
python test_runner.py --video path/to/video.mp4
```

## Run The LangGraph Pipeline

`main.py` now consumes live vision events instead of mock events. Stable non-safe detections are forwarded into the LangGraph flow:

```bash
python main.py --video 0 --display
python main.py --video path/to/video.mp4 --camera-id lobby-cam
```

If `GROQ_API_KEY` is not configured, the graph still runs with deterministic fallbacks for fusion and risk.

## Tests

```bash
pytest -q
```

Current test coverage includes event-shape validation for the vision pipeline and decision mapping checks.
