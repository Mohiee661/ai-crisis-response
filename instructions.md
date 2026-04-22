# AI Crisis Response Instructions

## Runtime Flow

```text
detection -> fusion -> risk -> decision -> action
```

- `vision_agent.py` handles detection and returns a normalized `VisionEvent`.
- `main.py` takes stable live events and runs them through the LangGraph pipeline.
- `test_runner.py` runs the deterministic non-LLM path.

## Vision Agent

Use:

```python
from vision_agent import process_frame

event = process_frame(frame, camera_id="camera-0", frame_index=12)
```

Event format:

```python
{
    "camera_id": "camera-0",
    "frame_index": 12,
    "fire": False,
    "smoke": False,
    "person": False,
    "fall_detected": False,
    "confidence": 0.0,
}
```

Notes:

- Model loading is lazy and configurable through `FIRE_MODEL_PATH`, `PERSON_MODEL_PATH`, or `MODEL_DIR`.
- Fire and smoke detection rely only on the configured fire/smoke YOLO model.
- The module logs through `logging`; it does not print raw event lines directly.
- `camera_id` and `frame_index` are included so the pipeline can be extended to multiple camera streams.

## LangGraph Pipeline

`main.py` accepts real detections from `vision_agent.process_frame(...)`.

- Fusion uses the live vision event plus a small context object.
- Risk considers recent camera-local history.
- Decision maps the normalized danger state to the existing alert actions.
- Action produces the final dispatch payload used by `alert_system.py`.

Run it with:

```bash
python main.py --video 0 --display
python main.py --video path/to/video.mp4 --camera-id entrance-cam
```

If Groq credentials are missing, the pipeline falls back to deterministic fusion and risk outputs.

## Deterministic Runner

Run:

```bash
python test_runner.py --video 0 --display
python test_runner.py --video path/to/video.mp4
```

This path still applies 3-frame temporal stability before mapping to alerts.

## Local Assets

The repository does not require bundled sample videos.

- Use `--video 0` for a webcam.
- Use `--video path/to/file.mp4` for a local video.
- Keep `.pt` model files local; they are ignored by Git.

## Tests

```bash
pytest -q
```

Included tests cover:

- normalized vision event output
- danger and action mapping
