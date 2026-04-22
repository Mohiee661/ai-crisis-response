# AI Crisis Response

Python-based crisis response pipeline:

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
├── main.py
├── fire_model.pt
├── yolov8n.pt
└── videos/
    ├── fire.mp4
    └── fall.mp4
```

## Install

```bash
pip install -r requirements.txt
```

## Vision Usage

```python
from vision_agent import process_frame

event = process_frame(frame)
```

## Vision Output

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

## LLM Agent Demo

Create a local `.env` from `.env.example` and set `GROQ_API_KEY`, then run:

```bash
python main.py
```

Do not commit `.env` or model/video files.
