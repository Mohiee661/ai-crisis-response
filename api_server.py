import base64
import threading
import cv2
from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from datetime import datetime

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

state_lock = threading.Lock()
latest_state = {
    "frame": None,
    "decision": None,       # "ALERT_AMBULANCE" | "ALERT_FIRE_ENGINE" | "IGNORE"
    "severity": None,       # "LOW" | "MEDIUM" | "HIGH" | "CRITICAL"
    "trust_score": 0.0,
    "location": "Unknown",
    "current_video": "None",
    "lifecycle_state": "MONITORING", # MONITORING, DETECTED, CONFIRMED, DISPATCHED
    "confidence_explanation": [],
    "decision_reason": None,
    "signals": [],
    "llm_summary": None,
    "llm_link": None,
    "llm_confirmation": None,
    "system_health": {
        "model_status": "ACTIVE",
        "camera_status": "CONNECTED",
        "api_status": "ONLINE",
        "latency": "45ms"
    },
    "logs": [],
    "incident_locked": False,
}

_simulated_signals = []
_sim_lock = threading.Lock()

def encode_frame(frame):
    if frame is None:
        return None
    _, buf = cv2.imencode(".jpg", frame)
    return base64.b64encode(buf).decode()

def update_state(data):
    with state_lock:
        latest_state.update(data)

@app.get("/status")
def get_status():
    with state_lock:
        # Simulate slight latency variation
        import random
        latest_state["system_health"]["latency"] = f"{random.randint(40, 52)}ms"
        return latest_state

@app.post("/simulate")
def simulate_signal():
    with _sim_lock:
        _simulated_signals.append({
            "source": "Manual Trigger",
            "text": "Operator confirmed visual anomaly via dashboard.",
            "confidence": 0.90,
        })
    return {"ok": True}

@app.post("/reset")
def reset_incident():
    with state_lock:
        latest_state.update({
            "decision": None,
            "severity": None,
            "trust_score": 0.0,
            "location": "Unknown",
            "lifecycle_state": "MONITORING",
            "confidence_explanation": [],
            "decision_reason": None,
            "signals": [],
            "llm_summary": None,
            "llm_link": None,
            "llm_confirmation": None,
            "logs": [],
            "incident_locked": False,
        })
    return {"ok": True}

@app.post("/override")
def manual_override(payload: dict = Body(...)):
    action = payload.get("action")
    with state_lock:
        latest_state.update({
            "decision": action,
            "lifecycle_state": "DISPATCHED" if action != "IGNORE" else "MONITORING",
            "incident_locked": True if action != "IGNORE" else False,
            "decision_reason": f"Manual override by operator: {action}",
            "logs": latest_state["logs"] + [f"[Manual] Operator overrode system with: {action}"]
        })
    return {"ok": True}

def pop_simulated_signals():
    with _sim_lock:
        out = list(_simulated_signals)
        _simulated_signals.clear()
        return out

def start_server():
    def run():
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
    t = threading.Thread(target=run, daemon=True)
    t.start()
