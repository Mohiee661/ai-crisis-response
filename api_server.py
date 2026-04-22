import base64
import threading
import cv2
from fastapi import FastAPI
import uvicorn

app = FastAPI()

state_lock = threading.Lock()
latest_state = {
    "frame": None,
    "alerts": [],
    "logs": [],
    "metrics": {}
}

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
        return latest_state

def start_server():
    def run():
        uvicorn.run(app, host="0.0.0.0", port=8080, log_level="warning")
    
    t = threading.Thread(target=run, daemon=True)
    t.start()
