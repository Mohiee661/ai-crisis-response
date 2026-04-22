from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

from alert_system import send_alert
from decision_engine import decision_engine, detect_danger
from pipeline_support import (
    DEFAULT_CAMERA_ID,
    CrisisState,
    SocialSignal,
    TemporalEventTracker,
    configure_logging,
    get_env_bool,
    log_payload,
    normalize_video_source,
)
from social_agent import fetch_social_signals
from vision_agent import process_frame
import api_server


load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_VIDEO_SOURCE = os.getenv("CRISIS_VIDEO_SOURCE", "0")
DEFAULT_LOG_LEVEL = os.getenv("CRISIS_LOG_LEVEL", "INFO")
DEFAULT_GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
DEFAULT_LOCATION = os.getenv("CRISIS_LOCATION", "demo-site")
DEFAULT_DEMO_MODE = get_env_bool("DEMO_MODE", False)
MAX_HISTORY_ITEMS = 25

history_by_camera: dict[str, list[dict[str, Any]]] = defaultdict(list)
_llm: Any | None = None
_llm_disabled = False


def safe_parse(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"{.*}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return {"raw": text}


def confidence_band(score: float) -> str:
    if score >= 0.80:
        return "HIGH"
    if score >= 0.50:
        return "MEDIUM"
    return "LOW"


def normalize_choice(value: Any, allowed: set[str], fallback: str) -> str:
    candidate = str(value).upper()
    return candidate if candidate in allowed else fallback


def create_llm():
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None

    try:
        from langchain_groq import ChatGroq
    except ImportError:
        return None

    try:
        return ChatGroq(temperature=0, model=os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL))
    except Exception:
        return None


def get_llm():
    global _llm, _llm_disabled
    if _llm is not None:
        return _llm
    if _llm_disabled:
        return None

    _llm = create_llm()
    if _llm is None:
        _llm_disabled = True
        logger.warning(
            "Groq client unavailable; using deterministic fallbacks for fusion and risk agents."
        )
    return _llm


def invoke_json_agent(prompt: str, fallback: dict) -> dict:
    llm = get_llm()
    if llm is None:
        return fallback

    try:
        response = llm.invoke(prompt)
    except Exception as exc:
        logger.warning("LLM invocation failed; using fallback output: %s", exc)
        return fallback

    parsed = safe_parse(response.content)
    return parsed if "raw" not in parsed else fallback


def build_social_query(vision_event: dict) -> str:
    if detect_danger(vision_event) in {"FIRE", "BOTH"}:
        return "fire smoke emergency"
    if vision_event.get("validation_state") == "MONITOR":
        return "possible fire smoke report"
    if vision_event["fall_detected"]:
        return "medical emergency person fall"
    return "no active crisis"


def build_social_signal(vision_event: dict) -> SocialSignal:
    location = os.getenv("CRISIS_LOCATION", DEFAULT_LOCATION)
    social_signal = fetch_social_signals(build_social_query(vision_event), location=location)
    return {
        "type": social_signal["type"],
        "confidence": round(float(social_signal["confidence"]), 2),
        "source": social_signal.get("source", "social"),
        "text": social_signal["text"],
        "location": location,
    }


def build_initial_state(vision_event: dict, social: SocialSignal | None = None) -> CrisisState:
    return {
        "camera_id": vision_event["camera_id"],
        "frame_index": vision_event["frame_index"],
        "vision_event": vision_event,
        "social": social or build_social_signal(vision_event),
        "crisis": detect_danger(vision_event),
    }


def build_fusion_fallback(state: CrisisState) -> dict:
    crisis = state["crisis"]
    vision_event = state["vision_event"]
    social_signal = state["social"]
    validation_state = vision_event.get("validation_state", "SAFE")
    validation_reason = vision_event.get("validation_reason", "no validation details")
    social_type = normalize_choice(
        social_signal.get("type", "SAFE"),
        {"SAFE", "FIRE", "MEDICAL"},
        "SAFE",
    )

    vision_fire_confirmed = crisis in {"FIRE", "BOTH"}
    vision_medical_confirmed = crisis in {"MEDICAL", "BOTH"}
    vision_fire_candidate = (
        vision_fire_confirmed
        or validation_state == "MONITOR"
        or vision_event.get("raw_fire_detected", False)
        or vision_event.get("raw_smoke_detected", False)
    )
    social_fire = social_type == "FIRE" and social_signal["confidence"] >= 0.50
    social_medical = social_type == "MEDICAL" and social_signal["confidence"] >= 0.50

    if social_fire and vision_fire_candidate:
        return {
            "is_crisis": True,
            "crisis_type": "BOTH" if crisis == "BOTH" else "FIRE",
            "confidence": "HIGH",
            "reason": "Vision and social signals corroborate a fire event.",
        }

    if social_medical and vision_medical_confirmed:
        return {
            "is_crisis": True,
            "crisis_type": crisis,
            "confidence": "HIGH",
            "reason": "Vision and social signals corroborate a medical event.",
        }

    if vision_fire_confirmed or vision_medical_confirmed:
        return {
            "is_crisis": True,
            "crisis_type": crisis,
            "confidence": "MEDIUM",
            "reason": f"Single confirmed vision signal: {validation_reason}.",
        }

    if social_fire or social_medical or validation_state == "MONITOR":
        return {
            "is_crisis": False,
            "crisis_type": "SAFE",
            "confidence": "MEDIUM",
            "reason": (
                "Single weak signal only; keep monitoring. "
                f"Vision reason: {validation_reason}. Social signal: {social_signal['text']}"
            ),
        }

    return {
        "is_crisis": False,
        "crisis_type": "SAFE",
        "confidence": "LOW",
        "reason": "No corroborated crisis signal detected.",
    }


def build_risk_fallback(state: CrisisState) -> dict:
    history = state.get("history", [])
    repeated_similar_events = sum(
        1
        for item in history
        if item.get("crisis") == state["crisis"]
    )
    crisis = state["crisis"]

    if crisis == "BOTH":
        severity = "CRITICAL"
    elif crisis in {"FIRE", "MEDICAL"} and (
        repeated_similar_events >= 2 or state["vision_event"]["confidence"] >= 0.80
    ):
        severity = "HIGH"
    elif crisis in {"FIRE", "MEDICAL"}:
        severity = "MEDIUM"
    else:
        severity = "LOW"

    urgency = "IMMEDIATE" if severity in {"HIGH", "CRITICAL"} else "MONITOR"
    return {
        "severity": severity,
        "urgency": urgency,
        "note": f"{repeated_similar_events} similar event(s) observed for {state['camera_id']}.",
    }


def build_decision_output(state: CrisisState) -> dict:
    danger = normalize_choice(
        state["fusion_output"].get("crisis_type", state["crisis"]),
        {"SAFE", "FIRE", "MEDICAL", "BOTH"},
        state["crisis"],
    )
    action = decision_engine(danger)
    severity = state["risk_output"]["severity"]
    priority = "HIGH" if severity in {"HIGH", "CRITICAL"} else "MEDIUM"
    if action == "NO_ACTION":
        priority = "LOW"

    return {
        "danger": danger,
        "action": action,
        "priority": priority,
    }


def build_action_output(state: CrisisState) -> dict:
    action = state["decision_output"]["action"]
    danger = state["decision_output"]["danger"]
    tool = "send_alert" if action != "NO_ACTION" else "log_event"

    if action == "NO_ACTION":
        message = (
            f"No emergency escalation required for {state['camera_id']} "
            f"frame {state['frame_index']}."
        )
    else:
        message = (
            f"{danger} detected on {state['camera_id']} at frame {state['frame_index']}; "
            f"dispatching {action.lower()}."
        )

    return {
        "action": action,
        "tool": tool,
        "message": message,
    }


def fusion_agent(state: CrisisState) -> CrisisState:
    history = history_by_camera[state["camera_id"]]
    state["history"] = list(history)
    fallback = build_fusion_fallback(state)

    prompt = f"""
You are a crisis validation AI.

Vision event JSON:
{json.dumps(state["vision_event"], sort_keys=True)}

Context JSON:
{json.dumps(state["social"], sort_keys=True)}

Rules:
- If vision and social confirm the same crisis, return HIGH confidence.
- If only one signal is present or the vision signal is monitor-only, keep the result conservative.
- If there is no corroborated signal, return SAFE.

Return only valid JSON:
{{
    "is_crisis": true,
    "crisis_type": "SAFE/FIRE/MEDICAL/BOTH",
    "confidence": "LOW/MEDIUM/HIGH",
    "reason": "short reason"
}}
"""

    parsed = invoke_json_agent(prompt, fallback)
    state["fusion_output"] = {
        "is_crisis": bool(parsed.get("is_crisis", fallback["is_crisis"])),
        "crisis_type": normalize_choice(
            parsed.get("crisis_type", fallback["crisis_type"]),
            {"SAFE", "FIRE", "MEDICAL", "BOTH"},
            fallback["crisis_type"],
        ),
        "confidence": normalize_choice(
            parsed.get("confidence", fallback["confidence"]),
            {"LOW", "MEDIUM", "HIGH"},
            fallback["confidence"],
        ),
        "reason": str(parsed.get("reason", fallback["reason"])),
    }

    history.append(
        {
            "frame_index": state["frame_index"],
            "crisis": state["fusion_output"]["crisis_type"],
            "confidence": state["fusion_output"]["confidence"],
        }
    )
    if len(history) > MAX_HISTORY_ITEMS:
        del history[:-MAX_HISTORY_ITEMS]
    return state


def risk_agent(state: CrisisState) -> CrisisState:
    fallback = build_risk_fallback(state)
    prompt = f"""
You are a risk assessment AI.

Fusion output JSON:
{json.dumps(state["fusion_output"], sort_keys=True)}

Recent history JSON:
{json.dumps(state.get("history", []), sort_keys=True)}

Return only valid JSON:
{{
    "severity": "LOW/MEDIUM/HIGH/CRITICAL",
    "urgency": "MONITOR/IMMEDIATE",
    "note": "short reason"
}}
"""

    parsed = invoke_json_agent(prompt, fallback)
    state["risk_output"] = {
        "severity": normalize_choice(
            parsed.get("severity", fallback["severity"]),
            {"LOW", "MEDIUM", "HIGH", "CRITICAL"},
            fallback["severity"],
        ),
        "urgency": normalize_choice(
            parsed.get("urgency", fallback["urgency"]),
            {"MONITOR", "IMMEDIATE"},
            fallback["urgency"],
        ),
        "note": str(parsed.get("note", fallback["note"])),
    }
    return state


def decision_agent(state: CrisisState) -> CrisisState:
    state["decision_output"] = build_decision_output(state)
    return state


def action_agent(state: CrisisState) -> CrisisState:
    state["action_output"] = build_action_output(state)
    return state


def build_graph():
    graph = StateGraph(CrisisState)
    graph.add_node("fusion_node", fusion_agent)
    graph.add_node("risk_node", risk_agent)
    graph.add_node("decision_node", decision_agent)
    graph.add_node("action_node", action_agent)
    graph.set_entry_point("fusion_node")
    graph.add_edge("fusion_node", "risk_node")
    graph.add_edge("risk_node", "decision_node")
    graph.add_edge("decision_node", "action_node")
    graph.add_edge("action_node", END)
    return graph.compile()


def prioritize(events: list[CrisisState]) -> list[CrisisState]:
    action_score = {
        "ALERT_BOTH": 4,
        "ALERT_FIRE_STATION": 3,
        "ALERT_AMBULANCE": 2,
        "NO_ACTION": 1,
    }
    priority_score = {
        "HIGH": 3,
        "MEDIUM": 2,
        "LOW": 1,
    }

    def score(event: CrisisState) -> int:
        action = event["action_output"]["action"]
        priority = event["decision_output"]["priority"]
        return action_score.get(action, 0) + priority_score.get(priority, 0)

    return sorted(events, key=score, reverse=True)


def discover_demo_videos() -> list[str]:
    demo_glob = os.getenv("DEMO_VIDEO_GLOB", "videos/*.mp4")
    return sorted(glob.glob(demo_glob))


def run_demo_cycle(display: bool = True, max_frames: int | None = None) -> None:
    demo_videos = discover_demo_videos()
    if not demo_videos:
        raise RuntimeError("DEMO_MODE is enabled, but no demo videos were found.")

    for video_path in demo_videos:
        camera_id = Path(video_path).stem.replace(" ", "-")
        log_payload(
            logger,
            logging.INFO,
            "demo_video_start",
            {
                "camera_id": camera_id,
                "video_path": video_path,
            },
        )
        run_live_graph(
            video_source=video_path,
            camera_id=camera_id,
            display=display,
            max_frames=max_frames,
        )


def run_validation_mode():
    videos = ["videos/fire.mp4", "videos/fall.mp4"]
    summary = {
        "total_frames": 0,
        "fire_frames": 0,
        "false_fire_frames": 0,
        "fall_frames": 0
    }
    
    tracker = TemporalEventTracker()
    for video_path in videos:
        if not os.path.exists(video_path):
            logger.error(f"Validation video not found: {video_path}")
            continue
            
        camera_id = Path(video_path).stem
        cap = cv2.VideoCapture(normalize_video_source(video_path))
        frame_index = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            live_event = process_frame(frame, camera_id=camera_id, frame_index=frame_index)
            stable_event = tracker.stabilize(live_event, frame)
            
            summary["total_frames"] += 1
            if stable_event.get("fire", False) and stable_event.get("validation_state", "SAFE") == "ALERT":
                if "fire" in camera_id.lower():
                    summary["fire_frames"] += 1
                else:
                    summary["false_fire_frames"] += 1
                    
            if stable_event.get("person", False) or stable_event.get("fall_detected", False):
                if "fall" in camera_id.lower():
                    summary["fall_frames"] += 1
                
            print(f"{camera_id} Frame {frame_index}: Fire={stable_event.get('fire')} Smoke={stable_event.get('smoke')} Person={stable_event.get('person')} Conf={stable_event.get('confidence', 0):.2f} State={stable_event.get('validation_state')}")
            frame_index += 1
            
        cap.release()
        
    print("\nValidation Summary:")
    print(json.dumps(summary, indent=2))


def run_live_graph(
    video_source: str,
    camera_id: str = DEFAULT_CAMERA_ID,
    display: bool = False,
    max_frames: int | None = None,
) -> None:
    app = build_graph()
    tracker = TemporalEventTracker()
    cap = cv2.VideoCapture(normalize_video_source(video_source))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video source: {video_source}")

    api_server.start_server()
    FRAME_SKIP = 2

    frame_index = 0
    try:
        while True:
            if max_frames is not None and frame_index >= max_frames:
                break

            ret, frame = cap.read()
            if not ret:
                break

            live_event = process_frame(frame, camera_id=camera_id, frame_index=frame_index)
            stable_event = tracker.stabilize(live_event, frame)
            danger = detect_danger(stable_event)
            attention_state = danger if danger != "SAFE" else stable_event.get("validation_state", "SAFE")

            log_payload(
                logger,
                logging.INFO if attention_state != "SAFE" else logging.DEBUG,
                "stable_event",
                stable_event,
            )

            if tracker.should_invoke_graph(camera_id, attention_state):
                result = app.invoke(build_initial_state(stable_event))
                log_payload(
                    logger,
                    logging.INFO,
                    "graph_result",
                    {
                        "camera_id": camera_id,
                        "frame_index": frame_index,
                        "fusion_output": result["fusion_output"],
                        "risk_output": result["risk_output"],
                        "decision_output": result["decision_output"],
                        "action_output": result["action_output"],
                    },
                )

                action = result["action_output"]["action"]
                if tracker.should_send_alert(camera_id, action):
                    send_alert(action, camera_id=camera_id, details=result["action_output"]["message"])

            if display:
                cv2.namedWindow("AI Crisis Response Graph", cv2.WINDOW_NORMAL)
                cv2.imshow("AI Crisis Response Graph", frame)
                if cv2.waitKey(30) == 27:
                    break

            if frame_index % FRAME_SKIP == 0:
                api_server.update_state({
                    "frame": api_server.encode_frame(frame),
                    "metrics": {
                        "confidence": stable_event.get("confidence", 0.0),
                        "status": attention_state
                    }
                })

            frame_index += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the live crisis-response LangGraph pipeline.")
    parser.add_argument(
        "--video",
        default=DEFAULT_VIDEO_SOURCE,
        help="Path to a video file or webcam index such as 0.",
    )
    parser.add_argument(
        "--camera-id",
        default=DEFAULT_CAMERA_ID,
        help="Logical camera identifier used for state tracking and alerts.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap for frames processed during a run.",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Display the annotated video stream while processing.",
    )
    parser.add_argument(
        "--log-level",
        default=DEFAULT_LOG_LEVEL,
        help="Logging level (DEBUG, INFO, WARNING, ERROR).",
    )
    parser.add_argument(
        "--demo-mode",
        action="store_true",
        help="Cycle through the configured demo videos instead of a single input source.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run strict validation suite without invoking LLM/LangGraph.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.demo_mode or DEFAULT_DEMO_MODE:
        args.log_level = "WARNING"
    configure_logging(args.log_level)
    if args.validate:
        run_validation_mode()
    elif args.demo_mode or DEFAULT_DEMO_MODE:
        run_demo_cycle(display=args.display, max_frames=args.max_frames)
    else:
        run_live_graph(
            video_source=args.video,
            camera_id=args.camera_id,
            display=args.display,
            max_frames=args.max_frames,
        )
