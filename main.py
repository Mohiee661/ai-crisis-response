import json
import os
import re
from typing import TypedDict

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END


history = []


class State(TypedDict, total=False):
    vision_event: dict
    crisis: str
    social: dict
    fusion_output: dict
    risk_output: dict
    decision_output: dict
    action_output: dict
    history: list


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


def create_llm() -> ChatGroq:
    load_dotenv()
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY is missing. Copy .env.example to .env and set your key.")

    return ChatGroq(
        temperature=0,
        model="llama-3.1-8b-instant",
    )


llm = create_llm()


def fusion_agent(state: State) -> State:
    if "vision_event" not in state or "social" not in state:
        raise ValueError(f"Missing input data: {state}")

    prompt = f"""
You are a crisis validation AI.

Vision system detected:
{state["vision_event"]}

Social signals:
{state["social"]}

Return only valid JSON:
{{
    "is_crisis": true,
    "confidence": "LOW/MEDIUM/HIGH",
    "reason": "short reason"
}}
"""

    response = llm.invoke(prompt)
    state["fusion_output"] = safe_parse(response.content)
    state["history"] = history
    history.append(
        {
            "vision": state["vision_event"],
            "social": state["social"],
            "fusion": state["fusion_output"],
        }
    )
    return state


def risk_agent(state: State) -> State:
    prompt = f"""
You are a risk assessment AI.

Current event:
{state["fusion_output"]}

Previous events:
{state.get("history", [])}

Rules:
- If multiple similar events occur, increase severity.
- If repeated high-confidence events occur, mark CRITICAL.

Return only valid JSON:
{{
    "severity": "LOW/MEDIUM/HIGH/CRITICAL",
    "urgency": "MONITOR/IMMEDIATE",
    "note": "short reason"
}}
"""

    response = llm.invoke(prompt)
    state["risk_output"] = safe_parse(response.content)
    return state


def decision_agent(state: State) -> State:
    crisis_type = state["crisis"]

    prompt = f"""
You are an emergency decision system.

Crisis type:
{crisis_type}

Risk:
{state["risk_output"]}

Rules:
- FIRE or GAS LEAK means EVACUATE.
- PERSON COLLAPSE or MEDICAL means ALERT.
- Only EVACUATE for large-scale danger.

Return only valid JSON:
{{
    "decision": "EVACUATE/ALERT/IGNORE",
    "priority": "LOW/MEDIUM/HIGH"
}}
"""

    response = llm.invoke(prompt)
    state["decision_output"] = safe_parse(response.content)
    return state


def action_agent(state: State) -> State:
    prompt = f"""
You are an emergency execution system.

Input:
{state["decision_output"]}

Return only valid JSON:
{{
    "tool": "send_alert/log_event",
    "message": "short action message"
}}
"""

    response = llm.invoke(prompt)
    state["action_output"] = safe_parse(response.content)
    return state


def build_graph():
    graph = StateGraph(State)
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


def prioritize(events: list[State]) -> list[State]:
    priority_score = {
        "EVACUATE": 3,
        "ALERT": 2,
        "IGNORE": 1,
    }
    crisis_weight = {
        "FIRE": 3,
        "GAS LEAK": 3,
        "MEDICAL": 2,
    }

    def score(event: State) -> int:
        decision_score = priority_score.get(event["decision_output"]["decision"], 0)
        crisis_score = crisis_weight.get(event["crisis"], 1)
        return decision_score + crisis_score

    return sorted(events, key=score, reverse=True)


def run_demo() -> None:
    app = build_graph()
    
    # Mock vision events matching the new structure
    events = [
        {
            "vision_event": {
                "fire": True,
                "smoke": False,
                "person": False,
                "fall_detected": False,
                "confidence": 0.9,
            },
            "social": {
                "crisis": "fire",
                "confidence": 0.8,
                "text": "fire in mall food court",
            },
        },
        {
            "vision_event": {
                "fire": False,
                "smoke": False,
                "person": True,
                "fall_detected": True,
                "confidence": 0.85,
            },
            "social": {
                "crisis": "medical emergency",
                "confidence": 0.9,
                "text": "someone fainted near entrance",
            },
        },
    ]

    results = []
    for index, raw_event in enumerate(events, start=1):
        print(f"\n--- EVENT {index} ---\n")
        
        # Inject dynamic crisis label
        from vision_agent import event_to_crisis
        raw_event["crisis"] = event_to_crisis(raw_event["vision_event"])
        
        result = app.invoke(raw_event)
        results.append(result)
        print(result)

    print("\n--- PRIORITIZED RESPONSE ---\n")
    for event in prioritize(results):
        crisis = event.get("crisis", "UNKNOWN")
        decision = event.get("decision_output", {}).get("decision", "NO_DECISION")
        tool = event.get("action_output", {}).get("tool", "NO_TOOL")
        print(f"{crisis} -> {decision} | {tool}")


if __name__ == "__main__":
    run_demo()
