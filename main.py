import json
import os
import re
from typing import TypedDict

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph


history = []


class State(TypedDict, total=False):
    vision: dict
    social: dict
    fusion: dict
    risk: dict
    decision: dict
    action: dict
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
    if "vision" not in state or "social" not in state:
        raise ValueError(f"Missing input data: {state}")

    prompt = f"""
You are a crisis validation AI.

Vision system detected:
{state["vision"]}

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
    state["fusion"] = safe_parse(response.content)
    state["history"] = history
    history.append(
        {
            "vision": state["vision"],
            "social": state["social"],
            "fusion": state["fusion"],
        }
    )
    return state


def risk_agent(state: State) -> State:
    prompt = f"""
You are a risk assessment AI.

Current event:
{state["fusion"]}

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
    state["risk"] = safe_parse(response.content)
    return state


def decision_agent(state: State) -> State:
    crisis_type = state["vision"]["crisis"]

    prompt = f"""
You are an emergency decision system.

Crisis type:
{crisis_type}

Risk:
{state["risk"]}

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
    state["decision"] = safe_parse(response.content)
    return state


def action_agent(state: State) -> State:
    prompt = f"""
You are an emergency execution system.

Input:
{state["decision"]}

Return only valid JSON:
{{
    "tool": "send_alert/log_event",
    "message": "short action message"
}}
"""

    response = llm.invoke(prompt)
    state["action"] = safe_parse(response.content)
    return state


def build_graph():
    graph = StateGraph(State)
    graph.add_node("fusion", fusion_agent)
    graph.add_node("risk", risk_agent)
    graph.add_node("decision", decision_agent)
    graph.add_node("action", action_agent)
    graph.set_entry_point("fusion")
    graph.add_edge("fusion", "risk")
    graph.add_edge("risk", "decision")
    graph.add_edge("decision", "action")
    return graph.compile()


def prioritize(events: list[State]) -> list[State]:
    priority_score = {
        "EVACUATE": 3,
        "ALERT": 2,
        "IGNORE": 1,
    }
    crisis_weight = {
        "fire": 3,
        "gas leak": 3,
        "person collapse": 2,
        "medical emergency": 2,
    }

    def score(event: State) -> int:
        decision_score = priority_score.get(event["decision"]["decision"], 0)
        crisis_score = crisis_weight.get(event["vision"]["crisis"], 1)
        return decision_score + crisis_score

    return sorted(events, key=score, reverse=True)


def run_demo() -> None:
    app = build_graph()
    events = [
        {
            "vision": {
                "crisis": "fire",
                "confidence": 0.9,
                "location": "camera_1",
            },
            "social": {
                "crisis": "fire",
                "confidence": 0.8,
                "text": "fire in mall food court",
            },
        },
        {
            "vision": {
                "crisis": "person collapse",
                "confidence": 0.85,
                "location": "camera_2",
            },
            "social": {
                "crisis": "medical emergency",
                "confidence": 0.9,
                "text": "someone fainted near entrance",
            },
        },
    ]

    results = []
    for index, event in enumerate(events, start=1):
        print(f"\n--- EVENT {index} ---\n")
        result = app.invoke(event)
        results.append(result)
        print(result)

    print("\n--- PRIORITIZED RESPONSE ---\n")
    for event in prioritize(results):
        print(event["vision"]["crisis"], "->", event["decision"]["decision"], "|", event["action"]["tool"])


if __name__ == "__main__":
    run_demo()
