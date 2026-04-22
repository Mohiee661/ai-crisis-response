import os
import json
import re
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph
from typing import TypedDict


# ----------- MEMORY -----------

history = []


# ----------- STATE -----------

class State(TypedDict, total=False):
    vision: dict
    social: dict
    fusion: dict
    risk: dict
    decision: dict
    action: dict
    history: list


# ----------- SETUP -----------

load_dotenv()

llm = ChatGroq(
    temperature=0,
    model="llama-3.1-8b-instant"
)


# ----------- SAFE PARSE -----------

def safe_parse(text):
    try:
        return json.loads(text)
    except:
        try:
            json_str = re.search(r"{.*}", text, re.DOTALL).group()
            return json.loads(json_str)
        except:
            return {"raw": text}


# ----------- FUSION AGENT -----------

def fusion_agent(state):
    if "vision" not in state or "social" not in state:
        raise ValueError(f"Missing input data: {state}")

    data = state

    prompt = f"""
You are a crisis validation AI.

Vision system detected:
{data['vision']}

Social signals:
{data['social']}

STRICT RULES:

* Return ONLY valid JSON
* No explanation

Format:
{{
    "is_crisis": true/false,
    "confidence": "LOW/MEDIUM/HIGH",
    "reason": "short reason"
}}
"""

    response = llm.invoke(prompt)
    state["fusion"] = safe_parse(response.content)

    # Add memory
    state["history"] = history
    history.append({
        "vision": data["vision"],
        "social": data["social"],
        "fusion": state["fusion"]
    })

    return state


# ----------- RISK AGENT -----------

# ----------- RISK AGENT -----------

def risk_agent(state):
    fusion_output = state["fusion"]

    prompt = f"""
You are a risk assessment AI.

Current event:
{fusion_output}

Previous events:
{state.get("history", [])}

RULES:
- If multiple similar events → increase severity
- If repeated high confidence → CRITICAL

STRICT RULES:
- Return ONLY valid JSON
- No explanation
- No markdown
- No text before or after JSON

Format:
{{
  "severity": "LOW/MEDIUM/HIGH/CRITICAL",
  "urgency": "MONITOR/IMMEDIATE",
  "note": "short reason"
}}
"""

    response = llm.invoke(prompt)
    state["risk"] = safe_parse(response.content)
    return state
# ----------- DECISION AGENT -----------

def decision_agent(state):
    risk_output = state["risk"]
    crisis_type = state["vision"]["crisis"]

    prompt = f"""
You are an emergency decision system.

Crisis type:
{crisis_type}

Risk:
{risk_output}

RULES:
- FIRE or GAS LEAK → EVACUATE
- PERSON COLLAPSE or MEDICAL → ALERT (call emergency responders)
- Only EVACUATE if large-scale danger

STRICT RULES:
- Return ONLY valid JSON
- No explanation

Format:
{{
    "decision": "EVACUATE/ALERT/IGNORE",
    "priority": "LOW/MEDIUM/HIGH"
}}
"""

    response = llm.invoke(prompt)
    state["decision"] = safe_parse(response.content)
    return state
# ----------- ACTION AGENT -----------

def action_agent(state):
    decision_output = state["decision"]

    prompt = f"""
You are an emergency execution system.

Input:
{decision_output}

STRICT RULES:

* Return ONLY valid JSON

Format:
{{
    "tool": "send_alert/log_event",
    "message": "short action message"
}}
"""

    response = llm.invoke(prompt)
    state["action"] = safe_parse(response.content)
    return state


# ----------- GRAPH -----------

graph = StateGraph(State)

graph.add_node("fusion", fusion_agent)
graph.add_node("risk", risk_agent)
graph.add_node("decision", decision_agent)
graph.add_node("action", action_agent)

graph.set_entry_point("fusion")

graph.add_edge("fusion", "risk")
graph.add_edge("risk", "decision")
graph.add_edge("decision", "action")

app = graph.compile()


# ----------- INPUT -----------
events = [
    {
        "vision": {
            "crisis": "fire",
            "confidence": 0.9,
            "location": "camera_1"
        },
        "social": {
            "crisis": "fire",
            "confidence": 0.8,
            "text": "fire in mall food court"
        }
    },
    {
        "vision": {
            "crisis": "person collapse",
            "confidence": 0.85,
            "location": "camera_2"
        },
        "social": {
            "crisis": "medical emergency",
            "confidence": 0.9,
            "text": "someone fainted near entrance"
        }
    }
]

# ----------- RUN MULTIPLE EVENTS -----------

# ----------- RUN MULTIPLE EVENTS -----------

results = []

for i, event in enumerate(events):
    print(f"\n--- EVENT {i+1} ---\n")

    result = app.invoke(event)
    results.append(result)

    print(result)


# ----------- PRIORITIZATION -----------

def prioritize(events):
    priority_score = {
        "EVACUATE": 3,
        "ALERT": 2,
        "IGNORE": 1
    }

    crisis_weight = {
        "fire": 3,
        "gas leak": 3,
        "person collapse": 2,
        "medical emergency": 2
    }

    def score(e):
        decision_score = priority_score.get(e["decision"]["decision"], 0)
        crisis = e["vision"]["crisis"]
        crisis_score = crisis_weight.get(crisis, 1)

        return decision_score + crisis_score

    return sorted(events, key=score, reverse=True)


print("\n--- PRIORITIZED RESPONSE ---\n")

sorted_events = prioritize(results)

for e in sorted_events:
    print(
        e["vision"]["crisis"],
        "→",
        e["decision"]["decision"],
        "|",
        e["action"]["tool"]
    )