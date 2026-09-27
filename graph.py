import time
from typing import TypedDict

from langgraph.graph import END, StateGraph
from openai import OpenAI, RateLimitError

import actions
from config import COMPUTER_USE_MODEL, MAX_STEPS, OPENAI_API_KEY
from memory import recall_similar_tasks, save_task_result
from perception import screenshot_data_url

client = OpenAI(api_key=OPENAI_API_KEY)

# the computer-use model speaks its own key names; map to what pyautogui expects.
_KEY_ALIASES = {
    "RETURN": "enter",
    "ENTER": "enter",
    "ESC": "escape",
    "ESCAPE": "escape",
    "META": "win",
    "CMD": "win",
    "SUPER": "win",
}

DECIDE_INSTRUCTIONS = """You control a Windows computer via the computer tool. Given the goal
and memory of similar past tasks, take actions toward the goal using the computer tool.

To open an installed application: press the Windows key, type the app name, then press
Enter.

To open a website, do NOT hunt for a browser icon by clicking blindly. Use the Run dialog:
press Win+R, type the raw URL only (e.g. "https://mail.google.com", never prefixed with
"start" or any command name), then press Enter.

Once the goal is fully and visibly achieved, reply with a plain text message starting with
"DONE" (no more tool calls). If the goal looks impossible, or you keep repeating an action
that keeps failing, reply with a plain text message starting with "FAIL: <reason>".

SAFETY CHECKPOINT: before performing any action that is hard to undo or has a real-world
effect outside this screen — sending an email/message, deleting or overwriting a file,
submitting a form, making a payment or purchase, confirming an order, posting something
publicly, agreeing to terms, or anything similarly consequential — STOP before doing it.
Do not click Send/Delete/Submit/Confirm/Pay/Post yet. Instead reply with a plain text
message starting with "CONFIRM: " followed by a one-sentence description of exactly what
you are about to do (e.g. "CONFIRM: send the email to jane@example.com with subject X").
Wait for approval before proceeding. You will receive either "approved, proceed" (go ahead
with exactly that action) or "not approved" (do not perform it; ask for a different
approach or reply FAIL if there is none). Never skip this checkpoint for a risky action."""


class AgentState(TypedDict):
    goal: str
    history: list[dict]
    response_id: str | None
    call_id: str | None
    scale: float
    resume_reply: str | None
    status: str  # "running" | "done" | "failed" | "awaiting_confirmation"


def _create_response(**kwargs):
    for attempt in range(3):
        try:
            return client.responses.create(model=COMPUTER_USE_MODEL, tools=[{"type": "computer"}], **kwargs)
        except RateLimitError:
            if attempt == 2:
                raise
            time.sleep(5 * (attempt + 1))


def decide_node(state: AgentState) -> dict:
    if state.get("resume_reply"):
        resp = _create_response(
            previous_response_id=state["response_id"],
            input=[{"role": "user", "content": [{"type": "input_text", "text": state["resume_reply"]}]}],
        )
        scale = state["scale"]
    elif state["response_id"] is None:
        memory_hits = recall_similar_tasks(state["goal"])
        memory_text = "\n".join(memory_hits) if memory_hits else "(no relevant past tasks)"
        user_text = f"Goal: {state['goal']}\nPast similar tasks:\n{memory_text}"
        resp = _create_response(
            instructions=DECIDE_INSTRUCTIONS,
            input=[{"role": "user", "content": [{"type": "input_text", "text": user_text}]}],
        )
        scale = state["scale"]
    else:
        data_url, scale, _w, _h = screenshot_data_url()
        resp = _create_response(
            previous_response_id=state["response_id"],
            input=[
                {
                    "type": "computer_call_output",
                    "call_id": state["call_id"],
                    "output": {"type": "computer_screenshot", "image_url": data_url},
                }
            ],
        )

    call_item = next((item for item in resp.output if item.type == "computer_call"), None)
    if call_item is None:
        message_item = next((item for item in resp.output if item.type == "message"), None)
        text = ""
        if message_item is not None:
            text = "".join(c.text for c in message_item.content if hasattr(c, "text"))
        upper = text.strip().upper()
        if upper.startswith("FAIL"):
            status = "failed"
        elif "?" in text:
            # the model is asking permission/clarification (e.g. a CONFIRM
            # checkpoint) rather than reporting a finished outcome — always
            # route questions to the human instead of guessing.
            status = "awaiting_confirmation"
        else:
            # declarative completion report (whether or not it says "DONE"
            # verbatim) with no pending question — nothing left to confirm.
            status = "done"
        return {
            "status": status,
            "response_id": resp.id,
            "resume_reply": None,
            "history": state["history"] + [{"final_message": text}],
        }

    return {
        "status": "running",
        "response_id": resp.id,
        "call_id": call_item.call_id,
        "scale": scale,
        "resume_reply": None,
        "history": state["history"] + [{"pending_actions": [a.model_dump() for a in call_item.actions]}],
    }


def _map_action(raw: dict, scale: float) -> dict | None:
    kind = raw["type"]
    if kind == "screenshot":
        return None  # handled automatically by the next decide_node turn
    if kind == "click":
        return {"type": "click", "x": round(raw["x"] * scale), "y": round(raw["y"] * scale)}
    if kind == "move":
        return {"type": "move", "x": round(raw["x"] * scale), "y": round(raw["y"] * scale)}
    if kind == "double_click":
        return {"type": "double_click", "x": round(raw["x"] * scale), "y": round(raw["y"] * scale)}
    if kind == "scroll":
        return {
            "type": "scroll",
            "x": round(raw["x"] * scale),
            "y": round(raw["y"] * scale),
            "scroll_x": raw.get("scroll_x", 0),
            "scroll_y": raw.get("scroll_y", 0),
        }
    if kind == "drag":
        path = [{"x": round(p["x"] * scale), "y": round(p["y"] * scale)} for p in raw.get("path", [])]
        return {"type": "drag", "path": path}
    if kind == "type":
        return {"type": "type", "text": raw["text"]}
    if kind == "keypress":
        keys = [_KEY_ALIASES.get(k.upper(), k.lower()) for k in raw["keys"]]
        return {"type": "key", "keys": keys}
    if kind == "wait":
        return {"type": "wait", "seconds": 1.0}
    return {"type": kind, **raw}  # let actions.execute log+skip unrecognized types


def act_node(state: AgentState) -> dict:
    if state["status"] != "running":
        return {}
    last_batch = state["history"][-1].get("pending_actions", [])
    for raw in last_batch:
        mapped = _map_action(raw, state["scale"])
        if mapped is not None:
            actions.execute(mapped)
    return {}


def route_after_act(state: AgentState) -> str:
    if state["status"] == "awaiting_confirmation":
        return "pause"
    if state["status"] in ("done", "failed"):
        return "save_memory"
    if len(state["history"]) >= MAX_STEPS:
        return "save_memory"
    return "decide"


def save_memory_node(state: AgentState) -> dict:
    outcome = state["status"] if state["status"] in ("done", "failed") else "max_steps_reached"
    save_task_result(state["goal"], outcome, state["history"])
    return {}


def build_graph(checkpointer):
    graph = StateGraph(AgentState)
    graph.add_node("decide", decide_node)
    graph.add_node("act", act_node)
    graph.add_node("save_memory", save_memory_node)

    graph.set_entry_point("decide")
    graph.add_edge("decide", "act")
    graph.add_conditional_edges(
        "act", route_after_act, {"decide": "decide", "save_memory": "save_memory", "pause": END}
    )
    graph.add_edge("save_memory", END)

    return graph.compile(checkpointer=checkpointer)
