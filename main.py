import sys
import uuid

from db import get_checkpointer, init_db
from graph import build_graph


def main() -> None:
    if len(sys.argv) < 2:
        print('usage: python main.py "<goal>" [thread_id]')
        sys.exit(1)

    goal = sys.argv[1]
    thread_id = sys.argv[2] if len(sys.argv) > 2 else str(uuid.uuid4())

    init_db()
    checkpointer = get_checkpointer()
    app = build_graph(checkpointer)

    config = {"configurable": {"thread_id": thread_id}}

    existing = app.get_state(config)
    if existing.values:
        print(f"thread_id={thread_id} (resuming from checkpoint, {len(existing.values.get('history', []))} steps so far)")
        run_input = None  # None tells LangGraph to continue from the last checkpoint
    else:
        print(f"thread_id={thread_id} (new run)")
        run_input = {
            "goal": goal,
            "history": [],
            "response_id": None,
            "call_id": None,
            "scale": 1.0,
            "resume_reply": None,
            "status": "running",
        }

    final_state = app.invoke(run_input, config=config)

    while final_state["status"] == "awaiting_confirmation":
        confirm_text = final_state["history"][-1]["final_message"]
        print(f"\n[CONFIRMATION NEEDED] {confirm_text}")
        try:
            answer = input("Allow this action? [y/N]: ").strip().lower()
        except EOFError:
            answer = "n"  # no input available (e.g. piped/non-interactive) -> default to deny
        reply = "approved, proceed" if answer == "y" else "not approved"
        app.update_state(config, {"resume_reply": reply, "status": "running"})
        final_state = app.invoke(None, config=config)

    print(f"status={final_state['status']}")
    for step in final_state["history"]:
        print(step)


if __name__ == "__main__":
    main()
