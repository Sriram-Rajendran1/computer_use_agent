import time

import pyautogui

pyautogui.FAILSAFE = True  # move mouse to a screen corner to abort


def click(x: int, y: int) -> None:
    pyautogui.moveTo(x, y, duration=0.2)
    pyautogui.click()
    time.sleep(0.3)


def move(x: int, y: int) -> None:
    pyautogui.moveTo(x, y, duration=0.1)


def double_click(x: int, y: int) -> None:
    pyautogui.moveTo(x, y, duration=0.2)
    pyautogui.doubleClick()
    time.sleep(0.3)


def scroll(x: int, y: int, scroll_x: int, scroll_y: int) -> None:
    pyautogui.moveTo(x, y, duration=0.1)
    if scroll_y:
        pyautogui.scroll(-scroll_y)  # positive scroll_y = scroll down
    if scroll_x:
        pyautogui.hscroll(scroll_x)
    time.sleep(0.2)


def type_text(text: str) -> None:
    pyautogui.write(text, interval=0.02)
    time.sleep(0.2)


def press(keys: list[str]) -> None:
    pyautogui.hotkey(*keys)
    time.sleep(0.3)


def wait(seconds: float) -> None:
    time.sleep(seconds)


def drag(path: list[dict]) -> None:
    if not path:
        return
    pyautogui.moveTo(path[0]["x"], path[0]["y"], duration=0.1)
    pyautogui.mouseDown()
    for point in path[1:]:
        pyautogui.moveTo(point["x"], point["y"], duration=0.1)
    pyautogui.mouseUp()
    time.sleep(0.3)


def execute(action: dict) -> None:
    """Dispatch a decide-node action dict: {"type": ..., ...}. Unrecognized
    types are skipped (logged) rather than crashing the whole task."""
    kind = action["type"]
    if kind == "click":
        click(action["x"], action["y"])
    elif kind == "move":
        move(action["x"], action["y"])
    elif kind == "double_click":
        double_click(action["x"], action["y"])
    elif kind == "scroll":
        scroll(action["x"], action["y"], action.get("scroll_x", 0), action.get("scroll_y", 0))
    elif kind == "drag":
        drag(action.get("path", []))
    elif kind == "type":
        type_text(action["text"])
    elif kind == "key":
        press(action["keys"])
    elif kind == "wait":
        wait(action.get("seconds", 1.0))
    elif kind in ("done", "fail"):
        pass
    else:
        print(f"[actions] skipping unrecognized action type: {kind!r} ({action})")
