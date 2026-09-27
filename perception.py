import base64
import io

import mss
from PIL import Image

TARGET_WIDTH = 1280


def capture_screenshot() -> bytes:
    """Capture the primary monitor as PNG bytes, full resolution."""
    with mss.mss() as sct:
        raw = sct.grab(sct.monitors[1])
        img = Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()


def screenshot_data_url() -> tuple[str, float, int, int]:
    """Downscaled screenshot for the model, the scale factor to map its
    coordinates back to real screen pixels (real = model_coord * scale), and
    the resized (width, height) sent to the model."""
    png_bytes = capture_screenshot()
    img = Image.open(io.BytesIO(png_bytes))
    real_width = img.width
    scale = real_width / TARGET_WIDTH
    if scale > 1:
        target_height = round(img.height / scale)
        img = img.resize((TARGET_WIDTH, target_height), Image.LANCZOS)
    else:
        scale = 1.0

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}", scale, img.width, img.height
