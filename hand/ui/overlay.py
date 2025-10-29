import time
from typing import Deque

import cv2
import numpy as np

def draw_enhanced_ui(
    frame,
    state,
    active_gesture: str,
    inference_times: Deque[float],
    frame_count: int,
    fps_start_time: float,
    backend_name: str,
):
    """Draw enhanced UI with performance metrics.

    Matches the existing on-frame overlay but lives separate from controller.
    """
    h, w = frame.shape[:2]

    # Background overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 0), -1)
    frame = cv2.addWeighted(frame, 0.75, overlay, 0.25, 0)

    # State and gesture info
    try:
        state_color = {
            getattr(state, "NEUTRAL", None) or 1: (0, 0, 0),
            getattr(state, "DEBOUNCING", None) or 2: (255, 255, 0),
            getattr(state, "ACTIVE", None) or 3: (0, 255, 0),
            getattr(state, "RETURNING", None) or 4: (255, 150, 0),
        }
        state_name = state.name if hasattr(state, "name") else str(state)
        color = state_color.get(state, (255, 255, 255))
    except Exception:
        # Fallback if enum mapping fails
        color = (255, 255, 255)
        state_name = str(state)

    cv2.putText(frame, f"STATE: {state_name}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    active_text = active_gesture.upper() if (hasattr(state, "name") and state.name == "ACTIVE") else "NONE"
    cv2.putText(frame, f"ACTION: {active_text}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    if active_gesture == "pointer":
        cv2.putText(
            frame,
            "POINTER: Move index tip, flick thumb to click",
            (10, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )

    # Performance metrics
    if inference_times:
        avg_inference_time = float(np.mean(inference_times))
        fps = frame_count / max(time.time() - fps_start_time, 0.001)

        cv2.putText(
            frame,
            f"Inference: {avg_inference_time:.1f}ms",
            (10, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
        )
        cv2.putText(frame, f"FPS: {fps:.1f}", (200, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(
            frame,
            f"Backend: {backend_name.upper()}",
            (280, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 200, 0),
            1,
        )

    return frame
