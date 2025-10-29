import time
from typing import Any

import numpy as np

from physics_engine import Vector2D


def handle_pointer_mode(controller: Any, landmarks):
    """
    Calculates hand movement delta and applies force to the physics engine.
    The physics engine handles acceleration, friction, and cursor movement.
    """
    index_tip_landmark = controller.pointer_config["index_tip_landmark"]
    index_tip = landmarks[index_tip_landmark]
    current_pos = Vector2D(index_tip.x, index_tip.y)

    if controller.previous_index_pos is None:
        controller.previous_index_pos = current_pos
        detect_clicks(controller, landmarks, reset=True)
        return

    delta_pos = Vector2D(
        current_pos.x - controller.previous_index_pos.x,
        current_pos.y - controller.previous_index_pos.y,
    )

    controller.physics_engine.apply_cursor_force(delta_pos)
    controller.previous_index_pos = current_pos
    detect_clicks(controller, landmarks)


def detect_clicks(controller: Any, landmarks, reset: bool = False):
    if reset:
        controller.thumb_openness_history.clear()
        controller.flick_cycle_state = "closed"
        return
    # Calculate thumb openness (distance between thumb tip and index MCP)
    thumb_tip = np.array(
        [
            landmarks[controller.pointer_config["thumb_tip_landmark"]].x,
            landmarks[controller.pointer_config["thumb_tip_landmark"]].y,
        ]
    )
    index_mcp = np.array([landmarks[10].x, landmarks[10].y])  # Index MCP is landmark 5
    openness = np.linalg.norm(thumb_tip - index_mcp)

    controller.thumb_openness_history.append(openness)

    # Detect state changes for flick cycle
    if len(controller.thumb_openness_history) < 3:
        return

    recent_openness = list(controller.thumb_openness_history)[-3:]  # Last 3 frames for trend

    if controller.flick_cycle_state == "closed":
        if recent_openness[-1] - recent_openness[0] > controller.openness_threshold:
            controller.flick_cycle_state = "extending"
    elif controller.flick_cycle_state == "extending":
        if recent_openness[-1] > recent_openness[-2]:
            # Still extending
            pass
        elif recent_openness[-1] - recent_openness[-2] < -controller.openness_threshold / 2:
            controller.flick_cycle_state = "retracting"
    elif controller.flick_cycle_state == "retracting":
        if recent_openness[-1] < recent_openness[0] + (controller.openness_threshold / 2):
            # Completed retraction: register click
            now = time.time()
            if now - controller.last_flick_time < controller.pointer_config["double_flick_window"]:
                import pyautogui

                pyautogui.doubleClick()
            else:
                import pyautogui

                pyautogui.click()
            controller.last_flick_time = now
            controller.flick_cycle_state = "closed"

