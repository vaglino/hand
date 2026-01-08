import platform
import time
from typing import Any

import numpy as np
import pyautogui

from physics_engine import Vector2D
from hand.logic import pointer as pointer_logic


def handle_one_shot_actions(controller: Any, gesture: str) -> bool:
    """
    Handles discrete, one-shot actions and returns True if an action was fired.
    Behavior matches previous implementation inside EnhancedGestureController.
    """
    action_fired = False
    os_type = platform.system()

    if gesture == "maximize_window":
        if os_type == "Darwin":  # macOS
            # Toggles fullscreen for the active application
            pyautogui.hotkey("command", "ctrl", "f")
            print("ACTION: Toggled Fullscreen (macOS)")
        else:  # Windows
            pyautogui.hotkey("win", "up")
            print("ACTION: Maximized window (Windows)")
        action_fired = True
    elif gesture == "go_back":
        if os_type == "Darwin":  # macOS
            pyautogui.hotkey("command", "left")
            print("ACTION: Navigated back (macOS)")
        else:  # Windows
            pyautogui.hotkey("alt", "left")
            print("ACTION: Navigated back (Windows)")
        action_fired = True

    return action_fired


def apply_active_action(controller: Any, landmarks):
    """Apply original working physics - responsive and reliable."""
    if controller.state.name != "ACTIVE":
        return

    motion_features = controller.motion_extractor.extract_motion_features(landmarks)
    if "scroll" in controller.active_gesture:
        direction = Vector2D(0, 0)

        # --- FIX: Introduce a sensitivity multiplier ---
        velocity_magnitude = motion_features["palm_velocity"].magnitude()
        sensitivity_multiplier = 1.0  # Default sensitivity

        if "up" in controller.active_gesture:
            direction = Vector2D(0, 1)  # Scrolls content up
        elif "down" in controller.active_gesture:
            direction = Vector2D(0, -1)  # Scrolls content down
            # Compensate for the slower upward hand motion by boosting its effect
            sensitivity_multiplier = 1.5  # (Value can be tuned, e.g., 1.2 to 1.5)

        # Apply force using the adjusted velocity
        controller.physics_engine.apply_scroll_force(direction, velocity_magnitude * sensitivity_multiplier)
    elif "zoom" in controller.active_gesture:
        zoom_rate = motion_features["spread_velocity"]
        # Ensure the motion direction matches the gesture
        if ("in" in controller.active_gesture and zoom_rate > 0) or (
            "out" in controller.active_gesture and zoom_rate < 0
        ):
            controller.physics_engine.apply_zoom_force(zoom_rate)
    elif controller.active_gesture == "pointer":
        pointer_logic.handle_pointer_mode(controller, landmarks)

