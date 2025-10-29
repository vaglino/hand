from enum import Enum
from typing import Any

from hand.logic.actions import handle_one_shot_actions


class GestureState(Enum):
    NEUTRAL = 1
    DEBOUNCING = 2
    ACTIVE = 3
    RETURNING = 4


def update_enhanced_state_machine(
    controller: Any, predicted_label: str, confidence: float, smoothed_probs
):
    """Enhanced state machine with existing transition logic.

    Mutates controller state; mirrors previous implementation.
    """
    # Update neutral counter based on smoothed probabilities
    neutral_idx = (
        list(controller.label_encoder.classes_).index("neutral")
        if "neutral" in controller.label_encoder.classes_
        else -1
    )

    if neutral_idx >= 0 and smoothed_probs[neutral_idx] > 0.6:
        controller.neutral_counter += 1
    else:
        controller.neutral_counter = 0

    # State transitions
    if controller.state == GestureState.NEUTRAL:
        if (
            ("return" not in predicted_label)
            and ("neutral" not in predicted_label)
            and confidence > 0.8
        ):
            controller.state = GestureState.DEBOUNCING
            controller.debounce_candidate = predicted_label.replace("_start", "")
            controller.debounce_counter = 1

    elif controller.state == GestureState.DEBOUNCING:
        candidate_match = predicted_label.replace("_start", "") == controller.debounce_candidate

        if candidate_match and confidence > 0.8:
            controller.debounce_counter += 1
            if controller.debounce_counter >= controller.debounce_threshold:
                # Handle one-shot vs continuous actions
                controller.active_gesture = controller.debounce_candidate
                action_fired = handle_one_shot_actions(controller, controller.active_gesture)

                if action_fired:
                    # For one-shot actions, we don't stay in ACTIVE.
                    # Transition immediately to RETURNING to prevent re-triggering.
                    controller.state = GestureState.RETURNING
                    controller.neutral_counter = 0
                else:
                    # For continuous actions (scroll, zoom), transition to ACTIVE.
                    controller.state = GestureState.ACTIVE
                    controller.neutral_counter = 0
                    controller.prediction_smoother.reset()
                    if controller.active_gesture == "pointer":
                        controller.physics_engine.initialize_cursor()
                        controller.previous_index_pos = None
                        controller.neutral_threshold = 3
                        controller.thumb_openness_history.clear()
                        controller.flick_cycle_state = "closed"
        else:
            controller.state = GestureState.NEUTRAL
            controller.debounce_counter = 0

    elif controller.state == GestureState.ACTIVE:
        is_exiting_active_state = False
        if predicted_label == f"{controller.active_gesture}_return" and confidence > 0.6:
            is_exiting_active_state = True
            controller.state = GestureState.RETURNING
            controller.neutral_counter = 0
        elif controller.neutral_counter >= controller.neutral_threshold:
            is_exiting_active_state = True
            controller.state = GestureState.NEUTRAL
        elif (
            predicted_label != controller.active_gesture
            and "neutral" not in predicted_label
            and "return" not in predicted_label
            and confidence > (0.9 if controller.active_gesture == "pointer" else 0.85)
        ):
            is_exiting_active_state = True
            controller.state = GestureState.NEUTRAL

        if is_exiting_active_state:
            if controller.active_gesture == "pointer":
                controller.physics_engine.deactivate_pointer()
            if controller.state == GestureState.NEUTRAL:
                controller.active_gesture = "neutral"

    elif controller.state == GestureState.RETURNING:
        if controller.neutral_counter >= controller.neutral_threshold:
            if controller.active_gesture == "pointer":
                controller.physics_engine.deactivate_pointer()
            controller.state = GestureState.NEUTRAL
            controller.active_gesture = "neutral"

    if controller.state == GestureState.NEUTRAL:
        if controller.physics_engine.is_pointer_active:
            controller.physics_engine.deactivate_pointer()
        controller.neutral_threshold = 2
        controller.thumb_openness_history.clear()
        controller.flick_cycle_state = "closed"
