import numpy as np
from physics_engine import Vector2D


class GestureMotionExtractor:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.landmark_history = []

    def reset(self):  # <--- ADD THIS METHOD
        """Clears the history of landmarks."""
        self.landmark_history.clear()

    def extract_motion_features(self, landmarks):
        landmarks_array = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
        self.landmark_history.append(landmarks_array)
        if len(self.landmark_history) > self.window_size:
            self.landmark_history.pop(0)
        if len(self.landmark_history) < 2:
            return self._empty_features()
        return {
            "palm_velocity": self._calculate_palm_velocity(),
            "spread_velocity": self._calculate_spread_velocity(),
        }

    def _calculate_palm_velocity(self):
        if len(self.landmark_history) < 2:
            return Vector2D()
        palm_indices = [0, 5, 9, 13, 17]
        current_palm = np.mean(self.landmark_history[-1][palm_indices], axis=0)
        prev_palm = np.mean(self.landmark_history[-2][palm_indices], axis=0)
        velocity = current_palm - prev_palm
        return Vector2D(velocity[0] * 100, velocity[1] * 100)

    def get_thumb_velocity(self):
        if len(self.landmark_history) < 2:
            return Vector2D()
        thumb_indices = [4]
        current_thumb = np.mean(self.landmark_history[-1][thumb_indices], axis=0)
        prev_thumb = np.mean(self.landmark_history[-2][thumb_indices], axis=0)
        velocity = current_thumb - prev_thumb
        return Vector2D(velocity[0] * 100, velocity[1] * 100)

    def _calculate_spread_velocity(self):
        if len(self.landmark_history) < self.window_size:
            return 0.0
        half_window = self.window_size // 2
        spreads = []
        for i in range(self.window_size):
            landmarks = self.landmark_history[i]
            fingertips = [4, 8, 12, 16, 20]
            distances = [
                np.linalg.norm(landmarks[fingertips[j]] - landmarks[fingertips[k]])
                for j in range(len(fingertips))
                for k in range(j + 1, len(fingertips))
            ]
            spreads.append(sum(distances) / len(distances) if distances else 0)
        first_half_avg = (
            np.mean(spreads[:half_window]) if half_window > 0 and len(spreads) >= half_window else 0
        )
        second_half_avg = np.mean(spreads[half_window:]) if len(spreads) > half_window else 0
        return (second_half_avg - first_half_avg) * (30 / half_window) if half_window > 0 else 0

    def _empty_features(self):
        return {"palm_velocity": Vector2D(), "spread_velocity": 0.0}

