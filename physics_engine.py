from dataclasses import dataclass
import math

@dataclass
class Vector2D:
    x: float = 0.0
    y: float = 0.0

    def __iadd__(self, other: "Vector2D"):
        self.x += other.x
        self.y += other.y
        return self

    def __mul__(self, scalar: float):
        return Vector2D(self.x * scalar, self.y * scalar)

    def damp(self, coefficient: float):
        self.x *= coefficient
        self.y *= coefficient

    def magnitude(self) -> float:
        return math.hypot(self.x, self.y)

class TrackpadPhysicsEngine:
    """Simple physics engine for smooth trackpad like motion."""
    def __init__(self, friction: float = 0.95):
        self.momentum = Vector2D()
        self.zoom_velocity = 0.0
        self.friction = friction

    def apply_force(self, gesture_state: str, motion_features: dict):
        """Convert motion features into momentum changes."""
        vel = motion_features.get("velocity_vector", (0.0, 0.0))
        intensity = motion_features.get("gesture_intensity", 1.0)
        self.momentum += Vector2D(vel[0] * intensity, vel[1] * intensity)
        self.zoom_velocity += motion_features.get("zoom_delta", 0.0) * intensity

    def update(self, dt: float):
        self.momentum.damp(self.friction)
        self.zoom_velocity *= self.friction

    def get_scroll_delta(self) -> tuple:
        return self.momentum.x, self.momentum.y

    def get_zoom_delta(self) -> float:
        return self.zoom_velocity
