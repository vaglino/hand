# physics_engine.py - Core physics simulation for natural trackpad-like control

import numpy as np
import time
import pyautogui
from dataclasses import dataclass
from typing import Tuple, Optional
import math

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

@dataclass
class Vector2D:
    x: float = 0.0
    y: float = 0.0
    
    def __add__(self, other): return Vector2D(self.x + other.x, self.y + other.y)
    def __mul__(self, scalar): return Vector2D(self.x * scalar, self.y * scalar)
    def magnitude(self): return math.sqrt(self.x**2 + self.y**2)
    def normalize(self):
        mag = self.magnitude()
        return Vector2D(self.x/mag, self.y/mag) if mag > 0 else Vector2D(0, 0)

class TrackpadPhysicsEngine:
    def __init__(self):
        self.scroll_momentum = Vector2D()
        self.zoom_velocity = 0.0
        self.scroll_friction = 0.95
        self.zoom_friction = 0.88
        self.scroll_acceleration_factor = 10.0
        self.zoom_acceleration_factor = 0.12
        self.max_scroll_velocity = 150.0
        self.max_zoom_velocity = 0.5
        self.scroll_dead_zone = 0.5
        self.zoom_dead_zone = 0.005
        self.last_update_time = time.time()
        self.scroll_accumulator = Vector2D()
        self.zoom_accumulator = 0.0
        self.user_scroll_multiplier = 1.0
        self.user_zoom_multiplier = 1.0
        
    def apply_scroll_force(self, direction: Vector2D, intensity: float):
        effective_direction = direction.normalize()
        effective_intensity = intensity * self.scroll_acceleration_factor
        force = effective_direction * effective_intensity * self.user_scroll_multiplier
        self.scroll_momentum = self.scroll_momentum + force
        if self.scroll_momentum.magnitude() > self.max_scroll_velocity:
            self.scroll_momentum = self.scroll_momentum.normalize() * self.max_scroll_velocity
            
    def apply_zoom_force(self, zoom_rate: float, intensity: float):
        force = zoom_rate * self.zoom_acceleration_factor * intensity
        self.zoom_velocity += force * self.user_zoom_multiplier
        self.zoom_velocity = np.clip(self.zoom_velocity, -self.max_zoom_velocity, self.max_zoom_velocity)
        
    def update(self, dt: Optional[float] = None):
        current_time = time.time()
        if dt is None: dt = current_time - self.last_update_time
        self.last_update_time = current_time
        
        self.scroll_momentum = self.scroll_momentum * self.scroll_friction
        if self.scroll_momentum.magnitude() < self.scroll_dead_zone: self.scroll_momentum = Vector2D()
        
        self.zoom_velocity *= self.zoom_friction
        if abs(self.zoom_velocity) < self.zoom_dead_zone: self.zoom_velocity = 0.0
        
        self.scroll_accumulator = self.scroll_accumulator + self.scroll_momentum * dt
        self.zoom_accumulator += self.zoom_velocity * dt
        
    def execute_smooth_actions(self):
        if abs(self.scroll_accumulator.y) >= 1.0:
            y_pixels = int(self.scroll_accumulator.y)
            self.scroll_accumulator.y -= y_pixels
            if y_pixels != 0:
                pyautogui.scroll(y_pixels)
        
        ZOOM_ACTION_THRESHOLD = 0.1
        if self.zoom_accumulator > ZOOM_ACTION_THRESHOLD:
            pyautogui.hotkey('ctrl', '+')
            self.zoom_accumulator = 0
        elif self.zoom_accumulator < -ZOOM_ACTION_THRESHOLD:
            pyautogui.hotkey('ctrl', '-')
            self.zoom_accumulator = 0
    
    def reset_momentum(self):
        self.scroll_momentum = Vector2D()
        self.zoom_velocity = 0.0
        
    def get_physics_state(self):
        return {'scroll_momentum': (self.scroll_momentum.x, self.scroll_momentum.y), 'zoom_velocity': self.zoom_velocity}

class GestureMotionExtractor:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.landmark_history = []
    
    def extract_motion_features(self, landmarks):
        landmarks_array = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
        self.landmark_history.append(landmarks_array)
        if len(self.landmark_history) > self.window_size: self.landmark_history.pop(0)
        if len(self.landmark_history) < 2: return self._empty_features()
        return {'palm_velocity': self._calculate_palm_velocity(), 'spread_velocity': self._calculate_spread_velocity()}
    
    def _calculate_palm_velocity(self):
        if len(self.landmark_history) < 2: return Vector2D()
        palm_indices = [0, 5, 9, 13, 17]
        current_palm = np.mean(self.landmark_history[-1][palm_indices], axis=0)
        prev_palm = np.mean(self.landmark_history[-2][palm_indices], axis=0)
        velocity = current_palm - prev_palm
        return Vector2D(velocity[0] * 100, velocity[1] * 100)
    
    def _calculate_spread_velocity(self):
        if len(self.landmark_history) < self.window_size: return 0.0
        half_window = self.window_size // 2
        spreads = []
        for i in range(self.window_size):
            landmarks = self.landmark_history[i]
            fingertips = [4, 8, 12, 16, 20]
            distances = [np.linalg.norm(landmarks[fingertips[j]] - landmarks[fingertips[k]]) for j in range(len(fingertips)) for k in range(j + 1, len(fingertips))]
            spreads.append(sum(distances) / len(distances) if distances else 0)
        first_half_avg = np.mean(spreads[:half_window]) if half_window > 0 and len(spreads) >= half_window else 0
        second_half_avg = np.mean(spreads[half_window:]) if len(spreads) > half_window else 0
        return (second_half_avg - first_half_avg) * (30 / half_window) if half_window > 0 else 0
    
    def _empty_features(self):
        return {'palm_velocity': Vector2D(), 'spread_velocity': 0.0}