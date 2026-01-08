# physics_engine.py - Core physics simulation for natural trackpad-like control

import json
import numpy as np
import time
import pyautogui
from dataclasses import dataclass
from typing import Tuple, Optional
import math
import platform

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
    TARGET_FPS = 30.0

    def __init__(self, config=None):
        if config is None:
            with open('config.json', 'r') as f:
                config = json.load(f)['gesture_control_config']['physics_engine']

        # --- General Physics Settings ---
        self.screen_width, self.screen_height = pyautogui.size()
        self.last_update_time = time.time()
        self.zoom_key = 'command' if platform.system() == 'Darwin' else 'ctrl'
        
        # --- Scroll Physics ---
        scroll_config = config.get('scroll', {})
        self.scroll_acceleration_factor = scroll_config.get('acceleration_factor', 2.5)
        scroll_friction_per_frame = scroll_config.get('friction_coefficient', 0.92)
        self.scroll_base_friction = scroll_friction_per_frame ** self.TARGET_FPS
        self.max_scroll_velocity = scroll_config.get('max_velocity', 50.0)
        self.scroll_dead_zone = 0.5
        self.scroll_momentum = Vector2D()
        self.scroll_accumulator = Vector2D()
        
        # --- Zoom Physics ---
        zoom_config = config.get('zoom', {})
        self.zoom_acceleration_factor = zoom_config.get('acceleration_factor', 15.0)
        zoom_friction_per_frame = zoom_config.get('friction_coefficient', 0.88)
        self.zoom_base_friction = zoom_friction_per_frame ** self.TARGET_FPS
        self.max_zoom_velocity = zoom_config.get('max_velocity', 0.5)
        self.zoom_dead_zone = 0.001
        self.zoom_velocity = 0.0
        self.zoom_accumulator = 0.0
        
        # --- Pointer (Cursor) Physics ---
        pointer_config = config.get('pointer', {})
        self.pointer_input_scale = pointer_config.get('input_scale', 2.5)
        self.pointer_acceleration_factor = pointer_config.get('acceleration_factor', 120.0)
        pointer_friction_per_frame = pointer_config.get('friction_coefficient', 0.88)
        self.pointer_base_friction = pointer_friction_per_frame ** self.TARGET_FPS
        self.pointer_max_velocity = pointer_config.get('max_velocity', 3000.0)
        self.cursor_velocity = Vector2D()
        self.cursor_position = Vector2D(*pyautogui.position())
        self.is_pointer_active = False

    def initialize_cursor(self):
        """Syncs the engine's cursor state with the system's cursor."""
        self.cursor_position = Vector2D(*pyautogui.position())
        self.cursor_velocity = Vector2D()
        self.is_pointer_active = True
        
    def deactivate_pointer(self):
        """Stops the engine from controlling the cursor."""
        self.is_pointer_active = False

    def apply_cursor_force(self, hand_delta: Vector2D):
        """Applies force to the cursor based on hand movement."""
        if not self.is_pointer_active: return
            
        force = hand_delta * self.pointer_input_scale * self.pointer_acceleration_factor
        self.cursor_velocity += force
        
        if self.cursor_velocity.magnitude() > self.pointer_max_velocity:
            self.cursor_velocity = self.cursor_velocity.normalize() * self.pointer_max_velocity

    def apply_scroll_force(self, direction: Vector2D, intensity: float):
        effective_direction = direction.normalize()
        force = effective_direction * (intensity * self.scroll_acceleration_factor)
        self.scroll_momentum += force
        if self.scroll_momentum.magnitude() > self.max_scroll_velocity:
            self.scroll_momentum = self.scroll_momentum.normalize() * self.max_scroll_velocity
            
    def apply_zoom_force(self, zoom_rate: float):
        force = zoom_rate * self.zoom_acceleration_factor
        self.zoom_velocity += force
        self.zoom_velocity = np.clip(self.zoom_velocity, -self.max_zoom_velocity, self.max_zoom_velocity)
        
    def update(self, dt: Optional[float] = None):
        current_time = time.time()
        if dt is None: dt = current_time - self.last_update_time
        dt = min(dt, 0.1) # Prevent huge jumps in dt if the app hangs for a moment
        self.last_update_time = current_time
        
        # Update Scroll
        self.scroll_momentum *= (self.scroll_base_friction ** dt)
        if self.scroll_momentum.magnitude() < self.scroll_dead_zone: self.scroll_momentum = Vector2D()
        self.scroll_accumulator += self.scroll_momentum * dt
        
        # Update Zoom
        self.zoom_velocity *= (self.zoom_base_friction ** dt)
        if abs(self.zoom_velocity) < self.zoom_dead_zone: self.zoom_velocity = 0.0
        self.zoom_accumulator += self.zoom_velocity * dt
        
        # Update Cursor
        if self.is_pointer_active:
            self.cursor_velocity *= (self.pointer_base_friction ** dt)
            if self.cursor_velocity.magnitude() < 1.0:
                self.cursor_velocity = Vector2D()
            
            self.cursor_position += self.cursor_velocity * dt
            
            self.cursor_position.x = max(0, min(self.screen_width - 1, self.cursor_position.x))
            self.cursor_position.y = max(0, min(self.screen_height - 1, self.cursor_position.y))
        
    def execute_smooth_actions(self):
        if abs(self.scroll_accumulator.y) >= 1.0:
            y_pixels = int(self.scroll_accumulator.y)
            self.scroll_accumulator.y -= y_pixels
            if y_pixels != 0:
                pyautogui.scroll(y_pixels)
        
        ZOOM_ACTION_THRESHOLD = 0.1
        if self.zoom_accumulator > ZOOM_ACTION_THRESHOLD:
            pyautogui.hotkey(self.zoom_key, '+')
            self.zoom_accumulator = 0
        elif self.zoom_accumulator < -ZOOM_ACTION_THRESHOLD:
            pyautogui.hotkey(self.zoom_key, '-')
            self.zoom_accumulator = 0
        
        # Cursor
        if self.is_pointer_active:
            pyautogui.moveTo(self.cursor_position.x, self.cursor_position.y, _pause=False)
    
    def reset_momentum(self):
        self.scroll_momentum = Vector2D()
        self.zoom_velocity = 0.0
        self.cursor_velocity = Vector2D()
        
    def get_physics_state(self):
        return {'scroll_momentum': (self.scroll_momentum.x, self.scroll_momentum.y), 'zoom_velocity': self.zoom_velocity}

    # Motion extraction moved to hand.features.motion.GestureMotionExtractor
