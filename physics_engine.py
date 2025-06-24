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
    
    def __add__(self, other):
        return Vector2D(self.x + other.x, self.y + other.y)
    
    def __mul__(self, scalar):
        return Vector2D(self.x * scalar, self.y * scalar)
    
    def magnitude(self):
        return math.sqrt(self.x**2 + self.y**2)
    
    def normalize(self):
        mag = self.magnitude()
        if mag > 0:
            return Vector2D(self.x/mag, self.y/mag)
        return Vector2D(0, 0)

class TrackpadPhysicsEngine:
    """Physics simulation for natural trackpad-like gesture control"""
    
    def __init__(self):
        # Physics parameters
        self.scroll_momentum = Vector2D()
        self.zoom_velocity = 0.0
        self.position = Vector2D()
        
        # Tunable physics constants
        self.scroll_friction = 0.95  # Higher = longer glide
        self.zoom_friction = 0.88
        self.scroll_acceleration_factor = 10.0
        self.zoom_acceleration_factor = 0.12
        
        # Sensitivity curves
        self.scroll_sensitivity_base = 1.0
        self.zoom_sensitivity_base = 1.0
        
        # Limits and constraints
        self.max_scroll_velocity = 150.0
        self.max_zoom_velocity = 0.5
        self.scroll_dead_zone = 0.5
        self.zoom_dead_zone = 0.005
        
        # State tracking
        self.last_update_time = time.time()
        self.active_forces = {}
        self.gesture_intensities = {}
        
        # Smooth scrolling accumulator
        self.scroll_accumulator = Vector2D()
        self.zoom_accumulator = 0.0
        
        # User adaptation
        self.user_scroll_multiplier = 1.0
        self.user_zoom_multiplier = 1.0
        
    def apply_scroll_force(self, direction: Vector2D, intensity: float, raw_velocity: Optional[Vector2D] = None):
        """Apply scrolling force with proper physics"""
        # Use raw velocity if available for more direct control
        if raw_velocity:
            # Blend between gesture direction and raw motion
            effective_direction = (direction.normalize() * 0.3 + raw_velocity.normalize() * 0.7).normalize()
            effective_intensity = intensity * raw_velocity.magnitude() * self.scroll_acceleration_factor
        else:
            effective_direction = direction.normalize()
            effective_intensity = intensity * self.scroll_acceleration_factor
        
        # Apply exponential sensitivity curve for fine control
        adjusted_intensity = self._apply_sensitivity_curve(effective_intensity, self.scroll_sensitivity_base)
        
        # Apply force to momentum
        force = effective_direction * adjusted_intensity * self.user_scroll_multiplier
        self.scroll_momentum = self.scroll_momentum + force
        
        # Clamp to maximum velocity
        if self.scroll_momentum.magnitude() > self.max_scroll_velocity:
            self.scroll_momentum = self.scroll_momentum.normalize() * self.max_scroll_velocity
            
        self.active_forces['scroll'] = time.time()
        self.gesture_intensities['scroll'] = intensity
        
    def apply_zoom_force(self, zoom_rate: float, intensity: float):
        """Apply zooming force with smooth acceleration"""
        # BYPASS the faulty sensitivity curve and use direct scaling.
        # This correctly preserves the sign for both zoom-in and zoom-out.
        force = zoom_rate * self.zoom_acceleration_factor * intensity
        
        # Apply force to zoom velocity
        self.zoom_velocity += force * self.user_zoom_multiplier
        
        # Clamp to maximum velocity
        self.zoom_velocity = np.clip(self.zoom_velocity, -self.max_zoom_velocity, self.max_zoom_velocity)
        
        self.active_forces['zoom'] = time.time()
        self.gesture_intensities['zoom'] = intensity
        
        # You can keep this print for now to confirm the fix
        # print(f"Zoom Rate: {zoom_rate:.3f} | Vel: {self.zoom_velocity:.3f} | Accum: {self.zoom_accumulator:.3f}")
    
    def update(self, dt: Optional[float] = None):
        """Update physics simulation"""
        current_time = time.time()
        if dt is None:
            dt = current_time - self.last_update_time
        self.last_update_time = current_time
        
        # Apply friction to scroll momentum
        if 'scroll' not in self.active_forces or (current_time - self.active_forces.get('scroll', 0)) > 0.1:
            # No recent scroll force, apply friction
            self.scroll_momentum = self.scroll_momentum * self.scroll_friction
            
            # Stop if below dead zone
            if self.scroll_momentum.magnitude() < self.scroll_dead_zone:
                self.scroll_momentum = Vector2D()
        
        # Apply friction to zoom
        if 'zoom' not in self.active_forces or (current_time - self.active_forces.get('zoom', 0)) > 0.1:
            self.zoom_velocity *= self.zoom_friction
            
            if abs(self.zoom_velocity) < self.zoom_dead_zone:
                self.zoom_velocity = 0.0
        
        # Update position based on momentum
        self.position = self.position + self.scroll_momentum * dt
        
        # Accumulate motion for smooth execution
        self.scroll_accumulator = self.scroll_accumulator + self.scroll_momentum * dt
        self.zoom_accumulator += self.zoom_velocity * dt
        
    def execute_smooth_actions(self):
        """Execute accumulated actions with sub-pixel precision."""
        # --- Smooth scrolling (this part is fine) ---
        if abs(self.scroll_accumulator.x) >= 1.0 or abs(self.scroll_accumulator.y) >= 1.0:
            x_pixels = int(self.scroll_accumulator.x)
            y_pixels = int(self.scroll_accumulator.y)
            
            self.scroll_accumulator.x -= x_pixels
            self.scroll_accumulator.y -= y_pixels
            
            if x_pixels != 0:
                pyautogui.hscroll(x_pixels)
            if y_pixels != 0:
                pyautogui.scroll(y_pixels)
        
        # --- Corrected smooth zooming logic ---
        # A simple, stable threshold-based system
        ZOOM_ACTION_THRESHOLD = 0.1  # How much "zoom energy" is needed for one action

        if self.zoom_accumulator > ZOOM_ACTION_THRESHOLD:
            pyautogui.hotkey('ctrl', '+')
            self.zoom_accumulator = 0  # Reset after action to prevent runaway loops
        elif self.zoom_accumulator < -ZOOM_ACTION_THRESHOLD:
            pyautogui.hotkey('ctrl', '-')
            self.zoom_accumulator = 0  # Reset after action
    
    def _apply_sensitivity_curve(self, value: float, base_sensitivity: float) -> float:
        """Apply exponential sensitivity curve for fine control"""
        # Exponential curve: slow start, accelerating with intensity
        sign = np.sign(value)
        magnitude = abs(value)
        
        # Three-stage curve
        if magnitude < 0.3:
            # Fine control zone
            adjusted = magnitude * 0.5
        elif magnitude < 0.7:
            # Normal zone
            adjusted = 0.15 + (magnitude - 0.3) * 1.5
        else:
            # Acceleration zone
            adjusted = 0.75 + (magnitude - 0.7) * 3.0
            
        return sign * adjusted * base_sensitivity
    
    def reset_momentum(self):
        """Reset all momentum (for gesture release)"""
        self.scroll_momentum = Vector2D()
        self.zoom_velocity = 0.0
        
    def get_physics_state(self):
        """Get current physics state for debugging"""
        return {
            'scroll_momentum': (self.scroll_momentum.x, self.scroll_momentum.y),
            'zoom_velocity': self.zoom_velocity,
            'active_forces': list(self.active_forces.keys()),
            'gesture_intensities': dict(self.gesture_intensities)
        }
    
    def adapt_to_user(self, gesture_type: str, feedback: float):
        """Adapt sensitivity based on user feedback"""
        if gesture_type == 'scroll':
            self.user_scroll_multiplier *= (1.0 + feedback * 0.1)
            self.user_scroll_multiplier = np.clip(self.user_scroll_multiplier, 0.2, 3.0)
        elif gesture_type == 'zoom':
            self.user_zoom_multiplier *= (1.0 + feedback * 0.1)
            self.user_zoom_multiplier = np.clip(self.user_zoom_multiplier, 0.2, 3.0)

class GestureMotionExtractor:
    """Extract continuous motion parameters from hand landmarks"""
    
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.landmark_history = []
        self.velocity_history = []
        
    def extract_motion_features(self, landmarks):
        """Extract physics-relevant features from landmarks"""
        # Convert to numpy array
        landmarks_array = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
        
        # Add to history
        self.landmark_history.append(landmarks_array)
        if len(self.landmark_history) > self.window_size:
            self.landmark_history.pop(0)
        
        if len(self.landmark_history) < 2:
            return self._empty_features()
        
        # Calculate instantaneous features
        features = {
            'palm_velocity': self._calculate_palm_velocity(),
            'finger_spread': self._calculate_finger_spread(),
            'spread_velocity': self._calculate_spread_velocity(),
            'hand_rotation': self._calculate_hand_rotation(),
            'gesture_confidence': self._calculate_gesture_confidence(),
            'motion_smoothness': self._calculate_motion_smoothness(),
        }
        
        return features
    
    def _calculate_palm_velocity(self):
        """Calculate palm center velocity"""
        if len(self.landmark_history) < 2:
            return Vector2D()
        
        # Palm center is average of landmarks 0, 5, 9, 13, 17
        palm_indices = [0, 5, 9, 13, 17]
        
        current_palm = np.mean(self.landmark_history[-1][palm_indices], axis=0)
        prev_palm = np.mean(self.landmark_history[-2][palm_indices], axis=0)
        
        velocity = current_palm - prev_palm
        return Vector2D(velocity[0] * 100, velocity[1] * 100)  # Scale to pixels
    
    def _calculate_finger_spread(self):
        """Calculate average distance between fingertips"""
        current = self.landmark_history[-1]
        fingertips = [4, 8, 12, 16, 20]  # Thumb, index, middle, ring, pinky tips
        
        total_distance = 0
        count = 0
        for i in range(len(fingertips)):
            for j in range(i+1, len(fingertips)):
                dist = np.linalg.norm(current[fingertips[i]] - current[fingertips[j]])
                total_distance += dist
                count += 1
                
        return total_distance / count if count > 0 else 0
    
    def _calculate_spread_velocity(self):
        """Calculate rate of finger spread change over a smoothed window."""
        if len(self.landmark_history) < self.window_size:
            return 0.0

        # Use a more stable method: compare the average spread of the
        # first half of the window to the second half.
        half_window = self.window_size // 2
        
        # Calculate spread for each frame in the history
        spreads = []
        for i in range(self.window_size):
            landmarks = self.landmark_history[i]
            fingertips = [4, 8, 12, 16, 20]
            distances = []
            count = 0
            for j in range(len(fingertips)):
                for k in range(j + 1, len(fingertips)):
                    dist = np.linalg.norm(landmarks[fingertips[j]] - landmarks[fingertips[k]])
                    distances.append(dist)
                    count += 1
            spreads.append(sum(distances) / count if count > 0 else 0)

        # Average the first and second halves of the history
        first_half_avg = np.mean(spreads[:half_window])
        second_half_avg = np.mean(spreads[half_window:])

        # The velocity is the difference, scaled by FPS
        # This is much more stable than frame-to-frame calculation
        velocity = (second_half_avg - first_half_avg) * (30 / half_window)
        return velocity
    
    def _calculate_hand_rotation(self):
        """Estimate hand rotation for advanced gestures"""
        if len(self.landmark_history) < 2:
            return 0.0
            
        # Use wrist-to-middle-finger vector as reference
        current = self.landmark_history[-1]
        prev = self.landmark_history[-2]
        
        current_vec = current[12] - current[0]  # Middle finger MCP to wrist
        prev_vec = prev[12] - prev[0]
        
        # Calculate angle change
        cos_angle = np.dot(current_vec[:2], prev_vec[:2]) / (
            np.linalg.norm(current_vec[:2]) * np.linalg.norm(prev_vec[:2]) + 1e-6
        )
        angle_change = np.arccos(np.clip(cos_angle, -1, 1))
        
        # Determine rotation direction
        cross_product = current_vec[0] * prev_vec[1] - current_vec[1] * prev_vec[0]
        return angle_change if cross_product > 0 else -angle_change
    
    def _calculate_gesture_confidence(self):
        """Estimate confidence based on motion consistency"""
        if len(self.landmark_history) < 3:
            return 1.0
            
        # Check consistency of motion direction
        velocities = []
        for i in range(1, len(self.landmark_history)):
            palm_indices = [0, 5, 9, 13, 17]
            curr = np.mean(self.landmark_history[i][palm_indices], axis=0)
            prev = np.mean(self.landmark_history[i-1][palm_indices], axis=0)
            velocities.append(curr - prev)
        
        if len(velocities) < 2:
            return 1.0
            
        # Calculate variance in motion direction
        velocities = np.array(velocities)
        mean_velocity = np.mean(velocities, axis=0)
        variance = np.mean(np.sum((velocities - mean_velocity)**2, axis=1))
        
        # Convert to confidence (lower variance = higher confidence)
        confidence = np.exp(-variance * 100)
        return float(np.clip(confidence, 0.0, 1.0))
    
    def _calculate_motion_smoothness(self):
        """Calculate how smooth the motion is (for filtering)"""
        if len(self.landmark_history) < 3:
            return 1.0
            
        # Calculate acceleration variance
        accelerations = []
        for i in range(2, len(self.landmark_history)):
            v1 = self.landmark_history[i] - self.landmark_history[i-1]
            v2 = self.landmark_history[i-1] - self.landmark_history[i-2]
            accel = v1 - v2
            accelerations.append(np.mean(np.abs(accel)))
        
        if not accelerations:
            return 1.0
            
        # Lower variance in acceleration = smoother motion
        smoothness = 1.0 / (1.0 + np.std(accelerations) * 10)
        return float(smoothness)
    
    def _empty_features(self):
        """Return empty feature set"""
        return {
            'palm_velocity': Vector2D(),
            'finger_spread': 0.0,
            'spread_velocity': 0.0,
            'hand_rotation': 0.0,
            'gesture_confidence': 0.0,
            'motion_smoothness': 0.0,
        }