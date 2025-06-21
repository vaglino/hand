# calibration_tool.py - Interactive calibration for personalized gesture control

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import json
import os
from collections import deque
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple
from mediapipe.framework.formats import landmark_pb2

# Import physics engine for testing
from physics_engine import TrackpadPhysicsEngine, Vector2D, GestureMotionExtractor

@dataclass
class CalibrationProfile:
    """User-specific calibration settings"""
    hand_size_reference: float = 0.15
    scroll_sensitivity: float = 1.0
    zoom_sensitivity: float = 1.0
    gesture_thresholds: Dict[str, float] = None
    preferred_fps: int = 30
    motion_smoothing: float = 0.7
    dead_zones: Dict[str, float] = None
    
    def __post_init__(self):
        if self.gesture_thresholds is None:
            self.gesture_thresholds = {
                'scroll_vertical': 0.02,
                'scroll_horizontal': 0.02,
                'zoom': 0.01
            }
        if self.dead_zones is None:
            self.dead_zones = {
                'scroll': 0.01,
                'zoom': 0.005
            }
    
    def to_dict(self):
        return {
            'hand_size_reference': self.hand_size_reference,
            'scroll_sensitivity': self.scroll_sensitivity,
            'zoom_sensitivity': self.zoom_sensitivity,
            'gesture_thresholds': self.gesture_thresholds,
            'preferred_fps': self.preferred_fps,
            'motion_smoothing': self.motion_smoothing,
            'dead_zones': self.dead_zones
        }
    
    @classmethod
    def from_dict(cls, data):
        return cls(**data)

class GestureCalibrationTool:
    """Interactive calibration tool for personalized gesture control"""
    
    def __init__(self, model_path='hand_landmarker.task'):
        self.profile = CalibrationProfile()
        self.calibration_stage = 'menu'  # menu, hand_size, scroll, zoom, test
        self.calibration_data = {
            'hand_sizes': [],
            'scroll_speeds': [],
            'zoom_speeds': []
        }
        
        # MediaPipe setup
        self.results = None
        base_opts = python.BaseOptions(model_asset_path=model_path)
        opts = vision.HandLandmarkerOptions(
            base_options=base_opts,
            running_mode=vision.RunningMode.LIVE_STREAM,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            result_callback=self._process_result
        )
        self.landmarker = vision.HandLandmarker.create_from_options(opts)
        
        # Test physics engine
        self.test_physics = TrackpadPhysicsEngine()
        self.motion_extractor = GestureMotionExtractor()
        
        # UI state
        self.recording = False
        self.test_results = []
        self.instructions = {
            'menu': "Press [1] Hand Size, [2] Scroll, [3] Zoom, [4] Test, [S] Save, [Q] Quit",
            'hand_size': "Show your hand naturally. Press SPACE when ready.",
            'scroll': "Perform scroll gestures at different speeds. Press SPACE to finish.",
            'zoom': "Perform zoom gestures (pinch/spread). Press SPACE to finish.",
            'test': "Test your calibrated settings. Press SPACE to return to menu."
        }
        
    def _process_result(self, result: vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        self.results = result
    
    def _calibrate_hand_size(self, landmarks):
        """Calibrate reference hand size"""
        if not self.recording:
            return
        
        # Calculate hand size using multiple measurements
        measurements = []
        
        # Wrist to middle finger MCP
        dist1 = np.linalg.norm(
            np.array([landmarks[9].x, landmarks[9].y]) - 
            np.array([landmarks[0].x, landmarks[0].y])
        )
        measurements.append(dist1)
        
        # Wrist to index MCP
        dist2 = np.linalg.norm(
            np.array([landmarks[5].x, landmarks[5].y]) - 
            np.array([landmarks[0].x, landmarks[0].y])
        )
        measurements.append(dist2)
        
        # Palm width (index MCP to pinky MCP)
        dist3 = np.linalg.norm(
            np.array([landmarks[17].x, landmarks[17].y]) - 
            np.array([landmarks[5].x, landmarks[5].y])
        )
        measurements.append(dist3)
        
        hand_size = np.median(measurements)
        self.calibration_data['hand_sizes'].append(hand_size)
        
        return hand_size
    
    def _calibrate_scroll_speed(self, motion_features):
        """Calibrate scroll sensitivity based on user's natural speed"""
        if not self.recording or not motion_features:
            return
        
        velocity = motion_features['palm_velocity']
        if velocity.magnitude() > 0.01:  # Ignore tiny movements
            self.calibration_data['scroll_speeds'].append(velocity.magnitude())
    
    def _calibrate_zoom_speed(self, motion_features):
        """Calibrate zoom sensitivity"""
        if not self.recording or not motion_features:
            return
        
        spread_rate = abs(motion_features['spread_velocity'])
        if spread_rate > 0.001:
            self.calibration_data['zoom_speeds'].append(spread_rate)
    
    def _calculate_calibration_results(self):
        """Process calibration data and update profile"""
        # Hand size calibration
        if self.calibration_data['hand_sizes']:
            median_size = np.median(self.calibration_data['hand_sizes'])
            self.profile.hand_size_reference = float(median_size)
            print(f"Hand size calibrated: {median_size:.3f}")
        
        # Scroll sensitivity
        if self.calibration_data['scroll_speeds']:
            speeds = np.array(self.calibration_data['scroll_speeds'])
            
            # Find comfortable speed range (25th to 75th percentile)
            p25 = np.percentile(speeds, 25)
            p75 = np.percentile(speeds, 75)
            median_speed = np.median(speeds)
            
            # Set sensitivity to normalize median speed to 1.0
            self.profile.scroll_sensitivity = 1.0 / (median_speed * 10)
            self.profile.gesture_thresholds['scroll_vertical'] = float(p25 / 2)
            self.profile.gesture_thresholds['scroll_horizontal'] = float(p25 / 2)
            self.profile.dead_zones['scroll'] = float(p25 / 4)
            
            print(f"Scroll calibrated: sensitivity={self.profile.scroll_sensitivity:.2f}")
            print(f"  Threshold: {self.profile.gesture_thresholds['scroll_vertical']:.3f}")
        
        # Zoom sensitivity
        if self.calibration_data['zoom_speeds']:
            speeds = np.array(self.calibration_data['zoom_speeds'])
            
            p25 = np.percentile(speeds, 25)
            p75 = np.percentile(speeds, 75)
            median_speed = np.median(speeds)
            
            self.profile.zoom_sensitivity = 1.0 / (median_speed * 50)
            self.profile.gesture_thresholds['zoom'] = float(p25 / 2)
            self.profile.dead_zones['zoom'] = float(p25 / 4)
            
            print(f"Zoom calibrated: sensitivity={self.profile.zoom_sensitivity:.2f}")
            print(f"  Threshold: {self.profile.gesture_thresholds['zoom']:.3f}")
    
    def _test_calibration(self, motion_features):
        """Test calibrated settings with physics engine"""
        if not motion_features:
            return
        
        # Apply calibrated sensitivity
        self.test_physics.user_scroll_multiplier = self.profile.scroll_sensitivity
        self.test_physics.user_zoom_multiplier = self.profile.zoom_sensitivity
        
        # Test scroll
        velocity = motion_features['palm_velocity']
        if velocity.magnitude() > self.profile.gesture_thresholds['scroll_vertical']:
            self.test_physics.apply_scroll_force(
                velocity.normalize(),
                min(velocity.magnitude() * self.profile.scroll_sensitivity, 1.0)
            )
        
        # Test zoom
        spread_vel = motion_features['spread_velocity']
        if abs(spread_vel) > self.profile.gesture_thresholds['zoom']:
            self.test_physics.apply_zoom_force(
                spread_vel * self.profile.zoom_sensitivity,
                min(abs(spread_vel) * 10, 1.0)
            )
        
        # Update physics
        self.test_physics.update()
        
        # Record test results
        physics_state = self.test_physics.get_physics_state()
        self.test_results.append({
            'scroll_momentum': physics_state['scroll_momentum'],
            'zoom_velocity': physics_state['zoom_velocity']
        })
    
    def _draw_ui(self, frame):
        """Draw calibration UI"""
        h, w = frame.shape[:2]
        
        # Background overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 60), (0, 0, 0), -1)
        cv2.rectangle(overlay, (0, h-120), (w, h), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Title
        title = f"Calibration Tool - {self.calibration_stage.upper()}"
        cv2.putText(frame, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.8, (0, 255, 255), 2)
        
        # Instructions
        instructions = self.instructions.get(self.calibration_stage, "")
        cv2.putText(frame, instructions, (10, h-90), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, (255, 255, 255), 1)
        
        # Stage-specific UI
        if self.calibration_stage == 'hand_size':
            if self.calibration_data['hand_sizes']:
                current_size = self.calibration_data['hand_sizes'][-1]
                cv2.putText(frame, f"Hand size: {current_size:.3f}", 
                           (10, h-50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                
                # Draw hand size visualization
                center_x = w // 2
                center_y = h // 2
                radius = int(current_size * w)
                cv2.circle(frame, (center_x, center_y), radius, (0, 255, 0), 2)
        
        elif self.calibration_stage == 'scroll':
            if self.calibration_data['scroll_speeds']:
                recent_speeds = self.calibration_data['scroll_speeds'][-10:]
                avg_speed = np.mean(recent_speeds)
                cv2.putText(frame, f"Avg speed: {avg_speed:.3f}", 
                           (10, h-50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                
                # Speed visualization bar
                bar_width = int(avg_speed * 500)
                cv2.rectangle(frame, (10, h-30), (10 + bar_width, h-10), (0, 255, 0), -1)
        
        elif self.calibration_stage == 'zoom':
            if self.calibration_data['zoom_speeds']:
                recent_speeds = self.calibration_data['zoom_speeds'][-10:]
                avg_speed = np.mean(recent_speeds)
                cv2.putText(frame, f"Zoom rate: {avg_speed:.3f}", 
                           (10, h-50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 1)
        
        elif self.calibration_stage == 'test':
            # Show current settings
            y = 90
            cv2.putText(frame, "Current Settings:", (10, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y += 25
            cv2.putText(frame, f"Scroll Sensitivity: {self.profile.scroll_sensitivity:.2f}x", 
                       (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y += 20
            cv2.putText(frame, f"Zoom Sensitivity: {self.profile.zoom_sensitivity:.2f}x", 
                       (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Physics visualization
            if self.test_results:
                last_result = self.test_results[-1]
                momentum = last_result['scroll_momentum']
                if momentum[0] != 0 or momentum[1] != 0:
                    # Draw momentum vector
                    center_x, center_y = w // 2, h // 2
                    end_x = int(center_x + momentum[0] * 5)
                    end_y = int(center_y - momentum[1] * 5)
                    cv2.arrowedLine(frame, (center_x, center_y), (end_x, end_y), 
                                   (0, 255, 0), 3, tipLength=0.3)
        
        # Recording indicator
        if self.recording:
            cv2.circle(frame, (w-30, 30), 10, (0, 0, 255), -1)
            cv2.putText(frame, "RECORDING", (w-120, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        return frame
    
    def _save_profile(self):
        """Save calibration profile"""
        os.makedirs('gesture_data', exist_ok=True)
        
        profile_path = 'gesture_data/calibration_profile.json'
        with open(profile_path, 'w') as f:
            json.dump(self.profile.to_dict(), f, indent=2)
        
        print(f"\n✓ Calibration profile saved to {profile_path}")
        print(f"  Hand size: {self.profile.hand_size_reference:.3f}")
        print(f"  Scroll sensitivity: {self.profile.scroll_sensitivity:.2f}x")
        print(f"  Zoom sensitivity: {self.profile.zoom_sensitivity:.2f}x")
    
    def run(self):
        """Main calibration loop"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("\n=== Gesture Calibration Tool ===")
        print("This tool will personalize gesture control to your hand and preferences.\n")
        
        timestamp = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            
            # Process hand tracking
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp += 1
            self.landmarker.detect_async(mp_img, timestamp)
            
            # Process based on calibration stage
            if self.results and self.results.hand_landmarks:
                landmarks = self.results.hand_landmarks[0]
                
                # Draw hand
                proto = landmark_pb2.NormalizedLandmarkList()
                proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) 
                    for lm in landmarks
                ])
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, proto, mp.solutions.hands.HAND_CONNECTIONS)
                
                # Stage-specific processing
                if self.calibration_stage == 'hand_size':
                    self._calibrate_hand_size(landmarks)
                
                elif self.calibration_stage in ['scroll', 'zoom', 'test']:
                    motion_features = self.motion_extractor.extract_motion_features(landmarks)
                    
                    if self.calibration_stage == 'scroll':
                        self._calibrate_scroll_speed(motion_features)
                    elif self.calibration_stage == 'zoom':
                        self._calibrate_zoom_speed(motion_features)
                    elif self.calibration_stage == 'test':
                        self._test_calibration(motion_features)
            
            # Draw UI
            frame = self._draw_ui(frame)
            cv2.imshow('Gesture Calibration Tool', frame)
            
            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                self._save_profile()
            elif key == ord(' '):
                if self.calibration_stage == 'menu':
                    pass
                elif self.calibration_stage == 'hand_size':
                    if self.recording:
                        self.recording = False
                        self._calculate_calibration_results()
                        self.calibration_stage = 'menu'
                    else:
                        self.recording = True
                        self.calibration_data['hand_sizes'] = []
                elif self.calibration_stage in ['scroll', 'zoom']:
                    if self.recording:
                        self.recording = False
                        self._calculate_calibration_results()
                        self.calibration_stage = 'menu'
                    else:
                        self.recording = True
                        if self.calibration_stage == 'scroll':
                            self.calibration_data['scroll_speeds'] = []
                        else:
                            self.calibration_data['zoom_speeds'] = []
                elif self.calibration_stage == 'test':
                    self.calibration_stage = 'menu'
                    self.test_results = []
                    self.test_physics.reset_momentum()
            
            # Menu navigation
            elif self.calibration_stage == 'menu':
                if key == ord('1'):
                    self.calibration_stage = 'hand_size'
                    print("\nHand size calibration - Show your hand naturally")
                elif key == ord('2'):
                    self.calibration_stage = 'scroll'
                    print("\nScroll calibration - Perform scroll gestures")
                elif key == ord('3'):
                    self.calibration_stage = 'zoom'
                    print("\nZoom calibration - Perform pinch/spread gestures")
                elif key == ord('4'):
                    self.calibration_stage = 'test'
                    print("\nTest mode - Try your calibrated settings")
        
        cap.release()
        cv2.destroyAllWindows()
        self.landmarker.close()


def load_calibration_profile():
    """Load saved calibration profile"""
    profile_path = 'gesture_data/calibration_profile.json'
    if os.path.exists(profile_path):
        with open(profile_path, 'r') as f:
            data = json.load(f)
        return CalibrationProfile.from_dict(data)
    return CalibrationProfile()


if __name__ == "__main__":
    tool = GestureCalibrationTool()
    tool.run()