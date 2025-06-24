# gesture_recorder.py - Enhanced multi-modal gesture recording system

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import json
import os
from mediapipe.framework.formats import landmark_pb2
from collections import deque
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
import threading

# Reduced sequence length for faster response
GESTURE_SEQUENCE_LENGTH = 15
MOTION_BUFFER_SIZE = 60  # 2 seconds at 30fps

@dataclass
class RecordingSession:
    gesture_type: str
    start_time: float
    landmarks_sequence: List[List[List[float]]]
    motion_intensities: List[float]
    motion_directions: List[Dict[str, float]]
    transition_markers: List[int]  # Frame indices where transitions occur
    
    def to_dict(self):
        return asdict(self)

class EnhancedGestureRecorder:
    def __init__(self, model_path='hand_landmarker.task'):
        self.results = None
        self.recording_sessions = []
        
        # Gesture state data
        self.gesture_data = {
            'scroll_up': [], 'scroll_down': [], 'scroll_left': [], 'scroll_right': [],
            'zoom_in': [], 'zoom_out': [], 'neutral': []
        }
        
        # Continuous motion data
        self.motion_data = {
            'scroll_vertical': [], 'scroll_horizontal': [], 
            'zoom': [], 'combined': []
        }
        
        # Recording state
        self.recording_mode = None  # 'gesture' or 'motion'
        self.current_gesture = None
        self.current_session = None
        self.motion_buffer = deque(maxlen=MOTION_BUFFER_SIZE)
        
        # Real-time analysis
        self.prev_landmarks = None
        self.velocity_history = deque(maxlen=5)
        
        # MediaPipe setup
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
        
        # UI state
        self.show_analysis = True
        self.recording_intensity = 'medium'  # low, medium, high

    def _process_result(self, result: vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        self.results = result

    def _extract_landmarks(self, landmarks):
        """Extract normalized landmark coordinates"""
        return [[lm.x, lm.y, lm.z] for lm in landmarks]
    
    def _calculate_motion_intensity(self, landmarks_array):
        """Calculate real-time motion intensity"""
        if self.prev_landmarks is None:
            self.prev_landmarks = landmarks_array
            return 0.0
        
        # Calculate palm velocity
        palm_indices = [0, 5, 9, 13, 17]
        current_palm = np.mean(landmarks_array[palm_indices], axis=0)
        prev_palm = np.mean(self.prev_landmarks[palm_indices], axis=0)
        
        velocity = np.linalg.norm(current_palm - prev_palm) * 30  # Scale by FPS
        self.velocity_history.append(velocity)
        
        # Smooth velocity
        avg_velocity = np.mean(list(self.velocity_history))
        
        # Map to intensity (0-1 scale)
        intensity = np.tanh(avg_velocity * 10)  # Sigmoid-like mapping
        
        self.prev_landmarks = landmarks_array
        return float(intensity)
    
    def _calculate_motion_direction(self, landmarks_array):
        """Calculate motion direction vector"""
        if len(self.motion_buffer) < 2:
            return {'x': 0.0, 'y': 0.0, 'spread_rate': 0.0}
        
        # Palm motion
        palm_indices = [0, 5, 9, 13, 17]
        current_palm = np.mean(landmarks_array[palm_indices], axis=0)
        
        # Get palm from few frames ago for smoother direction
        old_idx = max(0, len(self.motion_buffer) - 5)
        old_landmarks = np.array(self.motion_buffer[old_idx])
        old_palm = np.mean(old_landmarks[palm_indices], axis=0)
        
        direction = current_palm - old_palm
        
        # Finger spread change
        current_spread = self._calculate_finger_spread(landmarks_array)
        old_spread = self._calculate_finger_spread(old_landmarks)
        spread_rate = (current_spread - old_spread) * 6  # Scale by frame difference
        
        return {
            'x': float(direction[0]),
            'y': float(direction[1]),
            'spread_rate': float(spread_rate)
        }
    
    def _calculate_finger_spread(self, landmarks):
        """Calculate average finger spread"""
        fingertips = [4, 8, 12, 16, 20]
        distances = []
        for i in range(len(fingertips)):
            for j in range(i+1, len(fingertips)):
                dist = np.linalg.norm(landmarks[fingertips[i]] - landmarks[fingertips[j]])
                distances.append(dist)
        return np.mean(distances)
    
    def _record_gesture_state(self, landmarks_array):
        """Record discrete gesture state for classification"""
        if self.current_gesture and len(self.motion_buffer) == GESTURE_SEQUENCE_LENGTH:
            # Save the sequence
            sequence = list(self.motion_buffer)
            self.gesture_data[self.current_gesture].append(sequence)
            print(f"✓ Recorded {self.current_gesture} gesture ({len(self.gesture_data[self.current_gesture])} total)")
            self.current_gesture = None
            self.motion_buffer.clear()
    
    def _record_continuous_motion(self, landmarks_array):
        """Record continuous motion data with annotations"""
        if not self.current_session:
            return
        
        # Add to session
        self.current_session.landmarks_sequence.append(landmarks_array.tolist())
        
        # Calculate and store motion metrics
        intensity = self._calculate_motion_intensity(landmarks_array)
        direction = self._calculate_motion_direction(landmarks_array)
        
        self.current_session.motion_intensities.append(intensity)
        self.current_session.motion_directions.append(direction)
        
        # Check for gesture transitions
        if len(self.current_session.landmarks_sequence) > 10:
            # Simple transition detection based on velocity change
            if len(self.velocity_history) > 1:
                vel_change = abs(self.velocity_history[-1] - self.velocity_history[-2])
                if vel_change > 0.3:  # Threshold for transition
                    frame_idx = len(self.current_session.landmarks_sequence) - 1
                    self.current_session.transition_markers.append(frame_idx)
    
    def _draw_ui(self, frame):
        """Enhanced UI with real-time motion analysis"""
        h, w = frame.shape[:2]
        
        # Top status bar
        cv2.rectangle(frame, (0, 0), (w, 50), (0, 0, 0), -1)
        
        if self.recording_mode == 'gesture':
            if self.current_gesture:
                progress = len(self.motion_buffer)
                text = f"Recording {self.current_gesture.upper()}: {progress}/{GESTURE_SEQUENCE_LENGTH}"
                color = (0, 0, 255)
                # Progress bar
                bar_width = int((progress / GESTURE_SEQUENCE_LENGTH) * w)
                cv2.rectangle(frame, (0, 45), (bar_width, 50), color, -1)
            else:
                text = "GESTURE MODE: [1-7] Record | [S] Save | [M] Motion Mode | [Q] Quit"
                color = (0, 255, 0)
        else:  # motion mode
            if self.current_session:
                duration = time.time() - self.current_session.start_time
                text = f"Recording {self.current_session.gesture_type.upper()} motion: {duration:.1f}s | [SPACE] Stop"
                color = (255, 0, 255)
            else:
                text = "MOTION MODE: [1-4] Start Recording | [G] Gesture Mode | [I] Intensity"
                color = (255, 255, 0)
        
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Side panel
        panel_x = w - 250
        cv2.rectangle(frame, (panel_x, 60), (w, h), (0, 0, 0), -1)
        
        # Gesture counts
        y = 90
        cv2.putText(frame, "Gesture Samples:", (panel_x + 10, y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y += 25
        for i, (gesture, samples) in enumerate(self.gesture_data.items()):
            cv2.putText(frame, f"[{i+1}] {gesture}: {len(samples)}", 
                       (panel_x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y += 20
        
        # Motion analysis (if enabled)
        if self.show_analysis and self.results and self.results.hand_landmarks:
            y += 20
            cv2.putText(frame, "Motion Analysis:", (panel_x + 10, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y += 25
            
            # Current intensity
            if self.prev_landmarks is not None:
                intensity = self._calculate_motion_intensity(
                    np.array(self._extract_landmarks(self.results.hand_landmarks[0]))
                )
                cv2.putText(frame, f"Intensity: {intensity:.2f}", 
                           (panel_x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
                # Intensity bar
                bar_len = int(intensity * 150)
                cv2.rectangle(frame, (panel_x + 10, y + 5), 
                             (panel_x + 10 + bar_len, y + 15), (0, 255, 255), -1)
                y += 30
                
                # Velocity
                if self.velocity_history:
                    vel = self.velocity_history[-1]
                    cv2.putText(frame, f"Velocity: {vel:.1f}", 
                               (panel_x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
                    y += 20
        
        # Recording settings
        y = h - 60
        cv2.putText(frame, f"Intensity Setting: {self.recording_intensity.upper()}", 
                   (panel_x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 255), 1)
        
        return frame
    
    def _draw_hand_with_trail(self, frame, landmarks):
        """Draw hand with motion trail effect"""
        # Standard hand drawing
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in landmarks
        ])
        mp.solutions.drawing_utils.draw_landmarks(
            frame, hand_landmarks_proto, mp.solutions.hands.HAND_CONNECTIONS,
            mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
            mp.solutions.drawing_styles.get_default_hand_connections_style())
        
        # Motion trail (if recording motion)
        if self.current_session and len(self.current_session.landmarks_sequence) > 1:
            h, w = frame.shape[:2]
            trail_length = min(10, len(self.current_session.landmarks_sequence))
            
            for i in range(1, trail_length):
                idx = len(self.current_session.landmarks_sequence) - i
                if idx >= 0:
                    old_landmarks = self.current_session.landmarks_sequence[idx]
                    # Draw fading trail for palm
                    palm_indices = [0, 5, 9, 13, 17]
                    palm_pos = np.mean([old_landmarks[j] for j in palm_indices], axis=0)
                    
                    x = int(palm_pos[0] * w)
                    y = int(palm_pos[1] * h)
                    alpha = 1.0 - (i / trail_length)
                    color = (int(255 * alpha), int(100 * alpha), int(255 * alpha))
                    cv2.circle(frame, (x, y), 3, color, -1)
    
    def save_data(self):
        """Save both gesture state and motion data"""
        os.makedirs('gesture_data', exist_ok=True)
        
        # Save gesture sequences
        gesture_path = 'gesture_data/gesture_sequences.json'
        with open(gesture_path, 'w') as f:
            json.dump(self.gesture_data, f, indent=2)
        
        # Save motion sessions
        motion_path = 'gesture_data/motion_sessions.json'
        motion_export = {}
        for session in self.recording_sessions:
            gesture_type = session.gesture_type
            if gesture_type not in motion_export:
                motion_export[gesture_type] = []
            motion_export[gesture_type].append(session.to_dict())
        
        with open(motion_path, 'w') as f:
            json.dump(motion_export, f, indent=2)
        
        print(f"\n✓ Data saved!")
        print(f"  Gestures: {gesture_path}")
        print(f"  Motion: {motion_path}")
        
        # Summary
        for gesture, samples in self.gesture_data.items():
            if samples:
                print(f"  {gesture}: {len(samples)} samples")
        
        print(f"\nMotion sessions: {len(self.recording_sessions)}")
    
    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("\n=== Enhanced Gesture Recorder ===")
        print("Two recording modes available:")
        print("1. GESTURE MODE - Quick gesture classification samples")
        print("2. MOTION MODE - Continuous motion tracking with intensity\n")
        
        self.recording_mode = 'gesture'
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
            
            # Handle hand detection
            if self.results and self.results.hand_landmarks:
                landmarks = self.results.hand_landmarks[0]
                landmarks_array = np.array(self._extract_landmarks(landmarks))
                
                # Update motion buffer
                self.motion_buffer.append(landmarks_array.tolist())
                
                # Mode-specific recording
                if self.recording_mode == 'gesture':
                    self._record_gesture_state(landmarks_array)
                else:
                    self._record_continuous_motion(landmarks_array)
                
                # Draw hand with effects
                self._draw_hand_with_trail(frame, landmarks)
            
            # Draw UI
            frame = self._draw_ui(frame)
            cv2.imshow('Enhanced Gesture Recorder', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.save_data()
            elif key == ord('m'):
                self.recording_mode = 'motion'
                print("Switched to MOTION recording mode")
            elif key == ord('g'):
                self.recording_mode = 'gesture'
                print("Switched to GESTURE recording mode")
            elif key == ord('i'):
                # Cycle intensity settings
                intensities = ['low', 'medium', 'high']
                current_idx = intensities.index(self.recording_intensity)
                self.recording_intensity = intensities[(current_idx + 1) % 3]
                print(f"Recording intensity: {self.recording_intensity}")
            elif key == ord(' '):
                # Stop current motion recording
                if self.current_session:
                    self.recording_sessions.append(self.current_session)
                    print(f"✓ Saved {self.current_session.gesture_type} motion session")
                    self.current_session = None
            
            # Number keys for gesture selection
            if self.recording_mode == 'gesture' and not self.current_gesture:
                gesture_map = {
                    ord('1'): 'scroll_up', ord('2'): 'scroll_down',
                    ord('3'): 'scroll_left', ord('4'): 'scroll_right',
                    ord('5'): 'zoom_in', ord('6'): 'zoom_out',
                    ord('7'): 'neutral'
                }
                if key in gesture_map:
                    self.current_gesture = gesture_map[key]
                    self.motion_buffer.clear()
                    print(f"Recording {self.current_gesture}...")
            
            elif self.recording_mode == 'motion' and not self.current_session:
                motion_map = {
                    ord('1'): 'scroll_vertical',
                    ord('2'): 'scroll_horizontal',
                    ord('3'): 'zoom',
                    ord('4'): 'combined'
                }
                if key in motion_map:
                    self.current_session = RecordingSession(
                        gesture_type=motion_map[key],
                        start_time=time.time(),
                        landmarks_sequence=[],
                        motion_intensities=[],
                        motion_directions=[],
                        transition_markers=[]
                    )
                    print(f"Started recording {motion_map[key]} motion...")
        
        cap.release()
        cv2.destroyAllWindows()
        self.landmarker.close()

if __name__ == "__main__":
    recorder = EnhancedGestureRecorder()
    recorder.run()