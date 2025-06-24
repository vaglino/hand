# enhanced_gesture_recorder.py - Continuous multi-gesture recording with transition learning

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
from typing import List, Dict, Optional, Tuple
import threading
from enum import Enum

# Recording parameters
GESTURE_SEQUENCE_LENGTH = 15
RECORDING_FPS = 30
GESTURE_DURATION = 1.0  # seconds per gesture
NEUTRAL_DURATION = 0.5  # seconds between gestures
TRANSITION_SAMPLES = 10  # frames to capture during transitions

class GesturePhase(Enum):
    NEUTRAL = "neutral"
    TRANSITIONING_TO_GESTURE = "transitioning_to_gesture"
    ACTIVE_GESTURE = "active_gesture"
    TRANSITIONING_TO_NEUTRAL = "transitioning_to_neutral"

@dataclass
class ContinuousGestureSequence:
    """Represents a continuous sequence of multiple gesture repetitions"""
    gesture_type: str
    start_time: float
    phases: List[Dict]  # List of {phase: GesturePhase, landmarks: List, timestamp: float}
    gesture_count: int
    recording_mode: str  # 'guided' or 'freestyle'
    
    def to_dict(self):
        data = asdict(self)
        # Convert enum to string
        for phase_data in data['phases']:
            if isinstance(phase_data['phase'], GesturePhase):
                phase_data['phase'] = phase_data['phase'].value
        return data

class TransitionAwareRecorder:
    def __init__(self, model_path='hand_landmarker.task'):
        self.results = None
        self.continuous_sequences = []
        
        # Recording state
        self.is_recording = False
        self.current_sequence = None
        self.recording_mode = 'guided'  # 'guided' or 'freestyle'
        
        # Gesture timing
        self.gesture_timer = 0
        self.current_phase = GesturePhase.NEUTRAL
        self.phase_start_time = 0
        self.gesture_repetitions = 5  # Number of times to repeat each gesture
        self.current_repetition = 0
        
        # Visual/Audio feedback
        self.metronome_enabled = True
        self.visual_guide_enabled = True
        
        # Motion analysis
        self.landmark_history = deque(maxlen=30)
        self.velocity_threshold = 0.02  # Threshold to detect motion start/end
        
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
        
        # Gesture definitions with instructions
        self.gesture_instructions = {
            'scroll_up': "Move hand upward smoothly, then return to center",
            'scroll_down': "Move hand downward smoothly, then return to center",
            'scroll_left': "Move hand left smoothly, then return to center",
            'scroll_right': "Move hand right smoothly, then return to center",
            'zoom_in': "Spread fingers apart, then return to neutral",
            'zoom_out': "Pinch fingers together, then return to neutral",
            'neutral': "Keep hand still in relaxed position"
        }
        
    def _process_result(self, result: vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        self.results = result
    
    def _extract_landmarks(self, landmarks):
        """Extract normalized landmark coordinates"""
        return [[lm.x, lm.y, lm.z] for lm in landmarks]
    
    def _detect_motion_phase(self, landmarks_array):
        """Detect the current motion phase based on velocity and acceleration"""
        self.landmark_history.append(landmarks_array)
        
        if len(self.landmark_history) < 5:
            return self.current_phase
        
        # Calculate recent motion
        recent_motion = self._calculate_motion_intensity()
        
        # State machine for phase detection
        if self.current_phase == GesturePhase.NEUTRAL:
            if recent_motion > self.velocity_threshold:
                return GesturePhase.TRANSITIONING_TO_GESTURE
                
        elif self.current_phase == GesturePhase.TRANSITIONING_TO_GESTURE:
            if recent_motion > self.velocity_threshold * 2:
                return GesturePhase.ACTIVE_GESTURE
            elif recent_motion < self.velocity_threshold:
                return GesturePhase.NEUTRAL
                
        elif self.current_phase == GesturePhase.ACTIVE_GESTURE:
            if recent_motion < self.velocity_threshold:
                return GesturePhase.TRANSITIONING_TO_NEUTRAL
                
        elif self.current_phase == GesturePhase.TRANSITIONING_TO_NEUTRAL:
            if recent_motion < self.velocity_threshold * 0.5:
                return GesturePhase.NEUTRAL
            elif recent_motion > self.velocity_threshold * 2:
                return GesturePhase.ACTIVE_GESTURE
        
        return self.current_phase
    
    def _calculate_motion_intensity(self):
        """Calculate current motion intensity from landmark history"""
        if len(self.landmark_history) < 2:
            return 0.0
        
        # Palm center motion
        palm_indices = [0, 5, 9, 13, 17]
        current_palm = np.mean(self.landmark_history[-1][palm_indices], axis=0)
        prev_palm = np.mean(self.landmark_history[-2][palm_indices], axis=0)
        
        velocity = np.linalg.norm(current_palm - prev_palm)
        return velocity
    
    def _start_guided_recording(self, gesture_type: str):
        """Start a guided recording session for a specific gesture"""
        self.is_recording = True
        self.recording_mode = 'guided'
        self.current_repetition = 0
        self.current_phase = GesturePhase.NEUTRAL
        self.phase_start_time = time.time()
        
        self.current_sequence = ContinuousGestureSequence(
            gesture_type=gesture_type,
            start_time=time.time(),
            phases=[],
            gesture_count=0,
            recording_mode='guided'
        )
        
        print(f"\n=== Recording {gesture_type} ===")
        print(f"Will record {self.gesture_repetitions} repetitions")
        print("Follow the visual guide and audio cues")
    
    def _update_guided_recording(self, landmarks_array):
        """Update guided recording with timing cues"""
        if not self.is_recording or self.recording_mode != 'guided':
            return
        
        current_time = time.time()
        phase_duration = current_time - self.phase_start_time
        
        # Guided state machine
        if self.current_phase == GesturePhase.NEUTRAL:
            if phase_duration >= NEUTRAL_DURATION:
                self.current_phase = GesturePhase.TRANSITIONING_TO_GESTURE
                self.phase_start_time = current_time
                
        elif self.current_phase == GesturePhase.TRANSITIONING_TO_GESTURE:
            # Wait for motion to start
            motion = self._calculate_motion_intensity()
            if motion > self.velocity_threshold or phase_duration > 0.5:
                self.current_phase = GesturePhase.ACTIVE_GESTURE
                self.phase_start_time = current_time
                
        elif self.current_phase == GesturePhase.ACTIVE_GESTURE:
            if phase_duration >= GESTURE_DURATION:
                self.current_phase = GesturePhase.TRANSITIONING_TO_NEUTRAL
                self.phase_start_time = current_time
                self.current_sequence.gesture_count += 1
                
        elif self.current_phase == GesturePhase.TRANSITIONING_TO_NEUTRAL:
            motion = self._calculate_motion_intensity()
            if motion < self.velocity_threshold or phase_duration > 0.5:
                self.current_phase = GesturePhase.NEUTRAL
                self.phase_start_time = current_time
                self.current_repetition += 1
                
                if self.current_repetition >= self.gesture_repetitions:
                    self._finish_recording()
        
        # Record phase data
        phase_data = {
            'phase': self.current_phase,
            'landmarks': landmarks_array.tolist(),
            'timestamp': current_time - self.current_sequence.start_time,
            'repetition': self.current_repetition
        }
        self.current_sequence.phases.append(phase_data)
    
    def _start_freestyle_recording(self, gesture_type: str):
        """Start freestyle recording where user controls timing"""
        self.is_recording = True
        self.recording_mode = 'freestyle'
        self.current_phase = GesturePhase.NEUTRAL
        
        self.current_sequence = ContinuousGestureSequence(
            gesture_type=gesture_type,
            start_time=time.time(),
            phases=[],
            gesture_count=0,
            recording_mode='freestyle'
        )
        
        print(f"\n=== Freestyle Recording: {gesture_type} ===")
        print("Perform the gesture multiple times at your own pace")
        print("Press SPACE when done")
    
    def _update_freestyle_recording(self, landmarks_array):
        """Update freestyle recording with automatic phase detection"""
        if not self.is_recording or self.recording_mode != 'freestyle':
            return
        
        # Detect phase automatically
        new_phase = self._detect_motion_phase(landmarks_array)
        
        # Count gestures on transition to active
        if self.current_phase != GesturePhase.ACTIVE_GESTURE and new_phase == GesturePhase.ACTIVE_GESTURE:
            self.current_sequence.gesture_count += 1
        
        self.current_phase = new_phase
        
        # Record phase data
        phase_data = {
            'phase': self.current_phase,
            'landmarks': landmarks_array.tolist(),
            'timestamp': time.time() - self.current_sequence.start_time,
            'gesture_count': self.current_sequence.gesture_count
        }
        self.current_sequence.phases.append(phase_data)
    
    def _finish_recording(self):
        """Finish current recording and save"""
        if self.current_sequence:
            self.continuous_sequences.append(self.current_sequence)
            print(f"\n✓ Recorded {self.current_sequence.gesture_type}")
            print(f"  Gestures performed: {self.current_sequence.gesture_count}")
            print(f"  Total frames: {len(self.current_sequence.phases)}")
            
        self.is_recording = False
        self.current_sequence = None
        self.current_phase = GesturePhase.NEUTRAL
        self.landmark_history.clear()
    
    def _draw_ui(self, frame):
        """Enhanced UI with phase visualization"""
        h, w = frame.shape[:2]
        
        # Main status bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        if self.is_recording:
            # Recording status
            gesture_type = self.current_sequence.gesture_type
            cv2.putText(frame, f"Recording: {gesture_type.upper()}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            # Phase indicator
            phase_color = {
                GesturePhase.NEUTRAL: (150, 150, 150),
                GesturePhase.TRANSITIONING_TO_GESTURE: (255, 255, 0),
                GesturePhase.ACTIVE_GESTURE: (0, 255, 0),
                GesturePhase.TRANSITIONING_TO_NEUTRAL: (255, 150, 0)
            }
            
            color = phase_color.get(self.current_phase, (255, 255, 255))
            cv2.putText(frame, f"Phase: {self.current_phase.value}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Progress indicator
            if self.recording_mode == 'guided':
                progress = self.current_repetition / self.gesture_repetitions
                bar_width = int(progress * (w - 20))
                cv2.rectangle(frame, (10, 70), (10 + bar_width, 78), (0, 255, 0), -1)
                cv2.rectangle(frame, (10, 70), (w - 10, 78), (100, 100, 100), 2)
                
                cv2.putText(frame, f"Rep: {self.current_repetition + 1}/{self.gesture_repetitions}", 
                           (w - 150, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            else:
                cv2.putText(frame, f"Gestures: {self.current_sequence.gesture_count}", 
                           (w - 150, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        else:
            # Menu
            cv2.putText(frame, "ENHANCED GESTURE RECORDER", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(frame, "[1-7] Guided Recording | [Shift+1-7] Freestyle | [S] Save | [Q] Quit", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Visual guide for guided mode
        if self.is_recording and self.recording_mode == 'guided' and self.visual_guide_enabled:
            self._draw_gesture_guide(frame)
        
        # Statistics panel
        self._draw_statistics_panel(frame)
        
        return frame
    
    def _draw_gesture_guide(self, frame):
        """Draw visual guide for current gesture phase"""
        h, w = frame.shape[:2]
        center_x, center_y = w // 2, h // 2
        
        if self.current_phase == GesturePhase.NEUTRAL:
            # Draw center target
            cv2.circle(frame, (center_x, center_y), 50, (150, 150, 150), 2)
            cv2.putText(frame, "READY", (center_x - 30, center_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 2)
            
        elif self.current_phase in [GesturePhase.TRANSITIONING_TO_GESTURE, GesturePhase.ACTIVE_GESTURE]:
            # Draw gesture direction guide
            if 'scroll_up' in self.current_sequence.gesture_type:
                cv2.arrowedLine(frame, (center_x, center_y), (center_x, center_y - 100), 
                               (0, 255, 0), 3, tipLength=0.3)
            elif 'scroll_down' in self.current_sequence.gesture_type:
                cv2.arrowedLine(frame, (center_x, center_y), (center_x, center_y + 100), 
                               (0, 255, 0), 3, tipLength=0.3)
            elif 'scroll_left' in self.current_sequence.gesture_type:
                cv2.arrowedLine(frame, (center_x, center_y), (center_x - 100, center_y), 
                               (0, 255, 0), 3, tipLength=0.3)
            elif 'scroll_right' in self.current_sequence.gesture_type:
                cv2.arrowedLine(frame, (center_x, center_y), (center_x + 100, center_y), 
                               (0, 255, 0), 3, tipLength=0.3)
            elif 'zoom' in self.current_sequence.gesture_type:
                if 'zoom_in' in self.current_sequence.gesture_type:
                    cv2.circle(frame, (center_x, center_y), 80, (0, 255, 0), 2)
                    cv2.putText(frame, "SPREAD", (center_x - 35, center_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    cv2.circle(frame, (center_x, center_y), 30, (0, 255, 0), 2)
                    cv2.putText(frame, "PINCH", (center_x - 30, center_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Action text
            if self.current_phase == GesturePhase.ACTIVE_GESTURE:
                cv2.putText(frame, "GO!", (center_x - 20, h - 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
        
        elif self.current_phase == GesturePhase.TRANSITIONING_TO_NEUTRAL:
            cv2.putText(frame, "RETURN", (center_x - 40, center_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 150, 0), 2)
    
    def _draw_statistics_panel(self, frame):
        """Draw recording statistics"""
        h, w = frame.shape[:2]
        panel_w = 250
        
        # Panel background
        overlay = frame.copy()
        cv2.rectangle(overlay, (w - panel_w, 100), (w, h - 100), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.8, overlay, 0.2, 0)
        
        y = 130
        cv2.putText(frame, "Recording Statistics", (w - panel_w + 10, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        y += 30
        cv2.putText(frame, f"Total Sequences: {len(self.continuous_sequences)}", 
                   (w - panel_w + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Count by gesture type
        gesture_counts = {}
        for seq in self.continuous_sequences:
            gesture_counts[seq.gesture_type] = gesture_counts.get(seq.gesture_type, 0) + 1
        
        y += 25
        for gesture, count in gesture_counts.items():
            cv2.putText(frame, f"{gesture}: {count} sequences", 
                       (w - panel_w + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
            y += 20
        
        # Current motion intensity
        if len(self.landmark_history) > 1:
            motion = self._calculate_motion_intensity()
            y += 20
            cv2.putText(frame, f"Motion: {motion:.3f}", 
                       (w - panel_w + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # Motion bar
            bar_width = int(min(motion * 500, panel_w - 20))
            cv2.rectangle(frame, (w - panel_w + 10, y + 5), 
                         (w - panel_w + 10 + bar_width, y + 15), (0, 255, 255), -1)
    
    def save_data(self):
        """Save continuous sequences with transition information"""
        os.makedirs('gesture_data', exist_ok=True)
        
        # Convert sequences to dict format
        sequences_data = []
        for seq in self.continuous_sequences:
            sequences_data.append(seq.to_dict())
        
        # Save to JSON
        output_path = 'gesture_data/continuous_sequences.json'
        with open(output_path, 'w') as f:
            json.dump(sequences_data, f, indent=2)
        
        print(f"\n✓ Saved {len(sequences_data)} continuous sequences")
        print(f"  Output: {output_path}")
        
        # Generate training data from sequences
        self._generate_training_data()
    
    def _generate_training_data(self):
        """Generate training data with transition awareness"""
        training_data = {
            'sequences': [],
            'labels': [],
            'phases': [],
            'contexts': []  # Previous gesture context
        }
        
        window_size = GESTURE_SEQUENCE_LENGTH
        stride = 3  # Sliding window stride for more samples
        
        for seq in self.continuous_sequences:
            phases = seq.phases
            
            # Track previous gestures for context
            prev_gesture = 'none'
            gesture_history = []
            
            for i in range(0, len(phases) - window_size, stride):
                # Extract window
                window_phases = phases[i:i + window_size]
                
                # Get landmarks
                landmarks = [p['landmarks'] for p in window_phases]
                
                # Track gesture changes
                for p in window_phases:
                    if p['phase'] == GesturePhase.ACTIVE_GESTURE or (isinstance(p['phase'], str) and p['phase'] == 'active_gesture'):
                        if seq.gesture_type != prev_gesture:
                            prev_gesture = seq.gesture_type
                            gesture_history.append(prev_gesture)
                
                # Determine primary label based on phase distribution
                phase_counts = {}
                for p in window_phases:
                    phase = p['phase']
                    if isinstance(phase, str):
                        phase_counts[phase] = phase_counts.get(phase, 0) + 1
                    else:
                        phase_counts[phase.value] = phase_counts.get(phase.value, 0) + 1
                
                # Label based on dominant phase and context
                dominant_phase = max(phase_counts, key=phase_counts.get)
                
                # More nuanced labeling
                if dominant_phase == 'active_gesture':
                    label = seq.gesture_type
                elif dominant_phase == 'transitioning_to_neutral':
                    # This is a return motion
                    label = f"{seq.gesture_type}_return"
                elif dominant_phase == 'transitioning_to_gesture':
                    # This is gesture initiation
                    label = f"{seq.gesture_type}_start"
                else:
                    label = 'neutral'
                
                # Context: what was the most recent previous gesture?
                if len(gesture_history) > 1:
                    context = gesture_history[-2]  # Previous different gesture
                else:
                    context = 'none'
                
                training_data['sequences'].append(landmarks)
                training_data['labels'].append(label)
                training_data['phases'].append([p['phase'] if isinstance(p['phase'], str) else p['phase'].value for p in window_phases])
                training_data['contexts'].append(context)
        
        # Also generate traditional gesture sequences for compatibility
        traditional_data = {
            'scroll_up': [], 'scroll_down': [], 'scroll_left': [], 'scroll_right': [],
            'zoom_in': [], 'zoom_out': [], 'neutral': []
        }
        
        for seq in self.continuous_sequences:
            # Extract just the active gesture portions
            for i in range(len(seq.phases) - window_size):
                window = seq.phases[i:i + window_size]
                
                # Check if this is mostly an active gesture
                active_count = sum(1 for p in window if (p['phase'] == 'active_gesture' if isinstance(p['phase'], str) else p['phase'].value == 'active_gesture'))
                
                if active_count > window_size * 0.7:  # 70% active
                    landmarks = [p['landmarks'] for p in window]
                    if seq.gesture_type in traditional_data:
                        traditional_data[seq.gesture_type].append(landmarks)
        
        # Save both datasets
        output_path = 'gesture_data/transition_aware_training_data.json'
        with open(output_path, 'w') as f:
            json.dump(training_data, f)
        
        traditional_path = 'gesture_data/gesture_sequences.json'
        with open(traditional_path, 'w') as f:
            json.dump(traditional_data, f, indent=2)
        
        print(f"✓ Generated {len(training_data['sequences'])} transition-aware samples")
        print(f"  Output: {output_path}")
        
        # Show label distribution
        from collections import Counter
        label_counts = Counter(training_data['labels'])
        print("\nLabel distribution:")
        for label, count in label_counts.most_common():
            print(f"  {label}: {count}")
        
        # Show traditional data counts
        print("\nTraditional gesture samples:")
        for gesture, samples in traditional_data.items():
            if samples:
                print(f"  {gesture}: {len(samples)}")
    
    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("\n=== Enhanced Gesture Recorder ===")
        print("\nFeatures:")
        print("- Continuous multi-gesture recording")
        print("- Automatic transition detection")
        print("- Guided and freestyle modes")
        print("- Visual and timing guides")
        print("\nPress numbers 1-7 for guided recording")
        print("Press Shift+1-7 for freestyle recording")
        
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
            
            # Handle landmarks
            if self.results and self.results.hand_landmarks:
                landmarks = self.results.hand_landmarks[0]
                landmarks_array = np.array(self._extract_landmarks(landmarks))
                
                # Update recording
                if self.is_recording:
                    if self.recording_mode == 'guided':
                        self._update_guided_recording(landmarks_array)
                    else:
                        self._update_freestyle_recording(landmarks_array)
                
                # Draw hand
                hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                hand_landmarks_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in landmarks
                ])
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, hand_landmarks_proto, mp.solutions.hands.HAND_CONNECTIONS)
            
            # Draw UI
            frame = self._draw_ui(frame)
            cv2.imshow('Enhanced Gesture Recorder', frame)
            
            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.save_data()
            elif key == ord(' '):
                if self.is_recording and self.recording_mode == 'freestyle':
                    self._finish_recording()
            
            # Number keys for gesture selection
            gesture_map = {
                ord('1'): 'scroll_up', ord('2'): 'scroll_down',
                ord('3'): 'scroll_left', ord('4'): 'scroll_right',
                ord('5'): 'zoom_in', ord('6'): 'zoom_out',
                ord('7'): 'neutral'
            }
            
            # Check for guided recording (numbers)
            if key in gesture_map and not self.is_recording:
                self._start_guided_recording(gesture_map[key])
            
            # Check for freestyle recording (Shift + numbers)
            # OpenCV doesn't directly support shift detection, so we use uppercase letters
            freestyle_map = {
                ord('!'): 'scroll_up', ord('@'): 'scroll_down',
                ord('#'): 'scroll_left', ord('$'): 'scroll_right',
                ord('%'): 'zoom_in', ord('^'): 'zoom_out',
                ord('&'): 'neutral'
            }
            
            if key in freestyle_map and not self.is_recording:
                self._start_freestyle_recording(freestyle_map[key])
        
        cap.release()
        cv2.destroyAllWindows()
        self.landmarker.close()

if __name__ == "__main__":
    recorder = TransitionAwareRecorder()
    recorder.run()