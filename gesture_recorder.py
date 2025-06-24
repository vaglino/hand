# gesture_recorder.py

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import json
import os
from mediapipe.framework.formats import landmark_pb2
from collections import deque, Counter # <--- FIX: Add Counter here
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
import threading
from enum import Enum

# Recording parameters
GESTURE_SEQUENCE_LENGTH = 15
RECORDING_FPS = 30
GESTURE_DURATION = 1.0
NEUTRAL_DURATION = 0.5
TRANSITION_SAMPLES = 10

class GesturePhase(Enum):
    NEUTRAL = "neutral"
    TRANSITIONING_TO_GESTURE = "transitioning_to_gesture"
    ACTIVE_GESTURE = "active_gesture"
    TRANSITIONING_TO_NEUTRAL = "transitioning_to_neutral"

@dataclass
class ContinuousGestureSequence:
    gesture_type: str
    start_time: float
    phases: List[Dict]
    gesture_count: int
    recording_mode: str
    
    def to_dict(self):
        data = asdict(self)
        for phase_data in data['phases']:
            if isinstance(phase_data['phase'], GesturePhase):
                phase_data['phase'] = phase_data['phase'].value
        return data

class TransitionAwareRecorder:
    def __init__(self, model_path='hand_landmarker.task'):
        self.results = None
        self.continuous_sequences = []
        self.is_recording = False
        self.current_sequence = None
        self.recording_mode = 'guided'
        self.gesture_timer = 0
        self.current_phase = GesturePhase.NEUTRAL
        self.phase_start_time = 0
        self.gesture_repetitions = 10
        self.current_repetition = 0
        self.visual_guide_enabled = True
        self.landmark_history = deque(maxlen=30)
        self.velocity_threshold = 0.02
        
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
        
        self.gesture_instructions = {
            'scroll_up': "Move hand upward smoothly, then return to center",
            'scroll_down': "Move hand downward smoothly, then return to center",
            'zoom_in': "Spread fingers apart, then return to neutral",
            'zoom_out': "Pinch fingers together, then return to neutral",
            'neutral': "Keep hand still in relaxed position"
        }
        
    def _process_result(self, result: vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        self.results = result
    
    def _extract_landmarks(self, landmarks):
        return [[lm.x, lm.y, lm.z] for lm in landmarks]
    
    def _detect_motion_phase(self, landmarks_array):
        self.landmark_history.append(landmarks_array)
        if len(self.landmark_history) < 5: return self.current_phase
        recent_motion = self._calculate_motion_intensity()
        if self.current_phase == GesturePhase.NEUTRAL:
            if recent_motion > self.velocity_threshold: return GesturePhase.TRANSITIONING_TO_GESTURE
        elif self.current_phase == GesturePhase.TRANSITIONING_TO_GESTURE:
            if recent_motion > self.velocity_threshold * 2: return GesturePhase.ACTIVE_GESTURE
            elif recent_motion < self.velocity_threshold: return GesturePhase.NEUTRAL
        elif self.current_phase == GesturePhase.ACTIVE_GESTURE:
            if recent_motion < self.velocity_threshold: return GesturePhase.TRANSITIONING_TO_NEUTRAL
        elif self.current_phase == GesturePhase.TRANSITIONING_TO_NEUTRAL:
            if recent_motion < self.velocity_threshold * 0.5: return GesturePhase.NEUTRAL
            elif recent_motion > self.velocity_threshold * 2: return GesturePhase.ACTIVE_GESTURE
        return self.current_phase
    
    def _calculate_motion_intensity(self):
        if len(self.landmark_history) < 2: return 0.0
        palm_indices = [0, 5, 9, 13, 17]
        current_palm = np.mean(self.landmark_history[-1][palm_indices], axis=0)
        prev_palm = np.mean(self.landmark_history[-2][palm_indices], axis=0)
        return np.linalg.norm(current_palm - prev_palm)
    
    def _start_guided_recording(self, gesture_type: str):
        self.is_recording = True
        self.recording_mode = 'guided'
        self.current_repetition = 0
        self.current_phase = GesturePhase.NEUTRAL
        self.phase_start_time = time.time()
        self.current_sequence = ContinuousGestureSequence(
            gesture_type=gesture_type, start_time=time.time(), phases=[],
            gesture_count=0, recording_mode='guided'
        )
        print(f"\n=== Recording {gesture_type} ===")
        print(f"Will record {self.gesture_repetitions} repetitions")
    
    def _update_guided_recording(self, landmarks_array):
        if not self.is_recording or self.recording_mode != 'guided': return
        current_time = time.time()
        phase_duration = current_time - self.phase_start_time
        if self.current_phase == GesturePhase.NEUTRAL:
            if phase_duration >= NEUTRAL_DURATION:
                self.current_phase = GesturePhase.TRANSITIONING_TO_GESTURE
                self.phase_start_time = current_time
        elif self.current_phase == GesturePhase.TRANSITIONING_TO_GESTURE:
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
                    return
        phase_data = {'phase': self.current_phase, 'landmarks': landmarks_array.tolist(),
                      'timestamp': current_time - self.current_sequence.start_time,
                      'repetition': self.current_repetition}
        self.current_sequence.phases.append(phase_data)
    
    def _start_freestyle_recording(self, gesture_type: str):
        self.is_recording = True
        self.recording_mode = 'freestyle'
        self.current_phase = GesturePhase.NEUTRAL
        self.current_sequence = ContinuousGestureSequence(
            gesture_type=gesture_type, start_time=time.time(), phases=[],
            gesture_count=0, recording_mode='freestyle'
        )
        print(f"\n=== Freestyle Recording: {gesture_type} ===")
        print("Perform the gesture multiple times. Press SPACE when done")
    
    def _update_freestyle_recording(self, landmarks_array):
        if not self.is_recording or self.recording_mode != 'freestyle': return
        new_phase = self._detect_motion_phase(landmarks_array)
        if self.current_phase != GesturePhase.ACTIVE_GESTURE and new_phase == GesturePhase.ACTIVE_GESTURE:
            self.current_sequence.gesture_count += 1
        self.current_phase = new_phase
        phase_data = {'phase': self.current_phase, 'landmarks': landmarks_array.tolist(),
                      'timestamp': time.time() - self.current_sequence.start_time,
                      'gesture_count': self.current_sequence.gesture_count}
        self.current_sequence.phases.append(phase_data)
    
    def _finish_recording(self):
        if self.current_sequence:
            self.continuous_sequences.append(self.current_sequence)
            print(f"\n✓ Recorded {self.current_sequence.gesture_type}")
        self.is_recording = False
        self.current_sequence = None
        self.current_phase = GesturePhase.NEUTRAL
        self.landmark_history.clear()
    
    def _draw_ui(self, frame):
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        if self.is_recording:
            gesture_type = self.current_sequence.gesture_type
            cv2.putText(frame, f"Recording: {gesture_type.upper()}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            phase_color = {GesturePhase.NEUTRAL: (150, 150, 150), GesturePhase.TRANSITIONING_TO_GESTURE: (255, 255, 0),
                           GesturePhase.ACTIVE_GESTURE: (0, 255, 0), GesturePhase.TRANSITIONING_TO_NEUTRAL: (255, 150, 0)}
            color = phase_color.get(self.current_phase, (255, 255, 255))
            cv2.putText(frame, f"Phase: {self.current_phase.value}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        else:
            cv2.putText(frame, "GESTURE RECORDER", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(frame, "[1-5] Guided | [Shift+1-5] Freestyle | [S] Save | [Q] Quit", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        if self.is_recording and self.recording_mode == 'guided' and self.visual_guide_enabled:
            self._draw_gesture_guide(frame)
        return frame
    
    def _draw_gesture_guide(self, frame):
        h, w = frame.shape[:2]
        center_x, center_y = w // 2, h // 2
        gesture_type = self.current_sequence.gesture_type
        
        if self.current_phase in [GesturePhase.TRANSITIONING_TO_GESTURE, GesturePhase.ACTIVE_GESTURE]:
            if 'scroll_up' in gesture_type:
                cv2.arrowedLine(frame, (center_x, center_y), (center_x, center_y - 100), (0, 255, 0), 3, tipLength=0.3)
            elif 'scroll_down' in gesture_type:
                cv2.arrowedLine(frame, (center_x, center_y), (center_x, center_y + 100), (0, 255, 0), 3, tipLength=0.3)
            elif 'zoom' in gesture_type:
                if 'zoom_in' in gesture_type:
                    cv2.circle(frame, (center_x, center_y), 80, (0, 255, 0), 2)
                else:
                    cv2.circle(frame, (center_x, center_y), 30, (0, 255, 0), 2)
    
    def save_data(self):
        os.makedirs('gesture_data', exist_ok=True)
        sequences_data = [seq.to_dict() for seq in self.continuous_sequences]
        output_path = 'gesture_data/continuous_sequences.json'
        with open(output_path, 'w') as f:
            json.dump(sequences_data, f, indent=2)
        print(f"\n✓ Saved {len(sequences_data)} continuous sequences to {output_path}")
        self._generate_training_data()
    
    def _generate_training_data(self):
        training_data = {'sequences': [], 'labels': [], 'phases': [], 'contexts': []}
        window_size = GESTURE_SEQUENCE_LENGTH
        stride = 3
        for seq in self.continuous_sequences:
            phases = seq.phases
            for i in range(0, len(phases) - window_size, stride):
                window_phases = phases[i:i + window_size]
                landmarks = [p['landmarks'] for p in window_phases]
                phase_counts = Counter(p['phase'].value if isinstance(p['phase'], GesturePhase) else p['phase'] for p in window_phases)
                dominant_phase = max(phase_counts, key=phase_counts.get)
                if dominant_phase == 'active_gesture': label = seq.gesture_type
                elif dominant_phase == 'transitioning_to_neutral': label = f"{seq.gesture_type}_return"
                elif dominant_phase == 'transitioning_to_gesture': label = f"{seq.gesture_type}_start"
                else: label = 'neutral'
                training_data['sequences'].append(landmarks)
                training_data['labels'].append(label)
        
        traditional_data = {'scroll_up': [], 'scroll_down': [], 'zoom_in': [], 'zoom_out': [], 'neutral': []}
        for seq in self.continuous_sequences:
            for i in range(len(seq.phases) - window_size):
                window = seq.phases[i:i + window_size]
                active_count = sum(1 for p in window if (p['phase'] == 'active_gesture' if isinstance(p['phase'], str) else p['phase'].value == 'active_gesture'))
                if active_count > window_size * 0.7 and seq.gesture_type in traditional_data:
                    traditional_data[seq.gesture_type].append([p['landmarks'] for p in window])

        output_path = 'gesture_data/transition_aware_training_data.json'
        with open(output_path, 'w') as f: json.dump(training_data, f)
        print(f"✓ Generated {len(training_data['sequences'])} transition-aware samples")
        
    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        timestamp = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp += 1
            self.landmarker.detect_async(mp_img, timestamp)
            if self.results and self.results.hand_landmarks:
                landmarks = self.results.hand_landmarks[0]
                landmarks_array = np.array(self._extract_landmarks(landmarks))
                if self.is_recording:
                    if self.recording_mode == 'guided': self._update_guided_recording(landmarks_array)
                    else: self._update_freestyle_recording(landmarks_array)
                hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                hand_landmarks_proto.landmark.extend([landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in landmarks])
                mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks_proto, mp.solutions.hands.HAND_CONNECTIONS)
            frame = self._draw_ui(frame)
            cv2.imshow('Enhanced Gesture Recorder', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('s'): self.save_data()
            elif key == ord(' ') and self.is_recording and self.recording_mode == 'freestyle': self._finish_recording()
            
            gesture_map = {
                ord('1'): 'scroll_up', ord('2'): 'scroll_down',
                ord('3'): 'zoom_in', ord('4'): 'zoom_out',
                ord('5'): 'neutral'
            }
            if key in gesture_map and not self.is_recording:
                self._start_guided_recording(gesture_map[key])

        cap.release()
        cv2.destroyAllWindows()
        self.landmarker.close()

if __name__ == "__main__":
    recorder = TransitionAwareRecorder()
    recorder.run()