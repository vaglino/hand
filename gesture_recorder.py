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
from collections import deque, Counter
from dataclasses import dataclass, asdict
from typing import List, Dict
from enum import Enum
import threading

# Recording parameters
GESTURE_SEQUENCE_LENGTH = 20  # Increased slightly for better context
RECORDING_FPS = 30
GESTURE_DURATION = 1.0
NEUTRAL_DURATION = 0.5

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
        # Convert Enum members to their string values for JSON serialization
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
        self.current_phase = GesturePhase.NEUTRAL
        self.phase_start_time = 0
        self.gesture_repetitions = 15 # Reduced for quicker recording sessions
        self.current_repetition = 0
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
            'maximize_window': "Show an open hand, then return to neutral",
            'go_back': "Swipe hand to the left, then return to center",
            'neutral': "Keep hand still in relaxed position"
        }
        
    def _process_result(self, result: vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        self.results = result
    
    def _extract_landmarks(self, landmarks):
        return [[lm.x, lm.y, lm.z] for lm in landmarks]
    
    def _calculate_motion_intensity(self):
        if len(self.landmark_history) < 2: return 0.0
        palm_indices = [0, 5, 9, 13, 17]
        # Ensure landmarks are numpy arrays for calculation
        current_landmarks = np.array(self.landmark_history[-1])
        prev_landmarks = np.array(self.landmark_history[-2])
        current_palm = np.mean(current_landmarks[palm_indices], axis=0)
        prev_palm = np.mean(prev_landmarks[palm_indices], axis=0)
        return np.linalg.norm(current_palm - prev_palm)
    
    def _start_guided_recording(self, gesture_type: str):
        self.is_recording = True
        self.current_repetition = 0
        self.current_phase = GesturePhase.NEUTRAL
        self.phase_start_time = time.time()
        self.current_sequence = ContinuousGestureSequence(
            gesture_type=gesture_type, start_time=time.time(), phases=[],
            gesture_count=0, recording_mode='guided'
        )
        print(f"\n=== Recording {gesture_type} ===")
        print(f"Will record {self.gesture_repetitions} repetitions. Get Ready...")
        # Add a small delay to get ready
        threading.Timer(2.0, self._begin_guided_sequence).start()

    def _begin_guided_sequence(self):
        if not self.is_recording: return
        self.current_phase = GesturePhase.TRANSITIONING_TO_GESTURE
        self.phase_start_time = time.time()
        
    def _update_guided_recording(self, landmarks_array):
        if not self.is_recording: return
        current_time = time.time()
        phase_duration = current_time - self.phase_start_time
        
        # This state machine guides the user through the gesture
        if self.current_phase == GesturePhase.TRANSITIONING_TO_GESTURE:
            if phase_duration >= GESTURE_DURATION:
                self.current_phase = GesturePhase.ACTIVE_GESTURE
                self.phase_start_time = current_time
        elif self.current_phase == GesturePhase.ACTIVE_GESTURE:
            if phase_duration >= GESTURE_DURATION:
                self.current_phase = GesturePhase.TRANSITIONING_TO_NEUTRAL
                self.phase_start_time = current_time
                self.current_sequence.gesture_count += 1
        elif self.current_phase == GesturePhase.TRANSITIONING_TO_NEUTRAL:
            if phase_duration >= NEUTRAL_DURATION:
                self.current_phase = GesturePhase.NEUTRAL
                self.phase_start_time = current_time
                self.current_repetition += 1
                if self.current_repetition >= self.gesture_repetitions:
                    self._finish_recording()
                    return
                else: # Start next repetition
                    self.current_phase = GesturePhase.TRANSITIONING_TO_GESTURE
                    self.phase_start_time = current_time + 0.5 # Pause before next

        # Record every frame regardless of state changes
        phase_data = {'phase': self.current_phase, 'landmarks': landmarks_array.tolist(),
                      'timestamp': current_time - self.current_sequence.start_time,
                      'repetition': self.current_repetition}
        self.current_sequence.phases.append(phase_data)

    def _finish_recording(self):
        if self.current_sequence:
            self.continuous_sequences.append(self.current_sequence)
            print(f"\n✓ Recorded {self.current_sequence.gesture_type} with {self.current_sequence.gesture_count} repetitions.")
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
            cv2.putText(frame, f"Recording: {gesture_type.upper()} ({self.current_repetition}/{self.gesture_repetitions})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            phase_color = {
                GesturePhase.NEUTRAL: (150, 150, 150), 
                GesturePhase.TRANSITIONING_TO_GESTURE: (255, 255, 0),
                GesturePhase.ACTIVE_GESTURE: (0, 255, 0), 
                GesturePhase.TRANSITIONING_TO_NEUTRAL: (255, 150, 0)
            }
            color = phase_color.get(self.current_phase, (255, 255, 255))
            cv2.putText(frame, f"Phase: {self.current_phase.value}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            self._draw_gesture_guide(frame)
        else:
            cv2.putText(frame, "GESTURE RECORDER", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(frame, "[1-7] Record Gestures | [S] Save & Append | [Q] Quit", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return frame
    
    def _draw_gesture_guide(self, frame):
        h, w = frame.shape[:2]
        center_x, center_y = w // 2, h // 2
        gesture_type = self.current_sequence.gesture_type
        phase = self.current_phase
        
        # Guide for active gesture phase
        if phase == GesturePhase.TRANSITIONING_TO_GESTURE or phase == GesturePhase.ACTIVE_GESTURE:
            if 'scroll_up' in gesture_type:
                cv2.arrowedLine(frame, (center_x, center_y - 50), (center_x, center_y + 50), (0, 255, 0), 4, tipLength=0.3)
            elif 'scroll_down' in gesture_type:
                cv2.arrowedLine(frame, (center_x, center_y + 50), (center_x, center_y - 50), (0, 255, 0), 4, tipLength=0.3)
            elif 'zoom_in' in gesture_type:
                cv2.arrowedLine(frame, (center_x - 30, center_y), (center_x - 80, center_y), (0, 255, 0), 4, tipLength=0.3)
                cv2.arrowedLine(frame, (center_x + 30, center_y), (center_x + 80, center_y), (0, 255, 0), 4, tipLength=0.3)
            elif 'zoom_out' in gesture_type:
                cv2.arrowedLine(frame, (center_x - 80, center_y), (center_x - 30, center_y), (0, 255, 0), 4, tipLength=0.3)
                cv2.arrowedLine(frame, (center_x + 80, center_y), (center_x + 30, center_y), (0, 255, 0), 4, tipLength=0.3)
            elif 'go_back' in gesture_type:
                cv2.arrowedLine(frame, (center_x + 80, center_y), (center_x - 80, center_y), (0, 255, 0), 4, tipLength=0.3)
            elif 'maximize_window' in gesture_type:
                cv2.putText(frame, "OPEN HAND", (center_x - 80, center_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        # Guide for returning to neutral
        elif phase == GesturePhase.TRANSITIONING_TO_NEUTRAL:
             cv2.putText(frame, "Return to Center", (center_x - 100, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 150, 0), 2)

    def save_data(self):
        if not self.continuous_sequences:
            print("No new data recorded in this session to save.")
            return

        os.makedirs('gesture_data', exist_ok=True)
        output_path_raw = 'gesture_data/continuous_sequences.json'
        
        all_sequences_data = []
        if os.path.exists(output_path_raw):
            try:
                with open(output_path_raw, 'r') as f:
                    existing_data = json.load(f)
                    if isinstance(existing_data, list):
                        all_sequences_data.extend(existing_data)
                        print(f"Loaded {len(existing_data)} existing sequences from {output_path_raw}.")
            except (json.JSONDecodeError, FileNotFoundError):
                print(f"Warning: Could not read existing data file {output_path_raw}. It will be overwritten.")

        new_sequences_data = [seq.to_dict() for seq in self.continuous_sequences]
        all_sequences_data.extend(new_sequences_data)
        
        with open(output_path_raw, 'w') as f:
            json.dump(all_sequences_data, f, indent=2)
        print(f"\n✓ Saved a total of {len(all_sequences_data)} raw sequences to {output_path_raw}")
        
        # Process and save the final training data from the combined sequences
        self._generate_training_data(all_sequences_data)
    
    def _generate_training_data(self, all_sequences_data):
        print("Generating training samples from all recorded sequences...")
        training_data = {'sequences': [], 'labels': []}
        window_size = GESTURE_SEQUENCE_LENGTH
        stride = 2 # Use a smaller stride to generate more samples

        for seq_obj in all_sequences_data:
            phases = seq_obj['phases']
            gesture_type = seq_obj['gesture_type']
            for i in range(0, len(phases) - window_size, stride):
                window_phases = phases[i:i + window_size]
                landmarks = [p['landmarks'] for p in window_phases]
                
                # Determine the label based on the dominant phase in the window
                phase_values = [p['phase'] for p in window_phases]
                phase_counts = Counter(phase_values)
                dominant_phase = phase_counts.most_common(1)[0][0]

                label = 'neutral' # Default label
                if gesture_type != 'neutral':
                    if dominant_phase == 'active_gesture':
                        label = gesture_type
                    elif dominant_phase == 'transitioning_to_neutral':
                        label = f"{gesture_type}_return"
                    # We can also label the start of the gesture
                    elif dominant_phase == 'transitioning_to_gesture':
                        label = f"{gesture_type}_start"
                
                # Only add non-empty sequences
                if landmarks:
                    training_data['sequences'].append(landmarks)
                    training_data['labels'].append(label)
        
        output_path_train = 'gesture_data/training_data.json'
        with open(output_path_train, 'w') as f:
            json.dump(training_data, f)
        print(f"✓ Generated {len(training_data['sequences'])} training samples.")
        print(f"Label distribution: {Counter(training_data['labels'])}")

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
                self.landmark_history.append(landmarks_array.tolist())

                if self.is_recording:
                    self._update_guided_recording(landmarks_array)
                    
                # Draw landmarks on screen
                hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                hand_landmarks_proto.landmark.extend([landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in landmarks])
                mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks_proto, mp.solutions.hands.HAND_CONNECTIONS)

            frame = self._draw_ui(frame)
            cv2.imshow('Gesture Recorder', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('s'): self.save_data()
            
            gesture_map = {
                ord('1'): 'scroll_up', ord('2'): 'scroll_down',
                ord('3'): 'zoom_in', ord('4'): 'zoom_out',
                ord('5'): 'neutral',
                ord('6'): 'maximize_window',
                ord('7'): 'go_back'
            }
            if key in gesture_map and not self.is_recording:
                self._start_guided_recording(gesture_map[key])

        cap.release()
        cv2.destroyAllWindows()
        self.landmarker.close()

if __name__ == "__main__":
    recorder = TransitionAwareRecorder()
    recorder.run()