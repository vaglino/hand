# /gesture_recorder.py (Updated)

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import json
import os
from mediapipe.framework.formats import landmark_pb2

STATE_SEQUENCE_LENGTH = 12  # shorter sequences for gesture state detection
MOTION_WINDOW = 40          # longer window for continuous motion capture

class GestureRecorder:
    def __init__(self, model_path='hand_landmarker.task',
                 state_sequence_length=STATE_SEQUENCE_LENGTH,
                 motion_window=MOTION_WINDOW):
        self.results = None
        self.state_sequence_length = state_sequence_length
        self.motion_window = motion_window
        self.current_sequence = []
        self.motion_buffer = []
        self.motion_recording = False
        self.motion_label = None
        gestures = ['scroll_up', 'scroll_down', 'zoom_in', 'zoom_out', 'neutral']
        self.data = {
            'gesture_states': {g: [] for g in gestures},
            'motion_streams': {g: {'sequences': [], 'intensities': [], 'velocities': []}
                               for g in gestures if g != 'neutral'}
        }
        
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

    def _process_result(self, result: vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        self.results = result

    def extract_landmarks(self, landmarks):
        if not landmarks: return None
        # Return raw coordinates, feature engineering will happen in training script
        return [[lm.x, lm.y, lm.z] for lm in landmarks]

    def draw_ui(self, frame, recording_label, waiting_for_hand, progress):
        h, w = frame.shape[:2]
        # Top status bar
        cv2.rectangle(frame, (0, 0), (w, 40), (0, 0, 0), -1)
        if waiting_for_hand:
            txt = f"Show hand to record '{recording_label.upper()}'"
            color = (0, 255, 255)
        elif recording_label:
            txt = f"Recording {recording_label.upper()}: {progress}/{self.state_sequence_length}"
            color = (0, 0, 255)
            # Progress bar
            bar_width = int((progress / self.state_sequence_length) * w)
            cv2.rectangle(frame, (0, 36), (bar_width, 40), color, -1)
        else:
            txt = "KEYS: [1-5] Record | [S] Save | [Q] Quit"
            color = (0, 255, 0)
        cv2.putText(frame, txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Side panel for sample counts
        y = 60
        for i, (g, seqs) in enumerate(self.data['gesture_states'].items()):
            key_num = i + 1
            cv2.putText(frame, f"[{key_num}] {g}: {len(seqs)} samples", (10, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y += 25
        return frame

    def save_data(self):
        os.makedirs('gesture_data', exist_ok=True)
        out_path = 'gesture_data/dataset.json'
        with open(out_path, 'w') as f:
            json.dump(self.data, f, indent=2)
        print(f"\n✓ Data saved to {out_path}!")
        for g, seqs in self.data['gesture_states'].items():
            print(f"  {g}: {len(seqs)} sequences")

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        print("\n=== Gesture Recorder ===")
        print(f"Will record {self.state_sequence_length} frames per gesture.")
        print("1: scroll_up | 2: scroll_down | 3: zoom_in | 4: zoom_out | 5: neutral")
        print("U/D/I/O: record motion for scroll_up/down/zoom_in/zoom_out")
        print("S: save | Q: quit\n")
        
        recording_label = None
        waiting_for_hand = False
        timestamp = 0
        last_known_landmarks = None

        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp += 1
            self.landmarker.detect_async(mp_img, timestamp)
            
            progress = len(self.current_sequence)
            hand_is_visible = self.results and self.results.hand_landmarks

            if waiting_for_hand:
                if hand_is_visible:
                    print(f"Hand detected! Starting capture for '{recording_label}'...")
                    waiting_for_hand = False
            
            elif recording_label:
                if hand_is_visible:
                    landmarks = self.results.hand_landmarks[0]
                    landmark_coords = self.extract_landmarks(landmarks)
                    self.current_sequence.append(landmark_coords)
                    last_known_landmarks = landmark_coords
                elif last_known_landmarks is not None:
                    # Hand lost, pad with last known position
                    self.current_sequence.append(last_known_landmarks)

                if len(self.current_sequence) == self.state_sequence_length:
                    self.data['gesture_states'][recording_label].append(self.current_sequence)
                    print(f"\n✓ Recorded '{recording_label}' sample ({self.state_sequence_length} frames). Ready for next.")
                    recording_label = None
                    self.current_sequence = []
                    last_known_landmarks = None

            elif self.motion_recording:
                if hand_is_visible:
                    landmarks = self.results.hand_landmarks[0]
                    coords = self.extract_landmarks(landmarks)
                    self.motion_buffer.append(coords)
                    last_known_landmarks = coords
                elif last_known_landmarks is not None:
                    self.motion_buffer.append(last_known_landmarks)

                if len(self.motion_buffer) >= self.motion_window:
                    # simple velocity estimation using wrist landmark
                    velocities = []
                    for i in range(1, len(self.motion_buffer)):
                        prev = self.motion_buffer[i-1][0]
                        curr = self.motion_buffer[i][0]
                        velocities.append((curr[0]-prev[0], curr[1]-prev[1]))
                    intensities = [np.linalg.norm(v) for v in velocities]
                    mean_int = float(np.mean(intensities)) if intensities else 0.0
                    self.data['motion_streams'][self.motion_label]['sequences'].append(self.motion_buffer)
                    self.data['motion_streams'][self.motion_label]['intensities'].append(mean_int)
                    self.data['motion_streams'][self.motion_label]['velocities'].append(velocities)
                    print(f"\n✓ Recorded motion sample for {self.motion_label}.")
                    self.motion_buffer = []
                    self.motion_recording = False

            if hand_is_visible:
                for hand_landmarks in self.results.hand_landmarks:
                    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                    hand_landmarks_proto.landmark.extend([
                        landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in hand_landmarks
                    ])
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, hand_landmarks_proto, mp.solutions.hands.HAND_CONNECTIONS,
                        mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                        mp.solutions.drawing_styles.get_default_hand_connections_style())

            frame = self.draw_ui(frame, recording_label, waiting_for_hand, progress)
            cv2.imshow('Gesture Recorder', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.save_data()
            elif not recording_label and not waiting_for_hand and not self.motion_recording:
                key_map = {'1': 'scroll_up', '2': 'scroll_down', '3': 'zoom_in', '4': 'zoom_out', '5': 'neutral'}
                motion_map = {'u': 'scroll_up', 'd': 'scroll_down', 'i': 'zoom_in', 'o': 'zoom_out'}
                char_key = chr(key) if key != 255 else None
                if char_key in key_map:
                    recording_label = key_map[char_key]
                    waiting_for_hand = True
                    self.current_sequence = []
                    last_known_landmarks = None
                    print(f"\nPressed '{char_key}'. Prepared for '{recording_label}'. Show your hand.")
                elif char_key in motion_map:
                    self.motion_label = motion_map[char_key]
                    self.motion_recording = True
                    self.motion_buffer = []
                    last_known_landmarks = None
                    print(f"\nRecording continuous motion for '{self.motion_label}'.")

        cap.release()
        cv2.destroyAllWindows()
        self.landmarker.close()

if __name__ == "__main__":
    recorder = GestureRecorder()
    recorder.run()