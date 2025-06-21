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

# !CHANGE! Shorter sequence for better responsiveness
SEQUENCE_LENGTH = 30

class GestureRecorder:
    def __init__(self, model_path='hand_landmarker.task', sequence_length=SEQUENCE_LENGTH):
        self.results = None
        self.sequence_length = sequence_length
        self.current_sequence = []
        self.gesture_data = {g: [] for g in ['scroll_up', 'scroll_down', 'zoom_in', 'zoom_out', 'neutral']}
        
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
            txt = f"Recording {recording_label.upper()}: {progress}/{self.sequence_length}"
            color = (0, 0, 255)
            # Progress bar
            bar_width = int((progress / self.sequence_length) * w)
            cv2.rectangle(frame, (0, 36), (bar_width, 40), color, -1)
        else:
            txt = "KEYS: [1-5] Record | [S] Save | [Q] Quit"
            color = (0, 255, 0)
        cv2.putText(frame, txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Side panel for sample counts
        y = 60
        for i, (g, seqs) in enumerate(self.gesture_data.items()):
            key_num = i + 1
            cv2.putText(frame, f"[{key_num}] {g}: {len(seqs)} samples", (10, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y += 25
        return frame

    def save_data(self):
        os.makedirs('gesture_data', exist_ok=True)
        out_path = 'gesture_data/sequences.json'
        with open(out_path, 'w') as f:
            json.dump(self.gesture_data, f, indent=2)
        print(f"\n✓ Data saved to {out_path}!")
        for g, seqs in self.gesture_data.items():
            print(f"  {g}: {len(seqs)} sequences")

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        print("\n=== Gesture Recorder ===")
        print(f"Will record {self.sequence_length} frames per gesture.")
        print("1: scroll_up | 2: scroll_down | 3: zoom_in | 4: zoom_out | 5: neutral")
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
                
                if len(self.current_sequence) == self.sequence_length:
                    self.gesture_data[recording_label].append(self.current_sequence)
                    print(f"\n✓ Recorded '{recording_label}' sample ({self.sequence_length} frames). Ready for next.")
                    recording_label = None
                    self.current_sequence = []
                    last_known_landmarks = None

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
            elif not recording_label and not waiting_for_hand:
                key_map = {'1': 'scroll_up', '2': 'scroll_down', '3': 'zoom_in', '4': 'zoom_out', '5': 'neutral'}
                char_key = chr(key) if key != 255 else None
                if char_key in key_map:
                    recording_label = key_map[char_key]
                    waiting_for_hand = True
                    self.current_sequence = []
                    last_known_landmarks = None
                    print(f"\nPressed '{char_key}'. Prepared for '{recording_label}'. Show your hand.")

        cap.release()
        cv2.destroyAllWindows()
        self.landmarker.close()

if __name__ == "__main__":
    recorder = GestureRecorder()
    recorder.run()