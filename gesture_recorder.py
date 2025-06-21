# /gesture_recorder.py (Updated with padding logic)

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import json
import os
from mediapipe.framework.formats import landmark_pb2

class GestureRecorder:
    def __init__(self, model_path='hand_landmarker.task', sequence_length=50):
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

    def extract_features(self, landmarks):
        if not landmarks: return None
        pts = np.array([[lm.x, lm.y] for lm in landmarks])
        return pts.flatten().tolist()

    def draw_ui(self, frame, recording_label, waiting_for_hand, progress):
        h, w = frame.shape[:2]
        if waiting_for_hand:
            txt = f"Show hand to start recording '{recording_label.upper()}'"
            color = (0, 255, 255)
        elif recording_label:
            txt = f"Recording {recording_label.upper()}: {progress}/{self.sequence_length}"
            color = (0, 0, 255)
        else:
            txt = "1-5: Prepare to Record | S: Save | Q: Quit"
            color = (0, 255, 0)
        cv2.putText(frame, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        y = 60
        for g, seqs in self.gesture_data.items():
            cv2.putText(frame, f"{g}: {len(seqs)} samples", (10, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
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
        cooldown = 0
        last_known_features = None # !NEW! To store the last valid features for padding

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

            # --- State 1: Waiting for user to press a key ---
            # (Handled by key press logic at the bottom)

            # --- State 2: Waiting for hand to appear to start recording ---
            if waiting_for_hand:
                if hand_is_visible:
                    print(f"Hand detected! Starting capture for '{recording_label}'...")
                    waiting_for_hand = False
                # While waiting, do nothing else with recording
            
            # --- State 3: Actively recording frames ---
            elif recording_label:
                if hand_is_visible:
                    # Hand is visible, append its features
                    landmarks = self.results.hand_landmarks[0]
                    features = self.extract_features(landmarks)
                    self.current_sequence.append(features)
                    last_known_features = features # Update last known features
                
                # !CHANGE! Hand is NOT visible, but we have started recording
                elif last_known_features is not None:
                    # Hand was visible before and now it's gone. Pad with the last known position.
                    print(f"Hand lost, padding frame {progress+1}/{self.sequence_length}...", end='\r')
                    self.current_sequence.append(last_known_features)
                
                # Check if the sequence is complete
                if len(self.current_sequence) == self.sequence_length:
                    self.gesture_data[recording_label].append(self.current_sequence)
                    print(f"\n✓ Recorded '{recording_label}' sample ({self.sequence_length} frames).")
                    
                    # Reset for next recording
                    recording_label = None
                    self.current_sequence = []
                    last_known_features = None
                    cooldown = 10 # 3-second cooldown

            # --- Drawing and UI ---
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

            if cooldown > 0:
                cooldown -= 1
                cv2.putText(frame, f"Ready in {cooldown//30+1}...", (frame.shape[1]//2 - 100, frame.shape[0]-20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

            frame = self.draw_ui(frame, recording_label, waiting_for_hand, progress)
            cv2.imshow('Gesture Recorder', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.save_data()
            elif cooldown == 0 and not recording_label and not waiting_for_hand:
                key_map = {'1': 'scroll_up', '2': 'scroll_down', '3': 'zoom_in', '4': 'zoom_out', '5': 'neutral'}
                char_key = chr(key)
                if char_key in key_map:
                    recording_label = key_map[char_key]
                    waiting_for_hand = True
                    self.current_sequence = []
                    last_known_features = None # Reset padding tracker
                    print(f"\nPressed '{char_key}'. Prepared for '{recording_label}'. Show your hand to the camera.")

        cap.release()
        cv2.destroyAllWindows()
        self.landmarker.close()

if __name__ == "__main__":
    recorder = GestureRecorder()
    recorder.run()