# /gesture_control.py (Definitively Corrected)

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import torch
import torch.nn as nn
import joblib
import pyautogui
from collections import deque
import os
import sys
from mediapipe.framework.formats import landmark_pb2

# --- CONFIGURATION ---
CONFIDENCE_THRESHOLD = 0.7  # Lowered threshold slightly to help with initial detection
SMOOTHING_WINDOW = 5

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GestureLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

class GestureController:
    def __init__(self, model_path='hand_landmarker.task'):
        if not os.path.exists('gesture_data/model.pth'):
            raise FileNotFoundError("Model not found! Run train_model.py first.")
        
        ckpt = torch.load('gesture_data/model.pth', map_location=device)
        self.scaler = joblib.load('gesture_data/scaler.pkl')
        self.label_encoder = joblib.load('gesture_data/label_encoder.pkl')
        
        self.model = GestureLSTM(
            input_size=ckpt['input_size'],
            hidden_size=ckpt['hidden_size'],
            num_layers=ckpt['num_layers'],
            num_classes=len(ckpt['classes'])
        ).to(device)
        self.model.load_state_dict(ckpt['model_state'])
        self.model.eval()

        self.sequence_length = ckpt['sequence_length']
        self.sequence_buffer = deque(maxlen=self.sequence_length)

        self.prediction_history = deque(maxlen=SMOOTHING_WINDOW)
        self.current_stable_gesture = 'neutral'
        self.last_action_gesture = 'neutral'
        
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
        
        print(f"\n✓ Gesture Controller Ready ({device})")
        print(f"Gestures: {', '.join(self.label_encoder.classes_)}")

    def _process_result(self, result: vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        self.results = result

    # --- !THE DEFINITIVE FIX IS HERE! ---
    def create_features(self, landmark_deque):
        """
        Processes a deque of landmarks into a feature array.
        This must be IDENTICAL to the logic in train_model.py
        """
        # Convert deque of lists to a single NumPy array. Shape: (50, 42)
        landmarks_array = np.array(list(landmark_deque))

        # Get wrist position (landmark 0) for all frames. Shape: (50, 2)
        wrist_pos = landmarks_array[:, 0:2]

        # Reshape for broadcasting. Shape: (50, 21, 2)
        landmarks_reshaped = landmarks_array.reshape(self.sequence_length, -1, 2)

        # Normalize by subtracting wrist position.
        # wrist_pos[:, np.newaxis, :] gives shape (50, 1, 2).
        # Broadcasting works: (50, 21, 2) - (50, 1, 2) = (50, 21, 2)
        normalized_landmarks = landmarks_reshaped - wrist_pos[:, np.newaxis, :]

        # Calculate velocity (frame-to-frame difference)
        # Prepend with a zero-velocity frame so dimensions match.
        velocity = np.diff(normalized_landmarks, axis=0, prepend=normalized_landmarks[0:1])

        # Flatten features back to a 2D array
        pos_features = normalized_landmarks.reshape(self.sequence_length, -1)
        vel_features = velocity.reshape(self.sequence_length, -1)
        
        # Concatenate position and velocity features. Shape: (50, 84)
        return np.concatenate([pos_features, vel_features], axis=1)


    def predict_gesture(self):
        if len(self.sequence_buffer) < self.sequence_length:
            return 'neutral', 0.0

        features = self.create_features(self.sequence_buffer)
        scaled_features = self.scaler.transform(features)
        model_input = torch.from_numpy(scaled_features).float().unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits = self.model(model_input)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            
        idx = probs.argmax()
        confidence = probs[idx]
        label = self.label_encoder.inverse_transform([idx])[0]
        
        return (label, confidence) if confidence > CONFIDENCE_THRESHOLD else ('neutral', 0.0)

    def perform_action(self, gesture):
        if gesture == 'scroll_up':
            pyautogui.scroll(100) # Made it slightly less aggressive
        elif gesture == 'scroll_down':
            pyautogui.scroll(-100)
        
        if gesture in ['zoom_in', 'zoom_out']:
            if gesture != self.last_action_gesture:
                if gesture == 'zoom_in':
                    pyautogui.hotkey('ctrl', '+')
                elif gesture == 'zoom_out':
                    pyautogui.hotkey('ctrl', '-')
                self.last_action_gesture = gesture
        
        if gesture == 'neutral':
            self.last_action_gesture = 'neutral'
        
    def draw_ui(self, frame, gesture, confidence):
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 40), (0, 0, 0), -1)
        text = f"Action: {self.current_stable_gesture.upper()} | Pred: {gesture} ({confidence:.2f})"
        color = (0, 255, 0) if self.current_stable_gesture != 'neutral' else (255, 255, 255)
        cv2.putText(frame, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return frame

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
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
            
            if self.results and self.results.hand_landmarks:
                landmarks = self.results.hand_landmarks[0]
                raw_landmarks = [[lm.x, lm.y] for lm in landmarks]
                self.sequence_buffer.append(raw_landmarks)
                last_known_landmarks = raw_landmarks
            elif last_known_landmarks is not None:
                self.sequence_buffer.append(last_known_landmarks)
            
            predicted_gesture, confidence = self.predict_gesture()

            self.prediction_history.append(predicted_gesture)
            if len(self.prediction_history) == SMOOTHING_WINDOW:
                if all(g == self.prediction_history[0] for g in self.prediction_history):
                    if self.current_stable_gesture != self.prediction_history[0]:
                        self.current_stable_gesture = self.prediction_history[0]
                        self.perform_action(self.current_stable_gesture)
            
            if self.results and self.results.hand_landmarks:
                for hand_landmarks_list in self.results.hand_landmarks:
                    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                    hand_landmarks_proto.landmark.extend([
                        landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in hand_landmarks_list
                    ])
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, hand_landmarks_proto, mp.solutions.hands.HAND_CONNECTIONS,
                        mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                        mp.solutions.drawing_styles.get_default_hand_connections_style())
            
            frame = self.draw_ui(frame, predicted_gesture, confidence)
            cv2.imshow('Gesture Controller', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
        self.landmarker.close()

if __name__ == "__main__":
    try:
        ctrl = GestureController()
        ctrl.run()
    except Exception as e:
        import traceback
        print(f"An error occurred: {e}")
        traceback.print_exc()