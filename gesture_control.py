# gesture_control.py (Smarter & Smoother)

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import torch
import torch.nn as nn
import joblib
import pyautogui
from collections import deque, Counter
import os
from mediapipe.framework.formats import landmark_pb2
# Import the feature creation function from the training script to ensure consistency
from train_model import create_features_for_sequence, GestureLSTM

# ---------- USER-TUNABLE PARAMETERS ----------
CONFIDENCE_THRESHOLD = 0.60   # Be more confident before acting
SMOOTHING_WINDOW     = 5      # A shorter window for faster reaction
ACTION_COOLDOWN_FRAMES = 15   # Cooldown in frames (e.g., 15 frames ≈ 0.5s at 30fps)
SCROLL_PIXELS        = 120
# ---------------------------------------------

pyautogui.FAILSAFE = False
pyautogui.PAUSE    = 0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GestureController:
    def __init__(self, model_path='hand_landmarker.task'):
        ckpt_path = 'gesture_data/model.pth'
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError("Model not found – run train_model.py first")

        ckpt = torch.load(ckpt_path, map_location=device)
        self.scaler = joblib.load('gesture_data/scaler.pkl')
        self.label_encoder = joblib.load('gesture_data/label_encoder.pkl')

        self.model = GestureLSTM(
            ckpt['input_size'], ckpt['hidden_size'], ckpt['num_layers'],
            len(ckpt['classes']), ckpt.get('dropout', 0.2) # backwards compatible
        ).to(device)
        self.model.load_state_dict(ckpt['model_state'])
        self.model.eval()

        self.seq_len = ckpt['sequence_length']
        self.landmark_buffer = deque(maxlen=self.seq_len)
        self.pred_history = deque(maxlen=SMOOTHING_WINDOW)
        self.current_action = 'neutral'
        self.cooldown = 0

        # MediaPipe setup
        self.results = None
        base_opts = python.BaseOptions(model_asset_path=model_path)
        opts = vision.HandLandmarkerOptions(
            base_options=base_opts,
            running_mode=vision.RunningMode.LIVE_STREAM,
            num_hands=1, min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5, result_callback=self._process_result
        )
        self.landmarker = vision.HandLandmarker.create_from_options(opts)

        print(f"\n✓ Gesture Controller Ready ({device})")
        print("Gestures:", ", ".join(self.label_encoder.classes_))

    def _process_result(self, result: vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        self.results = result

    def _predict_gesture(self):
        if len(self.landmark_buffer) < self.seq_len:
            return 'neutral', 0.0
        
        # Use the exact same feature creation as in training
        features = create_features_for_sequence(self.landmark_buffer)
        
        # Scale the features
        scaled_features = self.scaler.transform(features)
        
        with torch.no_grad():
            tensor_in = torch.FloatTensor(scaled_features).unsqueeze(0).to(device)
            logits = self.model(tensor_in)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            
        pred_idx = probs.argmax()
        pred_label = self.label_encoder.inverse_transform([pred_idx])[0]
        confidence = probs[pred_idx]
        return pred_label, confidence

    def _update_action_state(self, label, conf):
        self.pred_history.append(label)
        
        if self.cooldown > 0:
            self.cooldown -= 1
            # If we are in cooldown, force action to neutral to prevent re-triggering
            self.current_action = 'neutral'
            return

        if len(self.pred_history) < SMOOTHING_WINDOW:
            return

        # Majority vote for stable prediction
        most_common_label, count = Counter(self.pred_history).most_common(1)[0]
        
        # Check for stability - is the most common gesture consistent enough?
        if count >= int(SMOOTHING_WINDOW * 0.8): # require 80% consistency in window
            stable_action = most_common_label
        else:
            stable_action = 'neutral'
            
        # Has the stable action changed, and is it not neutral?
        if stable_action != self.current_action and stable_action != 'neutral':
             # Check confidence ONLY when we are about to fire an action
            _, current_confidence = self._predict_gesture()
            if current_confidence > CONFIDENCE_THRESHOLD:
                self.current_action = stable_action
                self._perform_action(self.current_action)
                self.cooldown = ACTION_COOLDOWN_FRAMES # Start cooldown
                self.pred_history.clear() # Clear history after an action
        elif stable_action == 'neutral':
             self.current_action = 'neutral'


    def _perform_action(self, gesture):
        print(f"Action: {gesture}") # For debugging
        if gesture == 'scroll_up':
            pyautogui.scroll(SCROLL_PIXELS)
        elif gesture == 'scroll_down':
            pyautogui.scroll(-SCROLL_PIXELS)
        elif gesture == 'zoom_in':
            pyautogui.hotkey('ctrl', '+')
        elif gesture == 'zoom_out':
            pyautogui.hotkey('ctrl', '-')

    @staticmethod
    def _draw_hand(frame, landmark_list):
        proto = landmark_pb2.NormalizedLandmarkList()
        proto.landmark.extend([landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in landmark_list])
        mp.solutions.drawing_utils.draw_landmarks(
            frame, proto, mp.solutions.hands.HAND_CONNECTIONS,
            mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
            mp.solutions.drawing_styles.get_default_hand_connections_style())

    def _draw_overlay(self, frame, label, conf):
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
        status_text = f"Action: {self.current_action.upper()}"
        color = (0, 255, 0) if self.current_action != 'neutral' else (255, 255, 255)
        
        # Show a cooldown indicator
        if self.cooldown > 0:
             status_text += " (COOLDOWN)"
             color = (0, 165, 255) # Orange

        cv2.putText(frame, status_text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"Prediction: {label} ({conf:.2f})", (frame.shape[1] - 300, 28), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        return frame

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        timestamp = 0
        
        label, conf = 'neutral', 0.0

        while True:
            ok, frame = cap.read()
            if not ok: break
            frame = cv2.flip(frame, 1)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp += 1
            self.landmarker.detect_async(mp_img, timestamp)

            hand_is_visible = self.results and self.results.hand_landmarks
            
            if hand_is_visible:
                landmarks = self.results.hand_landmarks[0]
                self.landmark_buffer.append([[lm.x, lm.y, lm.z] for lm in landmarks])
                self._draw_hand(frame, landmarks)
                
                # Only predict if buffer is full
                if len(self.landmark_buffer) == self.seq_len:
                    label, conf = self._predict_gesture()
            else:
                # !CRITICAL! If hand is lost, clear the buffer.
                # This prevents old data from causing false predictions.
                self.landmark_buffer.clear()
                self.pred_history.clear()
                label, conf = 'neutral', 0.0
                self.current_action = 'neutral'
            
            self._update_action_state(label, conf)
            
            frame = self._draw_overlay(frame, label, conf)
            cv2.imshow('Gesture Controller', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        self.landmarker.close()

if __name__ == '__main__':
    GestureController().run()