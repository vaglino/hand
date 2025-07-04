# gesture_control.py

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from collections import deque
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import joblib
import os
from enum import Enum

from physics_engine import TrackpadPhysicsEngine, GestureMotionExtractor, Vector2D
# We will now import the model definition from train_model to avoid code duplication
from train_model import GestureClassifier, extract_features

# --- Control State Machine ---
class GestureState(Enum):
    NEUTRAL = 1   # Waiting for a gesture
    DEBOUNCING = 2 # A potential gesture has been detected, waiting for confirmation
    ACTIVE = 3    # A gesture is confirmed and actively controlling the system
    RETURNING = 4 # The active gesture is ending, waiting for hand to return to neutral

class GestureController:
    def __init__(self, model_path='hand_landmarker.task'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Initializing Gesture Controller on {self.device}...")
        
        self._load_models()
        self.physics_engine = TrackpadPhysicsEngine()
        self.motion_extractor = GestureMotionExtractor(window_size=5)
        self.landmark_buffer = deque(maxlen=self.sequence_length)
        
        # State machine variables
        self.state = GestureState.NEUTRAL
        self.active_gesture = "neutral"
        self.debounce_counter = 0
        self.debounce_candidate = "neutral"
        self.debounce_threshold = 2  # Needs to see gesture for 3 frames to confirm
        self.neutral_counter = 0
        self.neutral_threshold = 2   # Needs to see neutral for 5 frames to reset state

        # MediaPipe setup
        self.results = None
        base_opts = python.BaseOptions(model_asset_path=model_path)
        opts = vision.HandLandmarkerOptions(
            base_options=base_opts, running_mode=vision.RunningMode.LIVE_STREAM,
            num_hands=1, min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5, result_callback=self._process_result
        )
        self.landmarker = vision.HandLandmarker.create_from_options(opts)
        print("✓ Controller initialized successfully.")

    def _load_models(self):
        classifier_path = 'gesture_data/gesture_classifier.pth'
        if not os.path.exists(classifier_path):
            print(f"Error: Model not found at {classifier_path}. Please run train_model.py first.")
            exit()
            
        checkpoint = torch.load(classifier_path, map_location=self.device)
        self.model = GestureClassifier(
            input_size=checkpoint['input_size'],
            hidden_size=checkpoint['hidden_size'],
            num_layers=checkpoint['num_layers'],
            num_classes=checkpoint['num_classes']
        ).to(self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.model.eval()
        
        self.sequence_length = checkpoint['sequence_length']
        self.scaler = joblib.load('gesture_data/gesture_scaler.pkl')
        self.label_encoder = joblib.load('gesture_data/gesture_label_encoder.pkl')

    def _process_result(self, result: vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        self.results = result

    def _predict_gesture(self):
        if len(self.landmark_buffer) < self.sequence_length:
            return "neutral", 0.0

        features = extract_features(list(self.landmark_buffer))
        if features is None:
            return "neutral", 0.0

        # --- FIX: Correct scaling logic for a single 2D sample ---
        # The 'features' array is already in the correct 2D shape (seq_len, num_features)
        # for the scaler, which was trained on data of shape (N, num_features).
        features_scaled = self.scaler.transform(features)
        
        # Add the batch dimension (T, F) -> (1, T, F) for the model
        features_tensor = torch.FloatTensor(features_scaled).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(features_tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            pred_idx = np.argmax(probs)
            confidence = probs[pred_idx]
            label = self.label_encoder.inverse_transform([pred_idx])[0]
        
        return label, confidence

    def _update_state_machine(self, predicted_label, confidence):
        # --- Step 1: Update the neutral frame counter consistently ---
        # We use a more relaxed confidence threshold here. Any time we see a confident
        # neutral prediction, we increment the counter. Otherwise, we reset it.
        # This counter will be used by the state logic below.
        if predicted_label == 'neutral' and confidence > 0.6:
            self.neutral_counter += 1
        else:
            self.neutral_counter = 0

        # --- STATE: NEUTRAL ---
        if self.state == GestureState.NEUTRAL:
            # If we're not in a neutral state, reset debounce candidate.
            self.debounce_candidate = "neutral" 
            
            # Detect a potential new gesture. We look for non-neutral, non-return gestures.
            # A higher initial confidence helps prevent accidental activation.
            if 'return' not in predicted_label and 'neutral' not in predicted_label and confidence > 0.7:
                self.state = GestureState.DEBOUNCING
                # Clean up the label (e.g., 'scroll_up_start' becomes 'scroll_up')
                self.debounce_candidate = predicted_label.replace('_start', '')
                self.debounce_counter = 1
                
        # --- STATE: DEBOUNCING ---
        elif self.state == GestureState.DEBOUNCING:
            # Check if the prediction still matches our candidate gesture
            if predicted_label.replace('_start', '') == self.debounce_candidate:
                self.debounce_counter += 1
                if self.debounce_counter >= self.debounce_threshold:
                    self.state = GestureState.ACTIVE
                    self.active_gesture = self.debounce_candidate
                    self.neutral_counter = 0 # Reset neutral counter upon activation
            else:
                # Prediction changed or became neutral; abandon debouncing and go back to neutral
                self.state = GestureState.NEUTRAL
        
        # --- STATE: ACTIVE ---
        elif self.state == GestureState.ACTIVE:
            # Condition 1: An explicit "return" gesture is detected. This is the primary way to end.
            if predicted_label == f"{self.active_gesture}_return" and confidence > 0.5:
                self.state = GestureState.RETURNING
                self.neutral_counter = 0 # Reset counter as we enter the returning phase
            
            # Condition 2: The hand has returned to a neutral pose for several frames.
            # This is a robust fallback if the user doesn't make the explicit return gesture.
            elif self.neutral_counter >= self.neutral_threshold:
                self.state = GestureState.NEUTRAL
                self.active_gesture = "neutral"

            # Condition 3 (Optional but good): If a completely different gesture is detected, reset.
            elif predicted_label != self.active_gesture and 'neutral' not in predicted_label and confidence > 0.7:
                self.state = GestureState.NEUTRAL
                self.active_gesture = "neutral"

        # --- STATE: RETURNING ---
        elif self.state == GestureState.RETURNING:
            # The ONLY job of the RETURNING state is to wait for the hand to be verifiably neutral.
            # Now, because our neutral_counter is more reliable, this works correctly.
            if self.neutral_counter >= self.neutral_threshold:
                self.state = GestureState.NEUTRAL
                self.active_gesture = "neutral"

    def _apply_physics(self, landmarks):
        if self.state != GestureState.ACTIVE:
            return

        motion_features = self.motion_extractor.extract_motion_features(landmarks)
        if 'scroll' in self.active_gesture:
            direction = Vector2D(0, 0)
            if 'up' in self.active_gesture: direction = Vector2D(0, 1)
            elif 'down' in self.active_gesture: direction = Vector2D(0, -1)
            self.physics_engine.apply_scroll_force(direction, motion_features['palm_velocity'].magnitude())
        elif 'zoom' in self.active_gesture:
            zoom_rate = motion_features['spread_velocity']
            # Ensure the motion direction matches the gesture
            if ('in' in self.active_gesture and zoom_rate > 0) or ('out' in self.active_gesture and zoom_rate < 0):
                self.physics_engine.apply_zoom_force(zoom_rate, 1.0)
    
    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print("\n=== Gesture Control Active ===")
        print("System is ready. Press 'Q' to quit.\n")
        
        last_frame_time = time.time()
        timestamp = 0
        predicted_label, confidence = "neutral", 0.0

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
                landmarks_list = [[lm.x, lm.y, lm.z] for lm in landmarks]
                self.landmark_buffer.append(landmarks_list)

                predicted_label, confidence = self._predict_gesture()
                self._update_state_machine(predicted_label, confidence)
                self._apply_physics(landmarks)
                
                # Draw hand landmarks
                proto = landmark_pb2.NormalizedLandmarkList()
                proto.landmark.extend([landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in landmarks])
                mp.solutions.drawing_utils.draw_landmarks(frame, proto, mp.solutions.hands.HAND_CONNECTIONS)
            else:
                self.state = GestureState.NEUTRAL
                self.active_gesture = "neutral"
            
            # Physics update
            dt = time.time() - last_frame_time
            last_frame_time = time.time()
            self.physics_engine.update(dt)
            self.physics_engine.execute_smooth_actions()
            
            # --- UI Display ---
            cv2.rectangle(frame, (0,0), (frame.shape[1], 80), (0,0,0), -1)
            cv2.putText(frame, f"STATE: {self.state.name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"ACTION: {self.active_gesture.upper() if self.state == GestureState.ACTIVE else 'NONE'}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Pred: {predicted_label} ({confidence:.2f})", (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            cv2.imshow('Gesture Control', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        self.landmarker.close()

if __name__ == '__main__':
    controller = GestureController()
    controller.run()