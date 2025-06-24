# gesture_control.py

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import torch
import torch.nn as nn
import time
from collections import deque
from mediapipe.framework.formats import landmark_pb2
import warnings
warnings.filterwarnings('ignore')

from physics_engine import TrackpadPhysicsEngine, GestureMotionExtractor, Vector2D
import joblib
import os

DEBUG = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- FIX: Copied the new simple model definition here ---
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.5)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, context=None):
        lstm_out, _ = self.lstm(x)
        last_step_out = lstm_out[:, -1, :]
        out = self.dropout(last_step_out)
        out = self.fc(out)
        return {'gesture': out}

class EnhancedGestureController:
    def __init__(self, model_path='hand_landmarker.task'):
        print("Initializing Simplified Gesture Controller...")
        self._load_models()
        self.physics_engine = TrackpadPhysicsEngine()
        self.motion_extractor = GestureMotionExtractor(window_size=5)
        self.current_gesture = 'neutral'
        self.gesture_confidence = 0.0
        self.landmark_buffer = deque(maxlen=self.sequence_length)
        self.last_frame_time = time.time()
        self.fps_counter = deque(maxlen=30)
        self.results = None
        
        base_opts = python.BaseOptions(model_asset_path=model_path)
        opts = vision.HandLandmarkerOptions(
            base_options=base_opts, running_mode=vision.RunningMode.LIVE_STREAM,
            num_hands=1, min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5, result_callback=self._process_result
        )
        self.landmarker = vision.HandLandmarker.create_from_options(opts)
        print(f"✓ Controller initialized on {device}")

    def _load_models(self):
        model_path = 'gesture_data/gesture_model.pth' # Using new model name
        if not os.path.exists(model_path):
            print(f"Error: Model not found at {model_path}. Please run train_model.py.")
            exit()
            
        checkpoint = torch.load(model_path, map_location=device)
        self.model = SimpleLSTM(
            input_size=checkpoint['input_size'], hidden_size=64, num_layers=2,
            num_classes=checkpoint['num_classes']
        ).to(device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.model.eval()
        
        self.sequence_length = checkpoint['sequence_length']
        self.scaler = joblib.load('gesture_data/gesture_scaler.pkl')
        self.label_encoder = joblib.load('gesture_data/gesture_label_encoder.pkl')

    def _process_result(self, result: vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        self.results = result

    # --- FIX: Use the same simplified feature extraction as in training ---
    def _extract_features(self, landmarks_sequence):
        sequence = np.array(landmarks_sequence)
        features = []
        for frame in sequence:
            wrist = frame[0]
            relative_positions = (frame - wrist).flatten()
            features.append(relative_positions)
        return np.array(features)

    def _predict_gesture(self):
        if len(self.landmark_buffer) < self.sequence_length:
            return "neutral", 0.0

        features = self._extract_features(list(self.landmark_buffer))
        features_scaled = self.scaler.transform(features)
        features_tensor = torch.FloatTensor(features_scaled).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = self.model(features_tensor)['gesture']
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            pred_idx = np.argmax(probs)
            confidence = probs[pred_idx]
            label = self.label_encoder.inverse_transform([pred_idx])[0]
        
        return label, confidence

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print("\n=== Simplified Gesture Control Active ===")
        print("Press 'Q' to quit.\n")
        
        timestamp = 0
        actionable_gesture = "neutral"
        action_frames = 0
        MIN_ACTION_FRAMES = 3 # Frames a gesture must be held to be actioned

        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            
            # --- Hand Tracking ---
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp += 1
            self.landmarker.detect_async(mp_img, timestamp)
            
            # --- Prediction and State Logic ---
            if self.results and self.results.hand_landmarks:
                landmarks = self.results.hand_landmarks[0]
                landmarks_list = [[lm.x, lm.y, lm.z] for lm in landmarks]
                self.landmark_buffer.append(landmarks_list)

                predicted_label, confidence = self._predict_gesture()
                self.current_gesture = predicted_label
                self.gesture_confidence = confidence

                # --- FIX: Clean logic for acting on gestures ---
                # An "actionable" gesture is one that is NOT neutral and NOT a return motion
                is_actionable = "neutral" not in predicted_label and "_return" not in predicted_label
                
                if is_actionable and confidence > 0.7:
                    if predicted_label == actionable_gesture:
                        action_frames += 1
                    else:
                        actionable_gesture = predicted_label
                        action_frames = 1
                else:
                    actionable_gesture = "neutral"
                    action_frames = 0
                
                # --- Physics Application ---
                if action_frames >= MIN_ACTION_FRAMES:
                    motion_features = self.motion_extractor.extract_motion_features(landmarks)
                    if 'scroll' in actionable_gesture:
                        if 'up' in actionable_gesture: direction = Vector2D(0, 1)
                        elif 'down' in actionable_gesture: direction = Vector2D(0, -1)
                        elif 'left' in actionable_gesture: direction = Vector2D(-1, 0)
                        else: direction = Vector2D(1, 0)
                        self.physics_engine.apply_scroll_force(direction, motion_features['palm_velocity'].magnitude())
                    elif 'zoom' in actionable_gesture:
                        zoom_rate = motion_features['spread_velocity']
                        if 'in' in actionable_gesture and zoom_rate < 0: zoom_rate = abs(zoom_rate)
                        elif 'out' in actionable_gesture and zoom_rate > 0: zoom_rate = -abs(zoom_rate)
                        self.physics_engine.apply_zoom_force(zoom_rate, 1.0)

                # --- Drawing and Debug ---
                if DEBUG:
                    debug_text = f"Pred: {predicted_label:<25} | Conf: {confidence:.2f} | Action: {actionable_gesture if action_frames>=MIN_ACTION_FRAMES else 'None'}"
                    print(debug_text, end='\r')
                
                proto = landmark_pb2.NormalizedLandmarkList()
                proto.landmark.extend([landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in landmarks])
                mp.solutions.drawing_utils.draw_landmarks(frame, proto, mp.solutions.hands.HAND_CONNECTIONS)
            else:
                self.current_gesture = 'neutral'
            
            # --- Physics Update and UI ---
            frame_dt = time.time() - self.last_frame_time
            self.last_frame_time = time.time()
            self.physics_engine.update(frame_dt)
            self.physics_engine.execute_smooth_actions()
            cv2.putText(frame, f"Gesture: {self.current_gesture.upper()}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow('Gesture Control', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        self.landmarker.close()

if __name__ == '__main__':
    controller = EnhancedGestureController()
    controller.run()