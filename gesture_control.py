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

# Disable pyautogui failsafe
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GestureLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_classes=5):
        super().__init__()
        # 1-layer GRU
        self.gru = nn.GRU(input_size, hidden_size, num_layers=1, batch_first=True)
        self.fc  = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

class GestureController:
    def __init__(self, model_path='hand_landmarker.task'):
        # Load model checkpoint & preprocessing
        if not os.path.exists('gesture_data/model.pth'):
            raise FileNotFoundError("Model not found! Run train_model.py first.")
        ckpt = torch.load('gesture_data/model.pth', map_location=device)
        self.scaler = joblib.load('gesture_data/scaler.pkl')
        self.label_encoder = joblib.load('gesture_data/label_encoder.pkl')

        # Build and load network
        self.model = GestureLSTM(
            ckpt['input_size'],
            hidden_size=64,
            num_classes=len(ckpt['classes'])
        ).to(device)
        self.model.load_state_dict(ckpt['model_state'])
        self.model.eval()

        # Sequence buffer
        self.sequence_length = ckpt['sequence_length']
        self.sequence_buffer = deque(maxlen=self.sequence_length)

        # Async MediaPipe state
        self.results = None

        # Action rate limiting
        self.last_action_time = 0
        self.action_cooldown = 0.1

        # Throttle detection
        self.detect_every = 2           # run detect_async every N frames
        self.small_size  = (160, 120)   # downscale input for detection

        # Initialize MediaPipe Tasks API hand landmarker
        base_opts = python.BaseOptions(model_asset_path=model_path)
        opts = vision.HandLandmarkerOptions(
            base_options=base_opts,
            running_mode=vision.RunningMode.LIVE_STREAM,
            num_hands=1,
            min_hand_detection_confidence=0.3,
            min_hand_presence_confidence=0.3,
            min_tracking_confidence=0.3,
            result_callback=self._process_result
        )
        self.landmarker = vision.HandLandmarker.create_from_options(opts)

        print(f"\n✓ Gesture Controller Ready ({device})")
        print("Gestures: scroll_up, scroll_down, zoom_in, zoom_out")

    def _process_result(self, result: vision.HandLandmarkerResult,
                        output_image: mp.Image, timestamp_ms: int):
        # MediaPipe callback — store latest result
        self.results = result

    def extract_features(self, landmarks):
        if not landmarks:
            return None
        pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
        wrist = pts[0]
        norm = pts - wrist
        return norm[:, :2].flatten()

    def predict_gesture(self):
        seq = np.array(self.sequence_buffer)  # (L, F)
        flat = seq.reshape(-1, seq.shape[-1])
        scaled = self.scaler.transform(flat)
        scaled = scaled.reshape(1, self.sequence_length, -1)

        with torch.no_grad():
            t = torch.from_numpy(scaled).float().to(device)
            logits = self.model(t)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        idx = probs.argmax()
        label = self.label_encoder.inverse_transform([idx])[0]
        return label, float(probs[idx])

    def perform_action(self, gesture):
        now = time.time()
        if now - self.last_action_time < self.action_cooldown:
            return
        mod = "command" if sys.platform == "darwin" else "ctrl"
        def _zoom(d):
            pyautogui.keyDown(mod)
            pyautogui.scroll(200 * d)
            pyautogui.keyUp(mod)

        actions = {
            'scroll_up':   lambda: pyautogui.scroll(+10),
            'scroll_down': lambda: pyautogui.scroll(-10),
            'zoom_in':     lambda: _zoom(+1),
            'zoom_out':    lambda: _zoom(-1),
        }
        if gesture in actions:
            actions[gesture]()
            self.last_action_time = now

    def draw_ui(self, frame, gesture, confidence):
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0,0), (w,80), (0,0,0), -1)
        text = f"{gesture} ({confidence:.2f})"
        cv2.putText(frame, text, (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0,255,0) if gesture!='no_gesture' else (255,255,255), 2)
        cv2.putText(frame, "q: quit", (10,h-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
        return frame

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        fps = 0
        frame_count = 0
        start_t = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            # Convert once per loop
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 1) Queue detection only every N frames on a downscaled image
            if frame_count % self.detect_every == 0:
                small = cv2.resize(rgb, self.small_size)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=small)
                self.landmarker.detect_async(mp_img, int(time.time()*1000))

            # 2) If we have landmarks, extract features & classify
            gesture, confidence = 'no_gesture', 0.0
            if self.results and self.results.hand_landmarks:
                lm = self.results.hand_landmarks[0]
                feats = self.extract_features(lm)
                if feats is not None:
                    self.sequence_buffer.append(feats)
                    if len(self.sequence_buffer) == self.sequence_length:
                        gesture, confidence = self.predict_gesture()
                        self.perform_action(gesture)

                # draw landmarks
                for p in lm:
                    x, y = int(p.x * frame.shape[1]), int(p.y * frame.shape[0])
                    cv2.circle(frame, (x,y), 5, (0,255,0), -1)

            # 3) Overlay UI + FPS
            frame = self.draw_ui(frame, gesture, confidence)
            elapsed = time.time() - start_t
            if elapsed > 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                start_t = time.time()
            cv2.putText(frame, f"FPS: {fps:.1f}",
                        (frame.shape[1]-100,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

            # 4) Display & exit
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
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        print("Run in order:\n 1) python gesture_recorder.py\n 2) python train_model.py\n 3) python gesture_control.py")
