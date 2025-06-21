# gesture_control.py  –  smoother, less jittery version
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

# ---------- USER-TUNABLE PARAMETERS ----------
CONFIDENCE_THRESHOLD     = 0.75   # min avg confidence for a stable gesture
SMOOTHING_WINDOW         = 7      # frames used for majority vote
STABILITY_RATIO          = 0.7    # ≥ this fraction of same labels → stable
ACTION_COOLDOWN_FRAMES   = 20     # frames to ignore new commands after one fires
SCROLL_PIXELS            = 150    # amount scrolled per gesture
# ---------------------------------------------

pyautogui.FAILSAFE = False
pyautogui.PAUSE    = 0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GestureLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc   = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

class GestureController:
    def __init__(self, model_path='hand_landmarker.task'):
        ckpt_path = 'gesture_data/model.pth'
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError("Model not found – run train_model.py first")

        ckpt = torch.load(ckpt_path, map_location=device)
        self.scaler         = joblib.load('gesture_data/scaler.pkl')
        self.label_encoder  = joblib.load('gesture_data/label_encoder.pkl')

        self.model = GestureLSTM(
            ckpt['input_size'],
            ckpt['hidden_size'],
            ckpt['num_layers'],
            len(ckpt['classes'])
        ).to(device)
        self.model.load_state_dict(ckpt['model_state'])
        self.model.eval()

        self.seq_len          = ckpt['sequence_length']
        self.sequence_buffer  = deque(maxlen=self.seq_len)
        self.pred_history     = deque(maxlen=SMOOTHING_WINDOW)
        self.current_action   = 'neutral'
        self.cooldown         = 0

        # MediaPipe setup
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
        print("Gestures:", ", ".join(self.label_encoder.classes_))

    # ---------- MediaPipe callback ----------
    def _process_result(self, result: vision.HandLandmarkerResult,
                        output_image: mp.Image, timestamp_ms: int):
        self.results = result

    # ---------- Feature engineering ----------
    def _create_features(self, landmark_deque):
        arr   = np.array(landmark_deque, dtype=np.float32)   # (L, 21*2)
        arr   = arr.reshape(self.seq_len, 21, 2)             # (L,21,2)
        wrist = arr[:, 0, :]                                 # (L,2)
        norm  = arr - wrist[:, None, :]                      # (L,21,2)
        vel   = np.diff(norm, axis=0, prepend=norm[0:1])
        return np.concatenate([norm.reshape(self.seq_len, -1),
                               vel.reshape(self.seq_len, -1)], axis=1)

    def _predict_frame(self):
        if len(self.sequence_buffer) < self.seq_len:
            return 'neutral', 0.0
        feats  = self._create_features(self.sequence_buffer)
        feats  = self.scaler.transform(feats).astype(np.float32)
        with torch.no_grad():
            logits = self.model(torch.from_numpy(feats).unsqueeze(0).to(device))
            probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]
        idx = probs.argmax()
        return self.label_encoder.inverse_transform([idx])[0], probs[idx]

    # ---------- Smoothing + cooldown ----------
    def _update_action(self, label, conf):
        self.pred_history.append((label, conf))
        if len(self.pred_history) < SMOOTHING_WINDOW or self.cooldown > 0:
            self.cooldown = max(0, self.cooldown - 1)
            return

        labels      = [l for l, _ in self.pred_history]
        common, cnt = Counter(labels).most_common(1)[0]
        avg_conf    = np.mean([c for l, c in self.pred_history if l == common])

        stable = common if (cnt >= STABILITY_RATIO * SMOOTHING_WINDOW and
                            avg_conf > CONFIDENCE_THRESHOLD) else 'neutral'

        if stable != self.current_action:
            self.current_action = stable
            if stable != 'neutral':
                self._perform_action(stable)
                self.cooldown = ACTION_COOLDOWN_FRAMES

    def _perform_action(self, gesture):
        if gesture == 'scroll_up':
            pyautogui.scroll(SCROLL_PIXELS)
        elif gesture == 'scroll_down':
            pyautogui.scroll(-SCROLL_PIXELS)
        elif gesture == 'zoom_in':
            pyautogui.hotkey('ctrl', '+')
        elif gesture == 'zoom_out':
            pyautogui.hotkey('ctrl', '-')

    # ---------- Drawing ----------
    @staticmethod
    def _draw_hand(frame, landmark_list):
        for hand in landmark_list:
            proto = landmark_pb2.NormalizedLandmarkList()
            proto.landmark.extend([landmark_pb2.NormalizedLandmark(
                x=lm.x, y=lm.y, z=lm.z) for lm in hand])
            mp.solutions.drawing_utils.draw_landmarks(
                frame, proto, mp.solutions.hands.HAND_CONNECTIONS,
                mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                mp.solutions.drawing_styles.get_default_hand_connections_style())

    def _draw_overlay(self, frame):
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 35), (0, 0, 0), -1)
        col = (0, 255, 0) if self.current_action != 'neutral' else (255, 255, 255)
        cv2.putText(frame, f"Action: {self.current_action.upper()}",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
        return frame

    # ---------- Main loop ----------
    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        timestamp       = 0
        last_landmarks  = None

        while True:
            ok, frame = cap.read()
            if not ok: break
            frame = cv2.flip(frame, 1)

            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp += 1
            self.landmarker.detect_async(mp_img, timestamp)

            # Add landmarks to sequence buffer
            if self.results and self.results.hand_landmarks:
                pts = [[lm.x, lm.y] for lm in self.results.hand_landmarks[0]]
                self.sequence_buffer.append(pts)
                last_landmarks = pts
            elif last_landmarks is not None:
                self.sequence_buffer.append(last_landmarks)

            # Predict + update action
            lbl, conf = self._predict_frame()
            self._update_action(lbl, conf)

            # Draw overlays
            if self.results and self.results.hand_landmarks:
                self._draw_hand(frame, self.results.hand_landmarks)
            frame = self._draw_overlay(frame)
            cv2.imshow('Gesture Controller', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        self.landmarker.close()

if __name__ == '__main__':
    GestureController().run()
