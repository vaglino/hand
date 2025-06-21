import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import joblib
from collections import deque
import os

from train_model import create_state_features, STATE_SEQ_LEN
from physics_engine import TrackpadPhysicsEngine

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

class ContinuousMotionExtractor:
    def __init__(self):
        self.prev_landmarks = None
        self.prev_velocity = (0.0, 0.0)

    def extract(self, landmarks):
        if self.prev_landmarks is None:
            self.prev_landmarks = landmarks
            return {'velocity_vector': (0.0, 0.0), 'gesture_intensity': 0.0}
        prev = np.array(self.prev_landmarks[0])
        curr = np.array(landmarks[0])
        velocity = curr[:2] - prev[:2]
        intensity = float(np.linalg.norm(velocity))
        self.prev_landmarks = landmarks
        self.prev_velocity = velocity
        return {'velocity_vector': (float(velocity[0]), float(velocity[1])),
                'gesture_intensity': intensity}

class RealTimeGestureController:
    def __init__(self, model_path='hand_landmarker.task'):
        if not os.path.exists('gesture_data/state_classifier.pkl'):
            raise FileNotFoundError('Models not found. Run train_model.py first.')
        self.state_classifier = joblib.load('gesture_data/state_classifier.pkl')
        self.label_encoder = joblib.load('gesture_data/state_label_encoder.pkl')
        self.intensity_regressor = None
        if os.path.exists('gesture_data/intensity_regressor.pkl'):
            self.intensity_regressor = joblib.load('gesture_data/intensity_regressor.pkl')

        self.landmark_buffer = deque(maxlen=STATE_SEQ_LEN)
        self.motion_extractor = ContinuousMotionExtractor()
        self.physics = TrackpadPhysicsEngine()
        self.gesture_check_interval = 5
        self.frame_count = 0
        self.current_gesture_state = 'neutral'

        # MediaPipe setup
        base_opts = mp.tasks.python.BaseOptions(model_asset_path=model_path)
        opts = mp.tasks.python.vision.HandLandmarkerOptions(
            base_options=base_opts,
            running_mode=mp.tasks.python.vision.RunningMode.LIVE_STREAM,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            result_callback=self._process_result
        )
        self.landmarker = mp.tasks.python.vision.HandLandmarker.create_from_options(opts)
        self.results = None

    def _process_result(self, result, output_image, timestamp_ms):
        self.results = result

    def _predict_gesture_state(self):
        if len(self.landmark_buffer) < STATE_SEQ_LEN:
            return 'neutral'
        features = create_state_features(list(self.landmark_buffer))
        pred = self.state_classifier.predict([features])[0]
        return self.label_encoder.inverse_transform([pred])[0]

    def process_frame(self, landmarks):
        self.frame_count += 1
        self.landmark_buffer.append(landmarks)
        if self.frame_count % self.gesture_check_interval == 0:
            self.current_gesture_state = self._predict_gesture_state()
        if self.current_gesture_state != 'neutral':
            motion = self.motion_extractor.extract(landmarks)
            if self.intensity_regressor:
                vec = list(motion['velocity_vector']) + [motion['gesture_intensity']]
                scale = self.intensity_regressor.predict([vec])[0]
                motion['gesture_intensity'] = scale
            self.physics.apply_force(self.current_gesture_state, motion)

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        timestamp = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp += 1
            self.landmarker.detect_async(mp_img, timestamp)
            if self.results and self.results.hand_landmarks:
                lm = self.results.hand_landmarks[0]
                coords = [[p.x, p.y, p.z] for p in lm]
                self.process_frame(coords)
            self.physics.update(1.0)
            sx, sy = self.physics.get_scroll_delta()
            if self.current_gesture_state.startswith('scroll'):
                pyautogui.scroll(int(-sy * 500))
            zoom = self.physics.get_zoom_delta()
            if self.current_gesture_state.startswith('zoom'):
                if zoom > 0.01:
                    pyautogui.hotkey('ctrl', '+')
                elif zoom < -0.01:
                    pyautogui.hotkey('ctrl', '-')
            cv2.imshow('Gesture Control', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        self.landmarker.close()

if __name__ == '__main__':
    RealTimeGestureController().run()
