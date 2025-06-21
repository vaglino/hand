import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import json
import os
from collections import deque

class GestureRecorder:
    """
    Records fixed-length gesture sequences: starts when user presses a key,
    begins buffering once hand is detected, and stops when sequence_length
    frames have been collected or when the hand disappears after detection.
    """
    def __init__(self,
                 model_path='hand_landmarker.task',
                 sequence_length=30,
                 detect_every=2,
                 small_size=(160, 120),
                 miss_threshold=5):
        self.results = None
        self.sequence_length = sequence_length
        self.current_sequence = deque(maxlen=sequence_length)
        self.gesture_data = {g: [] for g in ['scroll_up','scroll_down','zoom_in','zoom_out','neutral']}

        # Async detection throttling
        self.detect_every = detect_every
        self.small_size = small_size
        self.frame_count = 0

        # Auto-stop parameters
        self.miss_threshold = miss_threshold
        self.miss_count = 0
        self.started_appending = False

        # MediaPipe async hand landmarker
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

    def _process_result(self, result: vision.HandLandmarkerResult,
                        output_image: mp.Image, timestamp_ms: int):
        self.results = result

    def extract_features(self, landmarks):
        if not landmarks:
            return None
        pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
        wrist = pts[0]
        norm = pts - wrist
        return norm[:, :2].flatten()

    def draw_ui(self, frame, recording=None):
        h, w = frame.shape[:2]
        if recording:
            txt = f"Recording {recording.upper()}: {len(self.current_sequence)}/{self.sequence_length}"
            color = (0,0,255)
        else:
            txt = "1-5: start | S: save | Q: quit"
            color = (0,255,0)
        cv2.putText(frame, txt, (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        y = 60
        for g, seqs in self.gesture_data.items():
            cv2.putText(frame, f"{g}: {len(seqs)}", (10,y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            y += 25
        return frame

    def save_data(self):
        os.makedirs('gesture_data', exist_ok=True)
        out = {g: seqs for g, seqs in self.gesture_data.items() if seqs}
        with open('gesture_data/sequences.json', 'w') as f:
            json.dump(out, f)
        print("\n✓ Data saved!")
        for g, seqs in out.items():
            print(f"  {g}: {len(seqs)} sequences")

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

        print("\n=== Gesture Recorder ===")
        print("1: scroll_up | 2: scroll_down | 3: zoom_in | 4: zoom_out | 5: neutral")
        print("S: save | Q: quit\n")

        recording = None
        cooldown = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            self.frame_count += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if self.frame_count % self.detect_every == 0:
                small = cv2.resize(rgb, self.small_size)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=small)
                self.landmarker.detect_async(mp_img, int(time.time()*1000))

            # Recording logic
            if recording:
                feats = None
                if self.results and self.results.hand_landmarks:
                    self.miss_count = 0
                    lm = self.results.hand_landmarks[0]
                    feats = self.extract_features(lm)
                else:
                    if self.started_appending:
                        self.miss_count += 1

                if feats is not None:
                    self.current_sequence.append(feats.tolist())
                    self.started_appending = True

                # Auto-stop on full buffer
                if len(self.current_sequence) == self.sequence_length:
                    self.gesture_data[recording].append(list(self.current_sequence))
                    print(f"✓ Recorded {recording} ({self.sequence_length} frames)")
                    recording = None
                    cooldown = 30
                    self.current_sequence.clear()
                    self.started_appending = False
                    self.miss_count = 0
                # Auto-stop on too many misses after start
                elif self.started_appending and self.miss_count >= self.miss_threshold:
                    self.gesture_data[recording].append(list(self.current_sequence))
                    print(f"✓ Recorded {recording} ({len(self.current_sequence)} frames)")
                    recording = None
                    cooldown = 30
                    self.current_sequence.clear()
                    self.started_appending = False
                    self.miss_count = 0

            # Draw landmarks
            if self.results and self.results.hand_landmarks:
                for lm in self.results.hand_landmarks[0]:
                    x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                    cv2.circle(frame, (x,y), 5, (0,255,0), -1)

            if cooldown > 0:
                cooldown -= 1
                cv2.putText(frame, "Ready...", (250,240),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

            frame = self.draw_ui(frame, recording)
            cv2.imshow('Gesture Recorder', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.save_data()
            elif cooldown == 0 and not recording:
                if key in map(ord, ['1','2','3','4','5']):
                    mapping = {'1':'scroll_up','2':'scroll_down',
                               '3':'zoom_in','4':'zoom_out','5':'neutral'}
                    recording = mapping[chr(key)]
                    self.current_sequence.clear()
                    self.started_appending = False
                    self.miss_count = 0

        cap.release()
        cv2.destroyAllWindows()
        self.landmarker.close()

if __name__ == "__main__":
    recorder = GestureRecorder()
    recorder.run()
