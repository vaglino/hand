import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class MediaPipeHandDetector:
    def __init__(
        self,
        model_path: str,
        result_callback,
        num_hands: int = 1,
        min_hand_detection_confidence: float = 0.5,
        min_hand_presence_confidence: float = 0.5,
        running_mode=vision.RunningMode.LIVE_STREAM,
    ):
        base_opts = python.BaseOptions(model_asset_path=model_path)
        opts = vision.HandLandmarkerOptions(
            base_options=base_opts,
            running_mode=running_mode,
            num_hands=num_hands,
            min_hand_detection_confidence=min_hand_detection_confidence,
            min_hand_presence_confidence=min_hand_presence_confidence,
            result_callback=result_callback,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(opts)

    def detect_async(self, mp_img: mp.Image, timestamp_ms: int):
        return self._landmarker.detect_async(mp_img, timestamp_ms)

    def close(self):
        return self._landmarker.close()

