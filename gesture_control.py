# enhanced_gesture_control.py - Optimized inference with HMM smoothing and TorchScript

import cv2
import mediapipe as mp
import numpy as np
import torch
import time
from collections import deque
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import joblib
import os
from typing import Optional, List, Tuple
import warnings
import json
from datetime import datetime

from hand.inference.engine import OptimizedInferenceEngine
from hand.logic.smoothing import PredictionSmoother
from hand.features.landmarks import LandmarkPreprocessor
from hand.models.tcn import EnhancedGestureClassifier  # for typing/reference only
from hand.features.motion import GestureMotionExtractor
from hand.capture.webcam import WebcamStream
from hand.ui.overlay import draw_enhanced_ui
from physics_engine import TrackpadPhysicsEngine, Vector2D
from hand.logic.state_machine import GestureState, update_enhanced_state_machine
from hand.logic.actions import apply_active_action, handle_one_shot_actions
from hand.detect.mediapipe_hand import MediaPipeHandDetector

class EnhancedGestureController:
    """Enhanced gesture controller with optimized inference and improved state management."""
    
    def __init__(self, model_path='hand_landmarker.task'):
        print("Initializing Gesture Controller...")
        
        # Load models and preprocessors
        with open('config.json', 'r') as f:
            full_config = json.load(f)
            self.config = full_config['gesture_control_config']
            self.pointer_config = full_config.get('pointer_config', {})
        self.camera_index = self.config.get('performance', {}).get('camera_index', 0)
        print(f"📹 Using camera index: {self.camera_index}")

        self._load_models()

        # Initialize components
        physics_config = self.config.get('physics_engine', {})
        self.physics_engine = TrackpadPhysicsEngine(config=physics_config)
        self.motion_extractor = GestureMotionExtractor(window_size=5)
        self.landmark_buffer = deque(maxlen=self.sequence_length)
        # Prediction interval (frames) from config for consistency
        self.check_interval_frames = (
            self.config.get('gesture_recognition', {}).get('check_interval_frames', 2)
        )
        
        # Prediction smoothing
        self.prediction_smoother = PredictionSmoother(
            num_classes=len(self.label_encoder.classes_),
            alpha=0.3,  # Moderate smoothing
            confidence_threshold=0.8
        )
        
        # State machine
        self.state = GestureState.NEUTRAL
        self.active_gesture = "neutral"
        self.debounce_counter = 0
        self.debounce_threshold = 5
        self.neutral_counter = 0
        self.neutral_threshold = 2
        
        # Performance monitoring
        self.inference_times = deque(maxlen=100)
        self.frame_count = 0
        self.fps_start_time = time.time()
        
        # MediaPipe setup
        self.results = None
        self.hand_detector = MediaPipeHandDetector(
            model_path=model_path,
            result_callback=self._process_result,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
        )
        
        # Check if OpenCV GUI is available
        self.gui_available = self._check_opencv_gui()
        if not self.gui_available:
            print("⚠️ OpenCV GUI not available - running in headless mode")
            print("Frame data will be saved to 'debug_frames' folder")
        
        print("✅ Controller initialized successfully")

        self.hand_was_present = False # <--- ADD THIS NEW ATTRIBUTE
        self.previous_index_pos = None
        self.flick_times = []
        self.anchor_mouse_pos = None
        self.thumb_openness_history = deque(maxlen=10)  # Buffer for thumb openness
        self.openness_threshold = 0.01  # Tune: distance change for "extended"
        self.flick_cycle_state = 'closed'  # Track thumb state: 'closed', 'extending', 'extended', 'retracting'
        self.last_flick_time = 0
    
    def _check_opencv_gui(self):
        """Check if OpenCV GUI functions are available."""
        try:
            # Try to create a test window
            test_img = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.imshow('test_window', test_img)
            cv2.waitKey(1)
            cv2.destroyWindow('test_window')
            return True
        except cv2.error:
            return False
    
    def _load_models(self):
        """Load all required models and preprocessors."""
        model_dir = 'gesture_data'
        
        # Load inference engine
        self.inference_engine = OptimizedInferenceEngine(model_dir)
        self.sequence_length = self.inference_engine.sequence_length
        
        # Load preprocessor and encoders
        try:
            # Try to load advanced preprocessor
            try:
                self.preprocessor = joblib.load(os.path.join(model_dir, 'landmark_preprocessor.pkl'))
                print("📦 Advanced preprocessor loaded")
            except:
                # Fallback: create a new preprocessor
                print("⚠️ Creating new preprocessor (advanced features may not work)")
                self.preprocessor = LandmarkPreprocessor()
            
            self.scaler = joblib.load(os.path.join(model_dir, 'enhanced_gesture_scaler.pkl'))
            self.label_encoder = joblib.load(os.path.join(model_dir, 'enhanced_gesture_label_encoder.pkl'))
            print("✅ All models and preprocessors loaded")
        except FileNotFoundError as e:
            print(f"❌ Error loading models: {e}")
            print("Please run enhanced training first")
            raise
    
    def _process_result(self, result: vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        """MediaPipe callback for hand detection results."""
        self.results = result
    
    def _predict_gesture(self) -> Tuple[str, float, np.ndarray]:
        """Predict gesture with enhanced preprocessing and smoothing."""
        if len(self.landmark_buffer) < self.sequence_length:
            return "neutral", 0.0, np.zeros(len(self.label_encoder.classes_))
        
        start_time = time.time()
        
        # Extract advanced features
        features = self.preprocessor.extract_advanced_features(list(self.landmark_buffer))
        if features is None:
            return "neutral", 0.0, np.zeros(len(self.label_encoder.classes_))
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Fast inference
        raw_probs = self.inference_engine.predict(features_scaled)
        
        # Apply prediction smoothing
        pred_idx, confidence, smoothed_probs = self.prediction_smoother.update(raw_probs)
        label = self.label_encoder.inverse_transform([pred_idx])[0]
        
        # Record inference time
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time * 1000)  # Convert to ms
        
        return label, confidence, smoothed_probs

    def _handle_one_shot_actions(self, gesture: str) -> bool:
        # Delegate to logic module (behavior unchanged)
        return handle_one_shot_actions(self, gesture)
    
    def _update_enhanced_state_machine(self, predicted_label: str, confidence: float, smoothed_probs: np.ndarray):
        # Delegate to logic module (behavior unchanged)
        return update_enhanced_state_machine(self, predicted_label, confidence, smoothed_probs)
    
    def _apply_active_action(self, landmarks):
        # Delegate to logic module (behavior unchanged)
        return apply_active_action(self, landmarks)
    
    # Pointer helpers moved to hand.logic.pointer (no longer needed here)
    
    
    
    def run(self):
        """Main control loop with enhanced performance monitoring."""
        # cap = cv2.VideoCapture(0)
        # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        # cap.set(cv2.CAP_PROP_FPS, 30)
        cap = WebcamStream(src=self.camera_index).start()
        cap.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.stream.set(cv2.CAP_PROP_FPS, 30)
        
        print("\n" + "="*60)
        print("GESTURE CONTROL ACTIVE")
        print("="*60)
        print("Inference engine ready")
        print("Performance monitoring enabled") 
        print("Prediction smoothing active")
        if self.gui_available:
            print("⌨️  Controls: 'Q' to quit, 'R' to reset state")
        else:
            print("⌨️  Controls: Ctrl+C to quit (headless mode)")
            print("📁 Debug frames saved to 'debug_frames' folder")
            os.makedirs('debug_frames', exist_ok=True)
        print("="*60 + "\n")
        
        last_frame_time = time.time()
        timestamp = 0
        predicted_label, confidence = "neutral", 0.0
        frame_save_counter = 0

        prediction_frame_counter = 0

        
        self.fps_start_time = time.time()
        
        try:
            while True:
                # ret, frame = cap.read()
                # if not ret:
                #     break
                frame = cap.read() # <--- NEW
                if frame is None:
                    break
                    
                frame = cv2.flip(frame, 1)
                
                # MediaPipe processing
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                timestamp += 1
                self.hand_detector.detect_async(mp_img, timestamp)
                
                # Calculate dt for physics
                dt = time.time() - last_frame_time
                last_frame_time = time.time()
                
                # Hand processing
                if self.results and self.results.hand_landmarks:
                     # If hand has just reappeared, reset everything to ensure a clean start
                    if not self.hand_was_present:
                        print("👋 Hand has reappeared. Resetting state.")
                        self.landmark_buffer.clear()
                        self.prediction_smoother.reset()
                        self.physics_engine.reset_momentum()
                        self.motion_extractor.reset() # We'll add this method
                        self.state = GestureState.NEUTRAL
                        prediction_frame_counter = 0
                        self.previous_index_pos = None
                        self.flick_times = []
                        self.anchor_mouse_pos = None
                        self.thumb_openness_history.clear()
                        self.flick_cycle_state = 'closed'
                    
                    self.hand_was_present = True
                    landmarks = self.results.hand_landmarks[0]
                    landmarks_list = [[lm.x, lm.y, lm.z] for lm in landmarks]
                    self.landmark_buffer.append(landmarks_list)
                    
                    # Predict at configured interval (gesture_recognition.check_interval_frames)
                    prediction_frame_counter += 1
                    if prediction_frame_counter >= self.check_interval_frames:
                        prediction_frame_counter = 0 # Reset counter

                        # Enhanced prediction and state management
                        predicted_label, confidence, smoothed_probs = self._predict_gesture()
                        self._update_enhanced_state_machine(predicted_label, confidence, smoothed_probs)

                    # --- END PREDICTION INTERVAL ---
                    # # Enhanced prediction and state management
                    # predicted_label, confidence, smoothed_probs = self._predict_gesture()
                    # self._update_enhanced_state_machine(predicted_label, confidence, smoothed_probs)
                    
                    # Apply physics for continuous gestures
                    self._apply_active_action(landmarks)
                    
                    # Draw hand landmarks
                    proto = landmark_pb2.NormalizedLandmarkList()
                    proto.landmark.extend([
                        landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) 
                        for lm in landmarks
                    ])
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, proto, mp.solutions.hands.HAND_CONNECTIONS
                    )
                else:
                    # --- NO HAND IS VISIBLE ---
                    if self.hand_was_present:
                        if self.physics_engine.is_pointer_active:
                            self.physics_engine.deactivate_pointer()
                    
                    self.hand_was_present = False
                    if self.state != GestureState.NEUTRAL:
                        self.state = GestureState.NEUTRAL
                        self.active_gesture = "neutral"
                        self.prediction_smoother.reset()
                        
                    self.previous_index_pos = None
                    self.flick_times = []
                    self.anchor_mouse_pos = None
                    self.thumb_openness_history.clear()
                    self.flick_cycle_state = 'closed'
                
                # Physics update with calculated dt
                self.physics_engine.update(dt)
                self.physics_engine.execute_smooth_actions()
                
                # UI and display
                frame = draw_enhanced_ui(
                    frame=frame,
                    state=self.state,
                    active_gesture=self.active_gesture,
                    inference_times=self.inference_times,
                    frame_count=self.frame_count,
                    fps_start_time=self.fps_start_time,
                    backend_name=self.inference_engine.backend,
                )
                
                if self.gui_available:
                    # Normal GUI mode
                    cv2.imshow('Enhanced Gesture Control', frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('r'):
                        # Reset state
                        self.state = GestureState.NEUTRAL
                        self.active_gesture = "neutral"
                        self.prediction_smoother.reset()
                        self.physics_engine.reset_momentum()
                        print("State reset")
                else:
                    # Headless mode - save frames periodically and check for interrupt
                    if frame_save_counter % 30 == 0:  # Save every 30 frames (~1 second)
                        frame_path = f"debug_frames/frame_{frame_save_counter:06d}.jpg"
                        cv2.imwrite(frame_path, frame)
                        print(f"Saved frame: {frame_path} | State: {self.state.name} | Gesture: {self.active_gesture}")
                    
                    frame_save_counter += 1
                    
                    # In headless mode, allow reset via console (you'd need to implement this)
                    # For now, just continue running
                
                self.frame_count += 1
        
        except KeyboardInterrupt:
            print("\n🛑 Shutting down...")
        
        finally:
            # Cleanup
            # cap.release()
            cap.stop() # <--- NEW
            if self.gui_available:
                try:
                    cv2.destroyAllWindows()
                except cv2.error:
                    pass  # Ignore errors during cleanup
            self.hand_detector.close()
            
            # Print performance summary
            if self.inference_times:
                print(f"\n" + "="*50)
                print("⚡ PERFORMANCE SUMMARY")
                print("="*50)
                print(f"Average inference time: {np.mean(self.inference_times):.2f}ms")
                print(f"Min inference time: {np.min(self.inference_times):.2f}ms")
                print(f"Max inference time: {np.max(self.inference_times):.2f}ms")
                print(f"Backend used: {self.inference_engine.backend.upper()}")
                print(f"Total frames processed: {self.frame_count}")
                if not self.gui_available:
                    print(f"Debug frames saved to: debug_frames/")
                print("="*50)

if __name__ == '__main__':
    controller = EnhancedGestureController()
    controller.run()
