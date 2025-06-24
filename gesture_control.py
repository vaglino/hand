# gesture_control.py - Revolutionary physics-based gesture control system

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import torch
import torch.nn as nn
import joblib
import time
from collections import deque
from mediapipe.framework.formats import landmark_pb2
import warnings
warnings.filterwarnings('ignore')

# Import physics engine and models
from physics_engine import TrackpadPhysicsEngine, GestureMotionExtractor, Vector2D
from train_model import LightweightGestureNet, AdvancedFeatureExtractor

# Performance constants
GESTURE_CHECK_INTERVAL = 2  # Check gesture state every N frames
MOTION_WINDOW_SIZE = 5      # Frames for motion smoothing
MIN_CONFIDENCE = 0.7       # Minimum confidence for gesture activation

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RevolutionaryGestureController:
    """Hybrid continuous gesture control with physics-based actuation"""
    
    def __init__(self, model_path='hand_landmarker.task'):
        print("Initializing Revolutionary Gesture Controller...")
        
        # Load models
        self._load_models()
        
        # Physics engine
        self.physics_engine = TrackpadPhysicsEngine()
        self.motion_extractor = GestureMotionExtractor(window_size=MOTION_WINDOW_SIZE)
        
        # State management
        self.current_gesture = 'neutral'
        self.gesture_confidence = 0.0
        self.frame_count = 0
        self.last_gesture_check = 0
        
        # Buffers
        self.landmark_buffer = deque(maxlen=self.sequence_length)
        self.raw_landmark_buffer = deque(maxlen=30)  # For motion extraction
        
        # Performance tracking
        self.fps_counter = deque(maxlen=30)
        self.last_frame_time = time.time()
        
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
        
        # Gesture mapping
        self.gesture_map = {
            'scroll_up': 'scroll_vertical',
            'scroll_down': 'scroll_vertical',
            'scroll_left': 'scroll_horizontal',
            'scroll_right': 'scroll_horizontal',
            'zoom_in': 'zoom',
            'zoom_out': 'zoom',
            'neutral': 'neutral'
        }
        
        print(f"✓ Controller initialized on {device}")
        print(f"✓ Available gestures: {', '.join(self.label_encoder.classes_)}")
        
    def _load_models(self):
        """Load trained models and preprocessors"""
        try:
            # Load feature extractor
            self.feature_extractor = joblib.load('gesture_data/feature_extractor.pkl')
            
            # Load gesture classifier
            if os.path.exists('gesture_data/gesture_classifier.pth'):
                # Neural network classifier
                checkpoint = torch.load('gesture_data/gesture_classifier.pth', map_location=device)
                self.classifier_type = 'neural_network'
                self.sequence_length = checkpoint['sequence_length']
                
                self.gesture_model = LightweightGestureNet(
                    checkpoint['input_size'],
                    checkpoint['num_classes']
                ).to(device)
                self.gesture_model.load_state_dict(checkpoint['model_state'])
                self.gesture_model.eval()
                
                classes = checkpoint['classes']
            else:
                # Random forest classifier
                self.gesture_model = joblib.load('gesture_data/gesture_classifier.pkl')
                info = joblib.load('gesture_data/classifier_info.pkl')
                self.classifier_type = 'random_forest'
                self.sequence_length = info['sequence_length']
                classes = info['classes']
            
            # Load preprocessors
            self.scaler = joblib.load('gesture_data/gesture_scaler.pkl')
            self.label_encoder = joblib.load('gesture_data/label_encoder.pkl')
            
            # Load motion models if available
            if os.path.exists('gesture_data/motion_models.pkl'):
                self.motion_models = joblib.load('gesture_data/motion_models.pkl')
                print("✓ Motion prediction models loaded")
            else:
                self.motion_models = None
                print("ℹ Motion models not found - using direct motion extraction")
                
        except Exception as e:
            raise RuntimeError(f"Failed to load models: {e}")
    
    def _process_result(self, result: vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        """MediaPipe callback"""
        self.results = result
    
    def _predict_gesture_state(self):
        """Fast gesture classification"""
        if len(self.landmark_buffer) < self.sequence_length:
            return 'neutral', 0.0
        
        # Extract features
        sequence = list(self.landmark_buffer)
        features = self.feature_extractor.extract_gesture_features(sequence)
        
        # Scale features
        features_scaled = self.scaler.transform([features])
        
        # Predict
        if self.classifier_type == 'neural_network':
            with torch.no_grad():
                inputs = torch.FloatTensor(features_scaled).to(device)
                outputs = self.gesture_model(inputs)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
                
            pred_idx = probs.argmax()
            confidence = probs[pred_idx]
        else:
            # Random forest
            pred_idx = self.gesture_model.predict(features_scaled)[0]
            probs = self.gesture_model.predict_proba(features_scaled)[0]
            confidence = probs[pred_idx]
        
        gesture = self.label_encoder.inverse_transform([pred_idx])[0]
        return gesture, float(confidence)
    
    def _extract_motion_parameters(self, motion_features):
        """Convert motion features to physics parameters"""
        palm_velocity = motion_features['palm_velocity']
        finger_spread_vel = motion_features['spread_velocity']
        confidence = motion_features['gesture_confidence']
        smoothness = motion_features['motion_smoothness']
        
        # Determine motion type and parameters based on current gesture
        if self.current_gesture in ['scroll_up', 'scroll_down', 'scroll_left', 'scroll_right']:
            # Scrolling motion
            if 'up' in self.current_gesture:
                direction = Vector2D(0, 1)
            elif 'down' in self.current_gesture:
                direction = Vector2D(0, -1)
            elif 'left' in self.current_gesture:
                direction = Vector2D(-1, 0)
            else:  # right
                direction = Vector2D(1, 0)
            
            # Use palm velocity magnitude as intensity
            intensity = min(palm_velocity.magnitude() * 2.0, 1.0)
            
            # Blend gesture direction with actual motion direction for natural feel
            actual_direction = palm_velocity.normalize()
            if actual_direction.magnitude() > 0:
                # 70% actual motion, 30% gesture direction
                blended_direction = (actual_direction * 0.7 + direction * 0.3).normalize()
            else:
                blended_direction = direction
            
            return {
                'type': 'scroll',
                'direction': blended_direction,
                'intensity': intensity * confidence * smoothness,
                'raw_velocity': palm_velocity
            }
            
        elif self.current_gesture in ['zoom_in', 'zoom_out']:
            # Zooming motion
            # The sign of finger_spread_vel already gives us the correct direction.
            # Positive = open (zoom in), Negative = close (zoom out).
            zoom_rate = finger_spread_vel 
            intensity = min(abs(finger_spread_vel) * 3.0, 1.0)
            
            return {
                'type': 'zoom',
                'zoom_rate': zoom_rate,
                'intensity': intensity * confidence * smoothness
            }
        
        return None
    
    def _update_physics(self, dt):
        """Update physics simulation"""
        # Update physics engine
        self.physics_engine.update(dt)
        
        # Execute smooth actions
        self.physics_engine.execute_smooth_actions()
    
    def _draw_overlay(self, frame):
        """Draw enhanced UI overlay"""
        h, w = frame.shape[:2]
        
        # Top bar with gradient
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 50), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Current state
        state_text = f"Gesture: {self.current_gesture.upper()}"
        state_color = (0, 255, 0) if self.current_gesture != 'neutral' else (150, 150, 150)
        cv2.putText(frame, state_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, state_color, 2)
        
        # Confidence bar
        if self.gesture_confidence > 0:
            conf_width = int(self.gesture_confidence * 200)
            cv2.rectangle(frame, (250, 15), (250 + conf_width, 35), (0, 255, 255), -1)
            cv2.rectangle(frame, (250, 15), (450, 35), (100, 100, 100), 2)
        
        # FPS counter
        if len(self.fps_counter) > 0:
            avg_fps = 1.0 / np.mean(list(self.fps_counter))
            cv2.putText(frame, f"FPS: {avg_fps:.1f}", (w - 100, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
        # Physics state visualization
        physics_state = self.physics_engine.get_physics_state()
        
        # Momentum indicator
        if self.current_gesture != 'neutral':
            # Draw momentum vector
            center_x, center_y = w // 2, h - 100
            momentum = physics_state['scroll_momentum']
            
            if momentum[0] != 0 or momentum[1] != 0:
                # Scale and draw momentum vector
                end_x = int(center_x + momentum[0] * 2)
                end_y = int(center_y - momentum[1] * 2)
                cv2.arrowedLine(frame, (center_x, center_y), (end_x, end_y), 
                               (0, 255, 0), 3, tipLength=0.3)
            
            # Zoom indicator
            zoom_vel = physics_state['zoom_velocity']
            if abs(zoom_vel) > 0.01:
                zoom_radius = int(30 + abs(zoom_vel) * 100)
                color = (0, 150, 255) if zoom_vel > 0 else (255, 150, 0)
                cv2.circle(frame, (w - 100, h - 100), zoom_radius, color, 2)
        
        # Gesture intensity indicator
        if 'gesture_intensities' in physics_state and physics_state['gesture_intensities']:
            y_pos = 70
            for gesture_type, intensity in physics_state['gesture_intensities'].items():
                cv2.putText(frame, f"{gesture_type}: ", (10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
                # Intensity bar
                bar_width = int(intensity * 100)
                cv2.rectangle(frame, (100, y_pos - 10), (100 + bar_width, y_pos), 
                             (0, 200, 255), -1)
                y_pos += 20
        
        return frame
    
    def _draw_hand_landmarks(self, frame, landmarks):
        """Draw hand with enhanced visualization"""
        # Convert to proto format
        proto = landmark_pb2.NormalizedLandmarkList()
        proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in landmarks
        ])
        
        # Draw with custom style
        mp.solutions.drawing_utils.draw_landmarks(
            frame, proto, mp.solutions.hands.HAND_CONNECTIONS,
            mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
            mp.solutions.drawing_styles.get_default_hand_connections_style()
        )
        
        # Add gesture-specific visualization
        if self.current_gesture != 'neutral':
            h, w = frame.shape[:2]
            
            # Highlight relevant landmarks
            if 'scroll' in self.current_gesture:
                # Highlight palm center
                palm_indices = [0, 5, 9, 13, 17]
                palm_x = int(np.mean([landmarks[i].x for i in palm_indices]) * w)
                palm_y = int(np.mean([landmarks[i].y for i in palm_indices]) * h)
                cv2.circle(frame, (palm_x, palm_y), 10, (0, 255, 255), 2)
                
            elif 'zoom' in self.current_gesture:
                # Highlight fingertips
                fingertips = [4, 8, 12, 16, 20]
                for tip_idx in fingertips:
                    x = int(landmarks[tip_idx].x * w)
                    y = int(landmarks[tip_idx].y * h)
                    cv2.circle(frame, (x, y), 5, (255, 0, 255), -1)
    
    def run(self):
        """Main control loop"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("\n=== Revolutionary Gesture Control Active ===")
        print("Experience smooth, trackpad-like control with your hands!")
        print("Press 'Q' to quit, 'R' to reset physics\n")
        
        timestamp = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            self.frame_count += 1
            
            # Track FPS
            current_time = time.time()
            frame_dt = current_time - self.last_frame_time
            self.fps_counter.append(frame_dt)
            self.last_frame_time = current_time
            
            # Process hand tracking
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp += 1
            self.landmarker.detect_async(mp_img, timestamp)
            
            # Process hand landmarks
            if self.results and self.results.hand_landmarks:
                landmarks = self.results.hand_landmarks[0]
                landmarks_list = [[lm.x, lm.y, lm.z] for lm in landmarks]
                
                # Update buffers
                self.landmark_buffer.append(landmarks_list)
                self.raw_landmark_buffer.append(landmarks)
                
                # Stage 1: Gesture state detection (periodic)
                if self.frame_count - self.last_gesture_check >= GESTURE_CHECK_INTERVAL:
                    if len(self.landmark_buffer) >= self.sequence_length:
                        gesture, confidence = self._predict_gesture_state()
                        
                        # Update state if confident
                        if confidence > MIN_CONFIDENCE:
                            if gesture != self.current_gesture:
                                print(f"Gesture changed: {self.current_gesture} → {gesture} ({confidence:.2f})")
                            self.current_gesture = gesture
                            self.gesture_confidence = confidence
                        elif self.current_gesture != 'neutral':
                            # Decay confidence if below threshold
                            self.gesture_confidence *= 0.9
                            if self.gesture_confidence < 0.3:
                                self.current_gesture = 'neutral'
                    
                    self.last_gesture_check = self.frame_count
                
                # Stage 2: Continuous motion extraction
                if self.current_gesture != 'neutral':
                    motion_features = self.motion_extractor.extract_motion_features(landmarks)
                    
                    # Stage 3: Convert to physics parameters
                    motion_params = self._extract_motion_parameters(motion_features)
                    
                    if motion_params:
                        # Apply forces to physics engine
                        if motion_params['type'] == 'scroll':
                            self.physics_engine.apply_scroll_force(
                                motion_params['direction'],
                                motion_params['intensity'],
                                motion_params.get('raw_velocity')
                            )
                        elif motion_params['type'] == 'zoom':
                            self.physics_engine.apply_zoom_force(
                                motion_params['zoom_rate'],
                                motion_params['intensity']
                            )
                
                # Draw hand visualization
                self._draw_hand_landmarks(frame, landmarks)
                
            else:
                # No hand detected - clear buffers and reset
                self.landmark_buffer.clear()
                self.raw_landmark_buffer.clear()
                self.current_gesture = 'neutral'
                self.gesture_confidence = 0.0
            
            # Update physics simulation
            self._update_physics(frame_dt)
            
            # Draw UI overlay
            frame = self._draw_overlay(frame)
            
            # Display
            cv2.imshow('Revolutionary Gesture Control', frame)
            
            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.physics_engine.reset_momentum()
                print("Physics reset")
            elif key == ord('+'):
                self.physics_engine.user_scroll_multiplier *= 1.2
                print(f"Scroll sensitivity: {self.physics_engine.user_scroll_multiplier:.2f}x")
            elif key == ord('-'):
                self.physics_engine.user_scroll_multiplier /= 1.2
                print(f"Scroll sensitivity: {self.physics_engine.user_scroll_multiplier:.2f}x")
        
        cap.release()
        cv2.destroyAllWindows()
        self.landmarker.close()

if __name__ == '__main__':
    import os
    if not os.path.exists('gesture_data/gesture_classifier.pth') and \
       not os.path.exists('gesture_data/gesture_classifier.pkl'):
        print("Error: No trained models found. Please run train_model.py first!")
        exit(1)
    
    controller = RevolutionaryGestureController()
    controller.run()