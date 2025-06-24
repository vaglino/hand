# enhanced_gesture_control.py - Gesture control with transition awareness

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import torch
import time
from collections import deque
from mediapipe.framework.formats import landmark_pb2
import warnings
warnings.filterwarnings('ignore')

# Import components
from physics_engine import TrackpadPhysicsEngine, GestureMotionExtractor, Vector2D
from transition_aware_model import TransitionAwareLSTM, GestureIntentionDetector
import joblib
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class EnhancedGestureController:
    """Gesture controller with transition awareness and intention detection"""
    
    def __init__(self, model_path='hand_landmarker.task'):
        print("Initializing Enhanced Gesture Controller...")
        
        # Load models and preprocessors
        self._load_models()
        
        # Physics engine with intention awareness
        self.physics_engine = TrackpadPhysicsEngine()
        self.motion_extractor = GestureMotionExtractor(window_size=5)
        
        # Intention detection
        self.intention_detector = GestureIntentionDetector(window_size=10)
        
        # Gesture state
        self.current_gesture = 'neutral'
        self.gesture_confidence = 0.0
        self.previous_gesture = 'neutral'
        self.in_transition = False
        self.gesture_commitment_frames = 0
        self.min_commitment_frames = 5
        
        # Buffers
        self.landmark_buffer = deque(maxlen=self.sequence_length)
        self.prediction_history = deque(maxlen=10)
        self.phase_history = deque(maxlen=30)
        
        # Performance
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
        
        print(f"✓ Enhanced controller initialized on {device}")
        print("✓ Transition-aware model loaded")
        print("✓ Intention detection enabled")
        
    def _load_models(self):
        """Load transition-aware models"""
        model_path = 'gesture_data/transition_aware_model.pth'
        
        # Check if enhanced model exists, fall back to original if not
        if not os.path.exists(model_path):
            print("Warning: Enhanced model not found. Please run enhanced_train_model.py")
            print("Falling back to original model...")
            # Load original model components
            self._load_original_models()
            return
        
        # Load enhanced model
        checkpoint = torch.load(model_path, map_location=device)
        
        self.model = TransitionAwareLSTM(
            input_size=checkpoint['input_size'],
            hidden_size=128,
            num_layers=2,
            num_classes=checkpoint['num_classes']
        ).to(device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.model.eval()
        
        self.sequence_length = checkpoint['sequence_length']
        self.classes = checkpoint['classes']
        self.context_classes = checkpoint['context_classes']
        self.transition_matrix = checkpoint.get('transition_matrix', {})
        
        # Load preprocessors
        self.scaler = joblib.load('gesture_data/transition_scaler.pkl')
        self.label_encoder = joblib.load('gesture_data/transition_label_encoder.pkl')
        self.context_encoder = joblib.load('gesture_data/context_encoder.pkl')
        
        # Map enhanced labels to base gestures
        self.label_to_gesture = {}
        for label in self.classes:
            if '_return' in label or '_start' in label:
                base_gesture = label.replace('_return', '').replace('_start', '')
                self.label_to_gesture[label] = base_gesture
            else:
                self.label_to_gesture[label] = label
        
    def _load_original_models(self):
        """Fallback to load original models"""
        # This would load the original model components
        # Implementation depends on your original model structure
        pass
    
    def _process_result(self, result: vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        self.results = result
    
    def _extract_features(self, landmarks_sequence):
        """Extract features for the model"""
        sequence = np.array(landmarks_sequence)
        features = []
        
        for t in range(len(sequence)):
            frame_features = []
            
            # Normalize to wrist
            wrist = sequence[t][0]
            relative_positions = sequence[t] - wrist
            frame_features.extend(relative_positions.flatten())
            
            # Velocity
            if t > 0:
                velocity = sequence[t] - sequence[t-1]
                frame_features.extend(velocity.flatten())
            else:
                frame_features.extend(np.zeros(21 * 3))
            
            # Acceleration
            if t > 1:
                acceleration = (sequence[t] - sequence[t-1]) - (sequence[t-1] - sequence[t-2])
                frame_features.extend(acceleration.flatten())
            else:
                frame_features.extend(np.zeros(21 * 3))
            
            # Additional features (simplified)
            features.append(frame_features)
        
        return np.array(features)
    
    def _predict_gesture(self):
        """Predict gesture with transition awareness"""
        if len(self.landmark_buffer) < self.sequence_length:
            return None, 0.0, 'neutral'
        
        # Extract features
        sequence = list(self.landmark_buffer)
        features = self._extract_features(sequence)
        
        # Reshape and scale
        features_flat = features.reshape(1, -1)
        features_scaled = self.scaler.transform(features_flat)
        features_scaled = features_scaled.reshape(1, *features.shape)
        
        # Get context (previous gesture)
        context_label = self.previous_gesture if self.previous_gesture != 'neutral' else 'none'
        try:
            context_idx = self.context_encoder.transform([context_label])[0]
        except:
            context_idx = self.context_encoder.transform(['none'])[0]
        
        # Model prediction
        with torch.no_grad():
            inputs = torch.FloatTensor(features_scaled).to(device)
            context = torch.LongTensor([context_idx]).to(device)
            
            outputs = self.model(inputs, context)
            
            # Gesture prediction
            gesture_probs = torch.softmax(outputs['gesture'], dim=1).cpu().numpy()[0]
            gesture_idx = gesture_probs.argmax()
            gesture_confidence = gesture_probs[gesture_idx]
            gesture_label = self.label_encoder.inverse_transform([gesture_idx])[0]
            
            # Transition prediction
            transition_probs = torch.softmax(outputs['transition'], dim=1).cpu().numpy()[0]
            transition_idx = transition_probs.argmax()
            transition_states = ['neutral', 'transitioning', 'active']
            transition_state = transition_states[transition_idx]
        
        # Map to base gesture
        base_gesture = self.label_to_gesture.get(gesture_label, gesture_label)
        
        return base_gesture, float(gesture_confidence), transition_state
    
    def _update_gesture_state(self, prediction, motion_features):
        """Update gesture state with intention detection"""
        if prediction is None:
            return
        
        gesture, confidence, transition_state = prediction
        
        # Analyze motion intention
        motion_data = {
            'velocity': motion_features['palm_velocity'].magnitude(),
            'acceleration': 0.0,  # Could calculate if needed
            'direction': [motion_features['palm_velocity'].x, motion_features['palm_velocity'].y]
        }
        
        intent_analysis = self.intention_detector.analyze_motion_intent(
            motion_data, self.current_gesture
        )
        
        # Record prediction
        self.prediction_history.append({
            'gesture': gesture,
            'confidence': confidence,
            'transition': transition_state,
            'intention': intent_analysis['confidence']
        })
        
        # State machine for gesture updates
        if transition_state == 'transitioning':
            self.in_transition = True
            
            # Only update gesture if it's intentional and confident
            if intent_analysis['is_intentional'] and confidence > 0.7:
                if gesture != 'neutral':
                    self.current_gesture = gesture
                    self.gesture_confidence = confidence
                    self.gesture_commitment_frames = 0
            else:
                # Return motion - go to neutral
                if not intent_analysis['is_intentional']:
                    self.current_gesture = 'neutral'
                    self.gesture_confidence = 0.0
                    
        elif transition_state == 'active':
            self.in_transition = False
            
            if gesture == self.current_gesture:
                self.gesture_commitment_frames += 1
                self.gesture_confidence = confidence
            else:
                # Gesture change
                if confidence > 0.7 and intent_analysis['is_intentional']:
                    self.previous_gesture = self.current_gesture
                    self.current_gesture = gesture
                    self.gesture_confidence = confidence
                    self.gesture_commitment_frames = 0
                    
        else:  # neutral
            self.in_transition = False
            if self.current_gesture != 'neutral':
                self.previous_gesture = self.current_gesture
            self.current_gesture = 'neutral'
            self.gesture_confidence = 0.0
            self.gesture_commitment_frames = 0
    
    def _should_execute_action(self):
        """Determine if action should be executed"""
        # Don't execute during transitions
        if self.in_transition:
            return False
        
        # Don't execute neutral
        if self.current_gesture == 'neutral':
            return False
        
        # Require minimum commitment
        if self.gesture_commitment_frames < self.min_commitment_frames:
            return False
        
        # Require minimum confidence
        if self.gesture_confidence < 0.65:
            return False
        
        # Check recent intention scores
        if self.prediction_history:
            recent_intentions = [p['intention'] for p in list(self.prediction_history)[-3:]]
            if np.mean(recent_intentions) < 0.5:
                return False
        
        return True
    
    def _apply_physics_action(self, motion_features):
        """Apply physics with intention awareness"""
        if not self._should_execute_action():
            return
        
        # Get intention score
        intention_score = 1.0
        if self.prediction_history:
            recent_intentions = [p['intention'] for p in list(self.prediction_history)[-3:]]
            intention_score = np.mean(recent_intentions)
        
        # Apply action based on gesture
        if 'scroll' in self.current_gesture:
            # Determine scroll direction
            if 'up' in self.current_gesture:
                direction = Vector2D(0, 1)
            elif 'down' in self.current_gesture:
                direction = Vector2D(0, -1)
            elif 'left' in self.current_gesture:
                direction = Vector2D(-1, 0)
            else:  # right
                direction = Vector2D(1, 0)
            
            # Blend with actual motion
            palm_velocity = motion_features['palm_velocity']
            if palm_velocity.magnitude() > 0.01:
                actual_direction = palm_velocity.normalize()
                direction = (direction * 0.3 + actual_direction * 0.7).normalize()
            
            # Apply force with intention scaling
            intensity = min(palm_velocity.magnitude() * 2.0, 1.0) * self.gesture_confidence
            adjusted_intensity = intensity * (0.5 + 0.5 * intention_score)
            
            self.physics_engine.apply_scroll_force(direction, adjusted_intensity)
            
        elif 'zoom' in self.current_gesture:
            zoom_rate = motion_features['spread_velocity']
            
            # Correct zoom direction based on gesture
            if 'zoom_in' in self.current_gesture and zoom_rate < 0:
                zoom_rate = abs(zoom_rate)
            elif 'zoom_out' in self.current_gesture and zoom_rate > 0:
                zoom_rate = -abs(zoom_rate)
            
            # Apply zoom with intention scaling
            intensity = min(abs(zoom_rate) * 3.0, 1.0) * self.gesture_confidence
            adjusted_intensity = intensity * (0.5 + 0.5 * intention_score)
            
            self.physics_engine.apply_zoom_force(zoom_rate, adjusted_intensity)
    
    def _draw_enhanced_ui(self, frame):
        """Draw UI with transition information"""
        h, w = frame.shape[:2]
        
        # Top panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Current state
        state_color = (0, 255, 0) if self.current_gesture != 'neutral' else (150, 150, 150)
        if self.in_transition:
            state_color = (255, 255, 0)
        
        cv2.putText(frame, f"Gesture: {self.current_gesture.upper()}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, state_color, 2)
        
        # Transition indicator
        if self.in_transition:
            cv2.putText(frame, "TRANSITIONING", (300, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Confidence bar
        conf_width = int(self.gesture_confidence * 200)
        cv2.rectangle(frame, (10, 50), (10 + conf_width, 65), (0, 255, 255), -1)
        cv2.rectangle(frame, (10, 50), (210, 65), (100, 100, 100), 2)
        cv2.putText(frame, f"{self.gesture_confidence:.2f}", (220, 62), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Intention indicator
        if self.prediction_history:
            intent = self.prediction_history[-1]['intention']
            intent_color = (0, 255, 0) if intent > 0.5 else (255, 0, 0)
            cv2.putText(frame, f"Intent: {intent:.2f}", (300, 62), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, intent_color, 1)
        
        # FPS
        if len(self.fps_counter) > 0:
            avg_fps = 1.0 / np.mean(list(self.fps_counter))
            cv2.putText(frame, f"FPS: {avg_fps:.1f}", (w - 100, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
        # Action indicator
        if self._should_execute_action():
            cv2.circle(frame, (w - 150, 60), 10, (0, 255, 0), -1)
            cv2.putText(frame, "ACTIVE", (w - 230, 65), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Physics visualization
        physics_state = self.physics_engine.get_physics_state()
        momentum = physics_state['scroll_momentum']
        
        if momentum[0] != 0 or momentum[1] != 0:
            center_x, center_y = w // 2, h - 100
            end_x = int(center_x + momentum[0] * 2)
            end_y = int(center_y - momentum[1] * 2)
            cv2.arrowedLine(frame, (center_x, center_y), (end_x, end_y), 
                           (0, 255, 0), 3, tipLength=0.3)
        
        return frame
    
    def run(self):
        """Main control loop"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("\n=== Enhanced Gesture Control Active ===")
        print("Now with transition awareness and intention detection!")
        print("Press 'Q' to quit, 'R' to reset physics\n")
        
        timestamp = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            
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
            
            # Process landmarks
            if self.results and self.results.hand_landmarks:
                landmarks = self.results.hand_landmarks[0]
                landmarks_list = [[lm.x, lm.y, lm.z] for lm in landmarks]
                
                # Update buffer
                self.landmark_buffer.append(landmarks_list)
                
                # Extract motion features
                motion_features = self.motion_extractor.extract_motion_features(landmarks)
                
                # Predict gesture with transitions
                prediction = self._predict_gesture()
                
                # Update state
                self._update_gesture_state(prediction, motion_features)
                
                # Apply physics if appropriate
                self._apply_physics_action(motion_features)
                
                # Draw hand
                proto = landmark_pb2.NormalizedLandmarkList()
                proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) 
                    for lm in landmarks
                ])
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, proto, mp.solutions.hands.HAND_CONNECTIONS)
            else:
                # No hand - clear state
                self.landmark_buffer.clear()
                self.current_gesture = 'neutral'
                self.gesture_confidence = 0.0
                self.gesture_commitment_frames = 0
            
            # Update physics
            self.physics_engine.update(frame_dt)
            self.physics_engine.execute_smooth_actions()
            
            # Draw UI
            frame = self._draw_enhanced_ui(frame)
            
            # Display
            cv2.imshow('Enhanced Gesture Control', frame)
            
            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.physics_engine.reset_momentum()
                print("Physics reset")
            elif key == ord('+'):
                self.physics_engine.user_scroll_multiplier *= 1.2
                print(f"Sensitivity: {self.physics_engine.user_scroll_multiplier:.2f}x")
            elif key == ord('-'):
                self.physics_engine.user_scroll_multiplier /= 1.2
                print(f"Sensitivity: {self.physics_engine.user_scroll_multiplier:.2f}x")
        
        cap.release()
        cv2.destroyAllWindows()
        self.landmarker.close()

if __name__ == '__main__':
    controller = EnhancedGestureController()
    controller.run()