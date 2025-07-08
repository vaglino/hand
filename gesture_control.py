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
from mediapipe.tasks.python import vision, BaseOptions
# from mediapipe.tasks.python.components.processors import GpuOptions
import joblib
import os
from enum import Enum
from typing import Optional, List, Tuple
import warnings
import torch.nn as nn
import torch.nn.functional as F
import pyautogui # <-- ADD THIS IMPORT
from threading import Thread

# Try to import ONNX runtime for optimized inference
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    warnings.warn("ONNX Runtime not available. Install with: pip install onnxruntime-gpu")

from physics_engine import TrackpadPhysicsEngine, GestureMotionExtractor, Vector2D

class WebcamStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while not self.stopped:
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True

# --- TCN MODEL DEFINITION (copied from enhanced_train_model) ---
class TemporalBlock(nn.Module):
    """Single temporal block with dilated convolution and residual connection."""
    
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, dropout=0.2):
        super().__init__()
        # Calculate causal padding (only pad left side)
        self.padding = (kernel_size - 1) * dilation
        
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, 
                              padding=0, dilation=dilation)  # No padding, we'll pad manually
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride,
                              padding=0, dilation=dilation)  # No padding, we'll pad manually
        self.bn2 = nn.BatchNorm1d(n_outputs)
        self.dropout2 = nn.Dropout(dropout)
        
        # Residual connection
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        
    def forward(self, x):
        # Apply causal padding (pad only the left side)
        if self.padding > 0:
            x = F.pad(x, (self.padding, 0))
        
        # First conv block
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout1(out)
        
        # Apply causal padding again for second conv
        if self.padding > 0:
            out = F.pad(out, (self.padding, 0))
        
        # Second conv block
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.dropout2(out)
        
        # Prepare residual - ensure same size as output
        if self.downsample is not None:
            # Apply same causal padding to input for residual
            if self.padding > 0:
                x_padded = F.pad(x, (self.padding, 0))
            else:
                x_padded = x
            res = self.downsample(x_padded)
            # Crop to match output size
            if res.size(2) > out.size(2):
                res = res[:, :, :out.size(2)]
        else:
            res = x
            # Crop residual to match output size if needed
            if res.size(2) > out.size(2):
                res = res[:, :, :out.size(2)]
        
        return F.relu(out + res)

class TemporalConvNet(nn.Module):
    """Temporal Convolutional Network for gesture classification."""
    
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            layers.append(TemporalBlock(in_channels, out_channels, kernel_size, 
                                      stride=1, dilation=dilation_size, 
                                      dropout=dropout))
            
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

class EnhancedGestureClassifier(nn.Module):
    """Enhanced gesture classifier using TCN architecture."""
    
    def __init__(self, input_size, num_classes, dropout=0.3):
        super().__init__()
        
        # TCN layers with increasing receptive field
        tcn_channels = [64, 96, 128, 160]
        self.tcn = TemporalConvNet(input_size, tcn_channels, kernel_size=3, dropout=dropout)
        
        # Global attention pooling
        self.attention = nn.Sequential(
            nn.Linear(tcn_channels[-1], 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(tcn_channels[-1], 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, features)
        x = x.transpose(1, 2)  # (batch_size, features, seq_len)
        
        # TCN processing
        tcn_out = self.tcn(x)  # (batch_size, channels, seq_len)
        tcn_out = tcn_out.transpose(1, 2)  # (batch_size, seq_len, channels)
        
        # Attention pooling
        attention_weights = F.softmax(self.attention(tcn_out), dim=1)
        context = torch.sum(attention_weights * tcn_out, dim=1)
        
        # Classification
        return self.classifier(context)

# --- ADVANCED PREPROCESSING MODULE (subset needed for inference) ---
class LandmarkPreprocessor:
    """Advanced landmark preprocessing with Procrustes alignment and filtering."""
    
    def __init__(self, filter_order=1, cutoff_freq=0.3):
        from scipy.signal import butter, filtfilt
        from scipy.spatial import procrustes
        
        self.filter_order = filter_order
        self.cutoff_freq = cutoff_freq
        self.b, self.a = butter(filter_order, cutoff_freq, btype='low')
        
    def procrustes_align_sequence(self, landmarks_sequence: np.ndarray) -> np.ndarray:
        """Apply Procrustes alignment to each frame against the first frame."""
        from scipy.spatial import procrustes
        
        if landmarks_sequence.shape[0] < 2:
            return landmarks_sequence
            
        aligned_sequence = np.zeros_like(landmarks_sequence)
        reference = landmarks_sequence[0]  # Use first frame as reference
        aligned_sequence[0] = reference
        
        for i in range(1, len(landmarks_sequence)):
            # Procrustes alignment: removes translation, scale, and rotation
            _, aligned_frame, _ = procrustes(reference, landmarks_sequence[i])
            aligned_sequence[i] = aligned_frame
            
        return aligned_sequence
    
    def apply_temporal_filter(self, landmarks_sequence: np.ndarray) -> np.ndarray:
        """Apply Butterworth filter to reduce jitter in landmark positions."""
        from scipy.signal import filtfilt
        
        if landmarks_sequence.shape[0] < 4:  # Need minimum frames for filtering
            return landmarks_sequence
            
        filtered_sequence = np.zeros_like(landmarks_sequence)
        
        # Filter each landmark coordinate independently
        for landmark_idx in range(landmarks_sequence.shape[1]):
            for coord_idx in range(landmarks_sequence.shape[2]):
                signal = landmarks_sequence[:, landmark_idx, coord_idx]
                filtered_signal = filtfilt(self.b, self.a, signal)
                filtered_sequence[:, landmark_idx, coord_idx] = filtered_signal
                
        return filtered_sequence
    
    def compute_finger_angles(self, landmarks: np.ndarray) -> np.ndarray:
        """Compute angles between finger segments for richer geometric features."""
        # Finger landmark indices: thumb, index, middle, ring, pinky
        finger_tips = [4, 8, 12, 16, 20]
        finger_bases = [2, 5, 9, 13, 17]
        
        angles = []
        for i, (tip, base) in enumerate(zip(finger_tips, finger_bases)):
            # Vector from base to tip
            finger_vec = landmarks[tip] - landmarks[base]
            
            # Angle with respect to palm normal (wrist to middle finger base)
            palm_vec = landmarks[9] - landmarks[0]
            
            # Calculate angle (handling potential division by zero)
            dot_product = np.dot(finger_vec[:2], palm_vec[:2])  # Use 2D projection
            norms = np.linalg.norm(finger_vec[:2]) * np.linalg.norm(palm_vec[:2])
            
            if norms > 1e-6:
                angle = np.arccos(np.clip(dot_product / norms, -1.0, 1.0))
            else:
                angle = 0.0
                
            angles.append(angle)
            
        return np.array(angles)
    
    
    
    def extract_advanced_features(self, landmarks_sequence: List) -> Optional[np.ndarray]:
        """Extract advanced features with Procrustes alignment and multi-temporal derivatives."""
        sequence = np.array(landmarks_sequence)
        if sequence.ndim != 3 or sequence.shape[1:] != (21, 3):
            return None
            
        try:
            # Step 1: Apply temporal filtering to reduce jitter
            filtered_sequence = self.apply_temporal_filter(sequence)
            
            # Step 2: Procrustes alignment for scale/rotation invariance
            aligned_sequence = self.procrustes_align_sequence(filtered_sequence)
            
            # Step 3: Normalize relative to wrist
            wrist = aligned_sequence[:, 0:1, :]
            relative_landmarks = (aligned_sequence - wrist).reshape(aligned_sequence.shape[0], -1)
            
            # Step 4: Multi-temporal velocity features (Δt = 1, 2, 3)
            velocities_1 = np.diff(relative_landmarks, axis=0, prepend=np.zeros((1, relative_landmarks.shape[1])))
            
            velocities_2 = np.zeros_like(velocities_1)
            if len(relative_landmarks) >= 3:
                velocities_2[2:] = relative_landmarks[2:] - relative_landmarks[:-2]
                
            velocities_3 = np.zeros_like(velocities_1)
            if len(relative_landmarks) >= 4:
                velocities_3[3:] = relative_landmarks[3:] - relative_landmarks[:-3]
            
            # Step 5: Finger angle features for each frame
            angle_features = []
            for frame_landmarks in aligned_sequence:
                angles = self.compute_finger_angles(frame_landmarks)
                angle_features.append(angles)
            angle_features = np.array(angle_features)
            
            # Step 6: Combine all features
            features = np.concatenate([
                relative_landmarks,    # Position features
                velocities_1,         # 1-frame velocity
                velocities_2,         # 2-frame velocity  
                velocities_3,         # 3-frame velocity
                angle_features        # Finger angle features
            ], axis=1)
            
            return features
            
        except Exception as e:
            # Fallback to basic features if advanced preprocessing fails
            warnings.warn(f"Advanced preprocessing failed: {e}. Using basic features.")
            wrist = sequence[:, 0:1, :]
            relative_landmarks = (sequence - wrist).reshape(sequence.shape[0], -1)
            velocities = np.diff(relative_landmarks, axis=0, prepend=np.zeros((1, relative_landmarks.shape[1])))
            return np.concatenate([relative_landmarks, velocities], axis=1)

# --- PREDICTION SMOOTHING WITH EXPONENTIAL MOVING AVERAGE ---
class PredictionSmoother:
    """Smooths predictions using exponential moving average to reduce flickering."""
    
    def __init__(self, num_classes: int, alpha: float = 0.3, confidence_threshold: float = 0.7):
        self.num_classes = num_classes
        self.alpha = alpha  # EMA smoothing factor
        self.confidence_threshold = confidence_threshold
        self.ema_probs = np.ones(num_classes) / num_classes  # Start with uniform distribution
        self.last_confident_prediction = None
        self.stable_frames = 0
        self.min_stable_frames = 3
        
    def update(self, raw_probs: np.ndarray) -> Tuple[str, float, np.ndarray]:
        """Update EMA and return smoothed prediction."""
        # Update exponential moving average
        self.ema_probs = self.alpha * raw_probs + (1 - self.alpha) * self.ema_probs
        
        # Get prediction from smoothed probabilities
        pred_idx = np.argmax(self.ema_probs)
        confidence = self.ema_probs[pred_idx]
        
        # Stability check: only change prediction if confident and stable
        if confidence > self.confidence_threshold:
            if self.last_confident_prediction == pred_idx:
                self.stable_frames += 1
            else:
                self.stable_frames = 0
                self.last_confident_prediction = pred_idx
        else:
            self.stable_frames = 0
        
        # Return prediction (use last stable if current is not stable enough)
        if self.stable_frames >= self.min_stable_frames or self.last_confident_prediction is None:
            final_prediction = pred_idx
        else:
            final_prediction = self.last_confident_prediction if self.last_confident_prediction is not None else pred_idx
            
        return final_prediction, confidence, self.ema_probs
    
    def reset(self):
        """Reset smoother state."""
        self.ema_probs = np.ones(self.num_classes) / self.num_classes
        self.last_confident_prediction = None
        self.stable_frames = 0

# --- OPTIMIZED INFERENCE ENGINE ---
class OptimizedInferenceEngine:
    """Handles model loading and inference with multiple backend options."""
    
    def __init__(self, model_dir: str = 'gesture_data'):
        self.model_dir = model_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.backend = None
        self.input_size = None
        self.sequence_length = None
        
        print(f"🚀 Initializing inference engine on {self.device}")
        self._load_optimal_model()
    
    def _load_optimal_model(self):
        """Load the best available model format for optimal inference."""
        torchscript_path = os.path.join(self.model_dir, 'enhanced_gesture_classifier_traced.pt')
        pytorch_path = os.path.join(self.model_dir, 'enhanced_gesture_classifier.pth')
        
        # Try TorchScript first (fastest)
        if os.path.exists(torchscript_path):
            try:
                print("📦 Loading TorchScript model for optimized inference...")
                self.model = torch.jit.load(torchscript_path, map_location=self.device)
                self.model.eval()
                self.backend = 'torchscript'
                
                # Load metadata from PyTorch checkpoint
                if os.path.exists(pytorch_path):
                    checkpoint = torch.load(pytorch_path, map_location='cpu')
                    self.input_size = checkpoint['input_size']
                    self.sequence_length = checkpoint['sequence_length']
                
                print("✅ TorchScript model loaded successfully")
                return
            except Exception as e:
                print(f"⚠️ Failed to load TorchScript model: {e}")
                print("📦 Falling back to PyTorch model...")
        
        # Fallback to PyTorch model
        if os.path.exists(pytorch_path):
            print("📦 Loading PyTorch TCN model...")
            checkpoint = torch.load(pytorch_path, map_location=self.device)
            
            # Create the enhanced TCN model
            self.model = EnhancedGestureClassifier(
                input_size=checkpoint['input_size'],
                num_classes=checkpoint['num_classes']
            ).to(self.device)
            
            self.model.load_state_dict(checkpoint['model_state'])
            self.model.eval()
            self.backend = 'pytorch'
            
            self.input_size = checkpoint['input_size']
            self.sequence_length = checkpoint['sequence_length']
            
            print("✅ PyTorch TCN model loaded successfully")
            return
        
        raise FileNotFoundError("❌ No trained model found. Please run enhanced training first.")
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """Fast inference with the loaded model."""
        if self.model is None:
            raise RuntimeError("No model loaded")
        
        # Convert to tensor and add batch dimension
        input_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if self.backend == 'torchscript':
                # TorchScript inference
                outputs = self.model(input_tensor)
            else:
                # Regular PyTorch inference
                outputs = self.model(input_tensor)
            
            # Convert to probabilities
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            
        return probs

# --- ENHANCED GESTURE STATE MACHINE ---
class GestureState(Enum):
    NEUTRAL = 1
    DEBOUNCING = 2
    ACTIVE = 3
    RETURNING = 4

class EnhancedGestureController:
    """Enhanced gesture controller with optimized inference and improved state management."""
    
    def __init__(self, model_path='hand_landmarker.task'):
        print("🚀 Initializing Enhanced Gesture Controller...")
        
        # Load models and preprocessor
        self._load_models()
        
        # Initialize components
        self.physics_engine = TrackpadPhysicsEngine()
        self.motion_extractor = GestureMotionExtractor(window_size=5)
        self.landmark_buffer = deque(maxlen=self.sequence_length)
        
        # Prediction smoothing
        self.prediction_smoother = PredictionSmoother(
            num_classes=len(self.label_encoder.classes_),
            alpha=0.25,  # Moderate smoothing
            confidence_threshold=0.75
        )
        
        # State machine
        self.state = GestureState.NEUTRAL
        self.active_gesture = "neutral"
        self.debounce_counter = 0
        self.debounce_threshold = 3
        self.neutral_counter = 0
        self.neutral_threshold = 4
        
        # Performance monitoring
        self.inference_times = deque(maxlen=100)
        self.frame_count = 0
        self.fps_start_time = time.time()
        
        # MediaPipe setup
        self.results = None
        # gpu_opts = GpuOptions()
        # base_opts = BaseOptions(
        #     model_asset_path=model_path,
        #     delegate=BaseOptions.Delegate.GPU)
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
        
        # Check if OpenCV GUI is available
        self.gui_available = self._check_opencv_gui()
        if not self.gui_available:
            print("⚠️ OpenCV GUI not available - running in headless mode")
            print("📝 Frame data will be saved to 'debug_frames' folder")
        
        print("✅ Enhanced controller initialized successfully")
    
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
        """
        Handles discrete, one-shot actions and returns True if an action was fired.
        These actions are triggered once when a gesture becomes active.
        """
        action_fired = False
        if gesture == 'maximize_window':
            pyautogui.hotkey('win', 'up')
            print("ACTION: Maximized window")
            action_fired = True
        elif gesture == 'go_back':
            pyautogui.hotkey('alt', 'left')
            print("ACTION: Navigated back")
            action_fired = True

        return action_fired
    
    def _update_enhanced_state_machine(self, predicted_label: str, confidence: float, smoothed_probs: np.ndarray):
        """Enhanced state machine with better transition logic."""
        
        # Update neutral counter based on smoothed probabilities
        neutral_idx = list(self.label_encoder.classes_).index('neutral') if 'neutral' in self.label_encoder.classes_ else -1
        
        if neutral_idx >= 0 and smoothed_probs[neutral_idx] > 0.6:
            self.neutral_counter += 1
        else:
            self.neutral_counter = 0
        
        # State transitions
        if self.state == GestureState.NEUTRAL:
            if ('return' not in predicted_label and 
                'neutral' not in predicted_label and 
                confidence > 0.8):
                
                self.state = GestureState.DEBOUNCING
                self.debounce_candidate = predicted_label.replace('_start', '')
                self.debounce_counter = 1
        
        elif self.state == GestureState.DEBOUNCING:
            candidate_match = predicted_label.replace('_start', '') == self.debounce_candidate
            
            if candidate_match and confidence > 0.7:
                self.debounce_counter += 1
                if self.debounce_counter >= self.debounce_threshold:
                    # --- MODIFICATION: Handle one-shot vs continuous actions ---
                    self.active_gesture = self.debounce_candidate
                    action_fired = self._handle_one_shot_actions(self.active_gesture)

                    if action_fired:
                        # For one-shot actions, we don't stay in ACTIVE.
                        # Transition immediately to RETURNING to prevent re-triggering.
                        self.state = GestureState.RETURNING
                        self.neutral_counter = 0
                    else:
                        # For continuous actions (scroll, zoom), transition to ACTIVE.
                        self.state = GestureState.ACTIVE
                        self.neutral_counter = 0
                        self.prediction_smoother.reset()  # Reset for clean active state
            else:
                self.state = GestureState.NEUTRAL
                self.debounce_counter = 0
        
        elif self.state == GestureState.ACTIVE:
            # Primary exit: explicit return gesture
            if (predicted_label == f"{self.active_gesture}_return" and confidence > 0.6):
                self.state = GestureState.RETURNING
                self.neutral_counter = 0
            
            # Secondary exit: sustained neutral state
            elif self.neutral_counter >= self.neutral_threshold:
                self.state = GestureState.NEUTRAL
                self.active_gesture = "neutral"
            
            # Tertiary exit: different gesture detected
            elif (predicted_label != self.active_gesture and 
                  'neutral' not in predicted_label and 
                  'return' not in predicted_label and 
                  confidence > 0.85):
                self.state = GestureState.NEUTRAL
                self.active_gesture = "neutral"
        
        elif self.state == GestureState.RETURNING:
            if self.neutral_counter >= self.neutral_threshold:
                self.state = GestureState.NEUTRAL
                self.active_gesture = "neutral"
    
    def _apply_enhanced_physics(self, landmarks):
        """Apply original working physics - responsive and reliable."""
        if self.state != GestureState.ACTIVE:
            return

        motion_features = self.motion_extractor.extract_motion_features(landmarks)
        if 'scroll' in self.active_gesture:
            direction = Vector2D(0, 0)
            
            # --- FIX: Introduce a sensitivity multiplier ---
            velocity_magnitude = motion_features['palm_velocity'].magnitude()
            sensitivity_multiplier = 1.0 # Default sensitivity

            if 'up' in self.active_gesture: 
                direction = Vector2D(0, 1) # Scrolls content up
            elif 'down' in self.active_gesture: 
                direction = Vector2D(0, -1) # Scrolls content down
                # Compensate for the slower upward hand motion by boosting its effect
                sensitivity_multiplier = 1.5 # (Value can be tuned, e.g., 1.2 to 1.5)

            # Apply force using the adjusted velocity
            self.physics_engine.apply_scroll_force(direction, velocity_magnitude * sensitivity_multiplier)
        elif 'zoom' in self.active_gesture:
            zoom_rate = motion_features['spread_velocity']
            # Ensure the motion direction matches the gesture
            if ('in' in self.active_gesture and zoom_rate > 0) or ('out' in self.active_gesture and zoom_rate < 0):
                self.physics_engine.apply_zoom_force(zoom_rate, 1.0)
    
    def _draw_enhanced_ui(self, frame):
        """Draw enhanced UI with performance metrics."""
        h, w = frame.shape[:2]
        
        # Background overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.75, overlay, 0.25, 0)
        
        # State and gesture info
        state_color = {
            GestureState.NEUTRAL: (0, 0, 0),
            GestureState.DEBOUNCING: (255, 255, 0),
            GestureState.ACTIVE: (0, 255, 0),
            GestureState.RETURNING: (255, 150, 0)
        }
        
        color = state_color.get(self.state, (255, 255, 255))
        cv2.putText(frame, f"STATE: {self.state.name}", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        active_text = self.active_gesture.upper() if self.state == GestureState.ACTIVE else 'NONE'
        cv2.putText(frame, f"ACTION: {active_text}", (10, 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Performance metrics
        if self.inference_times:
            avg_inference_time = np.mean(self.inference_times)
            fps = self.frame_count / max(time.time() - self.fps_start_time, 0.001)
            
            cv2.putText(frame, f"Inference: {avg_inference_time:.1f}ms", (10, 85), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(frame, f"FPS: {fps:.1f}", (200, 85), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(frame, f"Backend: {self.inference_engine.backend.upper()}", (280, 85), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
        
        return frame
    
    def run(self):
        """Main control loop with enhanced performance monitoring."""
        # cap = cv2.VideoCapture(0)
        # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        # cap.set(cv2.CAP_PROP_FPS, 30)
        cap = WebcamStream(src=0).start()
        cap.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.stream.set(cv2.CAP_PROP_FPS, 30)
        
        print("\n" + "="*60)
        print("🎯 ENHANCED GESTURE CONTROL ACTIVE")
        print("="*60)
        print("🚀 Optimized inference engine ready")
        print("📊 Performance monitoring enabled") 
        print("🎯 Advanced prediction smoothing active")
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
                self.landmarker.detect_async(mp_img, timestamp)
                
                # Calculate dt for physics
                dt = time.time() - last_frame_time
                last_frame_time = time.time()
                
                # Hand processing
                if self.results and self.results.hand_landmarks:
                    landmarks = self.results.hand_landmarks[0]
                    landmarks_list = [[lm.x, lm.y, lm.z] for lm in landmarks]
                    self.landmark_buffer.append(landmarks_list)
                    
                    # Enhanced prediction and state management
                    predicted_label, confidence, smoothed_probs = self._predict_gesture()
                    self._update_enhanced_state_machine(predicted_label, confidence, smoothed_probs)
                    
                    # Apply physics for continuous gestures
                    self._apply_enhanced_physics(landmarks)
                    
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
                    # No hand detected - return to neutral
                    if self.state != GestureState.NEUTRAL:
                        self.state = GestureState.NEUTRAL
                        self.active_gesture = "neutral"
                        self.prediction_smoother.reset()
                
                # Physics update with calculated dt
                self.physics_engine.update(dt)
                self.physics_engine.execute_smooth_actions()
                
                # UI and display
                frame = self._draw_enhanced_ui(frame)
                
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
                        print("🔄 State reset")
                else:
                    # Headless mode - save frames periodically and check for interrupt
                    if frame_save_counter % 30 == 0:  # Save every 30 frames (~1 second)
                        frame_path = f"debug_frames/frame_{frame_save_counter:06d}.jpg"
                        cv2.imwrite(frame_path, frame)
                        print(f"💾 Saved frame: {frame_path} | State: {self.state.name} | Gesture: {self.active_gesture}")
                    
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
            self.landmarker.close()
            
            # Print performance summary
            if self.inference_times:
                print(f"\n" + "="*50)
                print("📈 PERFORMANCE SUMMARY")
                print("="*50)
                print(f"⚡ Average inference time: {np.mean(self.inference_times):.2f}ms")
                print(f"🚀 Min inference time: {np.min(self.inference_times):.2f}ms")
                print(f"⏱️  Max inference time: {np.max(self.inference_times):.2f}ms")
                print(f"🖥️  Backend used: {self.inference_engine.backend.upper()}")
                print(f"🎞️  Total frames processed: {self.frame_count}")
                if not self.gui_available:
                    print(f"📁 Debug frames saved to: debug_frames/")
                print("="*50)

if __name__ == '__main__':
    controller = EnhancedGestureController()
    controller.run()