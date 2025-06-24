# transition_aware_model.py

# transition_aware_model.py - Advanced model that understands gesture transitions

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
import json
import time # I noticed this was missing but used in the adapter class

class TransitionAwareLSTM(nn.Module):
    """LSTM model that understands gesture transitions and context"""
    
    def __init__(self, input_size, hidden_size=128, num_layers=2, num_classes=15):
        super().__init__()
        
        # Feature extraction
        self.feature_conv = nn.Conv1d(input_size, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        
        # Bidirectional LSTM for temporal context
        self.lstm = nn.LSTM(
            64, hidden_size, num_layers,
            batch_first=True, bidirectional=True, dropout=0.3
        )
        
        # Attention mechanism for focusing on important transitions
        self.attention = nn.MultiheadAttention(
            hidden_size * 2, num_heads=4, dropout=0.2
        )
        
        # Context embedding (for previous gesture information)
        self.context_embedding = nn.Embedding(num_classes, 32)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2 + 32, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
        # Transition detection head
        self.transition_detector = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # neutral, transitioning, active
        )
        
    def forward(self, x, context=None):
        batch_size, seq_len, features = x.shape
        
        # Reshape for conv1d: (batch, features, seq_len)
        x = x.transpose(1, 2)
        
        # Extract features
        x = F.relu(self.bn1(self.feature_conv(x)))
        
        # Back to (batch, seq_len, features)
        x = x.transpose(1, 2)
        
        # LSTM processing
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Apply attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Global pooling
        pooled = torch.mean(attn_out, dim=1)
        
        # Add context if available
        if context is not None:
            context_emb = self.context_embedding(context)
            pooled = torch.cat([pooled, context_emb], dim=1)
        else:
            # Pad with zeros if no context
            zero_context = torch.zeros(batch_size, 32).to(pooled.device)
            pooled = torch.cat([pooled, zero_context], dim=1)
        
        # Classification
        gesture_logits = self.classifier(pooled)
        
        # Transition detection (using last hidden state)
        last_hidden = lstm_out[:, -1, :]
        transition_logits = self.transition_detector(last_hidden)
        
        return {
            'gesture': gesture_logits,
            'transition': transition_logits
        }

class GestureIntentionDetector:
    """Detects whether a motion is intentional or just returning to neutral"""
    
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.motion_history = deque(maxlen=window_size)
        self.phase_history = deque(maxlen=window_size)
        self.last_active_gesture = None
        self.frames_since_last_gesture = 0
        
    def analyze_motion_intent(self, motion_features: Dict, current_phase: str) -> Dict:
        """Analyze if current motion is intentional"""
        
        self.motion_history.append(motion_features)
        self.phase_history.append(current_phase)
        self.frames_since_last_gesture += 1
        
        if current_phase == 'active_gesture':
            self.frames_since_last_gesture = 0
        
        # Not enough history
        if len(self.motion_history) < 3:
            return {'is_intentional': True, 'confidence': 0.5, 'reason': 'insufficient_data'}
        
        # Calculate motion characteristics
        # --- FIX: Convert deque to list before slicing ---
        recent_velocities = [m['velocity'] for m in list(self.motion_history)[-5:]]
        recent_accelerations = [m.get('acceleration', 0) for m in list(self.motion_history)[-5:]]
        
        avg_velocity = np.mean(recent_velocities)
        velocity_variance = np.var(recent_velocities)
        avg_acceleration = np.mean(recent_accelerations)
        
        # Check for return motion pattern
        is_return_motion = self._detect_return_motion()
        
        # Decision logic
        intention_score = 1.0
        reasons = []
        
        # 1. Low velocity with high variance suggests hesitation
        if avg_velocity < 0.02 and velocity_variance > 0.01:
            intention_score *= 0.5
            reasons.append('hesitant_motion')
        
        # 2. Deceleration after recent gesture suggests return
        if self.frames_since_last_gesture < 15 and avg_acceleration < -0.01:
            intention_score *= 0.3
            reasons.append('decelerating_after_gesture')
        
        # 3. Opposite direction to recent gesture
        if is_return_motion:
            intention_score *= 0.2
            reasons.append('return_motion_detected')
        
        # 4. Consistent acceleration suggests intentional motion
        if avg_acceleration > 0.02 and velocity_variance < 0.005:
            intention_score *= 2.0
            reasons.append('consistent_acceleration')
        
        # 5. Long time since last gesture increases intention likelihood
        if self.frames_since_last_gesture > 30:
            intention_score *= 1.5
            reasons.append('sufficient_neutral_time')
        
        intention_score = np.clip(intention_score, 0.0, 1.0)
        
        return {
            'is_intentional': intention_score > 0.5,
            'confidence': intention_score,
            'reason': ', '.join(reasons) if reasons else 'normal_motion'
        }
    
    def _detect_return_motion(self) -> bool:
        """Detect if current motion is returning from a gesture"""
        if len(self.motion_history) < 5:
            return False
        
        # Get motion directions
        recent_directions = []
        for i in range(1, min(5, len(self.motion_history))):
            curr = self.motion_history[-i]
            prev = self.motion_history[-i-1]
            if 'direction' in curr and 'direction' in prev:
                recent_directions.append(curr['direction'])
        
        if len(recent_directions) < 2:
            return False
        
        # Check if directions are reversing
        dot_products = []
        for i in range(1, len(recent_directions)):
            # Calculate dot product between consecutive directions
            dir1 = np.array(recent_directions[i-1])
            dir2 = np.array(recent_directions[i])
            
            if np.linalg.norm(dir1) > 0 and np.linalg.norm(dir2) > 0:
                dot_product = np.dot(dir1, dir2) / (np.linalg.norm(dir1) * np.linalg.norm(dir2))
                dot_products.append(dot_product)
        
        # Negative dot products indicate opposite directions
        if dot_products and np.mean(dot_products) < -0.5:
            return True
        
        return False

class TransitionAwareGestureController:
    """Enhanced controller that handles transitions properly"""
    
    def __init__(self, model_path='gesture_data/transition_aware_model.pth'):
        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # Gesture state management
        self.current_gesture = 'neutral'
        self.gesture_confidence = 0.0
        self.previous_gesture = 'neutral'
        self.gesture_history = deque(maxlen=30)
        
        # Intention detection
        self.intention_detector = GestureIntentionDetector()
        
        # Transition handling
        self.in_transition = False
        self.transition_start_gesture = None
        self.transition_confidence_threshold = 0.7
        
        # Gesture commitment
        self.gesture_commitment_frames = 0
        self.min_commitment_frames = 5
        
    def _load_model(self, model_path):
        """Load the trained transition-aware model"""
        # This would load the actual trained model
        # For now, returning a dummy model for illustration
        return TransitionAwareLSTM(input_size=63, hidden_size=128, num_classes=15)
    
    def process_frame(self, landmarks, motion_features):
        """Process a single frame with transition awareness"""
        
        # Analyze motion intention
        intent_analysis = self.intention_detector.analyze_motion_intent(
            motion_features, self.current_gesture
        )
        
        # Get model prediction (pseudo-code for illustration)
        with torch.no_grad():
            # In real implementation, this would use actual features
            gesture_pred = self._get_gesture_prediction(landmarks)
            transition_pred = self._get_transition_prediction(landmarks)
        
        # Update gesture state based on predictions and intention
        self._update_gesture_state(gesture_pred, transition_pred, intent_analysis)
        
        # Record history
        self.gesture_history.append({
            'gesture': self.current_gesture,
            'confidence': self.gesture_confidence,
            'intention': intent_analysis['confidence'],
            'in_transition': self.in_transition
        })
        
        return {
            'gesture': self.current_gesture,
            'confidence': self.gesture_confidence,
            'is_intentional': intent_analysis['is_intentional'],
            'in_transition': self.in_transition,
            'should_execute': self._should_execute_action()
        }
    
    def _update_gesture_state(self, gesture_pred, transition_pred, intent_analysis):
        """Update gesture state with transition handling"""
        
        predicted_gesture = gesture_pred['class']
        gesture_confidence = gesture_pred['confidence']
        transition_state = transition_pred['state']  # neutral, transitioning, active
        
        # Handle transitions
        if transition_state == 'transitioning':
            if not self.in_transition:
                self.in_transition = True
                self.transition_start_gesture = self.current_gesture
            
            # During transition, maintain previous gesture unless high confidence
            if gesture_confidence > self.transition_confidence_threshold:
                # Check if this is an intentional new gesture or return motion
                if intent_analysis['is_intentional']:
                    self.current_gesture = predicted_gesture
                    self.gesture_confidence = gesture_confidence
                    self.gesture_commitment_frames = 0
                else:
                    # It's a return motion, don't change gesture
                    self.current_gesture = 'neutral'
                    self.gesture_confidence = 0.0
        
        elif transition_state == 'active':
            self.in_transition = False
            
            # Only update if intentional and confident
            if intent_analysis['is_intentional'] and gesture_confidence > 0.6:
                if self.current_gesture == predicted_gesture:
                    self.gesture_commitment_frames += 1
                else:
                    self.gesture_commitment_frames = 0
                
                self.current_gesture = predicted_gesture
                self.gesture_confidence = gesture_confidence
        
        else:  # neutral
            self.in_transition = False
            self.current_gesture = 'neutral'
            self.gesture_confidence = 0.0
            self.gesture_commitment_frames = 0
    
    def _should_execute_action(self):
        """Determine if the current gesture should trigger an action"""
        
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
        if self.gesture_confidence < 0.7:
            return False
        
        # Check intention
        recent_intents = [h['intention'] for h in list(self.gesture_history)[-5:]]
        if recent_intents and np.mean(recent_intents) < 0.5:
            return False
        
        return True
    
    def _get_gesture_prediction(self, landmarks):
        """Get gesture prediction from model (placeholder)"""
        # In real implementation, this would:
        # 1. Extract features from landmarks
        # 2. Pass through the model
        # 3. Return prediction and confidence
        return {
            'class': 'scroll_up',
            'confidence': 0.8
        }
    
    def _get_transition_prediction(self, landmarks):
        """Get transition state prediction (placeholder)"""
        return {
            'state': 'active',
            'confidence': 0.9
        }

# Utility function to update the existing physics engine
def create_intention_aware_physics_adapter(physics_engine):
    """Adapter to make physics engine intention-aware"""
    
    class IntentionAwarePhysicsEngine:
        def __init__(self, base_engine):
            self.base_engine = base_engine
            self.intention_threshold = 0.5
            self.last_intentional_time = 0
            
        def apply_scroll_force(self, direction, intensity, intention_score=1.0):
            """Apply force only if intentional"""
            if intention_score > self.intention_threshold:
                # Scale intensity by intention confidence
                adjusted_intensity = intensity * (0.5 + 0.5 * intention_score)
                self.base_engine.apply_scroll_force(direction, adjusted_intensity)
                self.last_intentional_time = time.time()
            else:
                # Dampen unintentional movements
                self.base_engine.scroll_momentum = self.base_engine.scroll_momentum * 0.9
        
        def apply_zoom_force(self, zoom_rate, intensity, intention_score=1.0):
            """Apply zoom only if intentional"""
            if intention_score > self.intention_threshold:
                adjusted_intensity = intensity * (0.5 + 0.5 * intention_score)
                self.base_engine.apply_zoom_force(zoom_rate, adjusted_intensity)
                self.last_intentional_time = time.time()
            else:
                # Dampen unintentional movements
                self.base_engine.zoom_velocity *= 0.9
        
        def __getattr__(self, name):
            """Delegate other methods to base engine"""
            return getattr(self.base_engine, name)
    
    return IntentionAwarePhysicsEngine(physics_engine)