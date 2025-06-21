# train_model.py - Multi-model training pipeline for hybrid gesture control

import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
import sys
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Configuration
GESTURE_SEQUENCE_LENGTH = 12
NUM_LANDMARKS = 21
FEATURE_DIM = NUM_LANDMARKS * 9  # pos(3) + vel(3) + accel(3)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Using device: {device}")

# ============== Feature Engineering ==============

class AdvancedFeatureExtractor:
    """Extract robust features for both gesture classification and motion prediction"""
    
    def __init__(self):
        self.hand_size_percentile = 50
        self.rotation_reference = None
        
    def extract_gesture_features(self, sequence: List[List[List[float]]]) -> np.ndarray:
        """Extract features for gesture classification (lightweight)"""
        sequence = np.array(sequence)  # Shape: (T, 21, 3)
        
        # Normalize to wrist-relative coordinates
        wrist = sequence[:, 0, :]
        relative_coords = sequence - wrist[:, np.newaxis, :]
        
        # Robust hand size estimation
        hand_sizes = []
        for t in range(len(sequence)):
            # Use multiple landmark pairs for robustness
            distances = [
                np.linalg.norm(relative_coords[t, 9] - relative_coords[t, 0]),  # Middle MCP to wrist
                np.linalg.norm(relative_coords[t, 5] - relative_coords[t, 0]),  # Index MCP to wrist
                np.linalg.norm(relative_coords[t, 17] - relative_coords[t, 0]), # Pinky MCP to wrist
            ]
            hand_sizes.append(np.median(distances))
        
        # Use median hand size for normalization
        median_hand_size = np.median(hand_sizes)
        if median_hand_size < 1e-6:
            median_hand_size = 1.0
            
        scaled_coords = relative_coords / median_hand_size
        
        # Temporal features
        velocity = np.diff(scaled_coords, axis=0, prepend=scaled_coords[0:1])
        acceleration = np.diff(velocity, axis=0, prepend=velocity[0:1])
        
        # Additional gesture-specific features
        features_per_frame = []
        
        for t in range(len(sequence)):
            frame_features = []
            
            # Basic coordinates, velocity, acceleration
            frame_features.extend(scaled_coords[t].flatten())
            frame_features.extend(velocity[t].flatten())
            frame_features.extend(acceleration[t].flatten())
            
            # Finger spread
            fingertips = [4, 8, 12, 16, 20]
            spreads = []
            for i in range(len(fingertips)):
                for j in range(i+1, len(fingertips)):
                    dist = np.linalg.norm(scaled_coords[t, fingertips[i]] - scaled_coords[t, fingertips[j]])
                    spreads.append(dist)
            frame_features.append(np.mean(spreads))
            frame_features.append(np.std(spreads))
            
            # Palm orientation (using cross product of palm vectors)
            palm_vec1 = scaled_coords[t, 5] - scaled_coords[t, 0]   # Index MCP to wrist
            palm_vec2 = scaled_coords[t, 17] - scaled_coords[t, 0]  # Pinky MCP to wrist
            palm_normal = np.cross(palm_vec1, palm_vec2)
            palm_normal = palm_normal / (np.linalg.norm(palm_normal) + 1e-6)
            frame_features.extend(palm_normal)
            
            # Hand openness (average finger extension)
            finger_bases = [1, 5, 9, 13, 17]  # MCPs
            finger_tips = [4, 8, 12, 16, 20]   # Tips
            extensions = []
            for base, tip in zip(finger_bases, finger_tips):
                extension = np.linalg.norm(scaled_coords[t, tip] - scaled_coords[t, base])
                extensions.append(extension)
            frame_features.append(np.mean(extensions))
            
            features_per_frame.append(frame_features)
        
        # Flatten all features
        return np.array(features_per_frame).flatten()
    
    def extract_motion_features(self, landmarks_sequence: List[List[List[float]]], 
                              motion_data: Optional[Dict] = None) -> Dict[str, np.ndarray]:
        """Extract continuous motion features for physics prediction"""
        sequence = np.array(landmarks_sequence)
        
        if len(sequence) < 2:
            return {
                'velocity_x': np.array([0.0]),
                'velocity_y': np.array([0.0]),
                'acceleration_x': np.array([0.0]),
                'acceleration_y': np.array([0.0]),
                'zoom_rate': np.array([0.0]),
                'smoothness': np.array([1.0]),
                'confidence': np.array([1.0])
            }
        
        # Palm center tracking
        palm_indices = [0, 5, 9, 13, 17]
        palm_centers = np.mean(sequence[:, palm_indices], axis=1)
        
        # Velocity calculation with smoothing
        velocities = np.diff(palm_centers, axis=0) * 30  # Assuming 30 FPS
        
        # Apply Gaussian smoothing for more stable velocity
        from scipy.ndimage import gaussian_filter1d
        smooth_velocities = gaussian_filter1d(velocities, sigma=1.0, axis=0)
        
        # Acceleration
        accelerations = np.diff(smooth_velocities, axis=0) * 30
        
        # Zoom rate from finger spread
        finger_spreads = []
        for t in range(len(sequence)):
            fingertips = [4, 8, 12, 16, 20]
            distances = []
            for i in range(len(fingertips)):
                for j in range(i+1, len(fingertips)):
                    dist = np.linalg.norm(sequence[t, fingertips[i]] - sequence[t, fingertips[j]])
                    distances.append(dist)
            finger_spreads.append(np.mean(distances))
        
        finger_spreads = np.array(finger_spreads)
        zoom_rates = np.diff(finger_spreads) * 30
        
        # Motion smoothness (lower jerk = smoother)
        if len(accelerations) > 0:
            jerk = np.diff(accelerations, axis=0)
            smoothness = 1.0 / (1.0 + np.std(np.linalg.norm(jerk, axis=1)))
        else:
            smoothness = 1.0
        
        # Confidence based on motion consistency
        if len(velocities) > 2:
            velocity_variance = np.std(np.linalg.norm(velocities, axis=1))
            confidence = np.exp(-velocity_variance)
        else:
            confidence = 1.0
        
        # Pad arrays to same length
        max_len = len(sequence) - 1
        
        return {
            'velocity_x': self._pad_array(smooth_velocities[:, 0], max_len),
            'velocity_y': self._pad_array(smooth_velocities[:, 1], max_len),
            'acceleration_x': self._pad_array(accelerations[:, 0] if len(accelerations) > 0 else np.array([0]), max_len),
            'acceleration_y': self._pad_array(accelerations[:, 1] if len(accelerations) > 0 else np.array([0]), max_len),
            'zoom_rate': self._pad_array(zoom_rates, max_len),
            'smoothness': np.full(max_len, smoothness),
            'confidence': np.full(max_len, confidence)
        }
    
    def _pad_array(self, arr: np.ndarray, target_length: int) -> np.ndarray:
        """Pad array to target length"""
        if len(arr) >= target_length:
            return arr[:target_length]
        else:
            return np.pad(arr, (0, target_length - len(arr)), mode='edge')

# ============== Models ==============

class LightweightGestureNet(nn.Module):
    """Fast CNN for gesture classification"""
    def __init__(self, input_size, num_classes, dropout=0.3):
        super().__init__()
        
        # 1D CNN for temporal patterns
        self.conv1 = nn.Conv1d(input_size // GESTURE_SEQUENCE_LENGTH, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        
        # Global pooling and classification
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # Reshape for 1D CNN: (batch, features_per_frame, sequence_length)
        batch_size = x.shape[0]
        x = x.view(batch_size, -1, GESTURE_SEQUENCE_LENGTH)
        
        # CNN layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Global pooling
        x = self.global_pool(x).squeeze(-1)
        
        # Classification
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class MotionIntensityNet(nn.Module):
    """Network for predicting continuous motion parameters"""
    def __init__(self, input_size, hidden_size=64):
        super().__init__()
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, 
                           batch_first=True, bidirectional=True)
        
        # Separate heads for different motion parameters
        self.velocity_head = nn.Linear(hidden_size * 2, 2)  # x, y velocity scale
        self.zoom_head = nn.Linear(hidden_size * 2, 1)     # zoom intensity
        self.confidence_head = nn.Linear(hidden_size * 2, 1)  # motion confidence
        
    def forward(self, x):
        # x shape: (batch, sequence, features)
        lstm_out, _ = self.lstm(x)
        
        # Use last timestep output
        last_output = lstm_out[:, -1, :]
        
        velocity = self.velocity_head(last_output)
        zoom = self.zoom_head(last_output)
        confidence = torch.sigmoid(self.confidence_head(last_output))
        
        return {
            'velocity': velocity,
            'zoom': zoom,
            'confidence': confidence
        }

# ============== Training Functions ==============

def train_gesture_classifier(X_train, y_train, X_test, y_test, epochs=50):
    """Train lightweight gesture classifier"""
    print("\n=== Training Gesture Classifier ===")
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)
    
    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Model
    num_classes = len(np.unique(y_train))
    model = LightweightGestureNet(X_train.shape[1], num_classes).to(device)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    
    best_acc = 0.0
    best_model_state = None
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Evaluation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        avg_loss = train_loss / len(train_loader)
        
        scheduler.step(avg_loss)
        
        if accuracy > best_acc:
            best_acc = accuracy
            best_model_state = model.state_dict().copy()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    
    print(f"Best Accuracy: {best_acc:.2f}%")
    
    # Restore best model
    model.load_state_dict(best_model_state)
    
    return model, best_acc

def train_random_forest_classifier(X_train, y_train, X_test, y_test):
    """Train Random Forest as alternative fast classifier"""
    print("\n=== Training Random Forest Classifier ===")
    
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        n_jobs=-1,
        random_state=42
    )
    
    rf.fit(X_train, y_train)
    
    # Evaluate
    train_acc = rf.score(X_train, y_train) * 100
    test_acc = rf.score(X_test, y_test) * 100
    
    print(f"Train Accuracy: {train_acc:.2f}%")
    print(f"Test Accuracy: {test_acc:.2f}%")
    
    return rf, test_acc

def train_motion_predictor(motion_data: Dict, epochs=30):
    """Train motion intensity predictor"""
    print("\n=== Training Motion Predictor ===")
    
    # Prepare data
    X_sequences = []
    y_velocities = []
    y_zooms = []
    
    feature_extractor = AdvancedFeatureExtractor()
    
    for gesture_type, sessions in motion_data.items():
        for session in sessions:
            if len(session['landmarks_sequence']) < 5:
                continue
                
            # Extract motion features
            motion_features = feature_extractor.extract_motion_features(
                session['landmarks_sequence']
            )
            
            # Create sequences
            seq_len = min(len(motion_features['velocity_x']), 30)
            for i in range(len(session['landmarks_sequence']) - seq_len):
                # Input: landmark sequence
                seq = session['landmarks_sequence'][i:i+seq_len]
                features = feature_extractor.extract_gesture_features(seq)
                X_sequences.append(features)
                
                # Target: average velocity and zoom for next few frames
                end_idx = min(i + seq_len + 5, len(motion_features['velocity_x']))
                avg_vel_x = np.mean(motion_features['velocity_x'][i+seq_len:end_idx])
                avg_vel_y = np.mean(motion_features['velocity_y'][i+seq_len:end_idx])
                avg_zoom = np.mean(motion_features['zoom_rate'][i+seq_len:end_idx])
                
                y_velocities.append([avg_vel_x, avg_vel_y])
                y_zooms.append(avg_zoom)
    
    if not X_sequences:
        print("No motion data available for training")
        return None
    
    # Convert to arrays
    X = np.array(X_sequences)
    y_vel = np.array(y_velocities)
    y_zoom = np.array(y_zooms)
    
    print(f"Motion training samples: {len(X)}")
    
    # Scale features
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train simple regression model for now
    from sklearn.ensemble import RandomForestRegressor
    
    velocity_model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
    velocity_model.fit(X_scaled, y_vel)
    
    zoom_model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
    zoom_model.fit(X_scaled, y_zoom)
    
    return {
        'velocity_model': velocity_model,
        'zoom_model': zoom_model,
        'scaler': scaler
    }

# ============== Main Training Pipeline ==============

def main():
    print("Loading training data...")
    
    # Load gesture sequences
    if not os.path.exists('gesture_data/gesture_sequences.json'):
        print("Error: gesture_sequences.json not found. Run gesture_recorder.py first.")
        sys.exit(1)
    
    with open('gesture_data/gesture_sequences.json', 'r') as f:
        gesture_data = json.load(f)
    
    # Filter empty entries
    gesture_data = {k: v for k, v in gesture_data.items() if v}
    
    if not gesture_data:
        print("Error: No gesture data found.")
        sys.exit(1)
    
    # Prepare gesture classification data
    feature_extractor = AdvancedFeatureExtractor()
    X_gestures = []
    y_gestures = []
    
    print("\nExtracting gesture features...")
    for gesture, sequences in gesture_data.items():
        for seq in sequences:
            if len(seq) == GESTURE_SEQUENCE_LENGTH:
                features = feature_extractor.extract_gesture_features(seq)
                X_gestures.append(features)
                y_gestures.append(gesture)
    
    X_gestures = np.array(X_gestures)
    print(f"Total gesture samples: {len(X_gestures)}")
    print(f"Feature dimension: {X_gestures.shape[1]}")
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_gestures)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_gestures, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train multiple classifiers and pick best
    print("\nTraining classifiers...")
    
    # 1. Neural Network
    nn_model, nn_acc = train_gesture_classifier(
        X_train_scaled, y_train, X_test_scaled, y_test, epochs=50
    )
    
    # 2. Random Forest
    rf_model, rf_acc = train_random_forest_classifier(
        X_train_scaled, y_train, X_test_scaled, y_test
    )
    
    # Choose best model
    if rf_acc >= nn_acc:
        print(f"\nUsing Random Forest (accuracy: {rf_acc:.2f}%)")
        best_classifier = rf_model
        classifier_type = 'random_forest'
    else:
        print(f"\nUsing Neural Network (accuracy: {nn_acc:.2f}%)")
        best_classifier = nn_model
        classifier_type = 'neural_network'
    
    # Load and train motion predictor if available
    motion_models = None
    if os.path.exists('gesture_data/motion_sessions.json'):
        print("\nLoading motion data...")
        with open('gesture_data/motion_sessions.json', 'r') as f:
            motion_data = json.load(f)
        
        if motion_data:
            motion_models = train_motion_predictor(motion_data)
    
    # Save models
    print("\n=== Saving Models ===")
    os.makedirs('gesture_data', exist_ok=True)
    
    # Save gesture classifier
    if classifier_type == 'neural_network':
        torch.save({
            'model_state': best_classifier.state_dict(),
            'model_type': 'neural_network',
            'input_size': X_gestures.shape[1],
            'num_classes': len(label_encoder.classes_),
            'classes': label_encoder.classes_.tolist(),
            'sequence_length': GESTURE_SEQUENCE_LENGTH
        }, 'gesture_data/gesture_classifier.pth')
    else:
        joblib.dump(best_classifier, 'gesture_data/gesture_classifier.pkl')
        joblib.dump({
            'model_type': 'random_forest',
            'classes': label_encoder.classes_.tolist(),
            'sequence_length': GESTURE_SEQUENCE_LENGTH
        }, 'gesture_data/classifier_info.pkl')
    
    # Save preprocessors
    joblib.dump(scaler, 'gesture_data/gesture_scaler.pkl')
    joblib.dump(label_encoder, 'gesture_data/label_encoder.pkl')
    joblib.dump(feature_extractor, 'gesture_data/feature_extractor.pkl')
    
    # Save motion models if trained
    if motion_models:
        joblib.dump(motion_models, 'gesture_data/motion_models.pkl')
        print("✓ Motion models saved")
    
    print("\n✓ All models saved successfully!")
    print(f"Gesture classifier accuracy: {max(nn_acc, rf_acc):.2f}%")

if __name__ == "__main__":
    main()