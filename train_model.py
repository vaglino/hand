# /train_model.py (Massively Improved)

import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os
import sys

# --- HYPERPARAMETERS ---
SEQUENCE_LENGTH = 30 # Must match recorder
NUM_LANDMARKS = 21
# !NEW! Features: pos(3), vel(3), accel(3) -> 9 per landmark
INPUT_SIZE = NUM_LANDMARKS * 9 
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.4 # Increased dropout for better generalization
EPOCHS = 70 # Can use fewer epochs with better features
BATCH_SIZE = 32
LEARNING_RATE = 0.0005

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- MODEL DEFINITION (Same as before) ---
class GestureLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# --- FEATURE ENGINEERING (The secret sauce) ---
def create_features_for_sequence(seq_landmarks):
    """
    Creates robust features (pos, vel, accel) for a single sequence.
    This function is now the single source of truth for feature creation.
    """
    seq_landmarks = np.array(seq_landmarks) # Shape: (L, 21, 3)
    
    # 1. Wrist-relative coordinates (translation invariance)
    wrist = seq_landmarks[:, 0, :] # wrist is landmark 0
    relative_coords = seq_landmarks - wrist[:, np.newaxis, :]
    
    # 2. Scale normalization (scale invariance)
    # Use distance between wrist(0) and middle_finger_mcp(9) as hand "size"
    hand_size = np.linalg.norm(relative_coords[:, 9, :], axis=1)
    # Avoid division by zero if hand size is somehow 0
    hand_size[hand_size < 1e-6] = 1 
    scaled_coords = relative_coords / hand_size[:, np.newaxis, np.newaxis]
    
    # 3. Temporal features (velocity and acceleration)
    # Prepend first element to keep array dimensions consistent after diff
    velocity = np.diff(scaled_coords, axis=0, prepend=scaled_coords[0:1, :, :])
    acceleration = np.diff(velocity, axis=0, prepend=velocity[0:1, :, :])
    
    # 4. Flatten and combine
    pos_features = scaled_coords.reshape(SEQUENCE_LENGTH, -1)
    vel_features = velocity.reshape(SEQUENCE_LENGTH, -1)
    accel_features = acceleration.reshape(SEQUENCE_LENGTH, -1)
    
    return np.concatenate([pos_features, vel_features, accel_features], axis=1)

# --- DATASET and AUGMENTATION ---
def augment_sequence(sequence):
    """Augments the position part of the feature vector."""
    num_coords = NUM_LANDMARKS * 3
    pos_data = sequence[:, :num_coords]
    
    # Add small noise
    noise = np.random.normal(0, 0.01, pos_data.shape).astype(np.float32)
    augmented_seq = pos_data + noise
    
    # Reshape back to (L, 21, 3) for rotation
    reshaped_seq = augmented_seq.reshape(SEQUENCE_LENGTH, -1, 3)
    
    # Apply a small random 3D rotation
    angle_x = np.random.uniform(-np.pi / 12, np.pi / 12)
    angle_y = np.random.uniform(-np.pi / 12, np.pi / 12)
    angle_z = np.random.uniform(-np.pi / 12, np.pi / 12)
    Rx = np.array([[1,0,0], [0,np.cos(angle_x),-np.sin(angle_x)], [0,np.sin(angle_x),np.cos(angle_x)]])
    Ry = np.array([[np.cos(angle_y),0,np.sin(angle_y)], [0,1,0], [-np.sin(angle_y),0,np.cos(angle_y)]])
    Rz = np.array([[np.cos(angle_z),-np.sin(angle_z),0], [np.sin(angle_z),np.cos(angle_z),0], [0,0,1]])
    rotation_matrix = Rz @ Ry @ Rx
    
    rotated_seq = np.dot(reshaped_seq, rotation_matrix.T)
    
    # Combine augmented pos data with original vel/accel data
    aug_pos_flat = rotated_seq.reshape(SEQUENCE_LENGTH, -1)
    vel_accel_data = sequence[:, num_coords:]
    return np.concatenate([aug_pos_flat, vel_accel_data], axis=1)

class GestureDataset(Dataset):
    def __init__(self, sequences, labels, augment=False):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)
        self.augment = augment
    
    def __len__(self): return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx].numpy()
        if self.augment:
            sequence = augment_sequence(sequence)
        return torch.FloatTensor(sequence), self.labels[idx]

# --- MAIN TRAINING SCRIPT ---
def train():
    print("Loading and processing data...")
    with open('gesture_data/sequences.json', 'r') as f:
        raw_data = json.load(f)

    filtered_data = {k: v for k, v in raw_data.items() if v}
    if not filtered_data:
        print("Error: No gesture data found. Please run gesture_recorder.py first.")
        sys.exit(1)

    X, y = [], []
    for gesture, sequences in filtered_data.items():
        for seq in sequences:
            # Check for data integrity
            if len(seq) == SEQUENCE_LENGTH:
                feature_vector = create_features_for_sequence(seq)
                X.append(feature_vector)
                y.append(gesture)

    if not X:
        print("Error: Could not process any sequences. Check data format.")
        sys.exit(1)

    X = np.array(X)
    print(f"Total sequences: {len(X)}")
    print(f"Feature vector shape: {X.shape}")

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    NUM_CLASSES = len(label_encoder.classes_)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    
    # Scale the data based on the training set
    scaler = StandardScaler()
    X_train_flat = X_train.reshape(-1, X_train.shape[-1])
    scaler.fit(X_train_flat)
    
    X_train_scaled = scaler.transform(X_train.reshape(-1, INPUT_SIZE)).reshape(X_train.shape)
    X_test_scaled = scaler.transform(X_test.reshape(-1, INPUT_SIZE)).reshape(X_test.shape)
    
    train_ds = GestureDataset(X_train_scaled, y_train, augment=True)
    test_ds = GestureDataset(X_test_scaled, y_test, augment=False)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
    
    model = GestureLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES, DROPOUT).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("\nTraining...")
    best_acc = 0.0
    for epoch in range(EPOCHS):
        model.train()
        for sequences, labels in train_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Evaluation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for sequences, labels in test_loader:
                sequences, labels = sequences.to(device), labels.to(device)
                outputs = model(sequences)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        acc = 100 * correct / total
        if (epoch + 1) % 10 == 0 or epoch == EPOCHS - 1:
            print(f'Epoch [{epoch+1}/{EPOCHS}], Test Accuracy: {acc:.2f}%')
            
        if acc > best_acc:
            best_acc = acc
            os.makedirs('gesture_data', exist_ok=True)
            torch.save({
                'model_state': model.state_dict(),
                'input_size': INPUT_SIZE,
                'hidden_size': HIDDEN_SIZE,
                'num_layers': NUM_LAYERS,
                'sequence_length': SEQUENCE_LENGTH,
                'classes': label_encoder.classes_.tolist()
            }, 'gesture_data/model.pth')
            joblib.dump(scaler, 'gesture_data/scaler.pkl')
            joblib.dump(label_encoder, 'gesture_data/label_encoder.pkl')

    print(f'\nBest Test Accuracy: {best_acc:.2f}%')
    print("✓ Model and preprocessing objects saved!")

if __name__ == "__main__":
    if not os.path.exists('gesture_data/sequences.json'):
        print("Error: sequences.json not found. Run gesture_recorder.py first.")
        sys.exit(1)
    train()