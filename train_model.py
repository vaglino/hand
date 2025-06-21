# /train_model.py (Updated)

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class GestureLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def augment_sequence(sequence):
    noise = np.random.normal(0, 0.003, sequence.shape).astype(np.float32)
    augmented_seq = sequence + noise
    angle = np.random.uniform(-np.pi / 18, np.pi / 18)
    cos, sin = np.cos(angle), np.sin(angle)
    rotation_matrix = np.array([[cos, -sin], [sin, cos]], dtype=np.float32)
    reshaped_seq = augmented_seq.reshape(augmented_seq.shape[0], -1, 2)
    rotated_seq = np.dot(reshaped_seq, rotation_matrix.T)
    return rotated_seq.reshape(augmented_seq.shape[0], -1)

class GestureDataset(Dataset):
    def __init__(self, sequences, labels, augment=False):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)
        self.augment = augment
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx].numpy()
        if self.augment:
            num_coords = sequence.shape[1] // 2
            pos_data = sequence[:, :num_coords]
            vel_data = sequence[:, num_coords:]
            aug_pos_data = augment_sequence(pos_data)
            sequence = np.concatenate([aug_pos_data, vel_data], axis=1)
        return torch.FloatTensor(sequence), self.labels[idx]

def create_features(raw_data):
    X, y = [], []
    for gesture, sequences in raw_data.items():
        for seq in sequences:
            landmarks = np.array(seq)
            wrist_pos = landmarks[:, 0:2]
            landmarks_reshaped = landmarks.reshape(landmarks.shape[0], -1, 2)
            normalized_landmarks = landmarks_reshaped - wrist_pos[:, np.newaxis, :]
            velocity = np.diff(normalized_landmarks, axis=0, prepend=normalized_landmarks[0:1])
            pos_features = normalized_landmarks.reshape(landmarks.shape[0], -1)
            vel_features = velocity.reshape(landmarks.shape[0], -1)
            feature_vector = np.concatenate([pos_features, vel_features], axis=1)
            X.append(feature_vector)
            y.append(gesture)
    return np.array(X), np.array(y)

def train():
    # --- HYPERPARAMETERS ---
    # !CHANGE! Updated sequence length to 50
    SEQUENCE_LENGTH = 50 
    INPUT_SIZE_FACTOR = 4
    NUM_LANDMARKS = 21
    INPUT_SIZE = NUM_LANDMARKS * INPUT_SIZE_FACTOR
    HIDDEN_SIZE = 128
    NUM_LAYERS = 2
    DROPOUT = 0.3
    EPOCHS = 150
    BATCH_SIZE = 32
    LEARNING_RATE = 0.0005

    print("Loading and processing data...")
    with open('gesture_data/sequences.json', 'r') as f:
        raw_data = json.load(f)

    # Filter out empty gesture lists before processing
    filtered_data = {k: v for k, v in raw_data.items() if v}
    if not filtered_data:
        print("No gesture data found in sequences.json. Please record gestures first.")
        sys.exit(1)

    X, y = create_features(filtered_data)
    print(f"Total sequences: {len(X)}")
    print(f"Feature vector shape: {X.shape}")

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    NUM_CLASSES = len(label_encoder.classes_)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

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
        total_loss = 0
        for sequences, labels in train_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

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
        if epoch % 10 == 0 or epoch == EPOCHS - 1:
            print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss/len(train_loader):.4f}, Test Accuracy: {acc:.2f}%')
            
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
    print("\n✓ Model and preprocessing saved!")

if __name__ == "__main__":
    if not os.path.exists('gesture_data/sequences.json'):
        print("No data found! Run gesture_recorder.py first.")
        sys.exit(1)
    train()