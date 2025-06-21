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

class GestureDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

class GestureLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_classes=5):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size,
                          num_layers=1, batch_first=True)
        self.fc  = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

def train():
    # Load raw data
    print("Loading data...")
    with open('gesture_data/sequences.json', 'r') as f:
        raw = json.load(f)

    X_raw, y_raw = [], []
    for gesture, seqs in raw.items():
        for seq in seqs:
            X_raw.append(seq)
            y_raw.append(gesture)
    print(f"Total sequences: {len(X_raw)}")

    # Determine max sequence length and feature dimension
    lengths = [len(s) for s in X_raw]
    max_len = max(lengths)
    print(f"Padding all sequences to length = {max_len}")

    # Determine feature dimension
    if len(X_raw) == 0 or len(X_raw[0]) == 0:
        print("No sequences or empty sequences found!")
        sys.exit(1)
    n_features = len(X_raw[0][0])

    # Pad shorter sequences with zeros; truncate longer ones if any
    X_padded = []
    for seq in X_raw:
        seq_len = len(seq)
        if seq_len < max_len:
            pad = [[0.0]*n_features] * (max_len - seq_len)
            new_seq = seq + pad
        else:
            new_seq = seq[:max_len]
        X_padded.append(new_seq)

    X = np.array(X_padded)  # shape (N, max_len, n_features)
    y = np.array(y_raw)
    print(f"Sequence array shape: {X.shape}")

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # Scale features
    n_samples, n_steps, _ = X_train.shape
    scaler = StandardScaler()
    X_train_flat = X_train.reshape(-1, n_features)
    X_train_scaled = scaler.fit_transform(X_train_flat).reshape(n_samples, n_steps, n_features)
    X_test_scaled = scaler.transform(X_test.reshape(-1, n_features))\
                         .reshape(X_test.shape[0], n_steps, n_features)

    # Create DataLoaders
    train_ds = GestureDataset(X_train_scaled, y_train)
    test_ds  = GestureDataset(X_test_scaled, y_test)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader  = DataLoader(test_ds, batch_size=32)

    # Build model
    model = GestureLSTM(n_features, hidden_size=64, num_classes=len(label_encoder.classes_))
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    print("\nTraining...")
    best_acc = 0.0
    epochs = 100
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Evaluate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
        acc = correct / total

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'gesture_data/best_model.pth')

        if epoch % 10 == 0:
            print(f"Epoch {epoch:2d} — Loss: {total_loss/len(train_loader):.4f}, Acc: {acc:.4f}")

    print(f"\nBest test accuracy: {best_acc:.4f}")

    # Save model + preprocessing
    os.makedirs('gesture_data', exist_ok=True)
    torch.save({
        'model_state': model.state_dict(),
        'input_size': n_features,
        'sequence_length': n_steps,
        'classes': label_encoder.classes_.tolist()
    }, 'gesture_data/model.pth')
    joblib.dump(scaler, 'gesture_data/scaler.pkl')
    joblib.dump(label_encoder, 'gesture_data/label_encoder.pkl')

    print("\n✓ Model and preprocessing saved! You can now run gesture_control.py")

if __name__ == "__main__":
    if not os.path.exists('gesture_data/sequences.json'):
        print("No data found! Run gesture_recorder.py first.")
        sys.exit(1)
    train()
