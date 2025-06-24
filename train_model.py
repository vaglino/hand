# train_model.py

import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
from collections import Counter
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.5)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, context=None):
        lstm_out, _ = self.lstm(x)
        last_step_out = lstm_out[:, -1, :]
        out = self.dropout(last_step_out)
        out = self.fc(out)
        return {'gesture': out}

class GestureDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

def extract_simple_features(landmarks_sequence):
    sequence = np.array(landmarks_sequence)
    features = []
    for frame in sequence:
        wrist = frame[0]
        relative_positions = (frame - wrist).flatten()
        features.append(relative_positions)
    return np.array(features)

def train_model(X_train, y_train, X_val, y_val, class_weights, num_epochs=100, batch_size=32):
    train_dataset = GestureDataset(X_train, y_train)
    val_dataset = GestureDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    input_size = X_train.shape[2]
    num_classes = len(class_weights)
    
    model = SimpleLSTM(input_size=input_size, hidden_size=64, num_layers=2, num_classes=num_classes).to(device)
    
    weights_tensor = torch.FloatTensor(class_weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.2, verbose=True)

    best_val_acc = 0
    patience_counter = 0
    max_patience = 20
    best_model_state = model.state_dict() # Initialize with current state

    print("Starting training with SIMPLIFIED model and features...")
    for epoch in range(num_epochs):
        model.train()
        for sequences, labels in train_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs['gesture'], labels)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences, labels = sequences.to(device), labels.to(device)
                outputs = model(sequences)
                loss = criterion(outputs['gesture'], labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs['gesture'], 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total
        scheduler.step(avg_val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1: >3}/{num_epochs}] | Val Acc: {val_acc:.2f}% | Val Loss: {avg_val_loss:.4f}")

        if patience_counter >= max_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
            
    model.load_state_dict(best_model_state)
    return model, best_val_acc

def main():
    data_path = 'gesture_data/transition_aware_training_data.json'
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Please run the recorder first.")
        return
        
    with open(data_path, 'r') as f:
        data = json.load(f)

    if not isinstance(data, dict) or 'sequences' not in data or not data['sequences']:
        print(f"Error: The data file '{data_path}' is empty or malformed.")
        print("Please delete the 'gesture_data' folder and re-record your gestures.")
        return

    print(f"Loaded {len(data['sequences'])} samples")

    X = np.array([extract_simple_features(seq) for seq in data['sequences']])
    labels = data['labels']

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)
    
    label_counts = Counter(y)
    num_classes = len(label_counts)
    class_weights = [len(y) / (num_classes * label_counts.get(i, 1)) for i in range(num_classes)]
    
    print("\nClasses and Weights:")
    for i, class_name in enumerate(label_encoder.classes_):
        print(f"  - {class_name:<25}: Count={label_counts[i]}, Weight={class_weights[i]:.2f}")

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_val_scaled = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)

    model, best_acc = train_model(X_train_scaled, y_train, X_val_scaled, y_val, class_weights)
    print(f"\nBest validation accuracy: {best_acc:.2f}%")
    
    print("\nSaving models...")
    os.makedirs('gesture_data', exist_ok=True)
    
    torch.save({
        'model_state': model.state_dict(), 'input_size': X.shape[2],
        'sequence_length': X.shape[1], 'num_classes': len(label_encoder.classes_),
        'classes': label_encoder.classes_.tolist(), 'best_accuracy': best_acc
    }, 'gesture_data/gesture_model.pth')
    
    joblib.dump(scaler, 'gesture_data/gesture_scaler.pkl')
    joblib.dump(label_encoder, 'gesture_data/gesture_label_encoder.pkl')

    print("\n--- Final Model Evaluation ---")
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for i in range(len(X_val_scaled)):
            seq = torch.FloatTensor(X_val_scaled[i:i+1]).to(device)
            outputs = model(seq)
            _, predicted = torch.max(outputs['gesture'], 1)
            all_preds.append(predicted.cpu().item())
            all_labels.append(y_val[i])
            
    report = classification_report(all_labels, all_preds, target_names=label_encoder.classes_, zero_division=0)
    print(report)
    print("\n✓ Training complete!")

if __name__ == "__main__":
    main() 