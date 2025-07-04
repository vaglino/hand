# train_model.py

import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter
import joblib
import os
import warnings
import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- NEW: Powerful Gesture Classifier Model ---
class GestureClassifier(nn.Module):
    """
    A more powerful model that uses CNN for feature extraction, Bi-LSTM for
    temporal analysis, and Attention to focus on key frames. This architecture
    is much better at distinguishing similar but distinct motions like
    'scroll_down' vs 'scroll_up_return'.
    """
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.5):
        super().__init__()
        # 1. Feature extraction from raw landmarks
        self.conv1 = nn.Conv1d(input_size, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        
        # 2. Bidirectional LSTM to capture temporal context from both directions
        self.lstm = nn.LSTM(64, hidden_size, num_layers, 
                            batch_first=True, bidirectional=True, dropout=dropout)
        
        # 3. Attention mechanism to weigh important time steps
        self.attention_fc = nn.Linear(hidden_size * 2, 1)

        # 4. Final classifier
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (batch_size, seq_len, features)
        
        # CNN feature extraction
        x = x.transpose(1, 2)  # (batch_size, features, seq_len)
        x = F.relu(self.bn1(self.conv1(x)))
        x = x.transpose(1, 2)  # (batch_size, seq_len, cnn_features)
        
        # Bi-LSTM processing
        lstm_out, _ = self.lstm(x) # (batch_size, seq_len, hidden_size*2)
        
        # Attention mechanism
        attention_weights = F.softmax(self.attention_fc(lstm_out), dim=1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1) # (batch_size, hidden_size*2)
        
        # Classification
        out = self.dropout(context_vector)
        out = self.fc(out)
        return out

class GestureDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

def extract_features(landmarks_sequence):
    """Extracts frame-by-frame features from a sequence of landmarks."""
    sequence = np.array(landmarks_sequence)
    if sequence.ndim != 3 or sequence.shape[1:] != (21, 3):
        return None # Skip malformed data

    # Normalize landmarks relative to the wrist
    wrist = sequence[:, 0:1, :]
    relative_landmarks = (sequence - wrist).reshape(sequence.shape[0], -1)

    # Calculate velocities (frame-to-frame difference)
    # --- FIX: Correctly prepend a row of zeros for the first frame's velocity ---
    velocities = np.diff(relative_landmarks, axis=0, prepend=np.zeros((1, relative_landmarks.shape[1])))
    
    # Combine features
    features = np.concatenate([relative_landmarks, velocities], axis=1)
    return features

def train(model, train_loader, val_loader, class_weights, num_epochs=100):
    weights_tensor = torch.FloatTensor(class_weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5, verbose=True)

    best_val_acc = 0
    patience_counter = 0
    max_patience = 20
    best_model_state = model.state_dict()

    print("\n--- Starting Training ---")
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for sequences, labels in train_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences, labels = sequences.to(device), labels.to(device)
                outputs = model(sequences)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total
        scheduler.step(avg_val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            print(f"Epoch {epoch+1:03d} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}% | ✨ New Best")
        else:
            patience_counter += 1
            if (epoch + 1) % 5 == 0:
                 print(f"Epoch {epoch+1:03d} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")


        if patience_counter >= max_patience:
            print(f"Early stopping at epoch {epoch+1}. Best validation accuracy: {best_val_acc:.2f}%")
            break
            
    model.load_state_dict(best_model_state)
    return model, best_val_acc

def main():
    data_path = 'gesture_data/training_data.json'
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Please run the recorder first.")
        return
        
    with open(data_path, 'r') as f:
        data = json.load(f)

    print(f"Loaded {len(data['sequences'])} sequences.")
    
    # Process features and labels
    sequences = [extract_features(seq) for seq in data['sequences']]
    labels = data['labels']
    
    # Filter out any malformed data
    valid_indices = [i for i, seq in enumerate(sequences) if seq is not None]
    X = np.array([sequences[i] for i in valid_indices])
    labels = [labels[i] for i in valid_indices]

    if len(X) == 0:
        print("Error: No valid sequences found in the data. Please re-record.")
        return

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)
    
    # Calculate class weights for imbalanced datasets
    label_counts = Counter(y)
    num_classes = len(label_counts)
    total_samples = len(y)
    class_weights = [total_samples / (num_classes * label_counts.get(i, 1)) for i in range(num_classes)]
    
    print("\n--- Data Preprocessing ---")
    print(f"Found {len(X)} valid samples.")
    print("Classes and Weights:")
    for i, class_name in enumerate(label_encoder.classes_):
        print(f"  - {class_name:<25}: Count={label_counts[i]}, Weight={class_weights[i]:.2f}")

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # --- FIX: Correctly reshape data for the scaler ---
    scaler = StandardScaler()
    num_train_samples, seq_len, num_features = X_train.shape
    X_train_reshaped = X_train.reshape(-1, num_features)
    scaler.fit(X_train_reshaped)
    X_train_scaled_reshaped = scaler.transform(X_train_reshaped)
    X_train_scaled = X_train_scaled_reshaped.reshape(num_train_samples, seq_len, num_features)

    num_val_samples, _, _ = X_val.shape
    X_val_reshaped = X_val.reshape(-1, num_features)
    X_val_scaled_reshaped = scaler.transform(X_val_reshaped)
    X_val_scaled = X_val_scaled_reshaped.reshape(num_val_samples, seq_len, num_features)

    # Initialize model
    model = GestureClassifier(
        input_size=X_train_scaled.shape[2],
        hidden_size=128,
        num_layers=2,
        num_classes=num_classes
    ).to(device)

    # Train model
    model, best_acc = train(model, 
                            DataLoader(GestureDataset(X_train_scaled, y_train), batch_size=32, shuffle=True),
                            DataLoader(GestureDataset(X_val_scaled, y_val), batch_size=32),
                            class_weights)
    
    print(f"\n✓ Training complete. Best validation accuracy: {best_acc:.2f}%")
    
    # --- Save artifacts ---
    print("\nSaving model and preprocessors...")
    os.makedirs('gesture_data', exist_ok=True)
    
    model_save_path = 'gesture_data/gesture_classifier.pth'
    torch.save({
        'model_state': model.state_dict(),
        'input_size': X_train_scaled.shape[2],
        'hidden_size': 128,
        'num_layers': 2,
        'num_classes': num_classes,
        'sequence_length': X_train_scaled.shape[1],
        'best_accuracy': best_acc
    }, model_save_path)
    joblib.dump(scaler, 'gesture_data/gesture_scaler.pkl')
    joblib.dump(label_encoder, 'gesture_data/gesture_label_encoder.pkl')
    print(f"✓ Model saved to {model_save_path}")

    # --- Final Evaluation ---
    print("\n--- Final Model Evaluation on Validation Set ---")
    model.eval()
    y_pred = []
    with torch.no_grad():
        val_data = torch.FloatTensor(X_val_scaled).to(device)
        outputs = model(val_data)
        _, predicted = torch.max(outputs, 1)
        y_pred = predicted.cpu().numpy()
        
    report = classification_report(y_val, y_pred, target_names=label_encoder.classes_, zero_division=0)
    print(report)

    # Plot confusion matrix
    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('gesture_data/confusion_matrix.png')
    print("✓ Confusion matrix saved to gesture_data/confusion_matrix.png")

if __name__ == "__main__":
    main()