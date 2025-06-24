# enhanced_train_model.py - Training script for transition-aware gesture recognition

import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import the transition-aware model
from transition_aware_model import TransitionAwareLSTM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class TransitionAwareDataset(Dataset):
    """Dataset that includes transition phases and context"""
    
    def __init__(self, sequences, labels, phases, contexts, label_encoder, context_encoder):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)
        self.phases = phases
        self.contexts = torch.LongTensor(contexts)
        self.label_encoder = label_encoder
        self.context_encoder = context_encoder
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return {
            'sequence': self.sequences[idx],
            'label': self.labels[idx],
            'context': self.contexts[idx],
            'phases': self.phases[idx]
        }

def extract_enhanced_features(landmarks_sequence):
    """Extract features that capture transition dynamics"""
    sequence = np.array(landmarks_sequence)
    features = []
    
    for t in range(len(sequence)):
        frame_features = []
        
        # Basic position features (normalized to wrist)
        wrist = sequence[t][0]
        relative_positions = sequence[t] - wrist
        frame_features.extend(relative_positions.flatten())
        
        # Velocity features
        if t > 0:
            velocity = sequence[t] - sequence[t-1]
            frame_features.extend(velocity.flatten())
        else:
            frame_features.extend(np.zeros(21 * 3))
        
        # Acceleration features
        if t > 1:
            acceleration = (sequence[t] - sequence[t-1]) - (sequence[t-1] - sequence[t-2])
            frame_features.extend(acceleration.flatten())
        else:
            frame_features.extend(np.zeros(21 * 3))
        
        # Hand configuration features
        # Finger spread
        fingertips = [4, 8, 12, 16, 20]
        spreads = []
        for i in range(len(fingertips)):
            for j in range(i+1, len(fingertips)):
                dist = np.linalg.norm(relative_positions[fingertips[i]] - relative_positions[fingertips[j]])
                spreads.append(dist)
        frame_features.extend(spreads)
        
        # Palm orientation
        palm_vec1 = relative_positions[5] - relative_positions[0]
        palm_vec2 = relative_positions[17] - relative_positions[0]
        palm_normal = np.cross(palm_vec1, palm_vec2)
        palm_normal = palm_normal / (np.linalg.norm(palm_normal) + 1e-6)
        frame_features.extend(palm_normal)
        
        features.append(frame_features)
    
    return np.array(features)

def create_transition_labels(labels, phases):
    """Create refined labels that distinguish transitions"""
    refined_labels = []
    
    for label, phase_seq in zip(labels, phases):
        # Count phase occurrences
        phase_counts = {}
        for phase in phase_seq:
            phase_counts[phase] = phase_counts.get(phase, 0) + 1
        
        # Determine refined label
        dominant_phase = max(phase_counts, key=phase_counts.get)
        
        if '_transition' in label:
            # Already a transition label
            refined_labels.append(label)
        elif dominant_phase in ['transitioning_to_gesture', 'transitioning_to_neutral']:
            # Refine to transition label
            base_gesture = label.replace('_transition', '')
            refined_labels.append(f"{base_gesture}_transition")
        else:
            # Keep original label
            refined_labels.append(label)
    
    return refined_labels

def train_transition_aware_model(X_train, y_train, X_val, y_val, contexts_train, contexts_val, 
                                num_epochs=100, batch_size=32):
    """Train the transition-aware model"""
    
    # Create datasets
    train_dataset = TransitionAwareDataset(
        X_train, y_train, None, contexts_train, None, None
    )
    val_dataset = TransitionAwareDataset(
        X_val, y_val, None, contexts_val, None, None
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Model parameters
    input_size = X_train.shape[2]
    num_classes = len(np.unique(y_train))
    
    # Initialize model
    model = TransitionAwareLSTM(
        input_size=input_size,
        hidden_size=128,
        num_layers=2,
        num_classes=num_classes
    ).to(device)
    
    # Loss functions
    gesture_criterion = nn.CrossEntropyLoss()
    transition_criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': []
    }
    
    best_val_acc = 0
    best_model_state = None
    patience_counter = 0
    max_patience = 20
    
    print("Starting training...")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch in train_loader:
            sequences = batch['sequence'].to(device)
            labels = batch['label'].to(device)
            contexts = batch['context'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(sequences, contexts)
            
            # Calculate losses
            gesture_loss = gesture_criterion(outputs['gesture'], labels)
            
            # For transition loss, create pseudo-labels based on gesture labels
            # This is simplified - in practice, you'd have actual transition labels
            transition_labels = (labels != 0).long()  # 0 = neutral, 1 = transitioning/active
            transition_loss = transition_criterion(outputs['transition'], transition_labels)
            
            # Combined loss
            total_loss = gesture_loss + 0.3 * transition_loss
            
            # Backward pass
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Statistics
            train_loss += total_loss.item()
            _, predicted = torch.max(outputs['gesture'], 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                sequences = batch['sequence'].to(device)
                labels = batch['label'].to(device)
                contexts = batch['context'].to(device)
                
                outputs = model(sequences, contexts)
                
                gesture_loss = gesture_criterion(outputs['gesture'], labels)
                transition_labels = (labels != 0).long()
                transition_loss = transition_criterion(outputs['transition'], transition_labels)
                total_loss = gesture_loss + 0.3 * transition_loss
                
                val_loss += total_loss.item()
                _, predicted = torch.max(outputs['gesture'], 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        # Calculate metrics
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        # Update history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}]")
            print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Early stopping
        if patience_counter >= max_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Restore best model
    model.load_state_dict(best_model_state)
    
    return model, history, best_val_acc

def analyze_transition_patterns(data):
    """Analyze transition patterns in the data"""
    print("\n=== Analyzing Transition Patterns ===")
    
    transition_matrix = {}
    
    for i in range(len(data['labels']) - 1):
        current = data['labels'][i]
        next_label = data['labels'][i + 1]
        
        if current not in transition_matrix:
            transition_matrix[current] = {}
        
        if next_label not in transition_matrix[current]:
            transition_matrix[current][next_label] = 0
        
        transition_matrix[current][next_label] += 1
    
    # Print common transitions
    print("\nMost common transitions:")
    transitions = []
    for from_gesture, to_gestures in transition_matrix.items():
        for to_gesture, count in to_gestures.items():
            if count > 10:  # Only show frequent transitions
                transitions.append((from_gesture, to_gesture, count))
    
    transitions.sort(key=lambda x: x[2], reverse=True)
    for from_g, to_g, count in transitions[:10]:
        print(f"  {from_g} → {to_g}: {count} times")
    
    return transition_matrix

def main():
    print("=== Enhanced Transition-Aware Training ===\n")
    
    # Load data
    data_path = 'gesture_data/transition_aware_training_data.json'
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        print("Please run the enhanced recorder first to generate transition-aware data.")
        return
    
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data['sequences'])} samples")
    
    # Analyze transition patterns
    transition_matrix = analyze_transition_patterns(data)
    
    # Extract features
    print("\nExtracting enhanced features...")
    X = []
    for seq in data['sequences']:
        features = extract_enhanced_features(seq)
        X.append(features)
    
    X = np.array(X)
    print(f"Feature shape: {X.shape}")
    
    # Process labels
    refined_labels = create_transition_labels(data['labels'], data['phases'])
    
    # Encode labels and contexts
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(refined_labels)
    
    context_encoder = LabelEncoder()
    # Add 'none' to possible contexts
    all_contexts = list(set(data['contexts'] + ['none']))
    context_encoder.fit(all_contexts)
    contexts = context_encoder.transform(data['contexts'])
    
    print(f"\nClasses: {label_encoder.classes_}")
    print(f"Contexts: {context_encoder.classes_}")
    
    # Split data
    X_train, X_val, y_train, y_val, contexts_train, contexts_val = train_test_split(
        X, y, contexts, test_size=0.2, random_state=42, stratify=y
    )
    
    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1]))
    X_train_scaled = X_train_scaled.reshape(X_train.shape)
    
    X_val_scaled = scaler.transform(X_val.reshape(-1, X_val.shape[-1]))
    X_val_scaled = X_val_scaled.reshape(X_val.shape)
    
    # Train model
    print("\nTraining transition-aware model...")
    model, history, best_acc = train_transition_aware_model(
        X_train_scaled, y_train, X_val_scaled, y_val,
        contexts_train, contexts_val,
        num_epochs=100, batch_size=32
    )
    
    print(f"\nBest validation accuracy: {best_acc:.2f}%")
    
    # Save model and preprocessors
    print("\nSaving models...")
    os.makedirs('gesture_data', exist_ok=True)
    
    # Save PyTorch model
    torch.save({
        'model_state': model.state_dict(),
        'input_size': X.shape[2],
        'sequence_length': X.shape[1],
        'num_classes': len(label_encoder.classes_),
        'classes': label_encoder.classes_.tolist(),
        'context_classes': context_encoder.classes_.tolist(),
        'transition_matrix': transition_matrix,
        'best_accuracy': best_acc
    }, 'gesture_data/transition_aware_model.pth')
    
    # Save preprocessors
    joblib.dump(scaler, 'gesture_data/transition_scaler.pkl')
    joblib.dump(label_encoder, 'gesture_data/transition_label_encoder.pkl')
    joblib.dump(context_encoder, 'gesture_data/context_encoder.pkl')
    
    # Generate confusion matrix for analysis
    print("\nAnalyzing model performance on transitions...")
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for i in range(len(X_val_scaled)):
            seq = torch.FloatTensor(X_val_scaled[i:i+1]).to(device)
            ctx = torch.LongTensor([contexts_val[i]]).to(device)
            
            outputs = model(seq, ctx)
            _, predicted = torch.max(outputs['gesture'], 1)
            
            all_preds.append(predicted.cpu().item())
            all_labels.append(y_val[i])
    
    # Analyze transition vs non-transition accuracy
    transition_correct = 0
    transition_total = 0
    non_transition_correct = 0
    non_transition_total = 0
    
    for pred, label in zip(all_preds, all_labels):
        label_name = label_encoder.inverse_transform([label])[0]
        pred_name = label_encoder.inverse_transform([pred])[0]
        
        if '_transition' in label_name:
            transition_total += 1
            if pred == label:
                transition_correct += 1
        else:
            non_transition_total += 1
            if pred == label:
                non_transition_correct += 1
    
    if transition_total > 0:
        print(f"\nTransition accuracy: {100 * transition_correct / transition_total:.2f}%")
    if non_transition_total > 0:
        print(f"Non-transition accuracy: {100 * non_transition_correct / non_transition_total:.2f}%")
    
    print("\n✓ Training complete!")
    print("The model now understands gesture transitions and can differentiate between")
    print("intentional gestures and return-to-neutral movements.")

if __name__ == "__main__":
    main()