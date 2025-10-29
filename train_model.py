# enhanced_train_model.py - Advanced gesture training with TCN and improved preprocessing

import json
import numpy as np
import torch
import torch.nn as nn
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
import math
from typing import List, Tuple, Optional
import time
import platform

warnings.filterwarnings('ignore')

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    # Check for MPS (Apple Silicon GPU)
    device = torch.device('mps')
else:
    device = torch.device('cpu')

from hand.features.landmarks import LandmarkPreprocessor

from hand.models.tcn import EnhancedGestureClassifier

# --- DATA AUGMENTATION ---
class GestureAugmentator:
    """Data augmentation for gesture sequences."""
    
    def __init__(self):
        pass
        
    def time_warp(self, sequence: np.ndarray, sigma: float = 0.2) -> np.ndarray:
        """Apply random time warping to the sequence."""
        seq_len = len(sequence)
        if seq_len < 4:
            return sequence
            
        # Generate smooth warping curve
        time_steps = np.linspace(0, 1, seq_len)
        warp_steps = time_steps + np.random.normal(0, sigma, seq_len)
        warp_steps = np.clip(warp_steps, 0, 1)
        warp_steps = np.sort(warp_steps)  # Maintain monotonicity
        
        # Interpolate sequence according to warped time
        warped_sequence = np.zeros_like(sequence)
        for feature_idx in range(sequence.shape[1]):
            warped_sequence[:, feature_idx] = np.interp(time_steps, warp_steps, sequence[:, feature_idx])
            
        return warped_sequence
    
    def add_noise(self, sequence: np.ndarray, noise_level: float = 0.02) -> np.ndarray:
        """Add Gaussian noise to the sequence."""
        noise = np.random.normal(0, noise_level, sequence.shape)
        return sequence + noise
    
    def scale_features(self, sequence: np.ndarray, scale_range: Tuple[float, float] = (0.9, 1.1)) -> np.ndarray:
        """Apply random scaling to features."""
        scale_factor = np.random.uniform(scale_range[0], scale_range[1])
        return sequence * scale_factor
    
    def augment_sequence(self, sequence: np.ndarray) -> np.ndarray:
        """Apply random combination of augmentations."""
        aug_sequence = sequence.copy()
        
        # Random time warping (50% chance)
        if np.random.random() < 0.5:
            aug_sequence = self.time_warp(aug_sequence)
            
        # Random noise (70% chance)
        if np.random.random() < 0.7:
            aug_sequence = self.add_noise(aug_sequence)
            
        # Random scaling (30% chance)
        if np.random.random() < 0.3:
            aug_sequence = self.scale_features(aug_sequence)
            
        return aug_sequence

# --- ENHANCED DATASET WITH AUGMENTATION ---
class EnhancedGestureDataset(Dataset):
    def __init__(self, sequences, labels, augment=False, augment_factor=2):
        self.original_sequences = sequences
        self.original_labels = labels
        self.augment = augment
        self.augmentator = GestureAugmentator() if augment else None
        
        # Generate augmented data for minority classes
        if augment:
            self.sequences, self.labels = self._create_balanced_dataset(augment_factor)
        else:
            self.sequences = torch.FloatTensor(sequences)
            self.labels = torch.LongTensor(labels)
    
    def _create_balanced_dataset(self, augment_factor):
        """Create balanced dataset with augmentation for minority classes.

        Oversamples each class up to the max class count to balance the dataset.
        """
        label_counts = Counter(self.original_labels)
        max_count = max(label_counts.values())
        
        augmented_sequences = []
        augmented_labels = []
        
        # Group sequences by label
        label_to_sequences = {}
        for seq, label in zip(self.original_sequences, self.original_labels):
            if label not in label_to_sequences:
                label_to_sequences[label] = []
            label_to_sequences[label].append(seq)
        
        # Balance classes with augmentation
        for label, sequences in label_to_sequences.items():
            current_count = len(sequences)
            # Target exactly the maximum class count for balance
            target_count = max_count
            
            # Add original sequences
            augmented_sequences.extend(sequences)
            augmented_labels.extend([label] * current_count)
            
            # Add augmented sequences if needed
            if target_count > current_count:
                needed = target_count - current_count
                for _ in range(needed):
                    # Pick random sequence from this class
                    base_seq = sequences[np.random.randint(0, len(sequences))]
                    aug_seq = self.augmentator.augment_sequence(base_seq)
                    augmented_sequences.append(aug_seq)
                    augmented_labels.append(label)
        
        return torch.FloatTensor(np.array(augmented_sequences)), torch.LongTensor(augmented_labels)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

# --- TRAINING WITH ENHANCED FEATURES ---
def train_enhanced_model(model, train_loader, val_loader, class_weights, num_epochs=120):
    """Train the enhanced model with advanced techniques."""
    weights_tensor = torch.FloatTensor(class_weights).to(device)
    # Reduce label smoothing for sharper decision boundaries
    criterion = nn.CrossEntropyLoss(weight=weights_tensor, label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.005)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.003, epochs=num_epochs, steps_per_epoch=len(train_loader)
    )
    
    best_val_acc = 0
    patience_counter = 0
    max_patience = 10
    best_model_state = model.state_dict()
    
    print("\n--- Starting Enhanced Training ---")
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        
        for sequences, labels in train_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
        
        # Validation
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
    """Main training function with enhanced preprocessing and model."""
    print(f"🚀 Using device: {device}")
    
    data_path = 'gesture_data/training_data.json'
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Please run the recorder first.")
        return
    
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data['sequences'])} sequences.")
    
    # Initialize enhanced preprocessor
    preprocessor = LandmarkPreprocessor()
    
    # Process features with advanced preprocessing
    print("Applying advanced preprocessing...")
    start_time = time.time()
    sequences = []
    labels = []
    for i, seq in enumerate(data['sequences']):
        if (i + 1) % 200 == 0:
            print(f"  ... processing sequence {i+1}/{len(data['sequences'])}")
        features = preprocessor.extract_advanced_features(seq)
        if features is not None:
            sequences.append(features)
            labels.append(data['labels'][i])
    
    print(f"Preprocessing completed in {time.time() - start_time:.2f} seconds")
    
    X = np.array(sequences)
    
    if len(X) == 0:
        print("Error: No valid sequences found. Please re-record.")
        return
    
    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)
    
    # Calculate class weights
    label_counts = Counter(y)
    num_classes = len(label_counts)
    total_samples = len(y)
    class_weights = [total_samples / (num_classes * label_counts.get(i, 1)) for i in range(num_classes)]
    
    print(f"\n--- Enhanced Data Summary ---")
    print(f"Processed {len(X)} valid samples")
    print(f"Feature dimension: {X.shape[2]} (includes multi-temporal and geometric features)")
    print("Class distribution:")
    for i, class_name in enumerate(label_encoder.classes_):
        print(f"  {class_name:<20}: {label_counts[i]:>4} samples (weight: {class_weights[i]:.2f})")
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    num_train_samples, seq_len, num_features = X_train.shape
    X_train_reshaped = X_train.reshape(-1, num_features)
    scaler.fit(X_train_reshaped)
    
    X_train_scaled = scaler.transform(X_train_reshaped).reshape(num_train_samples, seq_len, num_features)
    
    num_val_samples = X_val.shape[0]
    X_val_reshaped = X_val.reshape(-1, num_features)
    X_val_scaled = scaler.transform(X_val_reshaped).reshape(num_val_samples, seq_len, num_features)
    
    # Create datasets with augmentation
    train_dataset = EnhancedGestureDataset(X_train_scaled, y_train, augment=True, augment_factor=3)
    val_dataset = EnhancedGestureDataset(X_val_scaled, y_val, augment=False)
    
    # Set num_workers=0 on macOS to avoid multiprocessing issues with some PyTorch versions
    num_workers = 0 if platform.system() == 'Darwin' else 2
    print(f"Using {num_workers} workers for DataLoader.")
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=num_workers)
    
    print(f"Training set expanded to {len(train_dataset)} samples with augmentation")
    
    # Initialize enhanced model
    model = EnhancedGestureClassifier(
        input_size=num_features,
        num_classes=num_classes
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    model, best_acc = train_enhanced_model(model, train_loader, val_loader, class_weights)
    
    print(f"\n✓ Enhanced training complete! Best accuracy: {best_acc:.2f}%")
    
    # Convert to TorchScript for optimized inference
    model.eval()
    example_input = torch.randn(1, seq_len, num_features).to(device)
    traced_model = torch.jit.trace(model, example_input)
    
    # Save all artifacts
    print("\nSaving enhanced model and artifacts...")
    os.makedirs('gesture_data', exist_ok=True)
    
    # Save PyTorch model
    model_save_path = 'gesture_data/enhanced_gesture_classifier.pth'
    torch.save({
        'model_state': model.state_dict(),
        'input_size': num_features,
        'num_classes': num_classes,
        'sequence_length': seq_len,
        'best_accuracy': best_acc,
        'model_type': 'EnhancedTCN'
    }, model_save_path)
    
    # Save TorchScript model for fast inference
    traced_model.save('gesture_data/enhanced_gesture_classifier_traced.pt')
    
    # Save preprocessor and encoders
    joblib.dump(scaler, 'gesture_data/enhanced_gesture_scaler.pkl')
    joblib.dump(label_encoder, 'gesture_data/enhanced_gesture_label_encoder.pkl')
    joblib.dump(preprocessor, 'gesture_data/landmark_preprocessor.pkl')
    
    print(f"✓ Enhanced model saved to {model_save_path}")
    print("✓ TorchScript model saved for optimized inference")
    
    # Final evaluation
    print("\n--- Final Evaluation ---")
    model.eval()
    y_pred = []
    with torch.no_grad():
        val_data = torch.FloatTensor(X_val_scaled).to(device)
        outputs = model(val_data)
        _, predicted = torch.max(outputs, 1)
        y_pred = predicted.cpu().numpy()
    
    report = classification_report(y_val, y_pred, target_names=label_encoder.classes_, zero_division=0)
    print(report)
    
    # Save confusion matrix
    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=label_encoder.classes_, 
                yticklabels=label_encoder.classes_, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Enhanced Model Confusion Matrix')
    plt.tight_layout()
    plt.savefig('gesture_data/enhanced_confusion_matrix.png', dpi=300)
    print("✓ Enhanced confusion matrix saved")

if __name__ == "__main__":
    main()
