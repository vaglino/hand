"""
Experimental training script to compare different model architectures.
Runs all models on the same data and compares accuracy + inference speed.

Usage:
    python experiments/train_experimental.py

Results saved to experiments/results/
"""

import json
import os
import sys
import time
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hand.features.landmarks import LandmarkPreprocessor
from experiments.models import LightweightCNN, MLPClassifier, MiniTransformer, FastTCN
from experiments.models.lightweight_cnn import LightweightCNNv2
from hand.models.lightweight_cnn import LightweightCNNv3
from experiments.models.mlp_classifier import TemporalMLP, StatisticalMLP
from experiments.models.mini_transformer import TinyTransformer
from experiments.models.fast_tcn import FastTCNv2, DepthwiseTCN

# Also import original TCN for comparison
from hand.models.tcn import EnhancedGestureClassifier as OriginalTCN


# Device setup
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')


class GestureDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def load_and_preprocess_data(data_path: str) -> Tuple[np.ndarray, np.ndarray, LabelEncoder, int]:
    """Load and preprocess gesture data."""
    with open(data_path, 'r') as f:
        data = json.load(f)

    print(f"Loaded {len(data['sequences'])} sequences")

    preprocessor = LandmarkPreprocessor()
    sequences = []
    labels = []

    for i, seq in enumerate(data['sequences']):
        features = preprocessor.extract_advanced_features(seq)
        if features is not None:
            sequences.append(features)
            labels.append(data['labels'][i])

    X = np.array(sequences)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)

    return X, y, label_encoder, X.shape[1]  # seq_len


def mixup_data(x, y, alpha=0.2):
    """Apply mixup augmentation."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute mixup loss."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                num_epochs: int = 120, patience: int = 15, use_mixup: bool = False) -> Tuple[nn.Module, float, List[float]]:
    """Train a model and return best accuracy."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0008, weight_decay=0.02)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_acc = 0
    best_state = None
    patience_counter = 0
    val_accs = []

    for epoch in range(num_epochs):
        # Training
        model.train()
        for sequences, labels in train_loader:
            sequences, labels = sequences.to(device), labels.to(device)

            # Apply mixup
            if use_mixup:
                sequences, labels_a, labels_b, lam = mixup_data(sequences, labels, alpha=0.2)

            optimizer.zero_grad()
            outputs = model(sequences)

            if use_mixup:
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            else:
                loss = criterion(outputs, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences, labels = sequences.to(device), labels.to(device)
                outputs = model(sequences)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = 100 * correct / total
        val_accs.append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    model.to(device)

    return model, best_acc, val_accs


def measure_inference_speed(model: nn.Module, input_shape: Tuple[int, ...], num_runs: int = 100) -> float:
    """Measure average inference time in milliseconds."""
    model.eval()
    dummy_input = torch.randn(1, *input_shape).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)

    # Measure
    if device.type == 'cuda':
        torch.cuda.synchronize()

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    elapsed = (time.perf_counter() - start) / num_runs * 1000  # ms
    return elapsed


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    print(f"Using device: {device}")

    # Load data
    data_path = 'gesture_data/training_data.json'
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found")
        return

    X, y, label_encoder, seq_len = load_and_preprocess_data(data_path)
    num_features = X.shape[2]
    num_classes = len(label_encoder.classes_)

    print(f"\nData shape: {X.shape}")
    print(f"Classes: {list(label_encoder.classes_)}")

    # Split and scale
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_flat = X_train.reshape(-1, num_features)
    scaler.fit(X_train_flat)
    X_train_scaled = scaler.transform(X_train_flat).reshape(X_train.shape)
    X_val_scaled = scaler.transform(X_val.reshape(-1, num_features)).reshape(X_val.shape)

    train_dataset = GestureDataset(X_train_scaled, y_train)
    val_dataset = GestureDataset(X_val_scaled, y_val)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)

    # Define models to test
    models_to_test = {
        # 'LightweightCNN': lambda: LightweightCNN(num_features, num_classes),
        'LightweightCNNv3': lambda: LightweightCNNv3(num_features, num_classes),
    }

    results = {}

    print("\n" + "=" * 70)
    print("TRAINING AND EVALUATING MODELS")
    print("=" * 70)

    for name, model_fn in models_to_test.items():
        print(f"\n--- {name} ---")

        model = model_fn().to(device)
        num_params = count_parameters(model)
        print(f"Parameters: {num_params:,}")

        # Train
        train_start = time.time()
        model, best_acc, val_accs = train_model(model, train_loader, val_loader)
        train_time = time.time() - train_start

        # Measure inference speed
        inference_time = measure_inference_speed(model, (seq_len, num_features))

        print(f"Best accuracy: {best_acc:.2f}%")
        print(f"Training time: {train_time:.1f}s")
        print(f"Inference time: {inference_time:.3f}ms")

        results[name] = {
            'accuracy': best_acc,
            'parameters': num_params,
            'inference_ms': inference_time,
            'train_time_s': train_time,
            'val_history': val_accs
        }

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Model':<20} {'Accuracy':>10} {'Params':>12} {'Inference':>12}")
    print("-" * 70)

    # Sort by accuracy
    sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    for name, r in sorted_results:
        print(f"{name:<20} {r['accuracy']:>9.2f}% {r['parameters']:>11,} {r['inference_ms']:>10.3f}ms")

    # Find best accuracy and fastest
    best_acc_model = max(results.items(), key=lambda x: x[1]['accuracy'])
    fastest_model = min(results.items(), key=lambda x: x[1]['inference_ms'])

    print(f"\nBest accuracy: {best_acc_model[0]} ({best_acc_model[1]['accuracy']:.2f}%)")
    print(f"Fastest inference: {fastest_model[0]} ({fastest_model[1]['inference_ms']:.3f}ms)")

    # Save results
    os.makedirs('experiments/results', exist_ok=True)

    # Save as JSON
    json_results = {k: {kk: vv for kk, vv in v.items() if kk != 'val_history'} for k, v in results.items()}
    with open('experiments/results/model_comparison.json', 'w') as f:
        json.dump(json_results, f, indent=2)

    # Plot comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    names = list(results.keys())
    accs = [results[n]['accuracy'] for n in names]
    params = [results[n]['parameters'] for n in names]
    times = [results[n]['inference_ms'] for n in names]

    # Accuracy bar chart
    axes[0].barh(names, accs, color='steelblue')
    axes[0].set_xlabel('Accuracy (%)')
    axes[0].set_title('Model Accuracy')

    # Parameters bar chart
    axes[1].barh(names, params, color='coral')
    axes[1].set_xlabel('Parameters')
    axes[1].set_title('Model Size')

    # Inference time bar chart
    axes[2].barh(names, times, color='seagreen')
    axes[2].set_xlabel('Inference Time (ms)')
    axes[2].set_title('Inference Speed')

    plt.tight_layout()
    plt.savefig('experiments/results/model_comparison.png', dpi=150)
    print("\nResults saved to experiments/results/")


if __name__ == '__main__':
    main()
