import os
import torch
import numpy as np
from typing import Optional

from hand.models.tcn import EnhancedGestureClassifier
from hand.models.lightweight_cnn import LightweightCNN

try:
    import onnxruntime as ort  # noqa: F401

    ONNX_AVAILABLE = True
except Exception:
    ONNX_AVAILABLE = False


class OptimizedInferenceEngine:
    """Handles model loading and inference with multiple backend options."""

    def __init__(self, model_dir: str = "gesture_data"):
        self.model_dir = model_dir
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            # Check for MPS (Apple Silicon GPU)
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.model: Optional[torch.jit.ScriptModule | torch.nn.Module] = None
        self.backend: Optional[str] = None
        self.input_size: Optional[int] = None
        self.sequence_length: Optional[int] = None

        print(f"🚀 Initializing inference engine on {self.device}")
        self._load_optimal_model()

    def _load_optimal_model(self):
        """Load the best available model format for optimal inference."""
        torchscript_path = os.path.join(self.model_dir, "enhanced_gesture_classifier_traced.pt")
        pytorch_path = os.path.join(self.model_dir, "enhanced_gesture_classifier.pth")

        # Try TorchScript first (fastest)
        if os.path.exists(torchscript_path):
            try:
                print("📦 Loading TorchScript model for optimized inference...")
                self.model = torch.jit.load(torchscript_path, map_location=self.device)
                self.model.eval()
                self.backend = "torchscript"

                # Load metadata from PyTorch checkpoint
                if os.path.exists(pytorch_path):
                    checkpoint = torch.load(pytorch_path, map_location="cpu")
                    self.input_size = checkpoint["input_size"]
                    self.sequence_length = checkpoint["sequence_length"]

                print("✅ TorchScript model loaded successfully")
                return
            except Exception as e:
                print(f"⚠️ Failed to load TorchScript model: {e}")
                print("📦 Falling back to PyTorch model...")

        # Fallback to PyTorch model
        if os.path.exists(pytorch_path):
            checkpoint = torch.load(pytorch_path, map_location=self.device)
            model_type = checkpoint.get("model_type", "tcn")  # Default to TCN for backwards compatibility

            if model_type == "lightweight_cnn":
                print("📦 Loading PyTorch LightweightCNN model...")
                self.model = LightweightCNN(
                    input_size=checkpoint["input_size"], num_classes=checkpoint["num_classes"]
                ).to(self.device)
            else:
                print("📦 Loading PyTorch TCN model...")
                self.model = EnhancedGestureClassifier(
                    input_size=checkpoint["input_size"], num_classes=checkpoint["num_classes"]
                ).to(self.device)

            self.model.load_state_dict(checkpoint["model_state"])
            self.model.eval()
            self.backend = "pytorch"

            self.input_size = checkpoint["input_size"]
            self.sequence_length = checkpoint["sequence_length"]

            print(f"✅ PyTorch {model_type} model loaded successfully")
            return

        raise FileNotFoundError("❌ No trained model found. Please run enhanced training first.")

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Fast inference with the loaded model."""
        if self.model is None:
            raise RuntimeError("No model loaded")

        # Convert to tensor and add batch dimension
        input_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_tensor)
            # Convert to probabilities
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

        return probs

