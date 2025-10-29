import numpy as np


class PredictionSmoother:
    """Smooths predictions using exponential moving average to reduce flickering."""

    def __init__(self, num_classes: int, alpha: float = 0.3, confidence_threshold: float = 0.8):
        self.num_classes = num_classes
        self.alpha = alpha  # EMA smoothing factor
        self.confidence_threshold = confidence_threshold
        self.ema_probs = np.ones(num_classes) / num_classes  # Start with uniform distribution
        self.last_confident_prediction = None
        self.stable_frames = 0
        self.min_stable_frames = 3

    def update(self, raw_probs: np.ndarray):
        """Update EMA and return smoothed prediction."""
        # Update exponential moving average
        self.ema_probs = self.alpha * raw_probs + (1 - self.alpha) * self.ema_probs

        # Get prediction from smoothed probabilities
        pred_idx = int(np.argmax(self.ema_probs))
        confidence = float(self.ema_probs[pred_idx])

        # Stability check: only change prediction if confident and stable
        if confidence > self.confidence_threshold:
            if self.last_confident_prediction == pred_idx:
                self.stable_frames += 1
            else:
                self.stable_frames = 0
                self.last_confident_prediction = pred_idx
        else:
            self.stable_frames = 0

        # Return prediction (use last stable if current is not stable enough)
        if self.stable_frames >= self.min_stable_frames or self.last_confident_prediction is None:
            final_prediction = pred_idx
        else:
            final_prediction = (
                self.last_confident_prediction if self.last_confident_prediction is not None else pred_idx
            )

        return final_prediction, confidence, self.ema_probs

    def reset(self):
        """Reset smoother state."""
        self.ema_probs = np.ones(self.num_classes) / self.num_classes
        self.last_confident_prediction = None
        self.stable_frames = 0

