#!/bin/bash
# gesture_train.sh - ML pipeline with LightweightCNN and optimized inference

source venv/bin/activate
echo ""
echo "===================================================================="
echo "   GESTURE TRAINING - LightweightCNN"
echo "===================================================================="
echo ""
echo "FEATURES:"
echo "   - Procrustes alignment for rotation/scale invariance"
echo "   - Multi-temporal feature extraction (1, 2, 3-frame derivatives)"
echo "   - Butterworth filtering to reduce landmark jitter"
echo "   - Finger angle geometry features"
echo "   - LightweightCNN architecture (~100K params)"
echo "   - Advanced data augmentation (time-warping, noise, scaling)"
echo "   - Class balancing with minority oversampling"
echo "   - TorchScript compilation for faster inference"
echo ""
echo "PERFORMANCE:"
echo "   - Inference: ~0.25ms per frame"
echo "   - Training time: 5-10 minutes with data augmentation"
echo ""
echo "Starting training pipeline..."
echo ""

python3 train_lightweight_cnn.py

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Training failed! Check error messages above."
    echo ""
    echo "TROUBLESHOOTING:"
    echo "• Ensure you have recorded gesture data first (./gesture_record.sh)"
    echo "• Verify all dependencies are installed: pip install -r requirements.txt"
    echo ""
    read -p "Press Enter to continue..."
    exit 1
fi

echo ""
echo "===================================================================="
echo "   TRAINING COMPLETE"
echo "===================================================================="
echo ""
echo "Ready to run gesture control:"
echo "   ./gesture_control.sh"
echo ""
echo "Files created:"
echo "   - enhanced_gesture_classifier.pth (PyTorch model)"
echo "   - enhanced_gesture_classifier_traced.pt (TorchScript)"
echo "   - enhanced_gesture_scaler.pkl (feature scaler)"
echo "   - landmark_preprocessor.pkl (advanced preprocessor)"
echo "   - lightweight_cnn_confusion_matrix.png"
echo ""

read -p "Press Enter to continue..."
deactivate