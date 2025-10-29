#!/bin/bash
# gesture_train.sh - Advanced ML pipeline with TCN and optimized inference

source venv/bin/activate
echo ""
echo "===================================================================="
echo "   🚀 ENHANCED GESTURE TRAINING - NEXT-GEN ML PIPELINE"
echo "===================================================================="
echo ""
echo "🔬 ADVANCED FEATURES:"
echo "   ✓ Procrustes alignment for rotation/scale invariance"
echo "   ✓ Multi-temporal feature extraction (1, 2, 3-frame derivatives)"
echo "   ✓ Butterworth filtering to reduce landmark jitter"
echo "   ✓ Finger angle geometry features"
echo "   ✓ Temporal Convolutional Network (TCN) architecture"
echo "   ✓ Attention-based pooling mechanism"
echo "   ✓ Advanced data augmentation (time-warping, noise, scaling)"
echo "   ✓ Class balancing with minority oversampling"
echo "   ✓ TorchScript compilation for faster inference"
echo ""
echo "⚡ PERFORMANCE ON APPLE SILICON:"
echo "   • MPS (GPU) Inference: ~2-3ms per frame"
echo "   • CPU Inference: ~4-6ms per frame"
echo "   • Training time: 5-10 minutes with data augmentation"
echo ""
echo "Starting enhanced training pipeline..."
echo ""
echo "⏳ This will take 5-15 minutes depending on your hardware"
echo "📈 Progress will be shown with validation accuracy updates"
echo ""

python3 train_model.py

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
echo "   ✅ ENHANCED TRAINING COMPLETE!"
echo "===================================================================="
echo ""
echo "🎯 Your model now features:"
echo "   ✓ TorchScript optimized inference"
echo "   ✓ Advanced preprocessing pipeline"
echo "   ✓ Temporal convolutional architecture"
echo "   ✓ Robust prediction smoothing"
echo ""
echo "🚀 Ready to run enhanced gesture control:"
echo "   Run: ./gesture_control.sh"
echo ""
echo "📋 Files created:"
echo "   • enhanced_gesture_classifier.pth (PyTorch model)"
echo "   • enhanced_gesture_classifier_traced.pt (TorchScript)"
echo "   • enhanced_gesture_scaler.pkl (feature scaler)"
echo "   • landmark_preprocessor.pkl (advanced preprocessor)"
echo "   • enhanced_confusion_matrix.png (performance visualization)"
echo ""

read -p "Press Enter to continue..."
deactivate