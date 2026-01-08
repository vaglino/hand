#!/bin/bash
# setup.sh - Automated setup for gesture control on macOS/Linux

set -e # Exit immediately if a command exits with a non-zero status.

echo "Setting up Revolutionary Gesture Control System..."
echo ""

# ------------------------------------------------------------------
# 1. Check for Python
# ------------------------------------------------------------------
if ! command -v python3 &> /dev/null
then
    echo "ERROR: python3 could not be found."
    echo "Please install Python 3.8+."
    exit 1
fi

# ------------------------------------------------------------------
# 2. Create and Activate Virtual Environment
# ------------------------------------------------------------------
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    $HOME/.pyenv/versions/3.11.9/bin/python3.11 -m venv venv
fi
source venv/bin/activate
echo "Virtual environment is active."
echo ""

# ------------------------------------------------------------------
# 3. Install PyTorch with Apple Silicon (M-series) GPU Support
# ------------------------------------------------------------------
echo "=== Installing PyTorch with Apple Silicon (MPS) Support ==="
echo "This may take a few minutes..."
echo ""
pip install torch torchvision torchaudio

# ------------------------------------------------------------------
# 4. Verify PyTorch and MPS
# ------------------------------------------------------------------
echo ""
echo "Verifying PyTorch and MPS (Metal Performance Shaders) installation..."
python3 -c "import torch; success = torch.backends.mps.is_available(); print(f'PyTorch version: {torch.__version__}'); print(f'MPS available: {success}'); exit(0) if success else exit(1)"
if [ $? -ne 0 ]; then
    echo ""
    echo "***************************************************************"
    echo "WARNING: PyTorch was installed, but MPS is not available."
    echo "The program will use the CPU, which will be slower."
    echo "Ensure you are on an M-series Mac with macOS 12.3+."
    echo "***************************************************************"
    echo ""
else
    echo ""
    echo "SUCCESS: PyTorch has detected your Apple Silicon GPU (MPS)!"
    echo "Training will be accelerated."
    echo ""
fi
read -p "Press Enter to continue..."

# ------------------------------------------------------------------
# 5. Install other dependencies
# ------------------------------------------------------------------
echo "=== Installing other dependencies from requirements.txt ==="
echo ""
pip install -r requirements.txt
echo ""

# ------------------------------------------------------------------
# 6. Download Hand Tracking Model
# ------------------------------------------------------------------
if [ ! -f "hand_landmarker.task" ]; then
    echo "Downloading hand tracking model (256MB)..."
    curl -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
else
    echo "Hand tracking model already exists."
fi
echo ""

# Create data directory
mkdir -p gesture_data

# ------------------------------------------------------------------
# 7. Finish
# ------------------------------------------------------------------
echo "========================================="
echo "  ✓ Setup complete!"
echo "========================================="
echo ""
echo "Your system is now set up."
echo ""
echo "IMPORTANT: Make the scripts executable by running:"
echo "chmod +x gesture_*.sh"
echo ""
echo "To get started:"
echo "1. Run: ./gesture_record.sh"
echo "2. Run: ./gesture_train.sh (This will now use your Mac's GPU)"
echo "3. Run: ./gesture_control.sh"
echo ""
read -p "Press Enter to exit..."