:: setup.bat - Automated setup for gesture control with GPU support
@echo off
setlocal

echo Setting up Revolutionary Gesture Control System...
echo.

:: ------------------------------------------------------------------
:: 1. Check for Python
:: ------------------------------------------------------------------
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in your PATH.
    echo Please install Python 3.8+ and add it to your PATH.
    pause
    exit /b 1
)

:: ------------------------------------------------------------------
:: 2. Create and Activate Virtual Environment
:: ------------------------------------------------------------------
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)
call venv\Scripts\activate.bat
echo Virtual environment is active.
echo.

:: ------------------------------------------------------------------
:: 3. Install PyTorch with GPU Support (for CUDA 12.1)
:: ------------------------------------------------------------------
echo === Installing PyTorch with NVIDIA GPU Support ===
echo This may take a few minutes...
echo.
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

:: ------------------------------------------------------------------
:: 4. Verify PyTorch and CUDA
:: ------------------------------------------------------------------
echo.
echo Verifying PyTorch and CUDA installation...
python -c "import torch; success = torch.cuda.is_available(); print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {success}'); exit(0) if success else exit(1)"
if errorlevel 1 (
    echo.
    echo ***************************************************************
    echo ERROR: PyTorch was installed, but it cannot detect your GPU.
    echo Please check your NVIDIA driver installation.
    echo The program will continue with CPU-only mode.
    echo ***************************************************************
    echo.
) else (
    echo.
    echo SUCCESS: PyTorch has detected your NVIDIA GPU!
    echo Training will be accelerated.
    echo.
)
pause

:: ------------------------------------------------------------------
:: 5. Install other dependencies
:: ------------------------------------------------------------------
echo === Installing other dependencies from requirements.txt ===
echo.
pip install -r requirements.txt
echo.

:: ------------------------------------------------------------------
:: 6. Download Hand Tracking Model
:: ------------------------------------------------------------------
if not exist "hand_landmarker.task" (
    echo Downloading hand tracking model...
    curl -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
) else (
    echo Hand tracking model already exists.
)
echo.

:: Create data directory
if not exist "gesture_data" mkdir gesture_data

:: ------------------------------------------------------------------
:: 7. Finish
:: ------------------------------------------------------------------
echo =========================================
echo  ✓ Setup complete!
echo =========================================
echo.
echo Your system is now set up.
echo.
echo To get started:
echo 1. Run: gesture_record.bat
echo 2. Run: gesture_train.bat (This will now use your GPU)
echo 3. Run: gesture_control.bat
echo 4. Run: diagnostic_tool.py (To see live GPU stats)
echo.
pause