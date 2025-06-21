:: calibrate.bat - Run calibration tool
@echo off
call venv\Scripts\activate.bat
echo.
echo === Gesture Calibration Tool ===
echo.
echo This tool will personalize gesture control for your hand.
echo.
echo Steps:
echo 1. Calibrate hand size
echo 2. Calibrate scroll sensitivity
echo 3. Calibrate zoom sensitivity
echo 4. Test your settings
echo.
python calibration_tool.py
deactivate

:: diagnose.bat - Run diagnostic tool
@echo off
call venv\Scripts\activate.bat
echo.
echo === Gesture System Diagnostics ===
echo.
echo Real-time performance monitoring and analysis.
echo.
echo Features:
echo - FPS and latency tracking
echo - CPU/Memory usage
echo - Hand tracking quality
echo - Motion vector visualization
echo.
python diagnostic_tool.py
deactivate

:: physics_demo.bat - Interactive physics engine demo
@echo off
call venv\Scripts\activate.bat
echo.
echo === Physics Engine Demo ===
echo.
echo Interactive demonstration of the trackpad physics simulation.
echo.
python physics_demo.py
deactivate

:: quick_test.bat - Quick system test
@echo off
echo === Quick System Test ===
echo.

REM Check Python
echo Checking Python...
python --version
if errorlevel 1 (
    echo ERROR: Python not found!
    goto :error
)

REM Check virtual environment
echo.
echo Checking virtual environment...
if exist "venv\Scripts\activate.bat" (
    echo ✓ Virtual environment found
) else (
    echo ERROR: Virtual environment not found! Run setup.bat first.
    goto :error
)

REM Activate venv
call venv\Scripts\activate.bat

REM Check packages
echo.
echo Checking required packages...
python -c "import cv2; print('✓ OpenCV installed')"
python -c "import mediapipe; print('✓ MediaPipe installed')"
python -c "import torch; print('✓ PyTorch installed')"
python -c "import numpy; print('✓ NumPy installed')"
python -c "import sklearn; print('✓ Scikit-learn installed')"
python -c "import pyautogui; print('✓ PyAutoGUI installed')"

REM Check model files
echo.
echo Checking model files...
if exist "hand_landmarker.task" (
    echo ✓ Hand tracking model found
) else (
    echo ✗ Hand tracking model missing - will download during setup
)

if exist "gesture_data\gesture_classifier.pth" (
    echo ✓ Gesture classifier found
) else if exist "gesture_data\gesture_classifier.pkl" (
    echo ✓ Gesture classifier found
) else (
    echo ✗ Gesture classifier missing - run training first
)

REM Test imports
echo.
echo Testing system imports...
python -c "from physics_engine import TrackpadPhysicsEngine; print('✓ Physics engine OK')"

echo.
echo === Test Complete ===
echo.
pause
deactivate
exit /b 0

:error
echo.
echo Test failed! Please fix the errors above.
pause
exit /b 1

:: clean.bat - Clean temporary files
@echo off
echo Cleaning temporary files...
echo.

REM Remove Python cache
if exist "__pycache__" rmdir /s /q __pycache__
if exist ".pytest_cache" rmdir /s /q .pytest_cache

REM Remove diagnostic sessions (keep models)
if exist "gesture_data\diagnostic_session_*.json" del /q "gesture_data\diagnostic_session_*.json"

echo ✓ Cleanup complete
echo.
pause