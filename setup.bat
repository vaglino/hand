@echo off
echo Setting up Gesture Control...
echo.

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo Installing requirements...
pip install -r requirements.txt

REM Download model if not exists
if not exist "hand_landmarker.task" (
    echo.
    echo Downloading hand tracking model...
    curl -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
)

REM Create data directory
if not exist "gesture_data" mkdir gesture_data

echo.
echo ✓ Setup complete!
echo.
echo To use:
echo 1. Run: gesture_record.bat
echo 2. Run: gesture_train.bat  
echo 3. Run: gesture_control.bat
echo.
pause