@echo off
call venv\Scripts\activate.bat
echo.
echo === Gesture Recording ===
echo.
echo Instructions:
echo - Press 1-5 to record gestures
echo - Perform gesture continuously
echo - Recording stops automatically
echo - Press S to save, Q to quit
echo.
python gesture_recorder.py
deactivate