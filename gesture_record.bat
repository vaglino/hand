@echo off
call venv\Scripts\activate.bat
echo.
echo === Gesture Recording ===
echo.
echo Instructions:
echo - Press 1-5 to record gesture states
echo - Press U/D/I/O to record continuous motion
echo - Recording stops automatically
echo - Press S to save, Q to quit
echo.
python gesture_recorder.py
deactivate