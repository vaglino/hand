@echo off
call venv\Scripts\activate.bat
echo.
echo === Gesture Control (Trackpad Mode) ===
echo.
echo Controls:
echo - Swipe UP: Scroll up
echo - Swipe DOWN: Scroll down
echo - Spread fingers: Zoom in (Ctrl++)
echo - Close fingers: Zoom out (Ctrl+-)
echo - Press Q to quit
echo.
python gesture_control.py
deactivate