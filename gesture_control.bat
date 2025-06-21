:: gesture_control.bat - Revolutionary physics-based control
@echo off
call venv\Scripts\activate.bat
echo.
echo === Revolutionary Gesture Control ===
echo.
echo Experience trackpad-like control with your hands!
echo.
echo Controls:
echo - Swipe gestures for smooth scrolling
echo - Pinch/spread for natural zooming
echo - Physics-based momentum and deceleration
echo - [R] Reset physics
echo - [+/-] Adjust sensitivity
echo - [Q] Quit
echo.
echo Starting in 3 seconds...
timeout /t 3 /nobreak > nul
echo.
python gesture_control.py
deactivate