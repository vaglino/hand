:: gesture_record.bat - Enhanced recording with new features
@echo off
call venv\Scripts\activate.bat
echo.
echo === Enhanced Gesture Recorder ===
echo.
echo NEW FEATURES:
echo - Dual recording modes (Gesture + Motion)
echo - Real-time motion intensity visualization
echo - Continuous motion tracking
echo - Gesture transition detection
echo.
echo Instructions:
echo - [G] Gesture mode - Quick classification samples
echo - [M] Motion mode - Continuous motion recording
echo - [1-7] Record gestures/motion types
echo - [I] Change intensity setting
echo - [S] Save all data
echo - [Q] Quit
echo.
python gesture_recorder.py
deactivate