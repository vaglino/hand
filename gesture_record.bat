:: gesture_record.bat - Enhanced recording with new features
@echo off
call venv\Scripts\activate.bat
echo.
echo === Enhanced Gesture Recorder ===
echo.
echo This recorder captures continuous gesture sequences with transitions.
echo.
echo RECORDING MODES:
echo - Press a key to start recording for that gesture
echo.
echo GESTURES:
echo [1] Scroll Up        [2] Scroll Down
echo [3] Zoom In          [4] Zoom Out
echo [5] Neutral
echo [6] Maximize Window  [7] Go Back
echo.
echo TIPS:
echo - Follow the visual guides that appear.
echo - Perform each gesture about 10-15 times.
echo - Press 'S' to SAVE and APPEND your new gestures to the dataset.
echo.
python gesture_recorder.py
deactivate