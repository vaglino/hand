:: gesture_record.bat - Enhanced recording with new features
@echo off
call venv\Scripts\activate.bat
echo.
echo === Enhanced Gesture Recorder ===
echo.
echo This recorder captures continuous gesture sequences with transitions.
echo.
echo RECORDING MODES:
echo - Press 1-7: Guided recording (with timing cues)
echo - Press Shift+1-7: Freestyle recording (your own pace)
echo.
echo GESTURES:
echo [1] Scroll Up    [2] Scroll Down
echo [3] Scroll Left  [4] Scroll Right  
echo [5] Zoom In      [6] Zoom Out
echo [7] Neutral
echo.
echo TIPS:
echo - Follow the visual guides in guided mode
echo - Perform each gesture 5 times with returns to neutral
echo - The system will learn the transition patterns
echo.
python gesture_recorder.py
deactivate