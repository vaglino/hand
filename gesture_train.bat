@echo off
call venv\Scripts\activate.bat
echo.
echo === Training Gesture Models ===
echo.
python train_model.py
echo.
pause
deactivate