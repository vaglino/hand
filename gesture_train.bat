:: gesture_train.bat - Multi-model training system
@echo off
call venv\Scripts\activate.bat
echo.
echo === Multi-Model Training System ===
echo.
echo Training models:
echo 1. Lightweight gesture classifier (CNN/Random Forest)
echo 2. Motion intensity predictor
echo 3. User adaptation model
echo.
echo This may take a few minutes...
echo.
python train_model.py
echo.
echo Training complete! Check accuracy results above.
echo.
pause
deactivate