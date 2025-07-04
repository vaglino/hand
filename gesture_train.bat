:: gesture_train.bat - Multi-model training system
@echo off
call venv\Scripts\activate.bat
echo.
echo === Enhanced Model Training ===
echo.
echo Training transition-aware model that understands:
echo - Gesture transitions and return movements
echo - Motion intention detection
echo - Temporal context from previous gestures
echo.
echo This may take 5-10 minutes...
echo.
python train_model.py
echo.
echo Training complete! 
echo.
echo The model now understands transitions and won't misclassify
echo return movements as opposite gestures.
echo.
pause
deactivate