@echo off
echo Gesture Control System
echo =====================
echo.

call venv\Scripts\activate.bat

:menu
echo What would you like to do?
echo.
echo 1. Record gestures
echo 2. Train model
echo 3. Run gesture control
echo 4. Complete workflow (1 → 2 → 3)
echo Q. Quit
echo.
set /p choice="Enter choice: "

if /i "%choice%"=="1" goto record
if /i "%choice%"=="2" goto train
if /i "%choice%"=="3" goto control
if /i "%choice%"=="4" goto workflow
if /i "%choice%"=="q" goto end
echo Invalid choice!
goto menu

:record
echo.
python gesture_recorder.py
echo.
pause
goto menu

:train
echo.
python train_model.py
echo.
pause
goto menu

:control
echo.
python gesture_control.py
echo.
pause
goto menu

:workflow
echo.
echo Step 1: Recording gestures...
python gesture_recorder.py
echo.
echo Step 2: Training model...
python train_model.py
echo.
echo Step 3: Running control...
python gesture_control.py
goto menu

:end
deactivate
exit