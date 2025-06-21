:: test_physics.bat - Test physics engine independently
@echo off
call venv\Scripts\activate.bat
echo.
echo === Physics Engine Test ===
echo.
python -c "from physics_engine import TrackpadPhysicsEngine; engine = TrackpadPhysicsEngine(); print('Physics engine loaded successfully!')"
echo.
pause
deactivate