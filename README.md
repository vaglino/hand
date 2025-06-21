# Gesture Control System

Control your computer with hand gestures using deep learning.

## Quick Start

### 1. Setup (one time)
```bash
setup.bat
```
This will:
- Create a virtual environment
- Install all dependencies
- Download the hand tracking model

### 2. Record Training Data
```bash
gesture_record.bat
```
- Press 1-5 to record different gestures
- Collect 10-20 samples per gesture
- Press S to save

### 3. Train Model
```bash
gesture_train.bat
```
Trains an LSTM neural network on your gesture data.

### 4. Use Gesture Control
```bash
gesture_control.bat
```

## Gestures

| Gesture | Action |
|---------|--------|
| Swipe hand UP | Scroll up |
| Swipe hand DOWN | Scroll down |
| Spread fingers (open hand) | Zoom in |
| Close fingers (make fist) | Zoom out |
| Keep hand still | No action |

## Files

- `gesture_recorder.py` - Records gesture sequences
- `train_model.py` - Trains LSTM model
- `gesture_control.py` - Real-time gesture control
- `requirements.txt` - Python dependencies
- `setup.bat` - One-time setup
- `gesture_record.bat` - Record gestures
- `gesture_train.bat` - Train model
- `gesture_control.bat` - Run control

## Requirements

- Windows 10/11
- Python 3.8+
- Webcam
- NVIDIA GPU (optional, for faster training)

## Tips

- Record gestures in good lighting
- Keep hand fully visible in camera
- Make gestures distinct and clear
- Retrain if accuracy is low