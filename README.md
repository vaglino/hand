# Hand Gesture Computer Control

Control your computer with hand gestures using deep learning.

A physics-based system to control your computer with natural hand gestures. This project uses a ML pipeline to provide intuitive control, mimicking the feel of a trackpad but with video.

![Gestures](gestures_schematic.png)

### Key Features
- **Trackpad-like Physics:** Actions like scrolling and zooming have momentum and friction, creating a smooth experience.
- **State Machine:** Understands user intent by tracking gesture states (`ACTIVE`, `RETURNING`, `NEUTRAL`), eliminating false triggers between movements.
- **Temporal Convolutional Network (TCN):** A TCN with an Attention mechanism for recognizing gesture sequences.
- **Feature Engineering:** Higher robustness through Procrustes alignment (rotation/scale invariance), Butterworth filtering (jitter reduction), and multi-temporal feature extraction.
- **Performance optimizations:** The model is compiled to TorchScript for faster, real-time inference on both CPU and GPU.
- **Data Collection:** A guided recorder captures complete gesture cycles, including the crucial transition phases, ensuring high-quality training data.
- 
## Quick Start

### For Windows

1.  **Setup (one time)**
    ```bash
    setup.bat
    ```
    This will create a virtual environment, install dependencies, and download the necessary models.

2.  **Record Training Data**
    ```bash
    gesture_record.bat
    ```
    - Follow the on-screen prompts to record gestures.
    - Collect at least 5-10 samples per gesture for good results.
    - Press `S` to save your recordings.

3.  **Train Model**
    ```bash
    gesture_train.bat
    ```
    Trains the Temporal Convolutional Network on your gesture data. This may take several minutes.

4.  **Use Gesture Control**
    ```bash
    gesture_control.bat
    ```

### For macOS (Apple Silicon)

1.  **Setup (one time)**
    First, make the shell scripts executable:
    ```bash
    chmod +x setup.sh gesture_record.sh gesture_train.sh gesture_control.sh
    ```
    Then run the setup script:
    ```bash
    ./setup.sh
    ```
    This will create a virtual environment, install dependencies (including PyTorch with MPS for GPU acceleration), and download models.

2.  **Record Training Data**
    ```bash
    ./gesture_record.sh
    ```
    - Your external USB camera will be used by default (as configured in `config.json`).
    - Follow on-screen prompts to record gestures.
    - Press `S` to save.

3.  **Train Model**
    ```bash
    ./gesture_train.sh
    ```
    Trains the model using your Mac's GPU (MPS).

4.  **Use Gesture Control**
    ```bash
    ./gesture_control.sh
    ```
    **Note:** You may need to grant accessibility permissions to your terminal or IDE (e.g., VS Code) for the gesture controls (like scrolling and zooming) to work. Go to `System Settings > Privacy & Security > Accessibility` and add your terminal application.

## Gestures

The system supports both continuous and one-shot gestures.

| Gesture                 | Action (Windows) | Action (macOS)         | Type        |
|-------------------------|------------------|------------------------|-------------|
| Two-finger swipe down   | Scroll Up        | Scroll Up (Natural)    | Continuous  |
| Two-finger swipe up     | Scroll Down      | Scroll Down (Natural)  | Continuous  |
| Spread fingers apart    | Zoom In (`Ctrl`+`+`) | Zoom In (`Cmd`+`+`)      | Continuous  |
| Pinch fingers together  | Zoom Out (`Ctrl`+`-`) | Zoom Out (`Cmd`+`-`)     | Continuous  |
| Open hand               | Maximize Window  | Toggle Fullscreen      | One-shot    |
| Swipe hand left         | Go Back (Browser)| Go Back (Browser)      | One-shot    |
| Keep hand still/closed  | No action        | No action              | Neutral     |

## Files

- `gesture_recorder.py` - Records gesture sequences
- `train_model.py` - Trains the TCN model
- `gesture_control.py` - Real-time gesture control
- `physics_engine.py` - Simulates momentum for smooth control
- `requirements.txt` - Python dependencies
- `config.json` - Configuration for camera, physics, etc.

**Scripts:**
- `setup.bat` / `setup.sh` - One-time setup
- `gesture_record.bat` / `gesture_record.sh` - Record gestures
- `gesture_train.bat` / `gesture_train.sh` - Train model
- `gesture_control.bat` / `gesture_control.sh` - Run control

## Requirements

- Windows 10/11 or macOS (with Apple Silicon M1/M2/M3/M4)
- Python 3.8+
- Webcam (Internal or External USB)
- NVIDIA GPU (optional, for faster training on Windows)

## Tips

- Record gestures in good lighting
- Keep hand fully visible in camera
- Make gestures distinct and clear
- Retrain if accuracy is low