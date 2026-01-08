#!/bin/bash
# gesture_record.sh - Enhanced recording with new features

source venv/bin/activate
echo ""
echo "=== Enhanced Gesture Recorder ==="
echo ""
echo "This recorder captures continuous gesture sequences with transitions."
echo ""
echo "RECORDING MODES:"
echo "- Press 1-7: Guided recording (with timing cues)"
echo ""
echo "GESTURES:"
echo "[1] Scroll Up        [2] Scroll Down"
echo "[3] Zoom In          [4] Zoom Out"
echo "[5] Neutral          [6] Maximize Window"
echo "[7] Go Back          [8] Pointer (Cursor)"
echo ""
echo "TIPS:"
echo "- Follow the visual guides in guided mode"
echo "- Perform each gesture multiple times with returns to neutral"
echo "- The system will learn the transition patterns"
echo ""
python3 gesture_recorder.py
deactivate