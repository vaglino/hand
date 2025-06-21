# diagnostic_tool.py - Real-time diagnostics and performance monitoring

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import psutil
import torch
import matplotlib.pyplot as plt
from collections import deque
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional
import os
import json

@dataclass
class PerformanceMetrics:
    """System performance tracking"""
    fps: float = 0.0
    latency_ms: float = 0.0
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    hand_detection_confidence: float = 0.0
    gesture_inference_ms: float = 0.0
    physics_update_ms: float = 0.0
    
class DiagnosticTool:
    """Comprehensive diagnostic tool for gesture control system"""
    
    def __init__(self, model_path='hand_landmarker.task'):
        # Performance tracking
        self.metrics = PerformanceMetrics()
        self.fps_history = deque(maxlen=60)
        self.latency_history = deque(maxlen=60)
        self.cpu_history = deque(maxlen=60)
        self.confidence_history = deque(maxlen=60)
        
        # Timing measurements
        self.frame_times = deque(maxlen=30)
        self.last_frame_time = time.time()
        
        # System info
        self.gpu_available = torch.cuda.is_available()
        self.cpu_count = psutil.cpu_count()
        self.total_memory = psutil.virtual_memory().total / (1024**3)  # GB
        
        # MediaPipe setup
        self.results = None
        self.landmark_history = deque(maxlen=30)
        
        base_opts = python.BaseOptions(model_asset_path=model_path)
        opts = vision.HandLandmarkerOptions(
            base_options=base_opts,
            running_mode=vision.RunningMode.LIVE_STREAM,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            result_callback=self._process_result
        )
        self.landmarker = vision.HandLandmarker.create_from_options(opts)
        
        # Diagnostic modes
        self.show_graphs = False
        self.show_landmarks = True
        self.show_motion_vectors = True
        self.show_performance = True
        self.record_session = False
        self.session_data = []
        
        # Model diagnostics
        self.model_loaded = self._check_models()
        
        print("\n=== Gesture System Diagnostics ===")
        print(f"CPU: {self.cpu_count} cores")
        print(f"Memory: {self.total_memory:.1f} GB")
        print(f"GPU: {'Available' if self.gpu_available else 'Not available'}")
        print(f"Models: {'Loaded' if self.model_loaded else 'Not found'}")
        print("\nPress H for help")
        
    def _check_models(self):
        """Check if trained models exist"""
        required_files = [
            'gesture_data/feature_extractor.pkl',
            'gesture_data/gesture_scaler.pkl',
            'gesture_data/label_encoder.pkl'
        ]
        
        classifier_exists = (
            os.path.exists('gesture_data/gesture_classifier.pth') or 
            os.path.exists('gesture_data/gesture_classifier.pkl')
        )
        
        return classifier_exists and all(os.path.exists(f) for f in required_files)
    
    def _process_result(self, result: vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        self.results = result
        
        # Update detection confidence
        if result and result.hand_landmarks:
            # MediaPipe doesn't directly provide confidence, estimate from landmark visibility
            self.metrics.hand_detection_confidence = 0.95  # Placeholder
        else:
            self.metrics.hand_detection_confidence = 0.0
    
    def _update_performance_metrics(self):
        """Update all performance metrics"""
        current_time = time.time()
        
        # FPS calculation
        if self.frame_times:
            self.metrics.fps = len(self.frame_times) / sum(self.frame_times)
            self.fps_history.append(self.metrics.fps)
        
        # CPU and memory
        self.metrics.cpu_percent = psutil.cpu_percent(interval=0)
        self.metrics.memory_mb = psutil.Process().memory_info().rss / (1024**2)
        self.cpu_history.append(self.metrics.cpu_percent)
        
        # Confidence history
        self.confidence_history.append(self.metrics.hand_detection_confidence)
        
        # Latency estimation (simplified)
        if len(self.frame_times) > 0:
            self.metrics.latency_ms = np.mean(self.frame_times) * 1000
            self.latency_history.append(self.metrics.latency_ms)
    
    def _analyze_landmark_quality(self, landmarks):
        """Analyze hand tracking quality"""
        if len(self.landmark_history) < 2:
            return {}
        
        # Convert to numpy arrays
        current = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
        previous = np.array([[lm.x, lm.y, lm.z] for lm in self.landmark_history[-1]])
        
        # Calculate metrics
        movement = np.linalg.norm(current - previous, axis=1)
        jitter = np.std(movement)
        max_movement = np.max(movement)
        
        # Check for tracking issues
        issues = []
        if jitter > 0.05:
            issues.append("High jitter detected")
        if max_movement > 0.3:
            issues.append("Rapid movement detected")
        
        # Z-coordinate analysis (depth stability)
        z_variance = np.var(current[:, 2])
        if z_variance > 0.01:
            issues.append("Unstable depth tracking")
        
        return {
            'jitter': jitter,
            'max_movement': max_movement,
            'z_variance': z_variance,
            'issues': issues
        }
    
    def _draw_diagnostic_overlay(self, frame):
        """Draw comprehensive diagnostic information"""
        h, w = frame.shape[:2]
        
        # Performance panel (top-right)
        if self.show_performance:
            panel_w = 250
            overlay = frame.copy()
            cv2.rectangle(overlay, (w - panel_w, 0), (w, 200), (0, 0, 0), -1)
            frame = cv2.addWeighted(frame, 0.8, overlay, 0.2, 0)
            
            y = 20
            cv2.putText(frame, "PERFORMANCE", (w - panel_w + 10, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            
            y += 25
            cv2.putText(frame, f"FPS: {self.metrics.fps:.1f}", (w - panel_w + 10, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            y += 20
            cv2.putText(frame, f"Latency: {self.metrics.latency_ms:.1f}ms", 
                       (w - panel_w + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            y += 20
            cv2.putText(frame, f"CPU: {self.metrics.cpu_percent:.1f}%", 
                       (w - panel_w + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            y += 20
            cv2.putText(frame, f"Memory: {self.metrics.memory_mb:.0f}MB", 
                       (w - panel_w + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            y += 20
            cv2.putText(frame, f"Hand Conf: {self.metrics.hand_detection_confidence:.2f}", 
                       (w - panel_w + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Mini graphs
            if len(self.fps_history) > 10:
                self._draw_mini_graph(frame, list(self.fps_history), 
                                    (w - panel_w + 10, 140), (panel_w - 20, 40), 
                                    "FPS", target_value=30)
        
        # Landmark quality panel (left side)
        if self.results and self.results.hand_landmarks and len(self.landmark_history) > 0:
            quality = self._analyze_landmark_quality(self.results.hand_landmarks[0])
            
            if quality:
                y = h - 150
                cv2.putText(frame, "TRACKING QUALITY", (10, y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
                
                y += 25
                cv2.putText(frame, f"Jitter: {quality['jitter']:.4f}", (10, y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
                y += 20
                cv2.putText(frame, f"Max Move: {quality['max_movement']:.3f}", (10, y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
                # Issues
                if quality['issues']:
                    y += 25
                    for issue in quality['issues']:
                        cv2.putText(frame, f"! {issue}", (10, y), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                        y += 15
        
        # Recording indicator
        if self.record_session:
            cv2.circle(frame, (30, 30), 10, (0, 0, 255), -1)
            cv2.putText(frame, "RECORDING SESSION", (50, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Help text
        cv2.putText(frame, "[H] Help [G] Graphs [P] Perf [M] Motion [R] Record [Q] Quit", 
                   (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        return frame
    
    def _draw_mini_graph(self, frame, data, pos, size, label, target_value=None):
        """Draw a mini performance graph"""
        x, y = pos
        w, h = size
        
        if len(data) < 2:
            return
        
        # Normalize data
        data_array = np.array(data)
        data_min = np.min(data_array)
        data_max = np.max(data_array)
        
        if data_max - data_min < 0.1:
            data_max = data_min + 1
        
        # Draw background
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 50), -1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 1)
        
        # Draw data
        points = []
        for i, value in enumerate(data[-w:]):
            px = x + i
            py = y + h - int((value - data_min) / (data_max - data_min) * h)
            points.append((px, py))
        
        if len(points) > 1:
            pts = np.array(points, np.int32)
            cv2.polylines(frame, [pts], False, (0, 255, 0), 1)
        
        # Draw target line
        if target_value is not None:
            target_y = y + h - int((target_value - data_min) / (data_max - data_min) * h)
            cv2.line(frame, (x, target_y), (x + w, target_y), (0, 150, 255), 1)
        
        # Label
        cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
    
    def _draw_motion_vectors(self, frame, landmarks):
        """Visualize motion vectors between frames"""
        if not self.show_motion_vectors or len(self.landmark_history) < 1:
            return
        
        h, w = frame.shape[:2]
        current = landmarks
        previous = self.landmark_history[-1]
        
        # Key points to track
        key_points = {
            'palm': [0, 5, 9, 13, 17],  # Palm center
            'thumb': 4,
            'index': 8,
            'middle': 12,
            'ring': 16,
            'pinky': 20
        }
        
        for name, indices in key_points.items():
            if isinstance(indices, list):
                # Average position for palm
                curr_x = np.mean([current[i].x for i in indices]) * w
                curr_y = np.mean([current[i].y for i in indices]) * h
                prev_x = np.mean([previous[i].x for i in indices]) * w
                prev_y = np.mean([previous[i].y for i in indices]) * h
            else:
                # Single point
                curr_x = current[indices].x * w
                curr_y = current[indices].y * h
                prev_x = previous[indices].x * w
                prev_y = previous[indices].y * h
            
            # Draw motion vector
            if abs(curr_x - prev_x) > 1 or abs(curr_y - prev_y) > 1:
                cv2.arrowedLine(frame, 
                               (int(prev_x), int(prev_y)), 
                               (int(curr_x), int(curr_y)), 
                               (255, 0, 255), 2, tipLength=0.3)
    
    def _save_session_data(self):
        """Save diagnostic session data"""
        if not self.session_data:
            print("No session data to save")
            return
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"gesture_data/diagnostic_session_{timestamp}.json"
        
        os.makedirs('gesture_data', exist_ok=True)
        
        session_summary = {
            'timestamp': timestamp,
            'duration_seconds': len(self.session_data) / 30,  # Assuming 30 FPS
            'avg_fps': np.mean([d['fps'] for d in self.session_data]),
            'avg_latency_ms': np.mean([d['latency'] for d in self.session_data]),
            'avg_cpu_percent': np.mean([d['cpu'] for d in self.session_data]),
            'frames': self.session_data
        }
        
        with open(filename, 'w') as f:
            json.dump(session_summary, f, indent=2)
        
        print(f"\n✓ Session data saved to {filename}")
        print(f"  Duration: {session_summary['duration_seconds']:.1f}s")
        print(f"  Avg FPS: {session_summary['avg_fps']:.1f}")
    
    def _show_help(self):
        """Display help window"""
        help_text = """
        DIAGNOSTIC TOOL CONTROLS
        
        [H] - Show/hide this help
        [G] - Toggle performance graphs
        [L] - Toggle landmark visualization
        [M] - Toggle motion vectors
        [P] - Toggle performance panel
        [R] - Start/stop recording session
        [S] - Save session data
        [Q] - Quit
        
        WHAT TO LOOK FOR:
        - FPS should be stable around 30
        - Latency should be < 50ms
        - CPU usage should be < 30%
        - Jitter should be < 0.01
        - No tracking issues warnings
        
        Press any key to close...
        """
        
        # Create help window
        help_img = np.zeros((400, 500, 3), dtype=np.uint8)
        help_img[:] = (50, 50, 50)
        
        y = 30
        for line in help_text.strip().split('\n'):
            cv2.putText(help_img, line.strip(), (20, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y += 20
        
        cv2.imshow('Help', help_img)
        cv2.waitKey(0)
        cv2.destroyWindow('Help')
    
    def run(self):
        """Main diagnostic loop"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        timestamp = 0
        
        while True:
            frame_start = time.time()
            
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            
            # Process hand tracking
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp += 1
            self.landmarker.detect_async(mp_img, timestamp)
            
            # Update metrics
            self._update_performance_metrics()
            
            # Process landmarks
            if self.results and self.results.hand_landmarks:
                landmarks = self.results.hand_landmarks[0]
                
                # Draw landmarks if enabled
                if self.show_landmarks:
                    from mediapipe.framework.formats import landmark_pb2
                    proto = landmark_pb2.NormalizedLandmarkList()
                    proto.landmark.extend([
                        landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) 
                        for lm in landmarks
                    ])
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, proto, mp.solutions.hands.HAND_CONNECTIONS)
                
                # Draw motion vectors
                self._draw_motion_vectors(frame, landmarks)
                
                # Update history
                self.landmark_history.append(landmarks)
            
            # Draw diagnostics
            frame = self._draw_diagnostic_overlay(frame)
            
            # Record session data if enabled
            if self.record_session:
                self.session_data.append({
                    'timestamp': time.time(),
                    'fps': self.metrics.fps,
                    'latency': self.metrics.latency_ms,
                    'cpu': self.metrics.cpu_percent,
                    'memory': self.metrics.memory_mb,
                    'hand_detected': self.results is not None and self.results.hand_landmarks is not None
                })
            
            # Calculate frame time
            frame_time = time.time() - frame_start
            self.frame_times.append(frame_time)
            
            # Display
            cv2.imshow('Gesture System Diagnostics', frame)
            
            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('h'):
                self._show_help()
            elif key == ord('g'):
                self.show_graphs = not self.show_graphs
                print(f"Graphs: {'ON' if self.show_graphs else 'OFF'}")
            elif key == ord('l'):
                self.show_landmarks = not self.show_landmarks
                print(f"Landmarks: {'ON' if self.show_landmarks else 'OFF'}")
            elif key == ord('m'):
                self.show_motion_vectors = not self.show_motion_vectors
                print(f"Motion vectors: {'ON' if self.show_motion_vectors else 'OFF'}")
            elif key == ord('p'):
                self.show_performance = not self.show_performance
                print(f"Performance panel: {'ON' if self.show_performance else 'OFF'}")
            elif key == ord('r'):
                self.record_session = not self.record_session
                if self.record_session:
                    self.session_data = []
                    print("Recording session started")
                else:
                    print("Recording session stopped")
            elif key == ord('s'):
                self._save_session_data()
        
        cap.release()
        cv2.destroyAllWindows()
        self.landmarker.close()
        
        # Final summary
        print("\n=== Diagnostic Summary ===")
        if self.fps_history:
            print(f"Average FPS: {np.mean(list(self.fps_history)):.1f}")
        if self.latency_history:
            print(f"Average Latency: {np.mean(list(self.latency_history)):.1f}ms")
        if self.cpu_history:
            print(f"Average CPU: {np.mean(list(self.cpu_history)):.1f}%")


if __name__ == "__main__":
    tool = DiagnosticTool()
    tool.run()