# physics_demo.py - Interactive demonstration of the physics engine

import cv2
import numpy as np
import time
from physics_engine import TrackpadPhysicsEngine, Vector2D
import matplotlib.pyplot as plt
from collections import deque

class PhysicsEngineDemo:
    """Interactive demo to showcase physics engine capabilities"""
    
    def __init__(self):
        self.physics = TrackpadPhysicsEngine()
        self.width = 800
        self.height = 600
        
        # Demo state
        self.demo_mode = 'manual'  # manual, auto_scroll, auto_zoom, stress_test
        self.mouse_pressed = False
        self.last_mouse_pos = None
        self.auto_timer = 0
        
        # Visualization
        self.trail_points = deque(maxlen=100)
        self.momentum_history = deque(maxlen=200)
        self.zoom_history = deque(maxlen=200)
        
        # Performance tracking
        self.frame_times = deque(maxlen=60)
        self.last_frame_time = time.time()
        
        # Create window and mouse callback
        cv2.namedWindow('Physics Engine Demo')
        cv2.setMouseCallback('Physics Engine Demo', self._mouse_callback)
        
        print("\n=== Physics Engine Demo ===")
        print("\nControls:")
        print("- Click and drag: Apply scroll force")
        print("- Scroll wheel: Apply zoom force")
        print("- [1] Manual mode")
        print("- [2] Auto scroll demo")
        print("- [3] Auto zoom demo")
        print("- [4] Stress test")
        print("- [R] Reset physics")
        print("- [+/-] Adjust sensitivity")
        print("- [F] Toggle friction")
        print("- [Q] Quit\n")
    
    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse input for manual control"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.mouse_pressed = True
            self.last_mouse_pos = (x, y)
        
        elif event == cv2.EVENT_LBUTTONUP:
            self.mouse_pressed = False
            self.last_mouse_pos = None
        
        elif event == cv2.EVENT_MOUSEMOVE and self.mouse_pressed:
            if self.last_mouse_pos:
                # Calculate mouse velocity
                dx = x - self.last_mouse_pos[0]
                dy = y - self.last_mouse_pos[1]
                
                # Apply scroll force based on mouse movement
                direction = Vector2D(dx, -dy)  # Invert Y for natural scrolling
                intensity = min(direction.magnitude() / 100.0, 1.0)
                
                if intensity > 0.01:
                    self.physics.apply_scroll_force(direction, intensity)
                
            self.last_mouse_pos = (x, y)
        
        elif event == cv2.EVENT_MOUSEWHEEL:
            # Apply zoom force
            zoom_direction = 1 if flags > 0 else -1
            self.physics.apply_zoom_force(zoom_direction * 0.1, 0.8)
    
    def _apply_auto_scroll_demo(self):
        """Automated scroll demonstration"""
        phase = (self.auto_timer % 200) / 200.0
        
        if phase < 0.25:
            # Scroll right
            self.physics.apply_scroll_force(Vector2D(1, 0), 0.7)
        elif phase < 0.5:
            # Let momentum carry
            pass
        elif phase < 0.75:
            # Scroll up-left
            self.physics.apply_scroll_force(Vector2D(-0.7, 0.7), 0.8)
        else:
            # Let momentum carry
            pass
    
    def _apply_auto_zoom_demo(self):
        """Automated zoom demonstration"""
        phase = (self.auto_timer % 150) / 150.0
        
        if phase < 0.3:
            # Zoom in
            self.physics.apply_zoom_force(0.02, 0.8)
        elif phase < 0.6:
            # Let it settle
            pass
        elif phase < 0.9:
            # Zoom out
            self.physics.apply_zoom_force(-0.02, 0.8)
        else:
            # Let it settle
            pass
    
    def _apply_stress_test(self):
        """Stress test with rapid random inputs"""
        if self.auto_timer % 3 == 0:
            # Random scroll
            angle = np.random.random() * 2 * np.pi
            intensity = np.random.random() * 0.5 + 0.5
            direction = Vector2D(np.cos(angle), np.sin(angle))
            self.physics.apply_scroll_force(direction, intensity)
        
        if self.auto_timer % 7 == 0:
            # Random zoom
            zoom_rate = (np.random.random() - 0.5) * 0.1
            self.physics.apply_zoom_force(zoom_rate, 0.7)
    
    def _draw_visualization(self, frame):
        """Draw physics visualization"""
        h, w = frame.shape[:2]
        
        # Draw grid for reference
        grid_size = 50
        for x in range(0, w, grid_size):
            cv2.line(frame, (x, 0), (x, h), (50, 50, 50), 1)
        for y in range(0, h, grid_size):
            cv2.line(frame, (0, y), (w, y), (50, 50, 50), 1)
        
        # Get physics state
        state = self.physics.get_physics_state()
        momentum = state['scroll_momentum']
        zoom_vel = state['zoom_velocity']
        
        # Update histories
        self.momentum_history.append(momentum)
        self.zoom_history.append(zoom_vel)
        
        # Draw momentum vector
        center_x, center_y = w // 2, h // 2
        
        if momentum[0] != 0 or momentum[1] != 0:
            # Scale momentum for visualization
            end_x = int(center_x + momentum[0] * 10)
            end_y = int(center_y - momentum[1] * 10)
            
            # Draw arrow
            cv2.arrowedLine(frame, (center_x, center_y), (end_x, end_y), 
                           (0, 255, 0), 3, tipLength=0.3)
            
            # Add to trail
            self.trail_points.append((end_x, end_y))
        
        # Draw momentum trail
        if len(self.trail_points) > 1:
            for i in range(1, len(self.trail_points)):
                alpha = i / len(self.trail_points)
                color = (0, int(255 * alpha), 0)
                cv2.line(frame, self.trail_points[i-1], self.trail_points[i], color, 2)
        
        # Draw zoom indicator
        if abs(zoom_vel) > 0.001:
            zoom_radius = int(50 + abs(zoom_vel) * 500)
            color = (0, 150, 255) if zoom_vel > 0 else (255, 150, 0)
            cv2.circle(frame, (center_x, center_y), zoom_radius, color, 2)
        
        # Draw info panel
        panel_h = 200
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (250, panel_h), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.8, overlay, 0.2, 0)
        
        # Mode indicator
        y = 25
        cv2.putText(frame, f"Mode: {self.demo_mode.upper()}", (10, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Physics parameters
        y += 30
        cv2.putText(frame, "Physics State:", (10, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        y += 25
        cv2.putText(frame, f"Momentum: ({momentum[0]:.2f}, {momentum[1]:.2f})", 
                   (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        y += 20
        magnitude = np.sqrt(momentum[0]**2 + momentum[1]**2)
        cv2.putText(frame, f"Speed: {magnitude:.2f}", 
                   (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        y += 20
        cv2.putText(frame, f"Zoom: {zoom_vel:.3f}", 
                   (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        y += 25
        cv2.putText(frame, f"Friction: {self.physics.scroll_friction:.2f}", 
                   (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        y += 20
        cv2.putText(frame, f"Sensitivity: {self.physics.user_scroll_multiplier:.2f}x", 
                   (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Draw momentum graph
        if len(self.momentum_history) > 10:
            graph_y = h - 150
            graph_h = 100
            graph_w = 200
            
            # Background
            cv2.rectangle(frame, (10, graph_y), (10 + graph_w, graph_y + graph_h), 
                         (50, 50, 50), -1)
            cv2.rectangle(frame, (10, graph_y), (10 + graph_w, graph_y + graph_h), 
                         (100, 100, 100), 1)
            
            # Plot momentum magnitude
            points = []
            for i, mom in enumerate(list(self.momentum_history)[-graph_w:]):
                mag = np.sqrt(mom[0]**2 + mom[1]**2)
                x = 10 + i
                y = graph_y + graph_h - int(min(mag * 2, graph_h - 5))
                points.append((x, y))
            
            if len(points) > 1:
                pts = np.array(points, np.int32)
                cv2.polylines(frame, [pts], False, (0, 255, 0), 2)
            
            cv2.putText(frame, "Momentum", (15, graph_y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # FPS counter
        if len(self.frame_times) > 0:
            fps = 1.0 / np.mean(list(self.frame_times))
            cv2.putText(frame, f"FPS: {fps:.1f}", (w - 100, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
        
        return frame
    
    def run(self):
        """Main demo loop"""
        # Create blank canvas
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        while True:
            # Track frame time
            current_time = time.time()
            dt = current_time - self.last_frame_time
            self.frame_times.append(dt)
            self.last_frame_time = current_time
            
            # Clear frame
            frame[:] = (20, 20, 20)
            
            # Apply demo actions
            if self.demo_mode == 'auto_scroll':
                self._apply_auto_scroll_demo()
            elif self.demo_mode == 'auto_zoom':
                self._apply_auto_zoom_demo()
            elif self.demo_mode == 'stress_test':
                self._apply_stress_test()
            
            self.auto_timer += 1
            
            # Update physics
            self.physics.update(dt)
            
            # Note: In the real system, execute_smooth_actions() would
            # perform actual scrolling/zooming. Here we just simulate.
            
            # Draw visualization
            frame = self._draw_visualization(frame)
            
            # Help text
            cv2.putText(frame, "Click & Drag to scroll | Mouse wheel to zoom", 
                       (10, self.height - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (150, 150, 150), 1)
            
            # Display
            cv2.imshow('Physics Engine Demo', frame)
            
            # Handle keyboard
            key = cv2.waitKey(16) & 0xFF  # ~60 FPS
            
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.physics.reset_momentum()
                self.trail_points.clear()
                print("Physics reset")
            elif key == ord('1'):
                self.demo_mode = 'manual'
                print("Manual mode")
            elif key == ord('2'):
                self.demo_mode = 'auto_scroll'
                print("Auto scroll demo")
            elif key == ord('3'):
                self.demo_mode = 'auto_zoom'
                print("Auto zoom demo")
            elif key == ord('4'):
                self.demo_mode = 'stress_test'
                print("Stress test mode")
            elif key == ord('+'):
                self.physics.user_scroll_multiplier *= 1.2
                print(f"Sensitivity: {self.physics.user_scroll_multiplier:.2f}x")
            elif key == ord('-'):
                self.physics.user_scroll_multiplier /= 1.2
                print(f"Sensitivity: {self.physics.user_scroll_multiplier:.2f}x")
            elif key == ord('f'):
                # Toggle friction
                if self.physics.scroll_friction > 0.9:
                    self.physics.scroll_friction = 0.7
                    print("Low friction mode")
                else:
                    self.physics.scroll_friction = 0.92
                    print("Normal friction mode")
        
        cv2.destroyAllWindows()
        
        # Print summary
        print("\n=== Demo Summary ===")
        print(f"Average FPS: {1.0 / np.mean(list(self.frame_times)):.1f}")
        print("Physics engine performed smoothly!")


if __name__ == "__main__":
    demo = PhysicsEngineDemo()
    demo.run()