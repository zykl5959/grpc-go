#!/usr/bin/env python3
"""
Test script for Tennis Line Detection App
Creates synthetic test data to demonstrate the functionality.
"""

import cv2
import numpy as np
import time
import math
from tennis_line_detector import TennisLineApp, Point, BallDetection

class TennisSimulator:
    """Simulates a tennis court and ball movement for testing."""
    
    def __init__(self, width=1280, height=720):
        self.width = width
        self.height = height
        self.frame_count = 0
        
        # Court dimensions (relative to frame)
        self.court_left = width // 4
        self.court_right = 3 * width // 4
        self.court_top = height // 6
        self.court_bottom = 5 * height // 6
        
        # Ball simulation
        self.ball_x = width // 2
        self.ball_y = height // 2
        self.ball_vx = 3
        self.ball_vy = 2
        self.ball_radius = 15
        
    def generate_frame(self):
        """Generate a synthetic tennis court frame."""
        # Create black background
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Draw tennis court (green background)
        cv2.rectangle(frame, 
                     (self.court_left, self.court_top), 
                     (self.court_right, self.court_bottom), 
                     (0, 100, 0), -1)
        
        # Draw court lines (white)
        line_color = (255, 255, 255)
        line_thickness = 3
        
        # Outer boundary
        cv2.rectangle(frame, 
                     (self.court_left, self.court_top), 
                     (self.court_right, self.court_bottom), 
                     line_color, line_thickness)
        
        # Baseline (bottom line of court)
        baseline_y = self.court_bottom - 50
        cv2.line(frame, (self.court_left, baseline_y), 
                (self.court_right, baseline_y), line_color, line_thickness)
        
        # Service line
        service_y = baseline_y - 100
        cv2.line(frame, (self.court_left, service_y), 
                (self.court_right, service_y), line_color, line_thickness)
        
        # Center line
        center_x = (self.court_left + self.court_right) // 2
        cv2.line(frame, (center_x, self.court_top), 
                (center_x, self.court_bottom), line_color, line_thickness)
        
        # Sidelines
        cv2.line(frame, (self.court_left, self.court_top), 
                (self.court_left, self.court_bottom), line_color, line_thickness)
        cv2.line(frame, (self.court_right, self.court_top), 
                (self.court_right, self.court_bottom), line_color, line_thickness)
        
        # Update ball position
        self._update_ball_position()
        
        # Draw tennis ball (yellow-green)
        ball_color = (0, 255, 150)  # BGR format
        cv2.circle(frame, (int(self.ball_x), int(self.ball_y)), 
                  self.ball_radius, ball_color, -1)
        
        # Add some noise/texture
        noise = np.random.randint(0, 30, (self.height, self.width, 3), dtype=np.uint8)
        frame = cv2.add(frame, noise)
        
        self.frame_count += 1
        return frame
    
    def _update_ball_position(self):
        """Update ball position with realistic physics."""
        # Update position
        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy
        
        # Bounce off walls
        if self.ball_x <= self.ball_radius or self.ball_x >= self.width - self.ball_radius:
            self.ball_vx = -self.ball_vx
        
        if self.ball_y <= self.ball_radius or self.ball_y >= self.height - self.ball_radius:
            self.ball_vy = -self.ball_vy
        
        # Add some randomness
        if self.frame_count % 60 == 0:  # Change direction every 2 seconds at 30fps
            self.ball_vx += np.random.randint(-2, 3)
            self.ball_vy += np.random.randint(-2, 3)
            
            # Clamp velocities
            self.ball_vx = np.clip(self.ball_vx, -8, 8)
            self.ball_vy = np.clip(self.ball_vy, -8, 8)

class TestVideoSource:
    """Simulates a video source for testing."""
    
    def __init__(self):
        self.simulator = TennisSimulator()
        self.is_opened = True
        
    def isOpened(self):
        return self.is_opened
    
    def read(self):
        if not self.is_opened:
            return False, None
        
        frame = self.simulator.generate_frame()
        time.sleep(1/30)  # Simulate 30 FPS
        return True, frame
    
    def set(self, prop, value):
        # Ignore camera property settings for simulation
        pass
    
    def release(self):
        self.is_opened = False

def test_tennis_detection():
    """Test the tennis line detection with simulated data."""
    print("Testing Tennis Line Detection App with simulated data...")
    print("This will create a synthetic tennis court with a moving ball.")
    print("Press 'q' to quit the test.")
    
    # Create test app with simulated video source
    app = TennisLineApp(video_source=None)
    
    # Replace the video capture with our test source
    app.cap = TestVideoSource()
    
    try:
        app.run()
    except Exception as e:
        print(f"Test error: {e}")
    finally:
        print("Test completed.")

def test_ball_detection_accuracy():
    """Test ball detection accuracy with known positions."""
    print("Testing ball detection accuracy...")
    
    from tennis_line_detector import TennisBallDetector
    
    detector = TennisBallDetector()
    simulator = TennisSimulator()
    
    correct_detections = 0
    total_frames = 100
    
    for i in range(total_frames):
        frame = simulator.generate_frame()
        
        # Get actual ball position
        actual_x, actual_y = simulator.ball_x, simulator.ball_y
        
        # Detect ball
        ball = detector.detect_ball(frame)
        
        if ball:
            detected_x, detected_y = ball.center.x, ball.center.y
            distance = math.sqrt((actual_x - detected_x)**2 + (actual_y - detected_y)**2)
            
            # Consider detection correct if within 20 pixels
            if distance <= 20:
                correct_detections += 1
        
        if (i + 1) % 20 == 0:
            accuracy = (correct_detections / (i + 1)) * 100
            print(f"Frame {i+1}/{total_frames}: Accuracy = {accuracy:.1f}%")
    
    final_accuracy = (correct_detections / total_frames) * 100
    print(f"Final accuracy: {final_accuracy:.1f}% ({correct_detections}/{total_frames})")

def test_line_detection():
    """Test court line detection."""
    print("Testing court line detection...")
    
    from tennis_line_detector import TennisCourtDetector
    
    detector = TennisCourtDetector()
    simulator = TennisSimulator()
    
    # Generate a frame and detect lines
    frame = simulator.generate_frame()
    lines = detector.detect_court_lines(frame)
    
    print(f"Detected {len(lines)} court lines")
    print(f"Baseline Y: {detector.baseline_y}")
    print(f"Service line Y: {detector.service_line_y}")
    print(f"Left sideline X: {detector.sideline_left_x}")
    print(f"Right sideline X: {detector.sideline_right_x}")
    
    # Show the frame with detected lines
    test_frame = frame.copy()
    for line in lines:
        (x1, y1), (x2, y2) = line
        cv2.line(test_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    cv2.imshow('Line Detection Test', test_frame)
    cv2.waitKey(3000)  # Show for 3 seconds
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Tennis Line Detection - Test Suite")
    print("=" * 40)
    
    while True:
        print("\nChoose a test:")
        print("1. Full app test with simulation")
        print("2. Ball detection accuracy test")
        print("3. Court line detection test")
        print("4. Exit")
        
        choice = input("Enter choice (1-4): ").strip()
        
        if choice == "1":
            test_tennis_detection()
        elif choice == "2":
            test_ball_detection_accuracy()
        elif choice == "3":
            test_line_detection()
        elif choice == "4":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please enter 1-4.")