#!/usr/bin/env python3
"""
Tennis Line Detection App
Detects if a tennis ball hits on the line or outside the line using computer vision.
"""

import cv2
import numpy as np
import argparse
import time
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Optional
import math

@dataclass
class Point:
    x: float
    y: float

@dataclass
class BallDetection:
    center: Point
    radius: float
    confidence: float
    timestamp: float

class TennisCourtDetector:
    """Detects tennis court lines in the video frame."""
    
    def __init__(self):
        self.court_lines = []
        self.baseline_y = None
        self.service_line_y = None
        self.sideline_left_x = None
        self.sideline_right_x = None
        
    def detect_court_lines(self, frame):
        """Detect court lines using Hough Line Transform."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Apply morphological operations to clean up the edges
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # Detect lines using HoughLinesP
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, 
                               minLineLength=50, maxLineGap=10)
        
        self.court_lines = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                self.court_lines.append(((x1, y1), (x2, y2)))
                
        self._classify_lines(frame.shape)
        return self.court_lines
    
    def _classify_lines(self, frame_shape):
        """Classify detected lines as baselines, service lines, or sidelines."""
        h, w = frame_shape[:2]
        
        horizontal_lines = []
        vertical_lines = []
        
        for line in self.court_lines:
            (x1, y1), (x2, y2) = line
            angle = math.atan2(abs(y2 - y1), abs(x2 - x1)) * 180 / math.pi
            
            if angle < 30:  # Horizontal line
                horizontal_lines.append(line)
            elif angle > 60:  # Vertical line
                vertical_lines.append(line)
        
        # Sort horizontal lines by y-coordinate
        horizontal_lines.sort(key=lambda line: (line[0][1] + line[1][1]) / 2)
        
        # Assign baselines and service lines
        if len(horizontal_lines) >= 2:
            self.baseline_y = (horizontal_lines[-1][0][1] + horizontal_lines[-1][1][1]) / 2
            if len(horizontal_lines) >= 4:
                self.service_line_y = (horizontal_lines[-2][0][1] + horizontal_lines[-2][1][1]) / 2
        
        # Sort vertical lines by x-coordinate
        vertical_lines.sort(key=lambda line: (line[0][0] + line[1][0]) / 2)
        
        # Assign sidelines
        if len(vertical_lines) >= 2:
            self.sideline_left_x = (vertical_lines[0][0][0] + vertical_lines[0][1][0]) / 2
            self.sideline_right_x = (vertical_lines[-1][0][0] + vertical_lines[-1][1][0]) / 2

class TennisBallDetector:
    """Detects tennis balls in video frames."""
    
    def __init__(self):
        # HSV range for tennis ball (yellow-green)
        self.lower_hsv = np.array([20, 50, 50])
        self.upper_hsv = np.array([35, 255, 255])
        
        # Minimum and maximum radius for tennis ball
        self.min_radius = 5
        self.max_radius = 50
        
    def detect_ball(self, frame) -> Optional[BallDetection]:
        """Detect tennis ball in the frame."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create mask for tennis ball color
        mask = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)
        
        # Apply morphological operations
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Find the largest contour (most likely the ball)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Check if contour is circular enough
        area = cv2.contourArea(largest_contour)
        if area < 20:  # Too small
            return None
        
        # Get minimum enclosing circle
        (x, y), radius = cv2.minEnclosingCircle(largest_contour)
        
        if self.min_radius <= radius <= self.max_radius:
            # Calculate confidence based on circularity
            perimeter = cv2.arcLength(largest_contour, True)
            circularity = 4 * math.pi * area / (perimeter * perimeter)
            confidence = min(circularity, 1.0)
            
            return BallDetection(
                center=Point(x, y),
                radius=radius,
                confidence=confidence,
                timestamp=time.time()
            )
        
        return None

class LineCallDetector:
    """Determines if ball hits are in or out based on court lines."""
    
    def __init__(self, court_detector: TennisCourtDetector):
        self.court_detector = court_detector
        self.line_tolerance = 5  # pixels
        
    def is_ball_on_line(self, ball: BallDetection) -> Tuple[bool, str]:
        """
        Check if ball is on any court line.
        Returns (is_on_line, line_type)
        """
        x, y = ball.center.x, ball.center.y
        radius = ball.radius
        
        # Check baseline
        if self.court_detector.baseline_y is not None:
            if abs(y - self.court_detector.baseline_y) <= self.line_tolerance + radius:
                return True, "baseline"
        
        # Check service line
        if self.court_detector.service_line_y is not None:
            if abs(y - self.court_detector.service_line_y) <= self.line_tolerance + radius:
                return True, "service_line"
        
        # Check sidelines
        if self.court_detector.sideline_left_x is not None:
            if abs(x - self.court_detector.sideline_left_x) <= self.line_tolerance + radius:
                return True, "left_sideline"
                
        if self.court_detector.sideline_right_x is not None:
            if abs(x - self.court_detector.sideline_right_x) <= self.line_tolerance + radius:
                return True, "right_sideline"
        
        return False, ""
    
    def is_ball_out(self, ball: BallDetection) -> bool:
        """Check if ball is out of bounds."""
        x, y = ball.center.x, ball.center.y
        
        # Check if ball is beyond baseline
        if self.court_detector.baseline_y is not None:
            if y > self.court_detector.baseline_y + self.line_tolerance:
                return True
        
        # Check if ball is beyond sidelines
        if self.court_detector.sideline_left_x is not None:
            if x < self.court_detector.sideline_left_x - self.line_tolerance:
                return True
                
        if self.court_detector.sideline_right_x is not None:
            if x > self.court_detector.sideline_right_x + self.line_tolerance:
                return True
        
        return False

class TennisLineApp:
    """Main application class for tennis line detection."""
    
    def __init__(self, video_source=0):
        self.video_source = video_source
        self.cap = None
        
        # Initialize detectors
        self.court_detector = TennisCourtDetector()
        self.ball_detector = TennisBallDetector()
        self.line_call_detector = LineCallDetector(self.court_detector)
        
        # Ball tracking
        self.ball_history = deque(maxlen=30)  # Keep last 30 detections
        self.last_call = ""
        self.call_timestamp = 0
        
    def initialize_camera(self):
        """Initialize camera or video file."""
        self.cap = cv2.VideoCapture(self.video_source)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video source: {self.video_source}")
        
        # Set camera properties for better performance
        if isinstance(self.video_source, int):  # Camera
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
    
    def draw_court_lines(self, frame):
        """Draw detected court lines on the frame."""
        for line in self.court_detector.court_lines:
            (x1, y1), (x2, y2) = line
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Highlight specific court lines
        if self.court_detector.baseline_y is not None:
            cv2.line(frame, (0, int(self.court_detector.baseline_y)), 
                    (frame.shape[1], int(self.court_detector.baseline_y)), (0, 0, 255), 3)
            cv2.putText(frame, "Baseline", (10, int(self.court_detector.baseline_y) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        if self.court_detector.service_line_y is not None:
            cv2.line(frame, (0, int(self.court_detector.service_line_y)), 
                    (frame.shape[1], int(self.court_detector.service_line_y)), (255, 0, 0), 3)
            cv2.putText(frame, "Service Line", (10, int(self.court_detector.service_line_y) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    def draw_ball_detection(self, frame, ball: BallDetection):
        """Draw ball detection on the frame."""
        center = (int(ball.center.x), int(ball.center.y))
        radius = int(ball.radius)
        
        # Draw ball circle
        cv2.circle(frame, center, radius, (255, 255, 0), 2)
        cv2.circle(frame, center, 2, (255, 255, 0), -1)
        
        # Draw confidence
        cv2.putText(frame, f"Ball: {ball.confidence:.2f}", 
                   (center[0] - 50, center[1] - radius - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    def draw_line_call(self, frame, ball: BallDetection):
        """Draw line call result on the frame."""
        is_on_line, line_type = self.line_call_detector.is_ball_on_line(ball)
        is_out = self.line_call_detector.is_ball_out(ball)
        
        # Determine call
        if is_on_line:
            call = f"ON LINE ({line_type})"
            color = (0, 255, 255)  # Yellow
        elif is_out:
            call = "OUT"
            color = (0, 0, 255)  # Red
        else:
            call = "IN"
            color = (0, 255, 0)  # Green
        
        # Update last call with timestamp
        current_time = time.time()
        if call != "IN" or current_time - self.call_timestamp > 2:
            self.last_call = call
            self.call_timestamp = current_time
        
        # Draw call on screen
        cv2.putText(frame, self.last_call, (50, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3)
        
        # Draw ball position info
        cv2.putText(frame, f"Ball: ({int(ball.center.x)}, {int(ball.center.y)})", 
                   (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    def draw_ball_trail(self, frame):
        """Draw ball movement trail."""
        if len(self.ball_history) > 1:
            points = [(int(ball.center.x), int(ball.center.y)) for ball in self.ball_history]
            for i in range(1, len(points)):
                alpha = i / len(points)
                thickness = max(1, int(3 * alpha))
                cv2.line(frame, points[i-1], points[i], (255, 0, 255), thickness)
    
    def run(self):
        """Main application loop."""
        self.initialize_camera()
        
        print("Tennis Line Detection App")
        print("Press 'q' to quit, 'c' to calibrate court, 's' to save frame")
        print("Press 'r' to reset court detection")
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read frame")
                    break
                
                frame_count += 1
                
                # Detect court lines periodically or on first frame
                if frame_count == 1 or frame_count % 30 == 0:
                    self.court_detector.detect_court_lines(frame)
                
                # Detect ball
                ball = self.ball_detector.detect_ball(frame)
                
                # Draw court lines
                self.draw_court_lines(frame)
                
                if ball:
                    # Add to history
                    self.ball_history.append(ball)
                    
                    # Draw ball detection
                    self.draw_ball_detection(frame, ball)
                    
                    # Draw line call
                    self.draw_line_call(frame, ball)
                
                # Draw ball trail
                self.draw_ball_trail(frame)
                
                # Draw UI elements
                cv2.putText(frame, "Tennis Line Detector", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # Show frame
                cv2.imshow('Tennis Line Detection', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    print("Recalibrating court...")
                    self.court_detector.detect_court_lines(frame)
                elif key == ord('s'):
                    filename = f"tennis_frame_{int(time.time())}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"Saved frame as {filename}")
                elif key == ord('r'):
                    print("Resetting court detection...")
                    self.court_detector.court_lines = []
                    self.court_detector.baseline_y = None
                    self.court_detector.service_line_y = None
                    self.ball_history.clear()
        
        except KeyboardInterrupt:
            print("\nApplication interrupted by user")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Tennis Line Detection App')
    parser.add_argument('--video', type=str, default=0,
                       help='Video source (camera index or video file path)')
    parser.add_argument('--min-ball-radius', type=int, default=5,
                       help='Minimum ball radius in pixels')
    parser.add_argument('--max-ball-radius', type=int, default=50,
                       help='Maximum ball radius in pixels')
    parser.add_argument('--line-tolerance', type=int, default=5,
                       help='Line detection tolerance in pixels')
    
    args = parser.parse_args()
    
    # Convert video argument
    video_source = args.video
    if isinstance(video_source, str) and video_source.isdigit():
        video_source = int(video_source)
    
    try:
        app = TennisLineApp(video_source)
        
        # Apply custom parameters
        app.ball_detector.min_radius = args.min_ball_radius
        app.ball_detector.max_radius = args.max_ball_radius
        app.line_call_detector.line_tolerance = args.line_tolerance
        
        app.run()
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())