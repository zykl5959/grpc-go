#!/usr/bin/env python3
"""
Advanced Tennis Line Detection App
Enhanced version with trajectory prediction and improved accuracy.
"""

import cv2
import numpy as np
import argparse
import time
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Optional
import math
from config import *

@dataclass
class Point:
    x: float
    y: float
    
    def distance_to(self, other: 'Point') -> float:
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

@dataclass
class BallDetection:
    center: Point
    radius: float
    confidence: float
    timestamp: float
    velocity: Optional[Point] = None

class TennisCourtDetector:
    """Enhanced tennis court line detector with better accuracy."""
    
    def __init__(self):
        self.court_lines = []
        self.baseline_y = None
        self.service_line_y = None
        self.sideline_left_x = None
        self.sideline_right_x = None
        self.center_line_x = None
        self.court_corners = []
        self.court_calibrated = False
        
    def detect_court_lines(self, frame):
        """Enhanced court line detection with multiple algorithms."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply adaptive thresholding for better line detection
        adaptive_thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                              cv2.THRESH_BINARY, 11, 2)
        
        # Edge detection
        edges = cv2.Canny(blurred, CANNY_LOWER, CANNY_UPPER, apertureSize=3)
        
        # Combine adaptive threshold and edge detection
        combined = cv2.bitwise_or(edges, adaptive_thresh)
        
        # Apply morphological operations
        if ENABLE_MORPHOLOGY:
            kernel = np.ones((3, 3), np.uint8)
            combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        
        # Detect lines using HoughLinesP
        lines = cv2.HoughLinesP(combined, 1, np.pi/180, 
                               threshold=HOUGH_THRESHOLD,
                               minLineLength=HOUGH_MIN_LINE_LENGTH, 
                               maxLineGap=HOUGH_MAX_LINE_GAP)
        
        self.court_lines = []
        if lines is not None:
            # Filter and merge similar lines
            filtered_lines = self._filter_lines(lines)
            for line in filtered_lines:
                x1, y1, x2, y2 = line
                self.court_lines.append(((x1, y1), (x2, y2)))
                
        self._classify_lines(frame.shape)
        return self.court_lines
    
    def _filter_lines(self, lines):
        """Filter and merge similar lines."""
        if len(lines) == 0:
            return []
        
        # Group lines by angle and position
        horizontal_lines = []
        vertical_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = math.atan2(abs(y2 - y1), abs(x2 - x1)) * 180 / math.pi
            
            if angle < HORIZONTAL_ANGLE_THRESHOLD:
                horizontal_lines.append(line[0])
            elif angle > VERTICAL_ANGLE_THRESHOLD:
                vertical_lines.append(line[0])
        
        # Merge nearby lines
        merged_horizontal = self._merge_similar_lines(horizontal_lines, is_horizontal=True)
        merged_vertical = self._merge_similar_lines(vertical_lines, is_horizontal=False)
        
        return merged_horizontal + merged_vertical
    
    def _merge_similar_lines(self, lines, is_horizontal=True, threshold=20):
        """Merge lines that are close to each other."""
        if not lines:
            return []
        
        merged = []
        used = set()
        
        for i, line1 in enumerate(lines):
            if i in used:
                continue
                
            x1, y1, x2, y2 = line1
            group = [line1]
            used.add(i)
            
            for j, line2 in enumerate(lines):
                if j in used or j <= i:
                    continue
                
                x3, y3, x4, y4 = line2
                
                if is_horizontal:
                    # Check if y-coordinates are similar
                    if abs((y1 + y2) / 2 - (y3 + y4) / 2) < threshold:
                        group.append(line2)
                        used.add(j)
                else:
                    # Check if x-coordinates are similar
                    if abs((x1 + x2) / 2 - (x3 + x4) / 2) < threshold:
                        group.append(line2)
                        used.add(j)
            
            # Merge lines in group
            if group:
                merged_line = self._merge_line_group(group)
                merged.append(merged_line)
        
        return merged
    
    def _merge_line_group(self, group):
        """Merge a group of similar lines into one."""
        if len(group) == 1:
            return group[0]
        
        # Find the extents of all lines in the group
        all_x = []
        all_y = []
        
        for line in group:
            x1, y1, x2, y2 = line
            all_x.extend([x1, x2])
            all_y.extend([y1, y2])
        
        # Return line with maximum extents
        return [min(all_x), int(np.mean(all_y)), max(all_x), int(np.mean(all_y))]
    
    def _classify_lines(self, frame_shape):
        """Enhanced line classification."""
        h, w = frame_shape[:2]
        
        horizontal_lines = []
        vertical_lines = []
        
        for line in self.court_lines:
            (x1, y1), (x2, y2) = line
            angle = math.atan2(abs(y2 - y1), abs(x2 - x1)) * 180 / math.pi
            
            if angle < HORIZONTAL_ANGLE_THRESHOLD:
                horizontal_lines.append(line)
            elif angle > VERTICAL_ANGLE_THRESHOLD:
                vertical_lines.append(line)
        
        # Sort and classify horizontal lines
        horizontal_lines.sort(key=lambda line: (line[0][1] + line[1][1]) / 2)
        
        if len(horizontal_lines) >= 1:
            # Bottom line is likely baseline
            self.baseline_y = (horizontal_lines[-1][0][1] + horizontal_lines[-1][1][1]) / 2
            
            # Service line is typically before baseline
            if len(horizontal_lines) >= 2:
                self.service_line_y = (horizontal_lines[-2][0][1] + horizontal_lines[-2][1][1]) / 2
        
        # Sort and classify vertical lines
        vertical_lines.sort(key=lambda line: (line[0][0] + line[1][0]) / 2)
        
        if len(vertical_lines) >= 2:
            self.sideline_left_x = (vertical_lines[0][0][0] + vertical_lines[0][1][0]) / 2
            self.sideline_right_x = (vertical_lines[-1][0][0] + vertical_lines[-1][1][0]) / 2
            
            # Center line
            if len(vertical_lines) >= 3:
                center_idx = len(vertical_lines) // 2
                self.center_line_x = (vertical_lines[center_idx][0][0] + 
                                    vertical_lines[center_idx][1][0]) / 2

class TennisBallTracker:
    """Enhanced ball tracker with velocity and trajectory prediction."""
    
    def __init__(self):
        # HSV range for tennis ball
        self.lower_hsv = np.array(BALL_HSV_LOWER)
        self.upper_hsv = np.array(BALL_HSV_UPPER)
        
        # Detection parameters
        self.min_radius = BALL_MIN_RADIUS
        self.max_radius = BALL_MAX_RADIUS
        self.min_area = BALL_MIN_AREA
        
        # Tracking
        self.ball_history = deque(maxlen=BALL_TRAIL_LENGTH)
        self.kalman_filter = self._initialize_kalman_filter()
        
    def _initialize_kalman_filter(self):
        """Initialize Kalman filter for ball tracking."""
        kalman = cv2.KalmanFilter(4, 2)
        kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                           [0, 1, 0, 0]], np.float32)
        kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                          [0, 1, 0, 1],
                                          [0, 0, 1, 0],
                                          [0, 0, 0, 1]], np.float32)
        kalman.processNoiseCov = 0.03 * np.eye(4, dtype=np.float32)
        return kalman
        
    def detect_ball(self, frame) -> Optional[BallDetection]:
        """Enhanced ball detection with multiple methods."""
        # Method 1: Color-based detection
        ball_color = self._detect_by_color(frame)
        
        # Method 2: Template matching (if we have a good ball template)
        # ball_template = self._detect_by_template(frame)
        
        # Method 3: Machine learning-based detection (future enhancement)
        # ball_ml = self._detect_by_ml(frame)
        
        # For now, use color-based detection
        ball = ball_color
        
        if ball:
            # Calculate velocity if we have history
            if len(self.ball_history) > 0:
                prev_ball = self.ball_history[-1]
                dt = ball.timestamp - prev_ball.timestamp
                if dt > 0:
                    vx = (ball.center.x - prev_ball.center.x) / dt
                    vy = (ball.center.y - prev_ball.center.y) / dt
                    ball.velocity = Point(vx, vy)
            
            # Update Kalman filter
            measurement = np.array([[ball.center.x], [ball.center.y]], dtype=np.float32)
            self.kalman_filter.correct(measurement)
            
            # Add to history
            self.ball_history.append(ball)
        
        return ball
    
    def _detect_by_color(self, frame) -> Optional[BallDetection]:
        """Color-based ball detection."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create mask
        mask = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)
        
        # Apply morphological operations
        if ENABLE_MORPHOLOGY:
            kernel = np.ones((MORPHOLOGY_KERNEL_SIZE, MORPHOLOGY_KERNEL_SIZE), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Filter contours by area and shape
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= self.min_area:
                # Check circularity
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * math.pi * area / (perimeter * perimeter)
                    if circularity > 0.5:  # Reasonably circular
                        valid_contours.append((contour, area, circularity))
        
        if not valid_contours:
            return None
        
        # Select best contour (highest circularity * area score)
        best_contour = max(valid_contours, key=lambda x: x[1] * x[2])
        contour, area, circularity = best_contour
        
        # Get circle parameters
        (x, y), radius = cv2.minEnclosingCircle(contour)
        
        if self.min_radius <= radius <= self.max_radius:
            return BallDetection(
                center=Point(x, y),
                radius=radius,
                confidence=min(circularity, 1.0),
                timestamp=time.time()
            )
        
        return None
    
    def predict_trajectory(self, frames_ahead=PREDICTION_FRAMES) -> List[Point]:
        """Predict ball trajectory for the next few frames."""
        if len(self.ball_history) < 2:
            return []
        
        if not ENABLE_BALL_PREDICTION:
            return []
        
        # Use Kalman filter for prediction
        predictions = []
        for i in range(frames_ahead):
            prediction = self.kalman_filter.predict()
            pred_point = Point(float(prediction[0]), float(prediction[1]))
            predictions.append(pred_point)
        
        return predictions

class LineCallDetector:
    """Enhanced line call detection with improved accuracy."""
    
    def __init__(self, court_detector: TennisCourtDetector):
        self.court_detector = court_detector
        self.line_tolerance = LINE_TOLERANCE
        self.call_history = deque(maxlen=10)
        
    def make_call(self, ball: BallDetection) -> Tuple[str, str, float]:
        """
        Make a line call for the ball.
        Returns (call, line_type, confidence)
        """
        if not ball:
            return "UNKNOWN", "", 0.0
        
        x, y = ball.center.x, ball.center.y
        radius = ball.radius
        
        # Check each line type
        line_distances = self._calculate_line_distances(ball)
        
        # Find closest line
        closest_line = None
        min_distance = float('inf')
        
        for line_type, distance in line_distances.items():
            if distance < min_distance:
                min_distance = distance
                closest_line = line_type
        
        # Determine call based on distance
        if min_distance <= self.line_tolerance + radius:
            call = "ON LINE"
            confidence = max(0.5, 1.0 - (min_distance / (self.line_tolerance + radius)))
        elif self._is_ball_out(ball):
            call = "OUT"
            confidence = 0.8
        else:
            call = "IN"
            confidence = 0.7
        
        # Add to history for consistency checking
        self.call_history.append((call, closest_line, confidence))
        
        return call, closest_line or "", confidence
    
    def _calculate_line_distances(self, ball: BallDetection) -> dict:
        """Calculate distances from ball to all court lines."""
        x, y = ball.center.x, ball.center.y
        distances = {}
        
        if self.court_detector.baseline_y is not None:
            distances['baseline'] = abs(y - self.court_detector.baseline_y)
        
        if self.court_detector.service_line_y is not None:
            distances['service_line'] = abs(y - self.court_detector.service_line_y)
        
        if self.court_detector.sideline_left_x is not None:
            distances['left_sideline'] = abs(x - self.court_detector.sideline_left_x)
        
        if self.court_detector.sideline_right_x is not None:
            distances['right_sideline'] = abs(x - self.court_detector.sideline_right_x)
        
        if self.court_detector.center_line_x is not None:
            distances['center_line'] = abs(x - self.court_detector.center_line_x)
        
        return distances
    
    def _is_ball_out(self, ball: BallDetection) -> bool:
        """Check if ball is clearly out of bounds."""
        x, y = ball.center.x, ball.center.y
        tolerance = self.line_tolerance
        
        # Check baseline
        if self.court_detector.baseline_y is not None:
            if y > self.court_detector.baseline_y + tolerance:
                return True
        
        # Check sidelines
        if self.court_detector.sideline_left_x is not None:
            if x < self.court_detector.sideline_left_x - tolerance:
                return True
        
        if self.court_detector.sideline_right_x is not None:
            if x > self.court_detector.sideline_right_x + tolerance:
                return True
        
        return False

class AdvancedTennisApp:
    """Advanced tennis line detection application."""
    
    def __init__(self, video_source=0):
        self.video_source = video_source
        self.cap = None
        
        # Initialize components
        self.court_detector = TennisCourtDetector()
        self.ball_tracker = TennisBallTracker()
        self.line_call_detector = LineCallDetector(self.court_detector)
        
        # UI state
        self.last_call = ""
        self.call_timestamp = 0
        self.show_trail = True
        self.show_predictions = True
        self.show_debug = False
        
    def initialize_camera(self):
        """Initialize camera with optimal settings."""
        self.cap = cv2.VideoCapture(self.video_source)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video source: {self.video_source}")
        
        if isinstance(self.video_source, int):
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
            self.cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
    
    def draw_court_visualization(self, frame):
        """Draw court lines and zones."""
        # Draw all detected lines
        for line in self.court_detector.court_lines:
            (x1, y1), (x2, y2) = line
            cv2.line(frame, (x1, y1), (x2, y2), COLOR_COURT_LINES, 1)
        
        # Highlight important lines
        if self.court_detector.baseline_y is not None:
            y = int(self.court_detector.baseline_y)
            cv2.line(frame, (0, y), (frame.shape[1], y), COLOR_BASELINE, 3)
            cv2.putText(frame, "Baseline", (10, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_BASELINE, 2)
        
        if self.court_detector.service_line_y is not None:
            y = int(self.court_detector.service_line_y)
            cv2.line(frame, (0, y), (frame.shape[1], y), COLOR_SERVICE_LINE, 2)
            cv2.putText(frame, "Service", (10, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_SERVICE_LINE, 2)
        
        # Draw sidelines
        if self.court_detector.sideline_left_x is not None:
            x = int(self.court_detector.sideline_left_x)
            cv2.line(frame, (x, 0), (x, frame.shape[0]), COLOR_COURT_LINES, 2)
        
        if self.court_detector.sideline_right_x is not None:
            x = int(self.court_detector.sideline_right_x)
            cv2.line(frame, (x, 0), (x, frame.shape[0]), COLOR_COURT_LINES, 2)
    
    def draw_ball_visualization(self, frame, ball: BallDetection):
        """Draw ball detection and related information."""
        if not ball:
            return
        
        center = (int(ball.center.x), int(ball.center.y))
        radius = int(ball.radius)
        
        # Draw ball
        cv2.circle(frame, center, radius, COLOR_BALL, 2)
        cv2.circle(frame, center, 2, COLOR_BALL, -1)
        
        # Draw confidence
        cv2.putText(frame, f"Conf: {ball.confidence:.2f}", 
                   (center[0] - 40, center[1] - radius - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_BALL, 1)
        
        # Draw velocity if available
        if ball.velocity and self.show_debug:
            vel_end = (int(center[0] + ball.velocity.x * 0.1), 
                      int(center[1] + ball.velocity.y * 0.1))
            cv2.arrowedLine(frame, center, vel_end, (0, 255, 255), 2)
    
    def draw_trajectory_prediction(self, frame):
        """Draw predicted ball trajectory."""
        if not self.show_predictions:
            return
        
        predictions = self.ball_tracker.predict_trajectory()
        for i, pred in enumerate(predictions):
            alpha = 1.0 - (i / len(predictions))
            radius = max(2, int(5 * alpha))
            cv2.circle(frame, (int(pred.x), int(pred.y)), radius, 
                      (255, 255, 0), -1)
    
    def draw_ball_trail(self, frame):
        """Draw ball movement trail."""
        if not self.show_trail or len(self.ball_tracker.ball_history) < 2:
            return
        
        points = [(int(ball.center.x), int(ball.center.y)) 
                 for ball in self.ball_tracker.ball_history]
        
        for i in range(1, len(points)):
            alpha = i / len(points)
            thickness = max(1, int(3 * alpha))
            cv2.line(frame, points[i-1], points[i], COLOR_TRAIL, thickness)
    
    def draw_line_call(self, frame, ball: BallDetection):
        """Draw line call with enhanced visualization."""
        if not ball:
            return
        
        call, line_type, confidence = self.line_call_detector.make_call(ball)
        
        # Update display call
        current_time = time.time()
        if call != "IN" or current_time - self.call_timestamp > CALL_DISPLAY_DURATION:
            self.last_call = f"{call}"
            if line_type:
                self.last_call += f" ({line_type})"
            self.call_timestamp = current_time
        
        # Choose color based on call
        if "ON LINE" in call:
            color = COLOR_ON_LINE
        elif call == "OUT":
            color = COLOR_OUT
        else:
            color = COLOR_IN
        
        # Draw call
        cv2.putText(frame, self.last_call, (50, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3)
        
        # Draw confidence
        cv2.putText(frame, f"Confidence: {confidence:.2f}", (50, 140),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    def draw_ui(self, frame):
        """Draw user interface elements."""
        # Title
        cv2.putText(frame, "Advanced Tennis Line Detector", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_TEXT, 2)
        
        # Controls
        controls = [
            "Controls: Q-Quit, C-Calibrate, S-Save, R-Reset",
            "T-Toggle Trail, P-Toggle Predictions, D-Debug"
        ]
        
        for i, control in enumerate(controls):
            cv2.putText(frame, control, (10, frame.shape[0] - 40 + i * 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1)
    
    def run(self):
        """Main application loop."""
        self.initialize_camera()
        
        print("Advanced Tennis Line Detection App")
        print("Press 'q' to quit")
        print("Press 'c' to calibrate court")
        print("Press 's' to save frame")
        print("Press 'r' to reset detection")
        print("Press 't' to toggle ball trail")
        print("Press 'p' to toggle predictions")
        print("Press 'd' to toggle debug mode")
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read frame")
                    break
                
                frame_count += 1
                
                # Detect court lines periodically
                if frame_count == 1 or frame_count % COURT_DETECTION_INTERVAL == 0:
                    self.court_detector.detect_court_lines(frame)
                
                # Detect ball
                ball = self.ball_tracker.detect_ball(frame)
                
                # Draw visualizations
                self.draw_court_visualization(frame)
                self.draw_ball_visualization(frame, ball)
                
                if ball:
                    self.draw_line_call(frame, ball)
                
                self.draw_ball_trail(frame)
                self.draw_trajectory_prediction(frame)
                self.draw_ui(frame)
                
                # Show frame
                cv2.imshow('Advanced Tennis Line Detection', frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    print("Recalibrating court...")
                    self.court_detector.detect_court_lines(frame)
                elif key == ord('s'):
                    filename = f"tennis_advanced_{int(time.time())}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"Saved frame as {filename}")
                elif key == ord('r'):
                    print("Resetting detection...")
                    self.court_detector.__init__()
                    self.ball_tracker.ball_history.clear()
                elif key == ord('t'):
                    self.show_trail = not self.show_trail
                    print(f"Ball trail: {'ON' if self.show_trail else 'OFF'}")
                elif key == ord('p'):
                    self.show_predictions = not self.show_predictions
                    print(f"Predictions: {'ON' if self.show_predictions else 'OFF'}")
                elif key == ord('d'):
                    self.show_debug = not self.show_debug
                    print(f"Debug mode: {'ON' if self.show_debug else 'OFF'}")
        
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
    parser = argparse.ArgumentParser(description='Advanced Tennis Line Detection App')
    parser.add_argument('--video', type=str, default=0,
                       help='Video source (camera index or video file path)')
    parser.add_argument('--config', type=str, 
                       help='Configuration file path')
    
    args = parser.parse_args()
    
    # Convert video argument
    video_source = args.video
    if isinstance(video_source, str) and video_source.isdigit():
        video_source = int(video_source)
    
    try:
        app = AdvancedTennisApp(video_source)
        app.run()
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())