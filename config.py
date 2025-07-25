"""
Configuration settings for Tennis Line Detection App
"""

# Ball Detection Settings
BALL_HSV_LOWER = [20, 50, 50]    # Lower HSV values for tennis ball color
BALL_HSV_UPPER = [35, 255, 255]  # Upper HSV values for tennis ball color
BALL_MIN_RADIUS = 5              # Minimum ball radius in pixels
BALL_MAX_RADIUS = 50             # Maximum ball radius in pixels
BALL_MIN_AREA = 20               # Minimum contour area for ball detection

# Court Line Detection Settings
CANNY_LOWER = 50                 # Lower threshold for Canny edge detection
CANNY_UPPER = 150                # Upper threshold for Canny edge detection
HOUGH_THRESHOLD = 100            # Threshold for Hough line detection
HOUGH_MIN_LINE_LENGTH = 50       # Minimum line length for Hough detection
HOUGH_MAX_LINE_GAP = 10          # Maximum gap between line segments

# Line Call Settings
LINE_TOLERANCE = 5               # Tolerance in pixels for line calls
HORIZONTAL_ANGLE_THRESHOLD = 30  # Max angle for horizontal lines (degrees)
VERTICAL_ANGLE_THRESHOLD = 60    # Min angle for vertical lines (degrees)

# Display Settings
BALL_TRAIL_LENGTH = 30           # Number of ball positions to keep in trail
CALL_DISPLAY_DURATION = 2        # Duration to display calls in seconds
COURT_DETECTION_INTERVAL = 30    # Frames between court line re-detection

# Camera Settings
CAMERA_WIDTH = 1280              # Default camera width
CAMERA_HEIGHT = 720              # Default camera height
CAMERA_FPS = 30                  # Default camera FPS

# Colors (BGR format)
COLOR_BALL = (255, 255, 0)       # Cyan for ball detection
COLOR_COURT_LINES = (0, 255, 0)  # Green for court lines
COLOR_BASELINE = (0, 0, 255)     # Red for baseline
COLOR_SERVICE_LINE = (255, 0, 0) # Blue for service line
COLOR_TRAIL = (255, 0, 255)      # Magenta for ball trail
COLOR_ON_LINE = (0, 255, 255)    # Yellow for "on line" calls
COLOR_OUT = (0, 0, 255)          # Red for "out" calls
COLOR_IN = (0, 255, 0)           # Green for "in" calls
COLOR_TEXT = (255, 255, 255)     # White for text

# Advanced Settings
ENABLE_MORPHOLOGY = True         # Enable morphological operations
MORPHOLOGY_KERNEL_SIZE = 5       # Kernel size for morphological operations
ENABLE_BALL_PREDICTION = False   # Enable ball trajectory prediction
PREDICTION_FRAMES = 5            # Number of frames to predict ahead