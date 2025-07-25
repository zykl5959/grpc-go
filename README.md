# Tennis Line Detection App

A computer vision application that detects whether a tennis ball hits on the line or outside the line using real-time video analysis.

## Features

### Core Functionality
- **Real-time ball detection** using color-based computer vision
- **Court line detection** using Hough Transform and edge detection
- **Line call determination** (IN, OUT, or ON LINE) with confidence scoring
- **Ball trajectory tracking** with movement trail visualization
- **Support for multiple video sources** (webcam, video files)

### Advanced Features (tennis_advanced.py)
- **Enhanced ball tracking** with Kalman filtering
- **Trajectory prediction** for better accuracy
- **Improved line detection** with line merging and classification
- **Real-time confidence scoring** for line calls
- **Debug mode** with velocity vectors and additional information

## Installation

### Requirements
- Python 3.7 or higher
- OpenCV 4.5.0 or higher
- NumPy 1.19.0 or higher

### Setup
1. Clone or download the project files
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage
Run the basic tennis line detection app:
```bash
python tennis_line_detector.py
```

Run with a video file:
```bash
python tennis_line_detector.py --video path/to/video.mp4
```

Run with custom parameters:
```bash
python tennis_line_detector.py --min-ball-radius 8 --max-ball-radius 40 --line-tolerance 10
```

### Advanced Usage
Run the advanced version with enhanced features:
```bash
python tennis_advanced.py
```

### Testing
Run the test suite with synthetic data:
```bash
python test_tennis_app.py
```

## Controls

### Keyboard Controls
- **Q**: Quit the application
- **C**: Recalibrate court lines
- **S**: Save current frame as image
- **R**: Reset detection and clear history
- **T**: Toggle ball trail visualization (advanced version)
- **P**: Toggle trajectory prediction (advanced version)
- **D**: Toggle debug mode (advanced version)

## Configuration

### Basic Parameters
Edit `config.py` to adjust detection parameters:

```python
# Ball Detection
BALL_HSV_LOWER = [20, 50, 50]    # Lower HSV threshold
BALL_HSV_UPPER = [35, 255, 255]  # Upper HSV threshold
BALL_MIN_RADIUS = 5              # Minimum ball radius
BALL_MAX_RADIUS = 50             # Maximum ball radius

# Line Detection
LINE_TOLERANCE = 5               # Line call tolerance (pixels)
HOUGH_THRESHOLD = 100            # Hough transform threshold

# Display
BALL_TRAIL_LENGTH = 30           # Ball trail history length
```

### Command Line Arguments

#### Basic Version
- `--video`: Video source (camera index or file path)
- `--min-ball-radius`: Minimum ball radius in pixels
- `--max-ball-radius`: Maximum ball radius in pixels
- `--line-tolerance`: Line detection tolerance in pixels

#### Advanced Version
- `--video`: Video source (camera index or file path)
- `--config`: Configuration file path

## How It Works

### 1. Court Line Detection
- Uses **Canny edge detection** to find edges in the image
- Applies **Hough Line Transform** to detect straight lines
- **Classifies lines** as baselines, service lines, or sidelines
- **Merges similar lines** to reduce noise (advanced version)

### 2. Ball Detection
- Converts image to **HSV color space**
- Creates a **color mask** for tennis ball (yellow-green)
- Finds **circular contours** that match ball characteristics
- **Tracks ball movement** over time

### 3. Line Call Detection
- Calculates **distance from ball to nearest line**
- Accounts for **ball radius** in line calls
- Determines if ball is **IN**, **OUT**, or **ON LINE**
- Provides **confidence scoring** for each call

### 4. Enhanced Features (Advanced Version)
- **Kalman filtering** for smoother ball tracking
- **Trajectory prediction** using motion history
- **Multi-algorithm line detection** for better accuracy
- **Velocity calculation** and visualization

## Output Interpretation

### Line Calls
- **IN**: Ball is within court boundaries
- **OUT**: Ball is clearly outside court boundaries
- **ON LINE**: Ball touches or overlaps with a court line
- **Confidence**: Score from 0.0 to 1.0 indicating call reliability

### Visual Indicators
- **Green lines**: Detected court lines
- **Red line**: Baseline (back boundary)
- **Blue line**: Service line
- **Cyan circle**: Detected ball with confidence score
- **Magenta trail**: Ball movement history
- **Yellow dots**: Predicted trajectory (advanced version)

## Troubleshooting

### Common Issues

#### Ball Not Detected
- Adjust HSV color range in `config.py`
- Check lighting conditions
- Ensure ball is the correct size range
- Try different `--min-ball-radius` and `--max-ball-radius` values

#### Lines Not Detected
- Ensure good contrast between lines and court surface
- Adjust `HOUGH_THRESHOLD` in configuration
- Try manual calibration with 'C' key
- Check camera angle and positioning

#### Inaccurate Line Calls
- Increase `LINE_TOLERANCE` for more lenient calls
- Ensure court lines are properly detected
- Check camera perspective and distortion

### Performance Tips
- Use good lighting conditions
- Position camera for clear court view
- Minimize background distractions
- Use high-contrast court surfaces

## Technical Details

### Algorithms Used
- **Canny Edge Detection**: For finding court line edges
- **Hough Line Transform**: For detecting straight lines
- **HSV Color Filtering**: For ball detection
- **Contour Analysis**: For ball shape validation
- **Kalman Filtering**: For ball tracking (advanced)
- **Morphological Operations**: For noise reduction

### Coordinate System
- Origin (0,0) is top-left corner
- X increases rightward
- Y increases downward
- All measurements in pixels

### Performance
- **Real-time processing** at 30 FPS on modern hardware
- **Adaptive detection** with periodic line recalibration
- **Memory efficient** with bounded history buffers

## File Structure

```
tennis-line-detection/
├── tennis_line_detector.py    # Basic version
├── tennis_advanced.py         # Advanced version with enhanced features
├── config.py                  # Configuration settings
├── test_tennis_app.py         # Test suite with synthetic data
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Future Enhancements

### Planned Features
- Machine learning-based ball detection
- Court perspective correction
- Multi-ball tracking
- Statistical analysis and reporting
- Integration with tennis scoring systems
- Mobile app version

### Contributing
Contributions are welcome! Areas for improvement:
- Better ball detection algorithms
- Enhanced court calibration methods
- Performance optimizations
- Additional sports support

## License

This project is open source and available under the MIT License.

## Acknowledgments

- OpenCV community for computer vision algorithms
- Tennis community for domain knowledge and testing
- Contributors and testers who helped improve accuracy
