# Tennis Line Detection App - Quick Usage Guide

## What You've Got

I've created a complete tennis line detection application with the following components:

### Main Files:
- **`tennis_line_detector.py`** - Basic version with core functionality
- **`tennis_advanced.py`** - Enhanced version with trajectory prediction
- **`test_tennis_app.py`** - Test script with synthetic tennis ball simulation
- **`run_tennis_app.py`** - Easy launcher script
- **`config.py`** - Configuration settings
- **`requirements.txt`** - Dependencies
- **`setup.py`** - Installation script

## Quick Start

### 1. Basic Usage (Webcam)
```bash
python3 tennis_line_detector.py
```

### 2. Advanced Version (Webcam)
```bash
python3 tennis_advanced.py
```

### 3. Test with Simulation
```bash
python3 test_tennis_app.py
```

### 4. Use with Video File
```bash
python3 tennis_line_detector.py --source your_tennis_video.mp4
```

### 5. Easy Launcher (Auto-installs dependencies)
```bash
python3 run_tennis_app.py
```

## Key Features

### ✅ Real-time Ball Detection
- Detects yellow/green tennis balls using HSV color filtering
- Tracks ball movement with trail visualization
- Calculates ball velocity and trajectory

### ✅ Court Line Detection
- Automatically detects court lines using edge detection
- Identifies baselines, service lines, and sidelines
- Handles different court perspectives and lighting

### ✅ Line Call Determination
- **IN**: Ball lands clearly inside the court
- **OUT**: Ball lands clearly outside the court  
- **ON LINE**: Ball touches or overlaps with a line
- Confidence scoring for each call

### ✅ Advanced Features (tennis_advanced.py)
- Kalman filtering for smoother ball tracking
- Trajectory prediction for better accuracy
- Enhanced line detection with merging
- Debug mode with velocity vectors

## Controls

- **'q'** - Quit application
- **'d'** - Toggle debug mode (advanced version)
- **'r'** - Reset tracking
- **Space** - Pause/Resume

## Configuration

Edit `config.py` to adjust:
- Ball color detection ranges
- Court line detection sensitivity
- Display settings
- Performance parameters

## Example Output

When running, you'll see:
- Live video with ball tracking circle
- Ball movement trail
- Line call results ("IN", "OUT", "ON LINE")
- Confidence scores
- Ball velocity (advanced version)

## Troubleshooting

1. **No ball detected**: Adjust HSV color ranges in config.py
2. **Poor line detection**: Modify Canny/Hough parameters
3. **Performance issues**: Reduce frame resolution or processing frequency

The app works best with:
- Good lighting conditions
- Clear view of the tennis court
- Yellow/green tennis balls
- Stable camera position