#!/usr/bin/env python3
"""
Tennis Line Detection App Launcher
Simple script to check dependencies and run the tennis line detection app.
"""

import sys
import subprocess
import importlib.util
import os

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 7):
        print("Error: Python 3.7 or higher is required.")
        print(f"Current version: {sys.version}")
        return False
    return True

def check_package(package_name, install_name=None):
    """Check if a package is installed."""
    if install_name is None:
        install_name = package_name
    
    spec = importlib.util.find_spec(package_name)
    if spec is None:
        print(f"Package '{package_name}' not found.")
        response = input(f"Would you like to install {install_name}? (y/n): ").lower()
        if response == 'y':
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", install_name])
                print(f"Successfully installed {install_name}")
                return True
            except subprocess.CalledProcessError:
                print(f"Failed to install {install_name}")
                return False
        return False
    return True

def check_dependencies():
    """Check all required dependencies."""
    dependencies = [
        ("cv2", "opencv-python"),
        ("numpy", "numpy"),
    ]
    
    print("Checking dependencies...")
    all_good = True
    
    for package, install_name in dependencies:
        if not check_package(package, install_name):
            all_good = False
    
    return all_good

def run_app(app_type="basic", video_source=None):
    """Run the tennis line detection app."""
    if app_type == "advanced":
        script = "tennis_advanced.py"
    elif app_type == "test":
        script = "test_tennis_app.py"
    else:
        script = "tennis_line_detector.py"
    
    if not os.path.exists(script):
        print(f"Error: {script} not found in current directory.")
        return False
    
    cmd = [sys.executable, script]
    if video_source is not None:
        cmd.extend(["--video", str(video_source)])
    
    try:
        subprocess.run(cmd)
        return True
    except KeyboardInterrupt:
        print("\nApplication interrupted by user.")
        return True
    except Exception as e:
        print(f"Error running application: {e}")
        return False

def main():
    """Main launcher function."""
    print("Tennis Line Detection App Launcher")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check dependencies
    if not check_dependencies():
        print("\nSome dependencies are missing. Please install them manually:")
        print("pip install opencv-python numpy")
        sys.exit(1)
    
    print("\nAll dependencies are satisfied!")
    
    # Show menu
    while True:
        print("\nChoose an option:")
        print("1. Run basic tennis line detector")
        print("2. Run advanced tennis line detector")
        print("3. Run test with simulated data")
        print("4. Run with video file")
        print("5. Run with camera (specify camera index)")
        print("6. Exit")
        
        choice = input("\nEnter choice (1-6): ").strip()
        
        if choice == "1":
            print("Starting basic tennis line detector...")
            run_app("basic")
        
        elif choice == "2":
            print("Starting advanced tennis line detector...")
            run_app("advanced")
        
        elif choice == "3":
            print("Starting test with simulated data...")
            run_app("test")
        
        elif choice == "4":
            video_file = input("Enter path to video file: ").strip()
            if os.path.exists(video_file):
                print(f"Starting with video file: {video_file}")
                run_app("basic", video_file)
            else:
                print("Video file not found!")
        
        elif choice == "5":
            camera_idx = input("Enter camera index (0 for default camera): ").strip()
            try:
                camera_idx = int(camera_idx)
                print(f"Starting with camera {camera_idx}...")
                run_app("basic", camera_idx)
            except ValueError:
                print("Invalid camera index!")
        
        elif choice == "6":
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice. Please enter 1-6.")

if __name__ == "__main__":
    main()