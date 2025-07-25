#!/usr/bin/env python3
"""
Setup script for Tennis Line Detection App
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="tennis-line-detector",
    version="1.0.0",
    author="Tennis Line Detection Team",
    author_email="contact@tennislinedetector.com",
    description="A computer vision app that detects tennis ball line calls",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tennis-line-detector/tennis-line-detector",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Sports/Athletics",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Computer Vision",
        "Topic :: Multimedia :: Video :: Analysis",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
        ],
        "gpu": [
            "opencv-contrib-python>=4.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "tennis-detector=tennis_line_detector:main",
            "tennis-detector-advanced=tennis_advanced:main",
            "tennis-test=test_tennis_app:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)