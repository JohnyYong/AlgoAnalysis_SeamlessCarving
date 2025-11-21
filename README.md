# Seam Carver – Content-Aware Image Resizing

Team 25 · SIT Assignment 2

This project implements Seam Carving, a content-aware image resizing technique that removes the least important pixels (low-energy seams) to shrink images while preserving key visual content.

Features include dynamic programming seam selection, greedy seam selection, full visualization, timing breakdowns, and CI/CD automation.

## Features
- Dynamic Programming seam carving (globally optimal seams)
- Greedy seam carving (local minimum approach)
- Vertical and horizontal seam removal
- Energy computation using Sobel filters
- Pixel-by-pixel seam visualization
- Interactive or command-line execution
- Automatic build & test through GitHub Actions
- Cross-platform support (Windows, Linux, macOS)

## Project Structure
Assignment2_SeamCarving/

├── main.cpp

├── SeamCarver.cpp

├── SeamCarver.h

└── input.jpg     (optional test file)

## Build Instructions
Windows (Visual Studio 2022)
1. Install OpenCV
2. Configure Visual Studio Project
3. Add Include / Lib Paths

Linux (Ubuntu / Debian)
1. Install OpenCV
2. Build
3. Run

macOS (Homebrew)
1. Install OpenCV
2. Build
3. Run

## Usage
#### Run with arguments:

./seamcarver <inputPath> <targetWidth> <targetHeight>

Example:

./seamcarver dog.jpg 600 400

#### Interactive mode (no arguments):

Input the file path:

Input target width:

Input target height:

Output

A resized image will be saved to: output.jpg

## Seam Visualization

Each seam is displayed in red on a clone of the image before removal.
Visualization windows update every seam to show progress.

## Configuration Options
Switch between DP and Greedy

Inside SeamCarver.h:

#define USE_DP       // enable dynamic programming

// comment it out to use greedy algorithm

Disable visualization

Remove -DVISUALISE from the compiler flags or comment out related code.

## CI/CD (GitHub Actions)
Included workflow performs:
- OpenCV installation
- Automated building
- Optional sample execution
- Packaging into SeamCarverFile.zip
- Uploading build artifacts

Workflow file:

.github/workflows/SeamCarver.yml

## License
This project is developed for SIT coursework (Assignment 2) and is not intended for external redistribution.

## Acknowledgements
Avidan & Shamir — Seam Carving research

OpenCV team — image processing libraries
