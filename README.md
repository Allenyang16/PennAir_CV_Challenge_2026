# Penn Air Computer Vision Challenge

This repository contains my implementation for the Penn Air Computer Vision application challenge.

The goal of the project is to develop a computer vision algorithm that detects geometric shapes in images and video, traces their contours, and estimates their center positions.

## Approach

The project uses Python and OpenCV for image processing and shape detection.

The general detection pipeline is:

1. Read an image or a frame from the video.
2. Convert the image from BGR to HSV color space.
3. Generate binary masks using brightness and saturation thresholds.
4. Detect candidate shapes using `cv.findContours()`.
5. Filter small contours based on contour area.
6. Calculate the center of each detected shape using image moments.
7. Draw the detected contours and center coordinates onto the frame.

For the harder background-agnostic task, additional processing is used to handle color gradients and shapes whose pixels cannot be captured by a single HSV threshold.

For example, when a gradient causes one shape to be detected as multiple disconnected contours, nearby contour fragments can be combined and reconstructed using a convex hull.

## Current Progress

### Part 1 — Static Image Detection
- Detect shapes on the provided grassy background.
- Trace shape contours.
- Calculate and display shape centers.

### Part 2 — Video Detection
- Process the video as a stream of individual frames.
- Apply the static detection algorithm to each frame.
- Display detected contours and centers in real time.

### Part 3 — Background-Agnostic Detection
Work in progress.

Current experiments include:
- Combining multiple HSV masks.
- Filtering background noise using contour area.
- Handling gradient-colored shapes.
- Merging disconnected contour fragments.
- Reconstructing shapes using convex hulls and polygon approximation.

## Requirements

- Python
- OpenCV
- NumPy

Install the required packages with:

```bash
pip install opencv-python numpy