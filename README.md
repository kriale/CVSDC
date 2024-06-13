# CVSDC - Computer Vision Social Distance Control
This project is aimed at creating an algorithm for social distance monitoring through computer vision. The algorithm consists of three main stages: detection and tracking of people, coordinate transformation, and distance calculation between objects and violation detection.

## Directory Structure

- `datasets/` - This directory is used to store datasets for training and testing the algorithm.
- `results/` - This directory contains examples of the algorithm’s operation.
- `src/` - This directory contains the source code of the project.
- `test/` - This directory contains test scripts and datasets for checking the correctness and efficiency of the algorithm.
- `config.py` - This file is used to manage global project settings.
- `requirements.txt` - This file contains a list of all necessary dependencies for the project.

## Key Libraries Used

- OpenCV
- Ultralytics
- Supervision
- MotMetrics
- Numpy
- Matplotlib

## Implementation

In this implementation, the object tracking method is used for tracking. The Byte Track algorithm provided by the Supervision library was chosen for this purpose. For object detection, the YOLO detector provided by the Ultralytics library is used.

All the above functionalities are encapsulated in the `ByteTrackYOLOTracker` class. It provides the `update_tracker` method which is called for each frame of the video sequence to perform object detection on this frame and update the tracker data.

Functionality for transforming point coordinates between image and scene systems was encapsulated in the `CoordinatesConverter` class.

Social distance checking is implemented in the `SocialDistanceService` class. Its instance allows setting the threshold distance in meters through the `distance_threshold` parameter, which is set to 2 by default.