import os

# Get the path of the directory where the current file is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Main directories
DATASETS_PATH = os.path.join(BASE_DIR, 'datasets')
DEMO_DIR_PATH = os.path.join(BASE_DIR, 'demo')
RESOURCES_DIR_PATH = os.path.join(BASE_DIR, 'resources')

# TownCentre
TOWNCENTRE_DATASET_PATH = os.path.join(DATASETS_PATH, 'TownCentre')
GROUNDTRUTH_FILE_PATH = os.path.join(DATASETS_PATH, 'TownCentre-groundtruth.top')
CAMERA_PARAMETERS_FILE_PATH = os.path.join(DATASETS_PATH, 'TownCentre-calibration.ci')
VIDEO_PATH = os.path.join(DATASETS_PATH, 'TownCentreXVID.mp4')

# YOLO
YOLO_DIR_PATH = os.path.join(RESOURCES_DIR_PATH, 'yolo')
YOLO_V8_N_PATH = os.path.join(YOLO_DIR_PATH, 'yolov8n.pt')
YOLO_V8_X_PATH = os.path.join(YOLO_DIR_PATH, 'yolov8x.pt')
