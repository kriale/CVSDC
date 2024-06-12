import os

# Get the path of the directory where the current file is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Main directories
DATASETS_PATH = os.path.join(BASE_DIR, 'datasets')
RESULTS_DIR_PATH = os.path.join(BASE_DIR, 'results')
RESOURCES_DIR_PATH = os.path.join(BASE_DIR, 'resources')

# TownCentre
TOWNCENTRE_DATASET_PATH = os.path.join(DATASETS_PATH, 'TownCentre')
GROUNDTRUTH_FILE_PATH = os.path.join(TOWNCENTRE_DATASET_PATH, 'TownCentre-groundtruth.top')
PREDICTIONS_FILE_PATH = os.path.join(RESULTS_DIR_PATH, 'TownCentre-predictions.top')
CAMERA_PARAMETERS_FILE_PATH = os.path.join(TOWNCENTRE_DATASET_PATH, 'TownCentre-calibration.ci')
VIDEO_PATH = os.path.join(TOWNCENTRE_DATASET_PATH, 'TownCentreXVID.mp4')

# YOLO
YOLO_DIR_PATH = os.path.join(RESOURCES_DIR_PATH, 'yolo')
YOLO_V8_N_PATH = os.path.join(YOLO_DIR_PATH, 'yolov8n.pt')
YOLO_V8_X_PATH = os.path.join(YOLO_DIR_PATH, 'yolov8x.pt')

# Social Distance Control Parameters
DISTANCE_THRESHOLD = 2
LAST_FRAMES = 8
VIOLATION_PERCENTAGE = 0.8 # 4/5 frames to consider as violation
SOCIAL_DISTANCE_METRICS_RESULTS_PATH = os.path.join(RESULTS_DIR_PATH, 'social_distance_metrics.csv')
