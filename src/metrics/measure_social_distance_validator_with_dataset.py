import numpy as np

from config import PREDICTIONS_FILE_PATH, DISTANCE_THRESHOLD, LAST_FRAMES, VIOLATION_PERCENTAGE, \
    SOCIAL_DISTANCE_METRICS_RESULTS_PATH
from src.service.coordinates_converter import CoordinatesConverter
from src.service.mot_metrics_service import MotMetricsService
from src.service.social_distance_service import PeopleCoordinates, SocialDistanceService
from src.service.towncentre_video_service import TowncentreVideoService
from src.utils.detection_utils import DetectionUtils
import pandas as pd


def measure():
    dataset_service = TowncentreVideoService()
    coordinates_converter = CoordinatesConverter(dataset_service.get_camera_parameters())
    true_detections_dict = dataset_service.get_groundtruth_dict()
    pred_detections_dict = DetectionUtils.read_predictions_dict(PREDICTIONS_FILE_PATH)

    distance_service_pred = SocialDistanceService(DISTANCE_THRESHOLD, LAST_FRAMES, VIOLATION_PERCENTAGE)
    distance_service_true = SocialDistanceService(DISTANCE_THRESHOLD, LAST_FRAMES, VIOLATION_PERCENTAGE)

    true_violators_dict = get_violating_detections_dict(
        true_detections_dict, distance_service_true, coordinates_converter
    )
    pred_violators_dict = get_violating_detections_dict(
        pred_detections_dict, distance_service_pred, coordinates_converter
    )

    mot_metrics_service = MotMetricsService(
        groundtruth_dict=true_violators_dict,
        predictions_dict=pred_violators_dict
    )
    print('MOTA, MOTP for violators only: ', mot_metrics_service.calculate_mot_metrics())


def measure_matrix():
    dataset_service = TowncentreVideoService()
    coordinates_converter = CoordinatesConverter(dataset_service.get_camera_parameters())
    true_detections_dict = dataset_service.get_groundtruth_dict()
    pred_detections_dict = DetectionUtils.read_predictions_dict(PREDICTIONS_FILE_PATH)

    results = []

    for last_frames in range(1, 11):
        for violation_percentage in np.arange(0.1, 1.1, 0.1):
            distance_service_pred = SocialDistanceService(DISTANCE_THRESHOLD, last_frames, violation_percentage)
            distance_service_true = SocialDistanceService(DISTANCE_THRESHOLD, last_frames, violation_percentage)

            true_violators_dict = get_violating_detections_dict(
                true_detections_dict, distance_service_true, coordinates_converter
            )
            pred_violators_dict = get_violating_detections_dict(
                pred_detections_dict, distance_service_pred, coordinates_converter
            )

            mot_metrics_service = MotMetricsService(
                groundtruth_dict=true_violators_dict,
                predictions_dict=pred_violators_dict
            )
            mota, motp = mot_metrics_service.calculate_mot_metrics()

            results.append([last_frames, violation_percentage, mota, motp])

    df = pd.DataFrame(results, columns=['last_frames', 'violation_percentage', 'MOTA', 'MOTP'])
    df.to_csv(SOCIAL_DISTANCE_METRICS_RESULTS_PATH, index=False)


def get_violating_detections_dict(
        detections_dict: dict,
        distance_service: SocialDistanceService,
        coordinates_converter: CoordinatesConverter
    ) -> dict:
    violators_dict = {}
    for frame_index, detections in detections_dict.items():
        # Convert detections to PeopleCoordinates
        people_coordinates = PeopleCoordinates(
            object_ids=np.array([detection[0] for detection in detections]),
            sb_xy=np.array([coordinates_converter.convert_coordinates_to_scene(detection[5:7]) for detection in detections])
        )

        # Update violation pairs for the frame
        distance_service.update_violation_pairs(frame_index, people_coordinates)

        # Filter detections to include only the violators
        violators = [detection for detection in detections if detection[0] in distance_service.get_all_current_violators_set(frame_index)]
        if violators:
            violators_dict[frame_index] = violators

    return violators_dict


if __name__ == '__main__':
    measure()