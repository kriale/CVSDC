import os

import numpy as np

from config import DISTANCE_THRESHOLD, VIOLATION_PERCENTAGE, LAST_FRAMES, RESULTS_DIR_PATH, PREDICTIONS_FILE_PATH
from src.service.coordinates_converter import CoordinatesConverter
from src.service.social_distance_service import SocialDistanceService, PeopleCoordinates
from src.service.towncentre_video_service import TowncentreVideoService
from supervision import Color, Position
import supervision as sv
import cv2

from src.utils.detection_utils import DetectionUtils


def demo(result_video_path='demo_social_distance_monitoring_with_dataset.mp4'):
    dataset_service = TowncentreVideoService()
    # true_detections_dict = dataset_service.get_groundtruth_dict()
    pred_detections_dict = DetectionUtils.read_predictions_dict(PREDICTIONS_FILE_PATH)

    distance_service = SocialDistanceService(DISTANCE_THRESHOLD, LAST_FRAMES, VIOLATION_PERCENTAGE)
    coordinates_converter = CoordinatesConverter(dataset_service.get_camera_parameters())

    pred_people_annotator = sv.EllipseAnnotator(thickness=2, start_angle=-45, end_angle=235,
                                                color=Color(r=0, g=50, b=200))
    pred_dot_annotator = sv.DotAnnotator(color=Color(r=0, g=50, b=200), radius=3, position=Position.BOTTOM_CENTER)
    pred_box_annotator = sv.BoxAnnotator(thickness=4, text_thickness=2, text_scale=1,
                                         color=Color(r=0, g=50, b=255))

    def callback(frame: np.ndarray, index: int) -> np.ndarray:

        pred_detections = pred_detections_dict[index]
        pred_sv_detections = DetectionUtils.convert_detections_from_row_to_sv(pred_detections)

        # true_detections = true_detections_dict[index]
        # true_sv_detections = DetectionUtils.convert_detections_from_row_to_sv(true_detections)
        
        pred_people_coordinates = PeopleCoordinates(
            object_ids=np.array([detection[0] for detection in pred_detections]),
            sb_xy=np.array([coordinates_converter.convert_coordinates_to_scene(detection[5:7]) for detection in pred_detections])
        )

        distance_service.update_violation_pairs(index, pred_people_coordinates)

        violator_pairs = distance_service.get_all_current_violation_pairs(index)
        violator_set = distance_service.get_all_current_violators_set(index)
        violating_detections = DetectionUtils.filter_detections_by_tracker_ids(pred_sv_detections, violator_set)

        annotated_frame = frame.copy()
        annotated_frame = pred_people_annotator.annotate(scene=annotated_frame, detections=pred_sv_detections)
        annotated_frame = pred_dot_annotator.annotate(scene=annotated_frame, detections=pred_sv_detections)
        annotated_frame = pred_box_annotator.annotate(scene=annotated_frame, detections=violating_detections)

        for pair in violator_pairs:
            p1 = [(int((detection[0] + detection[2]) / 2), int(detection[3]))
                  for detection, tracker_id
                  in zip(pred_sv_detections.xyxy, pred_sv_detections.tracker_id)
                  if tracker_id == pair[0]
                  ][0]

            p2 = [(int((detection[0] + detection[2]) / 2), int(detection[3]))
                  for detection, tracker_id
                  in zip(pred_sv_detections.xyxy, pred_sv_detections.tracker_id)
                  if tracker_id == pair[1]
                  ][0]

            cv2.line(annotated_frame, p1, p2, (150, 0, 255), 2)

            center = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
            dist = distance_service.get_distance_for_pair(index, pair)
            cv2.putText(annotated_frame, f"{dist:.2f} m", tuple(center), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (150, 100, 255), 2)

        return annotated_frame

    sv.process_video(
        source_path=dataset_service.get_video_path(),
        target_path=os.path.join(RESULTS_DIR_PATH, result_video_path),
        callback=callback
    )


if __name__ == '__main__':
    demo('demo_social_distance_monitoring_with_dataset__8.mp4')