import os

import numpy as np
from supervision import Color, Position
import supervision as sv

from config import RESULTS_DIR_PATH, PREDICTIONS_FILE_PATH
from src.service.towncentre_video_service import TowncentreVideoService
from src.utils.detection_utils import DetectionUtils


def demo(result_video_path='demo_tracker_with_dataset.mp4'):
    dataset_service = TowncentreVideoService()
    true_detections_dict = dataset_service.get_groundtruth_dict()
    pred_detections_dict = DetectionUtils.read_predictions_dict(PREDICTIONS_FILE_PATH)

    # Creating annotations
    pred_box_annotator = sv.BoxAnnotator(thickness=4, text_thickness=2, text_scale=1,
                                         color=Color(r=0, g=50, b=255))
    true_box_annotator = sv.BoxAnnotator(thickness=4, text_thickness=2, text_scale=1,
                                         color=Color(r=255, g=0, b=0))
    trace_annotator = sv.TraceAnnotator(
        thickness=2,
        trace_length=50,
        position=Position.BOTTOM_CENTER,
        color=Color(r=0, g=50, b=240)
    )

    # This function processes each frame
    def callback(frame: np.ndarray, index: int) -> np.ndarray:
        pred_detections = pred_detections_dict[index]
        sv_pred_detections = DetectionUtils.convert_detections_from_row_to_sv(pred_detections)
        
        annotated_frame = frame.copy()
        annotated_frame = trace_annotator.annotate(
            scene=annotated_frame,
            detections=sv_pred_detections
        )

        if index in true_detections_dict:
            true_detections = true_detections_dict[index]
            true_box_labels = [
                f"#{tracker_id} TRUE"
                for tracker_id, x1, y1, x2, y2, _, _, _, _
                in true_detections
            ]
            sv_true_detections = DetectionUtils.convert_detections_from_row_to_sv(true_detections)
            annotated_frame = true_box_annotator.annotate(scene=annotated_frame,
                                                          detections=sv_true_detections,
                                                          labels=true_box_labels)

        pred_box_labels = [
            f"#{tracker_id} {confidence:0.2f}"
            for _, _, confidence, class_id, tracker_id
            in sv_pred_detections
        ]
        annotated_frame = pred_box_annotator.annotate(scene=annotated_frame, detections=sv_pred_detections,
                                                      labels=pred_box_labels)

        return annotated_frame

    sv.process_video(
        source_path=dataset_service.get_video_path(),
        target_path=os.path.join(RESULTS_DIR_PATH, result_video_path),
        callback=callback
    )


if __name__ == '__main__':
    demo()