import numpy as np
import supervision as sv
from ultralytics import YOLO

from config import YOLO_V8_N_PATH
from src.utils.detection_utils import DetectionUtils

DEFAULT_YOLO_MODEL_PATH = YOLO_V8_N_PATH
DEFAULT_BYTETRACK_PARAMS = {
    'track_thresh': 0.25,
    'track_buffer': 30,
    'match_thresh': 0.9,
    'frame_rate': 10
}


class ByteTrackYOLOTracker:
    """
    This class implements YOLO + ByteTrack MOT
    """
    def __init__(
            self,
            yolo_model_path=DEFAULT_YOLO_MODEL_PATH,
            bytetrack_params=None
    ):
        if bytetrack_params is None:
            bytetrack_params = DEFAULT_BYTETRACK_PARAMS

        self.__detection_model = YOLO(yolo_model_path)
        self.__byte_tracker = sv.ByteTrack(
            track_thresh=bytetrack_params['track_thresh'],
            track_buffer=bytetrack_params['track_buffer'],
            match_thresh=bytetrack_params['match_thresh'],
            frame_rate=bytetrack_params['frame_rate'],
        )
        self.__selected_classes = [0]  # pedestrians only
        self.__pred_detections_dict = dict()

    def update_tracker(self, frame, frame_index) -> sv.Detections:
        sv_detections = self.__detect(frame)
        sv_detections = self.__byte_tracker.update_with_detections(sv_detections)
        self.__pred_detections_dict[frame_index] = DetectionUtils.convert_detections_from_sv_to_row(sv_detections)
        return sv_detections

    def get_pred_detections(self):
        return self.__pred_detections_dict

    # ============= Private methods =============

    # Detects pedestrians on the frame
    def __detect(self, frame) -> sv.Detections:
        # Detect objects on the frame
        ultralytics_detections = self.__detection_model(frame, verbose=False)[0]
        # Convert detections from ultralytics to supervision format
        sv_detections = sv.Detections.from_ultralytics(ultralytics_detections)
        # Consider class id from __selected_classes define above
        sv_detections = sv_detections[np.isin(sv_detections.class_id, self.__selected_classes)]
        return sv_detections
