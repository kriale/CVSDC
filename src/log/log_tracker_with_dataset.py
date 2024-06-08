import supervision as sv

from config import PREDICTIONS_FILE_PATH
from src.service.byte_track_yolo_tracker import ByteTrackYOLOTracker
from src.service.towncentre_video_service import TowncentreVideoService
from src.utils.detection_utils import DetectionUtils


def log(frames_limit=None):
    """
    This function performs the tracking algorithm on the dataset and stores the result predicted detections in the file
    """
    pred_detections_dict = dict()
    dataset_service = TowncentreVideoService()
    dataset_video_path = dataset_service.get_video_path()

    tracker = ByteTrackYOLOTracker()

    frames_generator = sv.get_video_frames_generator(dataset_video_path)
    for frame_index, frame in enumerate(frames_generator):
        sv_detections = tracker.update_tracker(frame, frame_index)
        pred_detections_dict[frame_index] = DetectionUtils.convert_detections_from_sv_to_row(sv_detections)

        if frames_limit is not None and frame_index > frames_limit:
            break

    DetectionUtils.write_predictions_dict(pred_detections_dict, PREDICTIONS_FILE_PATH)


if __name__ == '__main__':
    log()
