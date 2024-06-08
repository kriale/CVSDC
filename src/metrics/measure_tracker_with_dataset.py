from config import PREDICTIONS_FILE_PATH
from src.service.mot_metrics_service import MotMetricsService
from src.service.towncentre_video_service import TowncentreVideoService
from src.utils.detection_utils import DetectionUtils


def measure():
    dataset_service = TowncentreVideoService()
    true_detections_dict = dataset_service.get_groundtruth_dict()

    pred_detections_dict = DetectionUtils.read_predictions_dict(PREDICTIONS_FILE_PATH)

    mot_metrics_service = MotMetricsService(
        groundtruth_dict=true_detections_dict,
        predictions_dict=pred_detections_dict
    )
    print(mot_metrics_service.calculate_mot_metrics())


if __name__ == '__main__':
    measure()