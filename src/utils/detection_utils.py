import numpy as np
import supervision as sv


class DetectionUtils:
    @staticmethod
    def convert_detections_from_sv_to_row(sv_detections: sv.Detections) -> []:
        return [
            (int(tracker_id), detection[0], detection[1], detection[2], detection[3],
             (detection[0] + detection[2]) / 2, detection[3], 0)
            for detection, tracker_id
            in zip(sv_detections.xyxy, sv_detections.tracker_id)
        ]

    @staticmethod
    def convert_detections_from_sv_to_row_dict(sv_detections_dict: dict) -> dict:
        row_detections_dict = {}
        for index in sv_detections_dict.keys():
            row_detections_dict[index] = DetectionUtils.convert_detections_from_sv_to_row(sv_detections_dict[index])
        return row_detections_dict

    @staticmethod
    def convert_detections_from_row_to_sv(row_detections: []) -> sv.Detections:
        if len(row_detections) == 0:
            return sv.Detections(
                xyxy=np.empty((0, 4)),
                tracker_id=np.empty(0),
            )
        else:
            return sv.Detections(
                xyxy=np.array([[x1, y1, x2, y2] for tracker_id, x1, y1, x2, y2, bdcx, bdcy, mask in row_detections]),
                tracker_id=np.array([tracker_id for tracker_id, x1, y1, x2, y2, bdcx, bdcy, mask in row_detections])
            )