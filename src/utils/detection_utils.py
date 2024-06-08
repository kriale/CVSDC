import numpy as np
import supervision as sv

FILE_STRUCTURE = 'frame_index,tracker_id,x1,y1,x2,y2,bdcx,bdcy,mask'


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

    @staticmethod
    def write_predictions_dict(predictions_dict: dict, file_path) -> None:
        with open(file_path, 'w') as f:
            f.write(FILE_STRUCTURE + '\n')
            for frame_index, detections in predictions_dict.items():
                for detection in detections:
                    f.write(str(frame_index) + ',' + ','.join(map(str, detection)) + '\n')

    @staticmethod
    def read_predictions_dict(file_path) -> dict:
        predictions_dict = {}
        with open(file_path, 'r') as f:
            file_structure = f.readline().strip()
            if file_structure != FILE_STRUCTURE:
                raise ValueError(
                    f'Unexpected file structure: {file_structure}. Expected: {FILE_STRUCTURE}')
            for line in f:
                frame_index, *detection = map(float, line.strip().split(','))
                frame_index = int(frame_index)
                if frame_index not in predictions_dict:
                    predictions_dict[frame_index] = []
                predictions_dict[frame_index].append(tuple(detection))
        return predictions_dict