import csv

from config import GROUNDTRUTH_FILE_PATH, CAMERA_PARAMETERS_FILE_PATH, VIDEO_PATH
from src.service.dataset_video_service import DatasetVideoService


class TowncentreVideoService(DatasetVideoService):
    def __init__(self):
        super().__init__(VIDEO_PATH, GROUNDTRUTH_FILE_PATH, CAMERA_PARAMETERS_FILE_PATH)

    def load_groundtruth_dict(self) -> dict:
        groundtruth_dict = {}
        with open(self.get_groundtruth_file_path(), 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                object_id = int(row[0])
                frame_id = int(row[1])
                hx1 = float(row[4])
                hy1 = float(row[5])
                hx2 = float(row[6])
                hy2 = float(row[7])
                x1 = float(row[8])
                y1 = float(row[9])
                x2 = float(row[10])
                y2 = float(row[11])
                if frame_id not in groundtruth_dict:
                    groundtruth_dict[frame_id] = []
                groundtruth_dict[frame_id].append((object_id, x1, y1, x2, y2, (x1 + x2) / 2, y2, 0))
        return groundtruth_dict

    def load_camera_parameters(self) -> dict:
        parameters = {}
        with open(self.get_camera_parameters_file_path(), 'r') as f:
            lines = f.readlines()
            for line in lines:
                key, value = line.strip().split(' = ')
                parameters[key] = float(value)
        return parameters
