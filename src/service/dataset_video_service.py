import os
from abc import ABC, abstractmethod


class DatasetVideoService(ABC):
    """
    This class provides method for working with video from a dataset
    """

    def __init__(self, video_path: os.path, groundtruth_file_path: os.path, camera_parameters_file_path: os.path):
        self.__video_path = video_path
        self.__groundtruth_file_path = groundtruth_file_path
        self.__camera_parameters_file_path = camera_parameters_file_path
        self.__groundtruth_dict = None
        self.__camera_parameters = None

    def get_video_path(self):
        return self.__video_path

    def get_groundtruth_file_path(self):
        return self.__groundtruth_file_path

    def get_camera_parameters_file_path(self):
        return self.__camera_parameters_file_path

    def get_groundtruth_dict(self):
        if self.__groundtruth_dict is None:
            self.__groundtruth_dict = self.load_groundtruth_dict()
        return self.__groundtruth_dict

    def get_camera_parameters(self):
        if self.__camera_parameters is None:
            self.__camera_parameters = self.load_camera_parameters()
        return self.__camera_parameters

    @abstractmethod
    def load_groundtruth_dict(self) -> dict:
        pass

    @abstractmethod
    def load_camera_parameters(self) -> dict:
        pass

    @staticmethod
    def load_video_paths(self, video_file_extension='.mp4') -> []:
        video_files = [f for f in os.listdir(self.__dataset_path) if f.endswith(video_file_extension)]
        video_paths = [os.path.join(self.__dataset_path, f) for f in video_files]
        return video_paths
