from abc import ABC, abstractmethod


class VideoDatasetService(ABC):
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    @abstractmethod
    def get_video_paths(self):
        pass

    @abstractmethod
    def get_annotations_file_path(self):
        pass

    @abstractmethod
    def get_groundtruth_dict(self):
        pass

    @abstractmethod
    def load_groundtruth_dict(self):
        pass
