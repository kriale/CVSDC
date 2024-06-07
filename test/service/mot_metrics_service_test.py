import unittest

from src.service.mot_metrics_service import MotMetricsService


class TestMotMetricsService(unittest.TestCase):
    def setUp(self):
        self.predictions_dict = {
            'frame_1': [
                (1, 10, 10, 20, 20),
                (2, 30, 30, 40, 40)
            ],
            'frame_2': [
                (3, 50, 50, 60, 60)
            ],
        }

        self.groundtruth_dict = {
            'frame_1': [
                (1, 10, 10, 20, 20),
                (2, 30, 30, 40, 40)
            ],
            'frame_2': [
                (3, 50, 50, 60, 60)
            ],
        }

        self.mot_metrics_service = MotMetricsService(self.predictions_dict, self.groundtruth_dict)

    def test_calculate_mot_metrics(self):
        mota, motp = self.mot_metrics_service.calculate_mot_metrics()

        self.assertIsNotNone(mota)
        self.assertIsNotNone(motp)
        self.assertIsInstance(mota, float)
        self.assertIsInstance(motp, float)


if __name__ == "__main__":
    unittest.main()
