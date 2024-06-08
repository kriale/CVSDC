import unittest
from unittest.mock import patch, MagicMock
import numpy as np

from src.service.byte_track_yolo_tracker import ByteTrackYOLOTracker


class TestByteTrackYOLOTracker(unittest.TestCase):

    @patch('src.service.byte_track_yolo_tracker.YOLO', autospec=True)
    @patch('src.service.byte_track_yolo_tracker.sv.ByteTrack', autospec=True)
    def setUp(self, mock_yolo, mock_bytetrack):
        self.mock_yolo = mock_yolo
        self.mock_yolo.return_value = MagicMock()

        self.mock_bytetrack = mock_bytetrack
        self.mock_bytetrack_instance = self.mock_bytetrack.return_value
        self.mock_bytetrack_instance.update_with_detections = MagicMock()

        self.tracker = ByteTrackYOLOTracker()

    def test_get_pred_detections(self):
        expected = self.tracker._ByteTrackYOLOTracker__pred_detections_dict
        result = self.tracker.get_pred_detections()
        self.assertEqual(result, expected)

    @patch('src.service.byte_track_yolo_tracker.sv.Detections.from_ultralytics', autospec=True)
    def test_detect(self, mock_from_ultralytics):
        fake_frame = np.random.rand(480, 640, 3)

        with patch.object(self.tracker, '_ByteTrackYOLOTracker__detection_model'):
            self.tracker._ByteTrackYOLOTracker__detect(fake_frame)

        mock_from_ultralytics.assert_called_once()


if __name__ == '__main__':
    unittest.main()
