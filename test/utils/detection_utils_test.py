import unittest
from unittest.mock import patch, mock_open

import numpy as np
import supervision as sv

from src.utils.detection_utils import DetectionUtils


class DetectionUtilsTest(unittest.TestCase):

    def setUp(self):
        self.detectionUtils = DetectionUtils()
        self.sv_detections = sv.Detections(
            xyxy=np.array([[1, 2, 3, 4], [5, 6, 7, 8]]),
            tracker_id=np.array([1, 2])
        )
        self.sv_detections_dict = {0: self.sv_detections}
        self.row_detections = [(1, 1, 2, 3, 4, 2, 4, 0), (2, 5, 6, 7, 8, 6, 8, 0)]

    def test_convert_detections_from_sv_to_row(self):
        expected = self.row_detections
        actual = self.detectionUtils.convert_detections_from_sv_to_row(self.sv_detections)
        self.assertEqual(actual, expected)

    def test_convert_detections_from_sv_to_row_dict(self):
        expected = {0: self.row_detections}
        actual = self.detectionUtils.convert_detections_from_sv_to_row_dict(self.sv_detections_dict)
        self.assertEqual(actual, expected)

    def test_convert_detections_from_row_to_sv(self):
        expected = self.sv_detections
        actual = self.detectionUtils.convert_detections_from_row_to_sv(self.row_detections)
        self.assertEqual(actual.xyxy.tolist(), expected.xyxy.tolist())
        self.assertEqual(actual.tracker_id.tolist(), expected.tracker_id.tolist())

    @patch('builtins.open', new_callable=mock_open,
           read_data="frame_index,tracker_id,x1,y1,x2,y2,bdcx,bdcy,mask\n5,19,6.0,7.0,8.0,9.0,3.0,2.0,0.0\n")
    def test_read_predictions_dict(self, mock_file):
        expected_output = {5: [(19, 6.0, 7.0, 8.0, 9.0, 3.0, 2.0, 0)]}
        result = DetectionUtils.read_predictions_dict('dummy_path')
        self.assertEqual(result, expected_output)

    @patch('builtins.open', new_callable=mock_open)
    def test_write_predictions_dict(self, mock_file):
        predictions_dict = {5: [(19, 6.0, 7.0, 8.0, 9.0, 3.0, 2.0, 0)]}
        DetectionUtils.write_predictions_dict(predictions_dict, 'dummy_path')
        mock_file.assert_called_once_with('dummy_path', 'w')
        mock_file().write.assert_any_call("frame_index,tracker_id,x1,y1,x2,y2,bdcx,bdcy,mask\n")
        mock_file().write.assert_any_call("5,19,6.0,7.0,8.0,9.0,3.0,2.0,0\n")


if __name__ == '__main__':
    unittest.main()
