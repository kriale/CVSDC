import unittest
from unittest.mock import patch, MagicMock

from src.service.towncentre_video_service import TowncentreVideoService


class TowncentreVideoServiceTest(unittest.TestCase):
    @patch('csv.reader')
    @patch('builtins.open', new_callable=MagicMock)
    def test_load_groundtruth_dict(self, mock_open, mock_csv_reader):
        mock_csv_reader.return_value = iter([
            ["0", "0", "1", "1", "270.828", "794.098", "309.037", "834.066", "235.925", "770.142", "371.546",
             "1101.029"]
        ])
        service = TowncentreVideoService()
        result = service.get_groundtruth_dict()
        expected = {0: [(0, 235.925, 770.142, 371.546, 1101.029, (235.925 + 371.546) / 2, 1101.029, 0)]}
        self.assertEqual(expected, result)

    @patch('builtins.open', new_callable=MagicMock)
    def test_load_camera_parameters(self, mock_open):
        mock_open.return_value.__enter__.return_value.readlines.return_value = [
            "FocalLengthX = 2696.35888671875",
            "FocalLengthY = 2696.35888671875",
            "PrincipalPointX = 959.5",
            "PrincipalPointY = 539.5",
            "Skew = 0.0",
            "TranslationX = -0.059883639216423035",
            "TranslationY = 3.83331298828125",
            "TranslationZ = 12.391121864318847",
            "RotationX = 0.6972491791820863",
            "RotationY = -0.4302962446956385",
            "RotationZ = 0.28876888503799525",
            "RotationW = 0.4952789668102726",
            "DistortionK1 = -0.6015060544013977",
            "DistortionK2 = 4.702037334442139",
            "DistortionP1 = -0.0004745212208945304",
            "DistortionP2 = -0.007822898216545582"
        ]
        service = TowncentreVideoService()
        result = service.get_camera_parameters()
        expected = {
            "FocalLengthX": 2696.35888671875,
            "FocalLengthY": 2696.35888671875,
            "PrincipalPointX": 959.5,
            "PrincipalPointY": 539.5,
            "Skew": 0.0,
            "TranslationX": -0.059883639216423035,
            "TranslationY": 3.83331298828125,
            "TranslationZ": 12.391121864318847,
            "RotationX": 0.6972491791820863,
            "RotationY": -0.4302962446956385,
            "RotationZ": 0.28876888503799525,
            "RotationW": 0.4952789668102726,
            "DistortionK1": -0.6015060544013977,
            "DistortionK2": 4.702037334442139,
            "DistortionP1": -0.0004745212208945304,
            "DistortionP2": -0.007822898216545582
        }
        self.assertEqual(expected, result)


if __name__ == '__main__':
    unittest.main()
