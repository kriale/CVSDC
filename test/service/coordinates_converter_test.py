import unittest
import numpy as np

from src.service.coordinates_converter import CoordinatesConverter


class TestCoordinatesConverter(unittest.TestCase):
    def setUp(self):
        self.params = {
            'FocalLengthX': 2696.35888671875,
            'FocalLengthY': 2696.35888671875,
            'PrincipalPointX': 959.5,
            'PrincipalPointY': 539.5,
            'Skew': 0,
            'TranslationX': -0.05988363921642303467,
            'TranslationY': 3.83331298828125,
            'TranslationZ': 12.39112186431884765625,
            'RotationX': 0.69724917918208628720,
            'RotationY': -0.43029624469563848566,
            'RotationZ': 0.28876888503799524877,
            'RotationW': 0.49527896681027261394,
            'DistortionK1': -0.60150605440139770508,
            'DistortionK2': 4.70203733444213867188,
            'DistortionP1': -0.00047452122089453042,
            'DistortionP2': -0.00782289821654558182
        }
        self.converter = CoordinatesConverter(self.params)
        self.image_point = np.array([100, -50])
        self.scene_point = np.array([-2, -2, 0])

    def test_convert_coordinates_to_scene(self):
        scene_point_converted = self.converter.convert_coordinates_to_scene(self.image_point)
        self.assertIsInstance(scene_point_converted, np.ndarray)
        self.assertEqual(scene_point_converted.shape, (3,))
        self.assertAlmostEqual(scene_point_converted[2], 0, places=5)

    def test_convert_coordinates_to_image(self):
        image_point_converted = self.converter.convert_coordinates_to_image(self.scene_point)
        self.assertIsInstance(image_point_converted, np.ndarray)
        self.assertEqual(image_point_converted.shape, (2,))

    def test_bijection_scene_image(self):
        scene_point_converted = self.converter.convert_coordinates_to_scene(self.image_point)
        image_point_back = self.converter.convert_coordinates_to_image(scene_point_converted)

        np.testing.assert_allclose(self.image_point, image_point_back, rtol=1e-5, atol=1e-8)

    def test_bijection_image_scene(self):
        image_point_converted = self.converter.convert_coordinates_to_image(self.scene_point)
        scene_point_back = self.converter.convert_coordinates_to_scene(image_point_converted)

        np.testing.assert_allclose(self.scene_point, scene_point_back, rtol=1e-5, atol=1e-8)

    def test_convert_coordinates_to_image__vertically_down__zero_point(self):
        self.params['RotationX'] = 0.9927129910375885
        self.params['RotationY'] = 0
        self.params['RotationZ'] = 0
        self.params['RotationW'] = -0.12050276936736662
        self.params['TranslationX'] = 0
        self.params['TranslationY'] = 0
        self.params['TranslationZ'] = 10

        self.converter = CoordinatesConverter(self.params)
        self.scene_point = np.array([0, 0, 0])

        image_point_converted = self.converter.convert_coordinates_to_image(self.scene_point)

        # Check position relatively to principal point
        self.assertEqual(image_point_converted[0], self.params['PrincipalPointX'])
        self.assertEqual(image_point_converted[1], self.params['PrincipalPointY'])

        # Projected point is on the image
        self.assertGreaterEqual(image_point_converted[0], 0)
        self.assertLessEqual(image_point_converted[0], self.params['PrincipalPointX'] * 2)
        self.assertGreaterEqual(image_point_converted[1], 0)
        self.assertLessEqual(image_point_converted[1], self.params['PrincipalPointY'] * 2)


if __name__ == '__main__':
    unittest.main()
