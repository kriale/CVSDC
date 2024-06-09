import unittest
from src.service.social_distance_service import SocialDistanceService, PeopleCoordinates
import numpy as np


class TestSocialDistanceService(unittest.TestCase):
    def setUp(self):
        self.social_distance_service = SocialDistanceService(2, 2, 0.5)

    def test_update_violation_pairs_with_no_previous_data(self):
        people_coordinates = PeopleCoordinates(np.array([1, 2]), np.array([[0, 0], [0, 2]]))
        all_curr_violation_pairs, new_curr_violation_pairs = self.social_distance_service.update_violation_pairs(0, people_coordinates)
        self.assertEqual(all_curr_violation_pairs, set())
        self.assertEqual(new_curr_violation_pairs, set())

    def test_update_violation_pairs_with_previous_data(self):
        people_coordinates0 = PeopleCoordinates(np.array([1, 2]), np.array([[0, 0], [0, 2]]))
        self.social_distance_service.update_violation_pairs(0, people_coordinates0)
        people_coordinates1 = PeopleCoordinates(np.array([1, 2]), np.array([[0, 0], [0, 1.5]]))
        all_curr_violation_pairs, new_curr_violation_pairs = self.social_distance_service.update_violation_pairs(1, people_coordinates1)
        self.assertEqual(all_curr_violation_pairs, {(1, 2)})
        self.assertEqual(new_curr_violation_pairs, {(1, 2)})

    def test_get_new_current_violation_pairs_with_previous_data(self):
        people_coordinates = PeopleCoordinates(np.array([1, 2]), np.array([[0, 0], [0, 2]]))
        self.social_distance_service.update_violation_pairs(0, people_coordinates)

        people_coordinates = PeopleCoordinates(np.array([1, 2]), np.array([[0, 0], [0, 1.5]]))
        self.social_distance_service.update_violation_pairs(1, people_coordinates)

        new_violation_pairs = self.social_distance_service.get_new_current_violation_pairs(1)
        self.assertEqual(new_violation_pairs, {(1, 2)})

    def test_violation_pairs_with_varying_params(self):
        self.social_distance_service = SocialDistanceService(last_frames=3, violation_percentage=0.2)
        people_coordinates = PeopleCoordinates(np.array([1, 2]), np.array([[0, 0], [0, 2]]))
        self.social_distance_service.update_violation_pairs(0, people_coordinates)
        people_coordinates = PeopleCoordinates(np.array([1, 2]), np.array([[0, 0], [0, 1.5]]))
        self.social_distance_service.update_violation_pairs(1, people_coordinates)
        people_coordinates = PeopleCoordinates(np.array([1, 2]), np.array([[0, 0], [0, 3]]))
        self.social_distance_service.update_violation_pairs(2, people_coordinates)
        all_curr_violation_pairs, new_curr_violation_pairs = self.social_distance_service.update_violation_pairs(3, people_coordinates)
        self.assertEqual(all_curr_violation_pairs, {(1, 2)})
        self.assertEqual(new_curr_violation_pairs, set())


if __name__ == '__main__':
    unittest.main()

