import numpy as np


class DistanceMatrix:
    """
    This class represents a distance matrix for a set of objects. The distance matrix is a 2D array where each element
    represents the distance between two objects. The objects are identified by their IDs.
    """

    def __init__(self, object_ids):
        """
        Initialize the DistanceMatrix with the given object IDs. The IDs are used to map the objects to their indices
        in the distance matrix.

        :param object_ids: A list of object IDs.
        """
        self.id_index_map = {id: index for index, id in enumerate(object_ids)}
        size = len(self.id_index_map)
        self.matrix = np.zeros((size, size))

    def add_distance(self, id1, id2, distance):
        """
        Add the distance between two objects to the distance matrix.

        :param id1: The ID of the first object.
        :param id2: The ID of the second object.
        :param distance: The distance between the two objects.
        """
        index1 = self.id_index_map[id1]
        index2 = self.id_index_map[id2]

        self.matrix[index1, index2] = distance
        self.matrix[index2, index1] = distance

    def get_distance(self, id1, id2):
        """
        Get the distance between two objects.

        :param id1: The ID of the first object.
        :param id2: The ID of the second object.
        :return: The distance between the two objects.
        """
        if (id1 not in self.id_index_map.keys()) or (id2 not in self.id_index_map.keys()):
            return None

        index1 = self.id_index_map[id1]
        index2 = self.id_index_map[id2]

        return self.matrix[index1, index2]

    def get_all_distances(self, id):
        """
        Get the distances between the given object and all other objects.

        :param id: The ID of the object.
        :return: A list of distances between the given object and all other objects.
        """
        index = self.id_index_map[id]
        return self.matrix[index, :]


class PeopleCoordinates:
    def __init__(self, object_ids: np.array, sb_xy: np.array):
        self.object_ids = object_ids
        self.sb_xy = sb_xy


class SocialDistanceService:
    """
    This class is responsible for managing and updating the violation pairs based on the social distance threshold,
    the number of last frames to consider, and the violation percentage.
    """

    def __init__(self, distance_threshold=2, last_frames: int = 5, violation_percentage: float = 0.8):
        """
        Initialize the SocialDistanceService with the given parameters.

        :param distance_threshold: The minimum distance between two people to consider it a violation.
        :param last_frames: The number of last frames to consider for calculating the violation pairs.
        :param violation_percentage: The minimum percentage of frames in which a pair of people violate the social distance to consider them a violation pair.
        """
        self.__distance_threshold = distance_threshold
        self.__last_frames = last_frames
        self.__violation_percentage = violation_percentage
        self.__distance_matrix_history = {}  # To store the distance matrix of each frame
        self.__violator_pairs_history = {}  # To store the violation pairs of each frame

    def update_violation_pairs(self, frame_index: int, people_coordinates: PeopleCoordinates) -> ():
        """
        Update the violation pairs for the given frame index and people coordinates.

        :param frame_index: The index of the frame.
        :param people_coordinates: The coordinates of the people in the frame.
        :return: A tuple containing all current violation pairs and new violation pairs.
        """
        if frame_index not in self.__distance_matrix_history.keys():
            self.__distance_matrix_history[frame_index] = self.__calculate_distance_matrix_for_single_frame(
                frame_index,
                people_coordinates
            )

        all_curr_violation_pairs = self.get_all_current_violation_pairs(frame_index)
        new_curr_violation_pairs = self.get_new_current_violation_pairs(frame_index)

        return all_curr_violation_pairs, new_curr_violation_pairs

    def get_all_current_violation_pairs(self, frame_index: int) -> set:
        """
        Get all current violation pairs for the given frame index.

        :param frame_index: The index of the frame.
        :return: A set of all current violation pairs.
        """
        if frame_index in self.__violator_pairs_history.keys():  # if violation pairs for this frame were already calculated
            return self.__violator_pairs_history[frame_index]

        return self.__calculate_all_current_violation_pairs(frame_index)

    def get_new_current_violation_pairs(self, frame_index: int) -> set:
        """
        Get new current violation pairs for the given frame index.

        :param frame_index: The index of the frame.
        :return: A set of new current violation pairs.
        """
        return self.__calculate_new_current_violation_pairs(frame_index)

    def get_distance_matrix(self, frame_index: int) -> DistanceMatrix:
        """
        Get the distance matrix for the given frame index.

        :param frame_index: The index of the frame.
        :return: The distance matrix for the frame.
        """
        return self.__distance_matrix_history[frame_index]

    def get_all_current_violators_set(self, frame_index: int) -> set:
        return self.__compute_violator_set(self.get_all_current_violation_pairs(frame_index))

    def get_new_current_violators_set(self, frame_index: int) -> set:
        return self.__compute_violator_set(self.get_new_current_violation_pairs(frame_index))

    # =================== Private methods ===================

    def __calculate_all_current_violation_pairs(self, frame_index: int) -> set:
        """
        Calculate all current violation pairs for the given frame index.

        :param frame_index: The index of the frame.
        :return: A set of all current violation pairs.
        """
        violator_pairs = set()

        relevant_frames = set(range(max(0, frame_index - self.__last_frames + 1), frame_index + 1))
        # Remove indexes that are not in the self.__distance_matrix_history.keys()
        relevant_frames = set(relevant_frames).intersection(self.__distance_matrix_history.keys())

        distance_matrix = self.get_distance_matrix(frame_index)
        for id1 in distance_matrix.id_index_map:
            for id2 in distance_matrix.id_index_map:
                if id1 < id2:  # To avoid duplicates
                    distance_history = [self.get_distance_matrix(i).get_distance(id1, id2) for i in relevant_frames]
                    filtered_distance_history = [distance for distance in distance_history if distance is not None]
                    actual_violations_count = [
                        distance < self.__distance_threshold
                        for distance
                        in filtered_distance_history
                    ].count(True)

                    if actual_violations_count / len(relevant_frames) >= self.__violation_percentage:
                        violator_pairs.add((id1, id2))

        self.__violator_pairs_history[frame_index] = violator_pairs
        return violator_pairs

    def __calculate_new_current_violation_pairs(self, frame_index: int) -> set:
        """
        Calculate new current violation pairs for the given frame index.

        :param frame_index: The index of the frame.
        :return: A set of new current violation pairs.
        """
        curr_violator_pairs = self.get_all_current_violation_pairs(frame_index)
        if frame_index == 0:
            return curr_violator_pairs

        prev_violator_pairs = self.get_all_current_violation_pairs(frame_index - 1)
        new_violator_pairs = curr_violator_pairs - prev_violator_pairs
        return new_violator_pairs

    def __calculate_distance_matrix_for_single_frame(self, frame_index,
                                                     people_coordinates: PeopleCoordinates) -> DistanceMatrix:
        """
        Calculate the distance matrix for a single frame.

        :param frame_index: The index of the frame.
        :param people_coordinates: The coordinates of the people in the frame.
        :return: The distance matrix for the frame.
        """
        distance_matrix = DistanceMatrix(people_coordinates.object_ids)

        # Calculate the distance between every pair of objects and store it in the distance matrix
        for i in range(len(people_coordinates.sb_xy)):
            for j in range(i + 1, len(people_coordinates.sb_xy)):
                distance = np.linalg.norm(people_coordinates.sb_xy[i] - people_coordinates.sb_xy[j])
                distance_matrix.add_distance(people_coordinates.object_ids[i], people_coordinates.object_ids[j],
                                             distance)

        self.__distance_matrix_history[frame_index] = distance_matrix
        return distance_matrix

    @staticmethod
    def __compute_violator_set(violator_pairs):
        """
        Compute the violator set from the given violator pairs.

        :param violator_pairs: A set of violator pairs.
        :return: A set of violators.
        """
        violator_set = set()
        for pair in violator_pairs:
            violator_set.add(pair[0])
            violator_set.add(pair[1])
        return violator_set
