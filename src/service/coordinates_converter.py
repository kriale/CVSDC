import numpy as np


class CoordinatesConverter:
    """
    camera_parameters - dictionary with camera params loaded from config file
    like 'TownCentre-calibration.ci'
    """
    def __init__(self, camera_parameters):
        self.parameters = camera_parameters
        self.camera_matrix, self.projection_matrix, self.transformation_matrix = self.__build_matrices(self.parameters)

    """
    Returns 3D scene point, Z=0
    """
    def convert_coordinates_to_scene(self, image_point: np.ndarray) -> np.ndarray:
        # convert image point to homogeneous coordinates
        image_point_homogeneous = np.array([image_point[0], image_point[1], 1, 1], dtype=float)

        # Extract the required parameters
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]
        tx = self.transformation_matrix[0, 3]
        ty = self.transformation_matrix[1, 3]
        tz = self.transformation_matrix[2, 3]

        r = self.transformation_matrix[:3, :3]
        r11, r12, r13, r21, r22, r23, r31, r32, r33 = r.flatten()

        x_i = image_point_homogeneous[0]
        y_i = image_point_homogeneous[1]

        # Calculate z using the provided formula
        z = -(r22 * r31 * tx * (
            -fx) * fy + r21 * r32 * tx * fx * fy + r12 * r31 * ty * fx * fy - r11 * r32 * ty * fx * fy - r12 * r21 * tz * fx * fy + r11 * r22 * tz * fx * fy) / (
                        r32 * r21 * cx * fy + r12 * r31 * cy * fx - r22 * r31 * cx * fy - r11 * r32 * cy * fx + r12 * r21 * fx * fy - r32 * r21 * x_i * fy - r11 * r22 * fx * fy + r22 * r31 * x_i * fy - r12 * r31 * y_i * fx + r11 * r32 * y_i * fx)
        # print('scene z: ', z)

        # calculate X, Y, Z using the provided formulas
        X = -((-r32 * cy + r22 * (-fy) + r32 * y_i) * (-tz * cx + tx * (-fx) + tz * x_i) - (
                    -r32 * cx + r12 * (-fx) + r32 * x_i) * (-tz * cy + ty * (-fy) + tz * y_i)) / (
                        (-r31 * cx + r11 * (-fx) + r31 * x_i) * (-r32 * cy + r22 * (-fy) + r32 * y_i) - (
                            -r32 * cx + r12 * (-fx) + r32 * x_i) * (-r31 * cy + r21 * (-fy) + r31 * y_i))
        Y = -(-r31 * tx * cy * fx + r31 * ty * cx * fy + r11 * tz * cy * fx - r21 * tz * cx * fy + r21 * tx * (
            -fx) * fy + r31 * tx * y_i * fx + r11 * ty * fx * fy - r31 * ty * x_i * fy + r21 * tz * x_i * fy - r11 * tz * y_i * fx) / (
                        -r32 * r21 * cx * fy - r12 * r31 * cy * fx + r22 * r31 * cx * fy + r11 * r32 * cy * fx + r12 * r21 * (
                    -fx) * fy + r32 * r21 * x_i * fy + r11 * r22 * fx * fy - r22 * r31 * x_i * fy + r12 * r31 * y_i * fx - r11 * r32 * y_i * fx)
        Z = 0  # Z is always 0 as per the document

        scene_point = np.array([X, Y, Z])
        return scene_point

    """
    Returns 2D image point
    """
    def convert_coordinates_to_image(self, scene_point: np.ndarray) -> np.ndarray:
        # convert scene point to homogeneous coordinates
        scene_point_homogeneous = np.append(scene_point, 1)
        # print('scene_point_homogeneous: ', scene_point_homogeneous)

        # apply camera transformation
        image_point_homogeneous = np.dot(self.transformation_matrix, scene_point_homogeneous)
        # print('image_point_homogeneous: ', image_point_homogeneous)

        # convert to non-homogeneous coordinates
        projected_point = np.dot(self.projection_matrix, image_point_homogeneous)
        # print('projected_point: ', projected_point)
        # print('image z: ', projected_point[2])

        image_point = projected_point[:2] / projected_point[2]
        # print('image_point: ', image_point)

        return image_point

    """
    Builds & returns three matrices: camera_matrix, projection_matrix, transformation_matrix
    """
    @staticmethod
    def __build_matrices(parameters: dict) -> ():
        # define camera intrinsic matrix
        camera_matrix = np.array([
            [parameters['FocalLengthX'], 0, parameters['PrincipalPointX']],
            [0, parameters['FocalLengthY'], parameters['PrincipalPointY']],
            [0, 0, 1]
        ])

        projection_matrix = np.zeros((4, 4))
        projection_matrix[:3, :3] = camera_matrix
        projection_matrix[3, 3] = 1

        # build rotation matrix from quaternion
        x, y, z, w = parameters['RotationX'], parameters['RotationY'], parameters['RotationZ'], parameters['RotationW']
        rotation_matrix = np.array([
            [1 - 2 * y ** 2 - 2 * z ** 2, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x ** 2 - 2 * z ** 2, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x ** 2 - 2 * y ** 2]
        ])
        # print('rotation_matrix:\n', rotation_matrix)
        # pitch = math.asin(2 * (w * y - z * x))
        # print('pitch (rad): ', pitch)
        # print('pitch (degrees): ', pitch / math.pi * 180)

        # define translation vector
        translation_vector = np.array(
            [parameters['TranslationX'], parameters['TranslationY'], parameters['TranslationZ']])
        # build 4x4 transformation matrix
        transformation_matrix = np.zeros((4, 4))
        transformation_matrix[:3, :3] = rotation_matrix
        transformation_matrix[:3, 3] = translation_vector
        transformation_matrix[3, 3] = 1
        # print('transformation_matrix:\n', transformation_matrix)

        return camera_matrix, projection_matrix, transformation_matrix
