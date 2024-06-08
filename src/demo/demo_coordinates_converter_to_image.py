import cv2
import supervision as sv
import numpy as np

from src.service.coordinates_converter import CoordinatesConverter
from src.service.towncentre_video_service import TowncentreVideoService


def demo():
    dataset_service = TowncentreVideoService()
    dataset_video_path = dataset_service.get_video_path()
    coordinates_converter = CoordinatesConverter(dataset_service.get_camera_parameters())

    frames_generator = sv.get_video_frames_generator(dataset_video_path)
    frame = frames_generator.__next__()

    grid_size = 50  # Grid size in meters
    num_points = 51  # Number of points in the grid
    x = np.linspace(-grid_size / 2, grid_size / 2, num_points)
    y = np.linspace(-grid_size / 2, grid_size / 2, num_points)
    grid_x, grid_y = np.meshgrid(x, y)

    image_points = np.zeros((num_points * num_points, 2))
    for i in range(num_points):
        for j in range(num_points):
            scene_point = np.array([grid_x[i, j], grid_y[i, j], 0])
            image_point = coordinates_converter.convert_coordinates_to_image(scene_point)
            image_points[i * num_points + j, :] = image_point

    # Draws the grid on the image
    for i in range(num_points):
        for j in range(num_points):
            image_point = image_points[i * num_points + j, :]
            cv2.circle(frame, tuple(image_point.astype(int)), radius=3, color=(0, 255, 200), thickness=-1)

    cv2.imshow('Image with grid', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    demo()