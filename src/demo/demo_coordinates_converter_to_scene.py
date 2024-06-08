import numpy as np

from src.service.coordinates_converter import CoordinatesConverter
from src.service.towncentre_video_service import TowncentreVideoService


def demo():
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

    # Создание объекта CoordinatesConverter
    dataset_service = TowncentreVideoService()
    coordinates_converter = CoordinatesConverter(dataset_service.get_camera_parameters())

    # Создание сетки точек на изображении (OXY)
    x = np.linspace(0, 1920, 10)  # изменение на размеры изображения
    y = np.linspace(0, 1080, 10)  # изменение на размеры изображения
    X, Y = np.meshgrid(x, y)

    image_points = np.column_stack((X.flatten(), Y.flatten()))

    # Преобразование точек на изображении в точки на сцене
    scene_points = [coordinates_converter.convert_coordinates_to_scene(point) for point in image_points]

    # Создание 3D-графика
    fig = plt.figure()

    # Визуализация оригинальных точек на изображении
    ax1 = fig.add_subplot(121)
    ax1.scatter(*zip(*image_points), color='blue', s=2)
    ax1.set_title('Image Points')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_xlim([0, 1920])  # изменение на размеры изображения
    ax1.set_ylim([0, 1080])  # изменение на размеры изображения

    # Визуализация преобразованных точек на сцене
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(*zip(*scene_points), color='blue', s=2)
    ax2.set_title('Scene Points')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')

    # Добавление схематического изображения камеры в виде пирамиды
    camera_parameters = dataset_service.get_camera_parameters()
    camera_location = np.array([camera_parameters[key] for key in
                                ['TranslationX', 'TranslationY', 'TranslationZ']])
    camera_orientation = np.array([camera_parameters[key] for key in
                                   ['RotationX', 'RotationY', 'RotationZ']])

    # Создание пирамиды: вершина - местоположение камеры, основание - плоскость изображения
    verts = [camera_location,
             camera_location + camera_orientation + np.array(
                 [1, 1, 1]),
             camera_location + camera_orientation + np.array(
                 [-1, -1, 1]),
             camera_location + camera_orientation + np.array(
                 [-1, 1, 1]),
             camera_location + camera_orientation + np.array(
                 [1, -1, 1])]

    faces = [[verts[0], verts[1], verts[2]],
             [verts[0], verts[1], verts[3]],
             [verts[0], verts[2], verts[4]],
             [verts[0], verts[3], verts[4]],
             [verts[1], verts[2], verts[3], verts[4]]]

    ax2.add_collection3d(Poly3DCollection(faces, facecolors='green', linewidths=1, edgecolors='g', alpha=0.5))

    ax2.text(*camera_location, 10, label='Camera', color='green')

    # Получение минимальных и максимальных значений для каждой оси
    min_x = min([point[0] for point in scene_points])
    max_x = max([point[0] for point in scene_points])
    min_y = min([point[1] for point in scene_points])
    max_y = max([point[1] for point in scene_points])

    # Создание и отображение полупрозрачного серого полигона, представляющего плоскость земли
    ground_vertices = np.array([[min_x, min_y, 0], [max_x, min_y, 0], [max_x, max_y, 0], [min_x, max_y, 0]])
    ground_faces = [ground_vertices]
    ax2.add_collection3d(Poly3DCollection(ground_faces, facecolors='gray', alpha=0.5))

    # Рисование перпендикуляра от камеры к плоскости земли
    line_vertices = np.array([camera_location, [camera_location[0], camera_location[1], 0]])
    ax2.add_collection3d(Line3DCollection([line_vertices], colors='black', linewidths=1))

    # Установка пределов по осям X, Y и Z
    ax2.set_xlim([min_x, max_x])
    ax2.set_ylim([min_y, max_y])
    ax2.set_zlim([0, camera_location[2]])

    plt.show()


if __name__ == '__main__':
    demo()