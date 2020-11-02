import numpy as np

class YOLO_v4:
    def __init__(self):
        self.coke_dimensions = [0.06, 0.06, 0.14]
        self.sheep_dimensions = [0.108, 0.223, 0.204]
        self.camera_w = 640
        self.camera_h = 480
        self.half_size_x = self.camera_w / 2
        self.half_size_y = self.camera_h / 2

        self.h_fov = 0.8517  # rad unit
        self.focal_length = (self.camera_h / 2) / np.tan(self.h_fov / 2)
    def calculate_relative_locations(self, box):
        # box(top_left_corner_x, top_left_corner_y, box_width, box_height, class)
        if box is None:
            return []

        object = []

        if box[4] == 0:
            true_height = self.sheep_dimensions[2]
        elif box[4] == 1:
            true_height = self.coke_dimensions[2]
        else:
            print("no class")

        pixel_height = box[3]
        pixel_center = float(box[0]+(box[2]/2)) - self.half_size_x

        theta = np.arctan(np.tan(self.h_fov/2) * pixel_center/self.half_size_x)
        distance = true_height/pixel_height * self.focal_length / np.cos(theta)

        horizontal_relative_distance = distance * np.sin(theta)
        vertical_relative_distance = distance * np.cos(theta)

        object.append([horizontal_relative_distance, vertical_relative_distance])

        return object
