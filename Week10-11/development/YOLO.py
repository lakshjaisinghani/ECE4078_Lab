import numpy as np
import matplotlib.pyplot as plt
import os, sys
import json
import math
import cv2

class YoloV5(object):
    # def __init__(self, weights, device):
    def __init__(self):
        # self.weights = weights
        # self.imgsz = 320
        # self.model = None
        # self.device = device
        #
        # self.classes = [0, 1]
        # self.agnostic_nms = False
        # self.conf_thres = 0.5
        # self.iou_thres = 0.4

        self.coke_dimensions = [0.06, 0.06, 0.14]
        self.sheep_dimensions = [0.108, 0.223, 0.204]
        self.camera_w = 640
        self.camera_h = 480
        self.half_size_x = self.camera_w / 2
        self.half_size_y = self.camera_h / 2

        self.h_fov = 0.8517  # rad unit
        self.focal_length = (self.camera_h / 2) / np.tan(self.h_fov / 2)

    # def setup(self):
    #     self.model = attempt_load(self.weights, map_location=self.device)
    #     self.model.eval()
    #     self.imgsz = check_img_size(
    #         self.imgsz, s=self.model.stride.max()
    #     )  # check img_size
    #
    # def forward(self, image):
    #     img = letterbox(image, new_shape=self.imgsz)[0]
    #     img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    #     # Convert
    #     img = np.ascontiguousarray(img)
    #     img = torch.from_numpy(img).to(self.device)
    #
    #     if img.ndimension() == 3:
    #         img = img.unsqueeze(0)
    #     img = img.float() / 255.0
    #
    #     pred = self.model(img)[0]
    #     pred = non_max_suppression(
    #         pred,
    #         self.conf_thres,
    #         self.iou_thres,
    #         classes=self.classes,
    #         agnostic=self.agnostic_nms,
    #     )[0]
    #
    #     if pred is None:
    #         return None
    #
    #     pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], image.shape).round()
    #
    #     for prediction in pred.split(1):
    #         prediction = prediction.squeeze()
    #         det_class = "sheep" if prediction[5] == 0 else "coke"
    #         confidence = float(prediction[4])
    #
    #     return pred

    def calculate_relative_locations(self, boxes, ClassID):
        # box(top_left_corner_x, top_left_corner_y, box_width, box_height, class)
        if boxes is None:
            return []

        object = []

        for i in range(len(boxes)):
            box = boxes[i]
            if ClassID[i] == 0:
                true_height = self.sheep_dimensions[2]
            elif ClassID[i] == 1:
                true_height = self.coke_dimensions[2]
            else:
                print("no class")

            pixel_height = box[3]
            pixel_center = float(box[0]+(box[2]/2)) - self.half_size_x

            theta = np.arctan(np.tan(self.h_fov/2) * pixel_center/self.half_size_x)
            distance = true_height/pixel_height * self.focal_length / np.cos(theta)

            horizontal_relative_distance = distance * np.sin(theta)
            vertical_relative_distance = distance * np.cos(theta)

            object.append([ClassID[i], horizontal_relative_distance, vertical_relative_distance])

        return object
if __name__ == "__main__":

    # # Location of the calibration files
    # currentDir = os.getcwd()
    # datadir = "{}/calibration/".format(currentDir)
    # # connect to the robot
    # ppi = penguinPiC.PenguinPi()
    #
    # # Perform Manual SLAM
    # operate = Operate(datadir, ppi)
    # operate.process()
    yolo = YoloV5()
    # boxes = np.zeros(5, dtype = np.int)
    # boxes[0] = 320
    # boxes[1] = 150
    # boxes[2] = 100
    # boxes[3] = 20
    # boxes[4] = 1
    boxes = []
    ClassID = []
    boxes.append([320, 150, 100, 20])
    boxes.append([100, 100, 100, 100])
    length = len(boxes)
    ClassID.append(1)
    ClassID.append(0)
    relative = yolo.calculate_relative_locations(boxes, ClassID)
    print(relative)
