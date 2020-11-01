import numpy as np
import cv2

class YOLO_v4:
    def __init__(self, weights, config):
        self.net = cv2.dnn.readNetFromDarknet(config, weights)
        # determine only the *output* layer names that we need from YOLO
        self.ln = self.net.getLayerNames()
        self.ln = [self.ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        # initialize the width and height of the frames in the video file
        self.W = None
        self.H = None
        self.COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),dtype="uint8")

        self.coke_dimensions = [0.06, 0.06, 0.14]
        self.sheep_dimensions = [0.108, 0.223, 0.204]
        self.camera_w = 640
        self.camera_h = 480
        self.half_size_x = self.camera_w / 2
        self.half_size_y = self.camera_h / 2
        self.h_fov = 0.8517  # rad unit
        self.focal_length = (self.camera_h / 2) / np.tan(self.h_fov / 2)
    def detect(self, frame ):
        if self.W is None or self.H is None:
            self.H, self.W = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)

        self.net.setInput(blob)
        layerOutputs = self.net.forward(self.ln)


        # initialize our lists of detected bounding boxes, confidences,
        # and class IDs, respectively
        boxes = []
        confidences = []
        classIDs = []
        # loop over each of the layer outputs
        for output in layerOutputs:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence (i.e., probability)
                # of the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > 0.5:
                    # scale the bounding box coordinates back relative to
                    # the size of the image, keeping in mind that YOLO
                    # actually returns the center (x, y)-coordinates of
                    # the bounding box followed by the boxes' width and
                    # height
                    box = detection[0:4] * np.array([self.W, self.H, self.W, self.H])
                    (centerX, centerY, width, height) = box.astype("int")
                    # use the center (x, y)-coordinates to derive the top
                    # and and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    # update our list of bounding box coordinates,
                    # confidences, and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)
                # apply non-maxima suppression to suppress weak, overlapping
                # bounding boxes
                idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)
                # ensure at least one detection exists
                if len(idxs) > 0:
                    # loop over the indexes we are keeping
                    for i in idxs.flatten():
                        # extract the bounding box coordinates
                        (x, y) = (boxes[i][0], boxes[i][1])
                        (w, h) = (boxes[i][2], boxes[i][3])
                        # draw a bounding box rectangle and label on the frame
                        color = [int(c) for c in self.COLORS[classIDs[i]]]
                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                        text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                        cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


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
