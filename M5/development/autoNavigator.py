import numpy as np
import matplotlib.pyplot as plt
import os, sys
import json
import math
import cv2
import cv2.aruco as cvAruco

import penguinPiC

# slam components
import slam.Slam as Slam
import slam.Robot as Robot
import slam.aruco_detector as aruco
import slam.Measurements as Measurements
import time

# yolo
from yolo.yolo import YOLO_v4

class Operate:
    def __init__(self, datadir, ppi):
        # Initialise
        self.ppi = ppi
        self.ppi.set_velocity(0, 0)
        self.img = np.zeros([240, 320, 3], dtype=np.uint8)
        self.aruco_img = np.zeros([240, 320, 3], dtype=np.uint8)

        # Get camera / wheel calibration info for SLAM
        camera_matrix, dist_coeffs, scale, baseline = self.getCalibParams(datadir)

        # SLAM components
        self.pibot = Robot.Robot(baseline, scale, camera_matrix, dist_coeffs)
        self.aruco_det = aruco.aruco_detector(self.pibot, marker_length=0.1)
        self.slam = Slam.Slam(self.pibot)

        # navigator components
        self.marker_list = []
        self.total_maker_num = 8
        self.fps = 5
        
        # initial location of the robot
        self.current_marker = 'start'
        self.saved_map = []
        self.startTime = 0
        self.counter = 0

        self.fig, self.ax = plt.subplots(1, 2)
        self.img_artist = self.ax[1].imshow(self.img)

        self.yolo = YOLO_v4("./yolo/yolov4-tiny-custom_2000.weights", "./yolo/yolov4-tiny-custom-1.cfg")

    def getCalibParams(self, datadir):
        # Imports camera / wheel calibration parameters
        fileK = "{}camera_calibration/intrinsic.txt".format(datadir)
        camera_matrix = np.loadtxt(fileK, delimiter=',')
        fileD = "{}camera_calibration/distCoeffs.txt".format(datadir)
        dist_coeffs = np.loadtxt(fileD, delimiter=',')
        fileS = "{}wheel_calibration/scale.txt".format(datadir)
        scale = np.loadtxt(fileS, delimiter=',')
        fileB = "{}wheel_calibration/baseline.txt".format(datadir)
        baseline = np.loadtxt(fileB, delimiter=',')

        return camera_matrix, dist_coeffs, scale, baseline

    def control(self, lv, rv):
        # Import teleoperation control signals)
        dt = time.time() - self.startTime

        drive_meas = Measurements.DriveMeasurement(lv, rv, dt)
        self.slam.predict(drive_meas)
        self.ppi.set_velocity(lv, rv, dt)
    
        self.startTime = time.time()
            
        
    def vision(self, turn):
        # Import camera input and ARUCO marker info
        self.img = self.ppi.get_image()
        lms, aruco_image = self.aruco_det.detect_marker_positions(self.img)

        # TODO: make sexi
        if self.counter % 5 == 0 and turn:
            objs = self.yolo.calculate_relative_locations(self.img)
            
        else if self.counter % 5 == 0:
            objs = self.yolo.calculate_relative_locations(self.img)

        self.counter = 0
        self.slam.add_landmarks(objs)
        self.slam.update(lms)
        return lms

    def action(self, lv, rv):
        turn = 0
        self.control(lv, rv)
        if  lv != rv:
            turn = 1
        _ = self.vision(turn)
        self.display(self.fig, self.ax)
        # self.write_map()

    def display(self, fig, ax):
        
        # Visualize SLAM
        ax[0].cla()
        self.slam.draw_slam_state(ax[0])

        ax[1].cla()
        ax[1].imshow(self.img[:, :, -1::-1])

        plt.pause(0.01)

    def write_map(self):
        # Output SLAM map as a json file
        map_dict = {"AR_tag_list":slam.taglist,
                    "map":slam.markers.tolist(),
                    "state_x": str(self.pibot.get_state()[0]),
                    "state_y":str(self.pibot.get_state()[1]), 
                    "state_z":str(self.pibot.get_state()[2]),
                    "covariance":slam.P[3:,3:].tolist()
                    }
        with open("slam.txt", 'w') as map_f:
            json.dump(map_dict, map_f, indent=2)

    def get_ids(self, image):
    # visualise ARUCO marker detection annotations
        aruco_params = cvAruco.DetectorParameters_create()
        aruco_params.minDistanceToBorder = 0
        aruco_params.adaptiveThreshWinSizeMax = 1000
        aruco_dict = cvAruco.Dictionary_get(cvAruco.DICT_4X4_100)
        marker_length = 0.1
        camera_matrix, dist_coeffs = self.pibot.camera_matrix, self.pibot.camera_dist

        corners, ids, rejected = cvAruco.detectMarkers(image, aruco_dict, parameters=aruco_params)
        rvecs, tvecs, _ = cvAruco.estimatePoseSingleMarkers(corners, marker_length, camera_matrix, dist_coeffs)

        return ids, corners, tvecs


    def spin_360(self):
        ''' spins the robot until first marker seen is 
            found again.
        '''

        # save all the seen markers and their estimated poses at each step
        measurements = []

        start_theta = self.slam.get_state_vector()[2]
        self.startTime = time.time()

        while True:

            # spinning and looking for markers at each step
            self.action(-30, 30)

            theta = self.slam.get_state_vector()[2]
            print("start theta : "+str(start_theta)+",  theta : " + str(theta))
        

            if abs(start_theta - theta) >= 2*math.pi:
                return measurements
            
            # pose estimation
            lm_measurement = self.vision()
            if len(lm_measurement) > 0:
                taglist = self.slam.taglist
                slam_markers = self.slam.markers.tolist()

                for i in range(len(taglist)):
                    # tag parameters
                    tag_id = taglist[i]
                    x_id   = slam_markers[0][i]
                    y_id   = slam_markers[1][i]

                    # robot parameters
                    x = self.slam.get_state_vector()[0]
                    y = self.slam.get_state_vector()[1]
                    dist =  dist = np.sqrt((x_id-x) ** 2 + (y_id - y) ** 2)

                    measurements.append([tag_id, dist, x_id, y_id])
                
            
    def find_target(self, target):
        target_id = [target[0]]
        
        self.startTime = time.time()
        while True:

            self.action(-20, 20)

            # get current frame
            curr = self.ppi.get_image()
            ids, _, _ = self.get_ids(curr)
            print("searching")
            if ids is not None:
                if target_id in ids:
                    return
            
    def center_target(self, target):

        target_id = [target[0]]

        self.startTime = time.time()
        while True:
            # get current frame
            curr = self.ppi.get_image()

            ids, corners, _ = self.get_ids(curr)
            
            if ids is not None:
                if target_id in ids:
                    
                    indx = list(ids).index(target_id)
                    corner = corners[indx]
                    centerX = (corner[0][0][0] + corner[0][1][0] + corner[0][2][0] + corner[0][3][0]) / 4
                    print("Centering...")
                    # stop if aruco is almost centered
                    if centerX < 280 :
                        self.action(-30, 30)
                    if centerX > 360:
                        self.action(-30, 30)
                    if 280 < centerX < 360:
                        return

    def move_forward(self, target):
        
        target_id = [target[0]]

        self.startTime = time.time()
        while True:
            curr = self.ppi.get_image()

            ids, corners, _ = self.get_ids(curr)

            if ids is not None:
                if target_id in ids:
                    indx = list(ids).index(target_id)
                    corner = corners[indx]
                    centerX = (corner[0][2][0] + corner[0][3][0]) / 2
                    centerY = (corner[0][2][1] + corner[0][3][1]) / 2
                    
                    print("x")
                    print(centerX)
                    print("y")
                    print(centerY)

                    # stop if aruco is almost centered
                    if centerY > 130:
                        # stop if aruco is almost centered
                        if centerX < 280 :
                            self.action(-30, 30)
                        if centerX > 360:
                            self.action(30, -30)
                        if 280 < centerX < 360:
                            self.action(30, 30)
                    else:
                        return
            else:
                return

    def process(self):

        # Main loop
        while len(self.marker_list) < self.total_maker_num:
            # Run SLAM
            self.ppi.set_velocity(0, 0)
            time.sleep(1)
            measurements = self.spin_360()
            self.ppi.set_velocity(0, 0)
            
            # expand the map by going to the nearest marker
            measurements = sorted(measurements, key=lambda x: x[1]) # sort seen markers by distance (closest first)
    
            if len(measurements) > 0:
                # add discovered markers to map
                for accessible_marker in measurements:
                    if self.current_marker != accessible_marker[0]: # avoid adding path to self
                        path = []
                        path.append(self.current_marker)
                        path.append(accessible_marker[0])
                        path.append(accessible_marker[1])
                        self.saved_map.append(path)
                        if accessible_marker[0] not in [found[0] for found in self.marker_list]: # avoid adding repeated marker
                            self.marker_list.append([accessible_marker[0], accessible_marker[2], accessible_marker[3]])
                        else:
                            continue 
                    else:
                        continue

            target = self.marker_list[-1]
            print(self.marker_list)
            
            self.find_target(target)
            self.center_target(target)
            self.ppi.set_velocity(0, 0)
            
            # drive towards
            self.move_forward(target)
            self.ppi.set_velocity(0, 0)


if __name__ == "__main__":

    # Location of the calibration files
    currentDir = os.getcwd()
    datadir = "{}/calibration/".format(currentDir)
    # connect to the robot
    ppi = penguinPiC.PenguinPi()

    # Perform Manual SLAM
    operate = Operate(datadir, ppi)
    operate.process()
