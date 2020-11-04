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
        self.travelled_markers = []
        self.seen_markers = []
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
            
        
    def vision(self, use_yolo):
        # Import camera input and ARUCO marker info
        self.img = self.ppi.get_image()
        lms, aruco_image = self.aruco_det.detect_marker_positions(self.img)
        objs = self.yolo.calculate_relative_locations(self.img)   
        self.slam.add_landmarks(lms, objs)
        self.slam.update(lms, objs)
        return lms

    def action(self, lv, rv, type):

        self.control(lv, rv)

        _ = self.vision(1)

        self.display(self.fig, self.ax)
        # write map
        self.write_map()

    def display(self, fig, ax):
        
        # Visualize SLAM
        ax[0].cla()
        self.slam.draw_slam_state(ax[0])

        ax[1].cla()
        ax[1].imshow(self.img[:, :, -1::-1])

        plt.pause(0.01)

    def write_map(self):
        map_f = "team_1_04.txt"
        s_cnt = 1 
        c_cnt = 1
        # sort marker list by id before printing

        marker_list = self.slam.markers
        taglist = list(self.slam.taglist)

        with open(map_f,'w') as f:
            f.write('id, x, y\n')
            for tag in taglist:
                indx  = taglist.index(tag)

                if tag <= -100:

                    f.write('sheep' + str(s_cnt) + ', ' + str(marker_list[0][indx]) + ', ' +  str(marker_list[1][indx]))
                    s_cnt += 1
                elif tag <= -1:
                    f.write('coke' + str(c_cnt) + ', ' + str(marker_list[0][indx]) + ', ' +  str(marker_list[1][indx]))
                    c_cnt += 1
                else:
                    f.write(str(tag) + ', ' + str(marker_list[0][indx]) + ', ' + str(marker_list[1][indx]))
                f.write('\n')
            f.write('\ncurrent id, accessible id, distance\n')
            for route in self.saved_map:
                f.write(str(route[0]) + ', ' + str(route[1]) + ', ' + str(route[2][0]))
                f.write('\n')
        #print('map saved!')

    def get_ids(self, image):
    # visualise ARUCO marker detection annotations
        aruco_params = cvAruco.DetectorParameters_create()
        aruco_params.minDistanceToBorder = 0
        aruco_params.adaptiveThreshWinSizeMax = 1000
        # aruco_params.maxMarkerPerimeterRate = 3
        # aruco_params.polygonalApproxAccuracyRate = 0.7
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
            self.action(-30, 30, "spin_360")

            theta = self.slam.get_state_vector()[2]
            print("start theta : "+str(start_theta)+",  theta : " + str(theta))
        

            if abs(start_theta - theta) >= 2*math.pi:
                return measurements
            
            # pose estimation
            lm_measurement = self.vision(1)
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

                    self.seen_markers = [s for s in list(self.slam.taglist) if s > 0]
                
            
    def find_target(self, target, direction):
        
        self.startTime = time.time()
        while True:
            if direction == 0:
                self.action(-30, 30, "find_target")
            elif direction == 1:
                self.action(30, -30, "find_target")
            # get current frame
            curr = self.ppi.get_image()
            ids, _, _ = self.get_ids(curr)
            print("searching")
            if ids is not None:
                if target in ids:
                    return
            
    def center_target(self, target):

        self.startTime = time.time()
        while True:
            # get current frame
            curr = self.ppi.get_image()

            ids, corners, _ = self.get_ids(curr)
            
            if ids is not None:
                if target in ids:
                    
                    indx = list(ids).index(target)
                    corner = corners[indx]
                    centerX = (corner[0][0][0] + corner[0][1][0] + corner[0][2][0] + corner[0][3][0]) / 4
                    print("Centering...")
                    # stop if aruco is almost centered
                    if centerX < 280 :
                        self.action(-30, 30, "center_target")
                    if centerX > 360:
                        self.action(30, -30, "center_target")
                    if 280 < centerX < 360:
                        return

    def move_forward(self, target):
        
        target_id = target

        self.startTime = time.time()
        while True:
            curr = self.ppi.get_image()

            ids, corners, _ = self.get_ids(curr)

            if ids is not None:
                if target in ids:
                    indx = list(ids).index(target)
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
                            self.action(-30, 30, " ")
                        if centerX > 360:
                            self.action(30, -30, " ")
                        if 280 < centerX < 360:
                            self.action(30, 30, "move_forward")
                    else:
                        self.travelled_markers.append(target)
                        self.current_marker = str(target)
                        return
            else:
                return
    
    def decide_target(self, current_viewable_ids):
        
        doable_ids = []
        for ids in current_viewable_ids:
            if ids in self.travelled_markers or ids < 0:
                continue
            doable_ids.append(ids)
        distances = []
        taglist = list(self.slam.taglist)
        spin_direction = []
        for x in doable_ids:
            indx  = taglist.index(x)
            tag_pos   = [self.slam.markers[:, indx][0], self.slam.markers[:, indx][1]]
            curr_pos = [self.slam.get_state_vector()[0][0], self.slam.get_state_vector()[1][0]]
            
            dist = ((tag_pos[0] - curr_pos[0]) ** 2 + (tag_pos[1] - curr_pos[1]) ** 2)**0.5
            distances.append(dist)

            tag_theta = tag_theta = np.arctan2(tag_pos[1]-curr_pos[1], tag_pos[0]-curr_pos[0])
            if tag_theta < 0:
                tag_theta = tag_theta+2*np.pi

            curr_theta = self.slam.get_state_vector()[2]
            curr_theta = curr_theta % (2*np.pi)
            if tag_theta - curr_theta <= np.pi:
                direction = 0
            elif tag_theta - curr_theta > np.pi:
                direction = 1

            spin_direction.append(direction)
        # min
        dist_to_go = min(distances)

        # max
        # dist_to_go = max(distances)
        target = doable_ids[distances.index(dist_to_go)]
        direction = spin_direction[distances.index(dist_to_go)]
        print("direction: "+str(direction))
        return target, direction

    def process(self):

        # Main loop
        while len(self.seen_markers) < self.total_maker_num:
            
            # Run SLAM
            self.ppi.set_velocity(0, 0)
            time.sleep(1)
            measurements = self.spin_360()
            self.ppi.set_velocity(0, 0)
            
            # expand the map by going to the nearest marker
            measurements = sorted(measurements, key=lambda x: x[1]) # sort seen markers by distance (closest first)
            current_viewable_ids = [ids for ids, _, _, _ in measurements]
            current_viewable_ids = list(dict.fromkeys(current_viewable_ids))
            print(current_viewable_ids)

            if len(measurements) > 0:
                # add discovered markers to map
                for accessible_marker in measurements:
                    existing = False
                    if self.current_marker != str(accessible_marker[0]) and accessible_marker[0] > 0: # avoid adding path to self
                        for route in self.saved_map:
                            if route[0] == self.current_marker and route[1] == accessible_marker[0]:
                                route[2] = accessible_marker[1]
                                existing = True
                        if not existing:
                            path = []
                            path.append(self.current_marker)
                            path.append(accessible_marker[0])
                            path.append(accessible_marker[1])
                            self.saved_map.append(path)
                    else:
                        continue

            # decide target
            target, direction = self.decide_target(current_viewable_ids)
            print("Target: " + str(target))
            
            self.find_target(target, direction)
            self.ppi.set_velocity(0, 0)
            time.sleep(2)
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
