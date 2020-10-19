# assumption of this basic vision-based auto nav implementation: 
# if the robot can see a marker, then that means there is a path to that marker

# implementation explanation
# the robot first spins 360 to find all visible markers and their estimated pose
# it then creates a list of markers that is reachable from current location and their estimated distance
# the robot now goes to the nearest reachable marker A
# once reached marker A, the robot spins 360 again to find all accessable markers from marker A and their estimated pose
# repeat until all markers are found or timed out

# import required modules
import time
import cv2
import numpy as np
import cv2.aruco as aruco

# don't forget to put penguinPiC.py in the same directory
import penguinPiC
ppi = penguinPiC.PenguinPi()

# initialize resulting map containing paths between markers
marker_list = []
saved_map = []
map_f = 'map.txt'
# there are 8 markers in total in the arena
total_marker_num = 8

# drive settings, feel free to change them
wheel_vel = 40
fps = 10

# camera calibration parameters (from M2: SLAM)
camera_matrix = np.loadtxt('camera_calibration/intrinsic.txt', delimiter=',')
dist_coeffs = np.loadtxt('camera_calibration/distCoeffs.txt', delimiter=',')
marker_length = 0.1

# display window for visulisation
# cv2.namedWindow('video', cv2.WINDOW_NORMAL);
# cv2.setWindowProperty('video', cv2.WND_PROP_AUTOSIZE, cv2.WINDOW_AUTOSIZE);
# font display options
font = cv2.FONT_HERSHEY_SIMPLEX
location = (0, 0)
font_scale = 1
font_col = (255, 255, 255)
line_type = 2

# initial location of the robot
robot_pose = [0,0]
current_marker = 'start'

# 15 minutes time-out to prevent being stuck during auto nav
timeout = time.time() + 60*15  

start_t = time.time()
init_time = time.time()

def capture():
    # get current frame
    curr = ppi.get_image()

    # visualise ARUCO marker detection annotations
    aruco_params = aruco.DetectorParameters_create()
    aruco_params.minDistanceToBorder = 0
    aruco_params.adaptiveThreshWinSizeMax = 1000
    aruco_dict = aruco.Dictionary_get(cv2.aruco.DICT_4X4_100)

    corners, ids, rejected = aruco.detectMarkers(curr, aruco_dict, parameters=aruco_params)
    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_length, camera_matrix, dist_coeffs)

    aruco.drawDetectedMarkers(curr, corners, ids) # for detected markers show their ids
    aruco.drawDetectedMarkers(curr, rejected, borderColor=(100, 0, 240)) # unknown squares

    id = []

    if ids is None:
        return id
    else:
        for i in range(len(ids)):
            idi = ids[i,0]
            id.append(idi)
    
    return id

def Check(count = 3):
    ids = []
    for i in range(count):
        ids.extend(capture())
    return ids

def Detection():
    measurements = []
    # get current frame
    curr = ppi.get_image()

    # visualise ARUCO marker detection annotations
    aruco_params = aruco.DetectorParameters_create()
    aruco_params.minDistanceToBorder = 0
    aruco_params.adaptiveThreshWinSizeMax = 1000
    aruco_dict = aruco.Dictionary_get(cv2.aruco.DICT_4X4_100)

    corners, ids, rejected = aruco.detectMarkers(curr, aruco_dict, parameters=aruco_params)
    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_length, camera_matrix, dist_coeffs)

    aruco.drawDetectedMarkers(curr, corners, ids) # for detected markers show their ids
    aruco.drawDetectedMarkers(curr, rejected, borderColor=(100, 0, 240)) # unknown squares

    if ids is None:
        return [[0]]
    else:
        for i in range(len(ids)):
            idi = ids[i,0]
            # if idi in seen_measurements
            #     measurements.append(lm_measurement)
            # else:
            #     seen_ids.append(idi)
            # ------------------------------------------------------------------------------------
            # TODO: this is a basic implementation of pose estimation, change it to improve your auto nav
            lm_tvecs = tvecs[ids==idi].T
            lm_bff2d = np.block([[lm_tvecs[2,:]],[-lm_tvecs[0,:]]])
            lm_bff2d = np.mean(lm_bff2d, axis=1).reshape(-1,1)
            # compute Euclidean distance between the robot and the marker
            dist = np.sqrt((lm_bff2d[0][0]-robot_pose[0]) ** 2 + (lm_bff2d[1][0]-robot_pose[1]) ** 2)
            # save marker measurements and distance
            lm_measurement = [idi, dist, lm_bff2d[0][0], lm_bff2d[1][0]]
            measurements.append(lm_measurement)
    return measurements

    # [[first_id, dist, x, y], [next_id, dist, x, y], ...]

def getCurrentId():
    current_id = Detection()
    turn_time = 1/fps
    while(len(current_id) > 1):
        ppi.set_velocity(-wheel_vel, wheel_vel, turn_time)    
        ppi.set_velocity(0, 0)
        current_id = Detection()
        
    return current_id

def Spin360():
    # closestId[id, x, y]
    past_id = 100
    turn_time = 1/fps
    measurement = getCurrentId()
    current_id = measurement[0][0]
    seen_id = []

    measurements = []

    # find the first_id
    while(current_id == 0):
        ppi.set_velocity(-wheel_vel, wheel_vel, turn_time)    
        ppi.set_velocity(0, 0)
        measurement = getCurrentId()
        current_id = measurement[0][0]
    else: 
        first_id = current_id

    past_id = current_id
    seen_id.append(current_id)
    measurements.append(measurement[0])

    while(past_id == current_id): 
        past_id = current_id
        ppi.set_velocity(-wheel_vel, wheel_vel, turn_time)    
        ppi.set_velocity(0, 0)
        measurement = getCurrentId()
        current_id = measurement[0][0]
    
    while(first_id != current_id):    
        ppi.set_velocity(-wheel_vel, wheel_vel, turn_time)    
        ppi.set_velocity(0, 0)
        measurement = getCurrentId()
        current_id = measurement[0][0]
        
        if current_id not in seen_id and current_id != 0: 
            seen_id.append(current_id)
            measurements.append(measurement[0])

    print("rotate by 360")

    return measurements



  
def DriveForward(closestID):
    # 2 times in a row 
    forward_vel = 100
    print("\ndrive forward\n")
    global robot_pose, current_marker
    drive_time = 1/fps * 5
    ids = Check()
    print(ids)

    while closestID[0] in ids:
        ppi.set_velocity(forward_vel, forward_vel, drive_time)    
        ppi.set_velocity(0, 0)
        ids = Check()
        print(ids)
    robot_pose = closestID[1:]
    current_marker = closestID[0]

def TurnToMarker(closestID):
    # closestID = Spin360()
    print("\nturn to the closest marker\n")
    turn_time = 1/fps
    id = getCurrentId()[0][0]
    
    while closestID[0] != id:
        ppi.set_velocity(-wheel_vel, wheel_vel, turn_time)    
        ppi.set_velocity(0, 0)
        id = getCurrentId()[0][0]
    
    
    # DriveForward(closestID)

# def GoBackToPrevious(ppi, id)
# id = [12, -0.5, 1.53]
# DriveForward(id)
# getCurrentId()
# print(len(Detection()))

while len(marker_list) < total_marker_num:
    measurements = Spin360()
    measurements = sorted(measurements, key=lambda x: x[1]) # sort seen markers by distance (closest first)
    closestId = measurements[0]
    print(closestId)

    if len(measurements) > 0:
        # add discovered markers to map
        for accessible_marker in measurements:
            if current_marker != accessible_marker[0]: # avoid adding path to self
                path = []
                path.append(current_marker)
                path.append(accessible_marker[0])
                path.append(accessible_marker[1])
                saved_map.append(path)
                if accessible_marker[0] not in [found[0] for found in marker_list]: # avoid adding repeated marker
                    marker_list.append([accessible_marker[0], accessible_marker[2], accessible_marker[3]])
                else:
                    continue
            else:
                continue

        # drive to the nearest marker by first turning towards it then driving straight
        # TODO: calculate the time the robot needs to spin towards the nearest marker
        # turn_time = ?
        
        TurnToMarker(closestId)
        # TODO: calculate the time the robot needs to drive straight to the nearest marker
        # drive_time = ?
        DriveForward(closestId)
        # TODO: you may implement an alterative approach that combines turning and driving forward

        # update the robot's pose to location of the marker it tries to reach
        # TODO: notice that the robot may not reach the marker, improve auto nav by improving the pose estimation
        # robot_pose = [measurements[0][2],measurements[0][3]]
        # current_marker = measurements[0][0]

        print('current map [current marker id, accessible marker id, distance]:\n',saved_map)
        print('current marker list [id, x, y]:\n',marker_list)

    else:
        print('no markers in sight!')
