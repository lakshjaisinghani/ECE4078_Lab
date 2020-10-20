import time
import cv2
import numpy as np
import cv2.aruco as aruco

import penguinPiC
ppi = penguinPiC.PenguinPi()

def measure_spin():
    ''' spins the robot until first marker seen is 
        found again.
    '''
    first_spin_marker_id = 0
    prev_marker_id       = 0
    cnt = 0
    flag = 0

    # save all the seen markers and their estimated poses at each step
    measurements = []
    seen_ids = []
    while True:

        # spinning and looking for markers at each step
        ppi.set_velocity(-wheel_vel, wheel_vel, 1/fps)
        ppi.set_velocity(0, 0)

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

        
        # compute a marker's estimated pose and distance to the robot
        if ids is None:
            continue
        else:
            for i in range(len(ids)):
                if cnt == 0:
                    first_spin_marker_id = ids[i,0]
                idi = ids[i,0]
                print(first_spin_marker_id, idi)

                # Some markers appear multiple times but should only be handled once.
                if idi in seen_ids:
                    if idi == first_spin_marker_id and prev_marker_id != first_spin_marker_id:
                        flag = 1
                else:
                    seen_ids.append(idi)

                cnt += 1
                prev_marker_id = idi
                
                # return if first marker has been seen
                if flag:
                    return measurements

                # pose estimation
                lm_tvecs = tvecs[ids==idi].T
                lm_bff2d = np.block([[lm_tvecs[2,:]],[-lm_tvecs[0,:]]])
                lm_bff2d = np.mean(lm_bff2d, axis=1).reshape(-1,1)
                # compute Euclidean distance between the robot and the marker
                dist = np.sqrt((lm_bff2d[0][0]-robot_pose[0]) ** 2 + (lm_bff2d[1][0]-robot_pose[1]) ** 2)
                # save marker measurements and distance
                lm_measurement = [idi, dist, lm_bff2d[0][0], lm_bff2d[1][0]]
                measurements.append(lm_measurement)

def spin_center(marker_id, direction, vel):
    while True:

        # rotate in opposite direction
        if not direction:
            vel *= -1 

        ppi.set_velocity(-vel, vel, 1/fps)
        ppi.set_velocity(0, 0)

        # get current frame
        curr = ppi.get_image()

        # visualise ARUCO marker detection annotations
        aruco_params = aruco.DetectorParameters_create()
        aruco_params.minDistanceToBorder = 0
        aruco_params.adaptiveThreshWinSizeMax = 1000
        aruco_dict = aruco.Dictionary_get(cv2.aruco.DICT_4X4_100)

        corner, ids, rejected = aruco.detectMarkers(curr, aruco_dict, parameters=aruco_params)
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corner, marker_length, camera_matrix, dist_coeffs)

        if ids is not None:
            if ids[0][0] == marker_id:
                centerX = (corner[0][0][0][0] + corner[0][0][1][0] + corner[0][0][2][0] + corner[0][0][3][0]) / 4
                centerY = (corner[0][0][0][1] + corner[0][0][1][1] + corner[0][0][2][1] + corner[0][0][3][1]) / 4
                center = (int(centerX), int(centerY))
                print(center)
                # stop if aruco is almost centered
                if 280 < centerX < 360:
                    return 0
                elif 240 < centerX < 280:
                    # slightly on left
                    return -1
                else if 360 < centerX < 400
                    # slightly on right

def get_dir(seen_ids, closest_marker_id):
    # TODO: figure spinning direction
    return True

if __name__ == "__main__":
    
    # initialize resulting map containing paths between markers
    marker_list = []
    saved_map = []
    map_f = 'map.txt'
    total_marker_num = 6

    # drive settings
    wheel_vel = 40
    fps = 10

    # camera calibration parameters (from M2: SLAM)
    camera_matrix = np.loadtxt('camera_calibration/intrinsic.txt', delimiter=',')
    dist_coeffs = np.loadtxt('camera_calibration/distCoeffs.txt', delimiter=',')
    marker_length = 0.1

    # wheel calibration parameters (from M2: SLAM)
    wheels_scale = np.loadtxt('wheel_calibration/scale.txt', delimiter=',')
    wheels_width = np.loadtxt('wheel_calibration/baseline.txt', delimiter=',')

    # initial location of the robot
    robot_pose = [0,0]
    current_marker = 'start'

    # 15 minutes time-out to prevent being stuck during auto nav
    timeout = time.time() + 60*15  
    start_t = time.time()
    
    seen_ids = []

    while len(marker_list) < total_marker_num:
        # spin until you see first marker
        # then stop
        measurements = measure_spin()
        ppi.set_velocity(0, 0)
        print("measurements done")
        
        # expand the map by going to the nearest marker
        measurements = sorted(measurements, key=lambda x: x[1]) # sort seen markers by distance (closest first)
        
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
            
            print(marker_list)
            
            # find closest marker and center
            # bot and stop
            # closest_marker_id = marker_list[0][0]
            # print(closest_marker_id)
            # ppi.set_velocity(0, 0)
            # direction = True
            # spin_center(closest_marker_id, direction, 30)
            # ppi.set_velocity(0, 0)

            break
        break

        #     #  move forward 
        #     dist_closest_marker = marker_list[0]
        #     dist_closest_marker = np.sqrt((marker_list[0][1]-robot_pose[0]) ** 2 + (marker_list[0][2]-robot_pose[1]) ** 2)
        #     time = dist_closest_marker/wheel_vel - 10
        #     print(time)
        #     ppi.set_velocity(wheel_vel, wheel_vel, time)
            
        #     # update robot pose
        #     robot_pose = [measurements[0][2],measurements[0][3]]
        #     current_marker = measurements[0][0]

        #     print('current map [current marker id, accessible marker id, distance]:\n',saved_map)
        #     print('current marker list [id, x, y]:\n',marker_list)

        # else:
        #     print('no markers in sight!')

        # # time out after 15 minutes
        # if time.time() > timeout:
        #     break
        

