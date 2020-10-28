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
# there are 6 markers in total in the arena
total_marker_num = 6

# drive settings, feel free to change them
wheel_vel = 50 
fps = 10


# camera calibration parameters (from M2: SLAM)
camera_matrix = np.loadtxt('camera_calibration/intrinsic.txt', delimiter=',')
dist_coeffs = np.loadtxt('camera_calibration/distCoeffs.txt', delimiter=',')
marker_length = 0.1

# display window for visulisation
cv2.namedWindow('video', cv2.WINDOW_NORMAL);
cv2.setWindowProperty('video', cv2.WND_PROP_AUTOSIZE, cv2.WINDOW_AUTOSIZE);
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


# repeat until all markers are found or until time out
while len(marker_list) < total_marker_num:
        # ------------------------------------------------------------------------------------
    # TODO: calculate the time the robot needs to spin 360 degrees


    spin_time = 3 #random time allows for more than one spin
    # ------------------------------------------------------------------------------------

    # save all the seen markers and their estimated poses at each step
    measurements = []
    seen_ids = []
    for step in range(int(spin_time*fps)):
        print ("looking for marker!!!")
        # spinning and looking for markers at each step
        #print(step)
        ppi.set_velocity(-wheel_vel, wheel_vel, 2/fps)
        #ppi.set_velocity(0, 0) #could be causing issues try muting
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

        # scale to 144p
        resized = cv2.resize(curr, (960, 720), interpolation = cv2.INTER_AREA)

        # add GUI text
        cv2.putText(resized, 'PenguinPi', (15, 50), font, font_scale, font_col, line_type)

        # visualisation
        cv2.imshow('video', resized)
        cv2.waitKey(1)


        # compute a marker's estimated pose and distance to the robot
        if ids is None: #if no marker visible
            continue
        else: #spotted a marker
            print(int(ids))
            for i in range(len(ids)):
                idi = ids[i,0]
                #corneri = int(corners[i][0][0][0])
                #print(corneri)
                # Some markers appear multiple times but should only be handled once.
                if idi in seen_ids:
                    continue
                else:
                    seen_ids.append(idi)
                # get pose estimation
                # ------------------------------------------------------------------------------------
                # 2TODO1: this is a basic implementation of pose estimation, change it to improve your auto nav
                #Improve the pose estimation of the ARUCO markers
                lm_tvecs = tvecs[ids==idi].T
                lm_bff2d = np.block([[lm_tvecs[2,:]],[-lm_tvecs[0,:]]])
                lm_bff2d = np.mean(lm_bff2d, axis=1).reshape(-1,1)

                # compute Euclidean distance between the robot and the marker
                dist = np.sqrt((lm_bff2d[0][0]-robot_pose[0]) ** 2 + (lm_bff2d[1][0]-robot_pose[1]) ** 2)

                # save marker measurements and distance
                lm_measurement = [idi, dist, lm_bff2d[0][0], lm_bff2d[1][0]]
                measurements.append(lm_measurement)
                #----------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------
    # expand the map by going to the nearest marker
    # 2TODO2: notice that the robot can get stuck always trying to reach the nearest marker, improve the search strategy to improve auto nav
    # Improve the search strategy for expanding the map at [line 125
    print("spin complete") #finished the innitial spin period
    measurements = sorted(measurements, key=lambda x: x[1]) # sort seen markers by distance (closest first)
    #so long as it sees a value will go through list of measurement markers to find closest unvisited target
    if len(measurements) > 0:
        targetMarker = measurements[0]
    else:
        targetMarket = -1

    #stop bot returning to old markers
    if len(marker_list) > 0:
        targetMarker = -1

        for i in range (len(measurements)):
            for z in range (len(marker_list)):
                marker = int (measurements[len(measurements)-(1+i)][0])

                if marker != int(marker_list[z][0]):
                    targetMarker = marker

    #infinite crash loop in case no visible markers NEED TO FIX THIS AND BREAK LOOP INSTEAD
    while targetMarker == -1:
        ppi.set_velocity(-0, 0, 2/fps)


    if len(measurements) > 0:
        # add discovered markers to map
        for accessible_marker in measurements:
            print(accessible_marker)
            if current_marker != accessible_marker[0]: # avoid adding path to self
                path = []
                path.append(current_marker)
                path.append(accessible_marker[0])
                path.append(accessible_marker[1])
                saved_map.append(path)
                if accessible_marker[0] not in [found[0] for found in marker_list]: # avoid adding repeated marker
                    marker_list.append([accessible_marker[0], accessible_marker[2], accessible_marker[3]])
                    last_seen_marker = accessible_marker[0]
                else:
                    continue
            else:
                continue

        print("target markers")   
        print(int(targetMarker[0]))
        # drive to the nearest marker by first turning towards it then driving straight
        # TODO: calculate the time the robot needs to spin towards the nearest marker
        #spin time (spin time = 360)
    search =1
    drive =0
    sv= 40
    adjust = 10
    lost = 0

    #after loop look for closest chosen marker (called target)
    while search:
        print("search")
        ppi.set_velocity(-sv, sv, 2/fps)
        curr = ppi.get_image()
        corners, ids, rejected = aruco.detectMarkers(curr, aruco_dict, parameters=aruco_params)
        

        if ids is None: #if no marker visible
            continue
        else: #spotted a marker
            ppi.set_velocity(0, -0, 0.5)
            for i in range(len(ids)): 
                idi = int(ids[i,0])

                if int(idi) == int(targetMarker[0]):
                    #takes new screen shot to look for ARUCO in front
                    ppi.set_velocity(0, -0, 1)
                    print("found")

                    cornerx = int(corners[i][0][0][0])
                    cornerx2 = int(corners[i][0][1][0]) #might need to change second last number to select corner
                    print(cornerx, cornerx2)
                    corneri = (cornerx + cornerx2)/2
                    print(corneri)

                    ppi.set_velocity(0, -0, 1)#delay
                    spinout =0
                    
                    while corneri < 295 or corneri > 425 and spinout<10:
                        curr = ppi.get_image()
                        corners, ids, rejected = aruco.detectMarkers(curr, aruco_dict, parameters=aruco_params)
                        print("lining up")
                        #recovery timer and double check the targetted marker
                        if ids != None:
                            print(ids)
                            for z in range(len(ids)): 
                                idi = ids[z,0]
                                if int(idi) == int(targetMarker[0]):
                                    i = z
                                    print(i)

                            cornerx = int(corners[i][0][0][0])
                            cornerx2 = int(corners[i][0][1][0]) #might need to change second last number to select corner
                            #print(cornerx, cornerx2)
                            corneri = (cornerx + cornerx2)/2
                            print("target = ", corneri)
                            #course correction stuff
                            if corneri > 300:
                                ppi.set_velocity(20, -20, 1/fps)
                                ppi.set_velocity(0, -0, 0.2)
                            if corneri < 420:
                                ppi.set_velocity(-20, 20, 1/fps)
                                ppi.set_velocity(0, -0, 0.2)
                        else:
                            print("lost")
                            spinout+=1
                            continue

                    if corneri >295 and corneri < 425:
                        search = 0
                        drive =1
                        print("target lined")
                        #delay
                        ppi.set_velocity(0, -0, 2)
                    #after locking on move towards goal
                    while drive and lost < 5:
                        print("driving")
                        ppi.set_velocity(30, 30, 2/fps)
                        curr = ppi.get_image()
                        corners, ids, rejected = aruco.detectMarkers(curr, aruco_dict, parameters=aruco_params)
                        if ids != None:
                            lost =0
                            for z in range(len(ids)): 
                                idi = ids[z,0]
                                if int(idi) == int(targetMarker[0]):
                                    i = z
                            cornerx = int(corners[i][0][0][0])
                            cornerx2 = int(corners[i][0][1][0]) #might need to change second last number to select corner
                            corneri = (cornerx + cornerx2)/2
                            print("midpoint = ", corneri)

                            ppi.set_velocity(25,25 , 5/fps)
                            print("corner dist ", cornerx2-cornerx)

                            if cornerx2-cornerx > 90:
                                drive = 0

                            if corneri < 290:
                                ppi.set_velocity(20, -20, 1/fps)
                                ppi.set_velocity(0, -0, 0.1)
                            if corneri > 430:
                                ppi.set_velocity(-20, 20, 1/fps)
                                ppi.set_velocity(0, -0, 0.1)
                        else:
                            lost +=1

                    
                    print("marker complete")

                    # TODO2: you may implement an alterative approach that combines turning and driving forward
                    # update the robot's pose to location of the marker it tries to reach
                    # 2TODO3: notice that the robot may not reach the marker, improve auto nav by improving the pose estimation
                    robot_pose = [measurements[0][2],measurements[0][3]]
                    current_marker = measurements[0][0]

                    print('current map [current marker id, accessible marker id, distance]:\n',saved_map)
                    print('current marker list [id, x, y]:\n',marker_list)
                    ppi.set_velocity(0, -0, 2)

    else:
        print('no markers in sight!')
    # ------------------------------------------------------------------------------------

    # time out after 15 minutes
    if time.time() > timeout:
        break

# # show time spent generating the map
# end_t = time.time()
# map_t = (end_t - start_t) / 60
# print('time spent generating the map (in minutes): ',map_t)

# # save results to map.txt
# # sort marker list by id before printing
# marker_list = sorted(marker_list, key=lambda x: x[0])
# with open(map_f,'w') as f:
#     f.write('id, x, y\n')
#     for markers in marker_list:
#         for marker in markers:
#             f.write(str(marker) + ',')
#         f.write('\n')
#     f.write('\ncurrent id, accessible id, distance\n')
#     for routes in saved_map:
#         for route in routes:
#             f.write(str(route) + ',')
#         f.write('\n')
# print('map saved!')























