import time
import math
import cv2
import numpy as np
import serial
import sys
import camera
import Extended_Kalman_Filter
import global_avoidance
import local_avoidance
import RepeatedTimer
import utilsRobot

from constant import DISTANCE_TOL,ANGLE_TOL,DISTANCE_WHEELS,LA_INCREMENT,LA_SPEED,FORWARD_SPEED,TURNING_SPEED,CALIBRATE_FACTOR

print(">> Starting Thymio")
from Thymio import Thymio

th = Thymio.serial(port="COM7", refreshing_rate=0.1) #Put your USB modem right
time.sleep(0.5) #Making sure Thymio has time to connect
th.set_var_array("leds.top",[0,0,0])
th.set_var_array('leds.bottom.right',[0,0,0])
th.set_var_array('leds.bottom.left',[0,0,0])
th.set_var_array("leds.circle",[0,0,0])

print(">> Starting Camera")
cam=camera.Camera()

print(">> Initializing variables")
GA = global_avoidance.Global_Avoidance()
ur = utilsRobot.utilsRobot(thymio=th)
la = local_avoidance.Local_Avoidance(th, ur)

GA.map=cam.take_picture()

print(">> Searching for goal")
if not GA.find_goal(GA.map):
    print("ERROR: Could not find goal")
    th.close(); cam.destroy(); sys.exit()

print(">> Searching for robot")
success, ur.coords, ur.angle, GA.patched_map = ur.find_robot(GA.patched_map)
if not success:
    print("ERROR: Could not find robot")
    th.close(); cam.destroy(); sys.exit()
GA.start_coords=ur.coords
ur.image_pixel_scale()

print(">> Searching for path")
GA.find_obstacles(GA.patched_map)
if not GA.find_path():
    print("ERROR: Could not find path")
    th.close(); cam.destroy(); sys.exit()
GA.display_path(GA.map, robot_coords=ur.coords, freeze=True, path=True, connections=True, corners=True)
GA.path.pop(0)

print(">> Initializing filter")
EKF = Extended_Kalman_Filter.Extended_Kalman_Filter()
EKF.Sigma = np.eye(5)
EKF.Mu = [ur.coords[0],ur.coords[1],ur.angle,0,0]
EKF.input_speed=CALIBRATE_FACTOR*50

print(">> Initializing threading functions")
motion_callback=RepeatedTimer.RepeatedTimer(None, ur.stop)
increment_callback=RepeatedTimer.RepeatedTimer(0.1, ur.increment_motor, la.side, LA_INCREMENT)

for goal in GA.path:
    print(">>> Aiming towards ",goal)
    while ((goal[1]-ur.coords[1])**2+(goal[0]-ur.coords[0])**2)**0.5>DISTANCE_TOL:
        goal_angle=math.atan2((goal[1]-ur.coords[1]),(goal[0]-ur.coords[0]))
        delta_angle=ur.find_delta_angle(ur.angle,goal_angle)

        print(">>> Robot estimated position: x:{:.2f}, y:{:.2f} angle:{:.2f}".format(ur.coords[0], ur.coords[1], ur.angle))
        print("    Distance from goal: {:.2f}, angle to goal: {:.2f}".format(((goal[1]-ur.coords[1])**2+(goal[0]-ur.coords[0])**2)**0.5, delta_angle*180/math.pi))
        while abs(delta_angle)>ANGLE_TOL:
            EKF.Mu = [ur.coords[0],ur.coords[1],ur.angle,0,0]

            motion_callback.interval, direction = ur.turn_angle(delta_angle, TURNING_SPEED)
            motion_callback.start()
            u=EKF.u_input(-direction*TURNING_SPEED*CALIBRATE_FACTOR,-direction*TURNING_SPEED*CALIBRATE_FACTOR, DISTANCE_WHEELS,ur.scaling_factor)

            while not ur.done_moving: continue
            motion_callback.stop()

            img=cam.take_picture()
            _, ur.coords, ur.angle,_=ur.find_robot(img)
            print(">>> Filtering")
            EKF.dt = time.time()-ur.time_stamp
            EKF.extended_kalman(u,EKF.capture_measurements(ur))
            ur.coords, ur.angle = (EKF.Mu[0],EKF.Mu[1]), EKF.Mu[2]
            print("    Result: \tPre filter: angle:{:.2f}\n\t\tPost filter: angle:{:.2f}".format(goal_angle,ur.angle))

            goal_angle=math.atan2((goal[1]-ur.coords[1]),(goal[0]-ur.coords[0]))
            delta_angle=ur.find_delta_angle(ur.angle,goal_angle)

        EKF.Mu = [ur.coords[0],ur.coords[1],ur.angle,0,0]
        u=EKF.u_input(FORWARD_SPEED*CALIBRATE_FACTOR,FORWARD_SPEED*CALIBRATE_FACTOR, DISTANCE_WHEELS,ur.scaling_factor)

        new_state="none"
        motion_callback.interval=ur.go_straight(((goal[1]-ur.coords[1])**2+(goal[0]-ur.coords[0])**2)**0.5, FORWARD_SPEED)
        motion_callback.start()
        while not ur.done_moving:
            time.sleep(0.001)
            if la.sawWall() or la.sawBlack():
                ur.stop(); motion_callback.stop(); increment_callback.stop()
                print("!!! Obstacle detected, switching to local avoidance")
                time.sleep(0.2)
                new_state="avoid"
                la.avoid_obstacle(LA_SPEED, new_state, motion_callback, increment_callback)

        motion_callback.stop(); increment_callback.stop()

        img=cam.take_picture()
        _, ur.coords, ur.angle,_=ur.find_robot(img)
        if new_state=="none":
            print(">>> Filtering")
            EKF.dt = time.time()-ur.time_stamp
            EKF.extended_kalman(u,EKF.capture_measurements(ur))

            ur.coords, ur.angle = (EKF.Mu[0],EKF.Mu[1]), EKF.Mu[2]
            print("    Result: \tPre filter: x:{:.2f} y:{:.2f}\n\t\tPost filter: x:{:.2f} y:{:.2f}".format(goal[0], goal[1], ur.coords[0], ur.coords[1]))
            GA.display_path(img, robot_coords=goal, est_robot_coords=ur.coords, freeze=False, path=False)
        else:
            print("!!! Searching for goal")
            EKF.Mu = [ur.coords[0],ur.coords[1],ur.angle,0,0]

        EKF.scaling_factor=ur.image_pixel_scale()

    print(">>> Arrived at ", goal)

ur.stop()
print(">>> Arrived at destination")
th.close(); cam.destroy()
