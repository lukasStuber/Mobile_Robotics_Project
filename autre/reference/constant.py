'''
constant.py
Module defining all the constants and magic numbers used to control the robot
Authors: Neil Chennoufi, Christopher Hémon, Jérémy Mayoraz, Arthur Rémondeau
'''
import math

DISTANCE_TOL=30
ANGLE_TOL=3/180*math.pi

DISTANCE_WHEELS=95 #distance between the wheels (in mm)
CALIBRATE_FACTOR=8/25 #Empirical factor found to get an accurate speed estimation.
ROBOT_LENGTH=100

D_REAL = 83         #real distance in mm between the two identifying points on the Thymio

EXT_SPEED=150
INT_SPEED=120

LA_INCREMENT=3
LA_MOVE_TIME=4
LA_SPEED=20

FORWARD_SPEED=190
TURNING_SPEED=70

ERODE_FACTOR=40
ERODE_FACTOR_CONNECTION_RATIO=0.5
