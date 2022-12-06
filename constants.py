'''
constant.py
Module defining all the constants and magic numbers used to control the robot
Authors: Neil Chennoufi, Christopher Hémon, Jérémy Mayoraz, Arthur Rémondeau
'''
import math

# control constants
WHEEL_DIST = 95 # mm
SPEED_TO_MMS = 0.32
# TIMERS
MOVE_INTERVAL = 0.21 # s # leave time for get_prox() and move() (100ms each)
ODOMETRY_INTERVAL = 0.025 # s
# MOVEMENT
STANDARD_SPEED = 200 # 32cm/5s
DIST_TOL = 10 # mm
ANGLE_TOL = 0.1 # rad
MAX_TIME = 0.3
# OBSTACLE AVOIDANCE
PROX_THRESHOLD = 1000
OBST_SPEED = 200
OBST_TURN_SPEED = 20
OBST_TIME = 1

WHEEL_RADIUS = 22   #[mm]

NOMINAL_AREA_LENGTH = 1600  # [mm]
NOMINAL_AREA_WIDTH = 840    # [mm]