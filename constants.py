import math

# environment constants
NOMINAL_AREA_LENGTH = 1475  # mm
NOMINAL_AREA_WIDTH = 730 # mm
NOISE_POS_XY = 0.64 # mm^2
NOISE_POS_THETA = 2 # rad^2
NOISE_MEASURE_XY = 0.8 # mm^2
NOISE_MEASURE_THETA = 0.5 # rad^2
# robot constants
WHEEL_DIST = 95 # mm
WHEEL_RADIUS = 22 # mm
SPEED_TO_MMS = 0.32
# timers
MOVE_INTERVAL = 0.21 # s # allow for get_prox() and move() (100ms each)
ODOMETRY_INTERVAL = 0.025 # s
IMAGE_PROCESSING_INTERVAL = 1.0 # s
# movement
STANDARD_SPEED = 200 # 32cm/5s
DIST_TOL = 10 # mm
ANGLE_TOL = 0.1 # rad
MAX_TIME = 0.3
# obstacle avoidance
PROX_THRESHOLD = 1000
OBST_SPEED = 200
OBST_TURN_SPEED = 20
OBST_TIME = 1