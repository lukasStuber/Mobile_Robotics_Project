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
TIME_FACTOR = 0.75
# OBSTACLE AVOIDANCE
PROX_THRESHOLD = 1000
OBST_SPEED = 200
OBST_TURN_SPEED = 20
OBST_TIME = 1

WHEEL_RADIUS = 22   #[mm]

NOMINAL_AREA_LENGTH = 1475  # [mm]
NOMINAL_AREA_WIDTH = 730    # [mm]

NOISE_POS_XY = 0.64         # [mm2]
NOISE_POS_THETA = 2         # [rad^2]
NOISE_MEASURE_XY = 0.8      # [mm2]
NOISE_MEASURE_THETA = 0.5     # [rad^2]

#PATH FINDING
angle_tolerance = 2*math.pi/180