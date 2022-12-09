from __future__ import print_function
import cv2 as cv
from parking_segmentation import *
from color_segmentation import *
from color_centroids import *
from discretize_map import *
from control import ThymioControl
from Kalman import Kalman
from constants import *
from RepeatedTimer import RepeatedTimer

# IMAGE PROCESSING
id_camera = 1
##[Parking segmentation]
corners, destination_corners = set_parking_limits(id_camera)
##[Color segmentation]
segmentation, refined_color_dict_HSV, kernels, openings = get_color_mask(id_camera, corners, destination_corners, real_size=(NOMINAL_AREA_LENGTH, NOMINAL_AREA_WIDTH))
cv.namedWindow("Segmentation Result")
cv.imshow("Segmentation Result", segmentation)
key = cv.waitKey(0)
cv.destroyWindow("Segmentation Result")
##[Thymio and objective localization]
centroids = {'goal': (0, 0), 'thymio': (0, 0), 'green': (0, 0), 'blue': (0, 0)}
theta_thymio = 0
localization = None

kalman = Kalman()
thymio = ThymioControl()

def compute_centroids():
    global centroids, theta_thymio, localization
    centroids, theta_thymio, localization = get_centroids(id_camera, corners, destination_corners, refined_color_dict_HSV, kernels, openings, prev_centroids=centroids, real_size=(NOMINAL_AREA_LENGTH, NOMINAL_AREA_WIDTH), real_time=False)
    print("centroids at", centroids['thymio'][0], centroids['thymio'][1])
    print("thymio angle at ", theta_thymio)
    kalman.state_correct(np.array([centroids['thymio'][0], centroids['thymio'][1], theta_thymio]))
    thymio.position = (kalman.x[0], kalman.x[1])
    thymio.angle = kalman.x[2]

def odometry():
    kalman.state_prop(thymio.speed_target)
    thymio.position = (kalman.x[0], kalman.x[1])
    thymio.angle = kalman.x[2]
    print(thymio.position[0], thymio.position[1], thymio.angle*180/math.pi)

# def plot_localization():
#     global localization
#     cv.namedWindow("Localization Result")
#     cv.imshow("Localization Result", localization)
#     key = cv.waitKey(30)
#     cv.destroyWindow("Localization Result")

# initialise position and path
compute_centroids()
kalman.set_state((centroids['thymio'][0], centroids['thymio'][1], theta_thymio))
thymio.position = (centroids['thymio'][0], centroids['thymio'][1])
thymio.angle = theta_thymio
path = discretize_map(segmentation, centroids)
thymio.set_path(path)

# start updating position and follow path
image_timer = RepeatedTimer(1.5, compute_centroids)
odometry_timer = RepeatedTimer(ODOMETRY_INTERVAL, odometry)
image_timer.start()
odometry_timer.start()
thymio.follow_path()