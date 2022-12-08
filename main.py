from __future__ import print_function
import cv2 as cv
from image_processing.parking_segmentation import *
from image_processing.color_segmentation import *
from image_processing.color_centroids import *
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
segmentation, refined_color_dict_HSV, kernels, openings = get_color_mask(id_camera, corners, destination_corners)
cv.namedWindow("Segmentation Result")
cv.imshow("Segmentation Result", segmentation)
key = cv.waitKey(0)
cv.destroyWindow("Segmentation Result")
##[Thymio and objective localization]
centroids = {'goal': (0, 0), 'thymio': (0, 0), 'green': (0, 0), 'blue': (0, 0)}
theta_thymio = 0
localization = None

# start kalman
# kalman = Kalman(NOISE_POS_XY, NOISE_POS_XY, NOISE_POS_THETA, NOISE_MEASURE_XY, NOISE_MEASURE_XY)
# start path following
thymio = ThymioControl(position=(0,0), angle=0, kalman=Kalman(NOISE_POS_XY, NOISE_POS_XY, NOISE_POS_THETA, NOISE_MEASURE_XY, NOISE_MEASURE_XY))

def compute_centroids():
    global centroids, theta_thymio, localization
    centroids, theta_thymio, localization = get_centroids(id_camera, corners, destination_corners, refined_color_dict_HSV, kernels, openings, prev_centroids=centroids, real_size=(NOMINAL_AREA_LENGTH, NOMINAL_AREA_WIDTH), real_time=False)
    #thymio.kalman.state_correct((centroids['thymio'][0], centroids['thymio'][1], theta_thymio))

def plot_localization():
    global localization
    cv.namedWindow("Localization Result")
    cv.imshow("Localization Result", localization)
    key = cv.waitKey(30)
    cv.destroyWindow("Localization Result")

# initialise position and path
thymio.kalman.set_state((centroids['thymio'][0], centroids['thymio'][1], theta_thymio))
path = discretize_map(segmentation)
thymio.set_path(path)
# start updating position and following path
timer_centroids = RepeatedTimer(1.5, compute_centroids)
timer_centroids.start()
thymio.follow_path()