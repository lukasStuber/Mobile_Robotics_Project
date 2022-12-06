from __future__ import print_function
import cv2 as cv
import argparse
import numpy as np
from image_processing.parking_segmentation import *
from image_processing.color_segmentation import *
from image_processing.color_centroids import *
from control import ThymioControl
from RepeatedTimer import RepeatedTimer

# import time

# image processing

id_camera = 0
##[Parking segmentation]
corners, destination_corners = set_parking_limits(id_camera)
##[Parking segmentation]

##[Color segmentation]
segmentation, refined_color_dict_HSV, kernels, openings = get_color_mask(id_camera, corners, destination_corners)
cv.namedWindow("Segmentation Result")
cv.imshow("Segmentation Result", segmentation)
key = cv.waitKey(0)
cv.destroyWindow("Segmentation Result")
##[Color segmentation]

##[Thymio and objective localization]
centroids = {'goal': (0, 0), 'thymio': (0, 0), 'green': (0, 0), 'blue': (0, 0)}
def compute_centroids():
    global centroids
    global theta_thymio
    global localization
    centroids, theta_thymio, localization = get_centroids(id_camera, corners, destination_corners, refined_color_dict_HSV, kernels, openings, prev_centroids=centroids, orig_frame=(120, 80), real_time=False)

def plot_localization():
    global localization
    cv.namedWindow("Localization Result")
    cv.imshow("Localization Result", localization)
    key = cv.waitKey(30)
    cv.destroyWindow("Localization Result")
##[Thymio and objective localization]

timer_centroids = RepeatedTimer(1.5, compute_centroids)
timer_centroids.start

# pathfinding
# start kalman, which calls image processing
# start path following
thymio = ThymioControl(position=(0,0), angle=0)
path = [(500,0)]
thymio.set_path(path)
thymio.follow_path()