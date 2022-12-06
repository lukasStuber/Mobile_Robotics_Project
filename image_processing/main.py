from __future__ import print_function
import cv2 as cv
import argparse
import numpy as np
from parking_segmentation import *
from color_segmentation import *
from color_centroids import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Code for Thresholding Operations using inRange.')
    parser.add_argument('--camera', help='Camera device number.', default=0, type=int)
    args = parser.parse_args()

    ##[Parking segmentation]
    corners, destination_corners = set_parking_limits(args.camera)
    ##[Parking segmentation]

    ##[Color segmentation]
    segmentation, refined_color_dict_HSV, kernels, openings = get_color_mask(args.camera, corners, destination_corners)
    cv.namedWindow("Segmentation Result")
    cv.imshow("Segmentation Result", segmentation)
    key = cv.waitKey(0)
    cv.destroyWindow("Segmentation Result")
    ##[Color segmentation]

    ##[Thymio and objective localization]
    cv.namedWindow("Localization Result")
    centroids = {'goal': (0, 0), 'thymio': (0, 0), 'green': (0, 0), 'blue': (0, 0)}
    while True:
        centroids, theta_thymio, localization = get_centroids(args.camera, corners, destination_corners, refined_color_dict_HSV, kernels, openings, prev_centroids=centroids, orig_frame=(120, 80), real_time=False)
        print(centroids['thymio'])
        cv.imshow("Localization Result", localization)
        key = cv.waitKey(30)
    cv.destroyWindow("Localization Result")
    ##[Thymio and objective localization]
