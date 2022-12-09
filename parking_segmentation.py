from __future__ import print_function
import cv2 as cv
import argparse
import numpy as np

max_value = 255
max_value_H = 360//2
low_H = 0
low_S = 0
low_V = 0
high_H = max_value_H
high_S = max_value
high_V = max_value
kernel_size = 3

window_detection_name = 'Parking Detection'
low_H_name = 'Low H'
low_S_name = 'Low S'
low_V_name = 'Low V'
high_H_name = 'High H'
high_S_name = 'High S'
high_V_name = 'High V'

## [HSV settings for mask]
def on_low_H_thresh_trackbar(val):
    global low_H
    global high_H
    low_H = val
    low_H = min(high_H-1, low_H)
    cv.setTrackbarPos(low_H_name, window_detection_name, low_H)

def on_high_H_thresh_trackbar(val):
    global low_H
    global high_H
    high_H = val
    high_H = max(high_H, low_H+1)
    cv.setTrackbarPos(high_H_name, window_detection_name, high_H)

def on_low_S_thresh_trackbar(val):
    global low_S
    global high_S
    low_S = val
    low_S = min(high_S-1, low_S)
    cv.setTrackbarPos(low_S_name, window_detection_name, low_S)

def on_high_S_thresh_trackbar(val):
    global low_S
    global high_S
    high_S = val
    high_S = max(high_S, low_S+1)
    cv.setTrackbarPos(high_S_name, window_detection_name, high_S)

def on_low_V_thresh_trackbar(val):
    global low_V
    global high_V
    low_V = val
    low_V = min(high_V-1, low_V)
    cv.setTrackbarPos(low_V_name, window_detection_name, low_V)

def on_high_V_thresh_trackbar(val):
    global low_V
    global high_V
    high_V = val
    high_V = max(high_V, low_V+1)
    cv.setTrackbarPos(high_V_name, window_detection_name, high_V)
## [HSV settings for mask]

## [kernel settings for mask]
def kernel_size_trackbar(val):
    global kernel_size
    min_kernel_size = 1
    max_kernel_size = 10
    kernel_size = max(min(val, max_kernel_size), min_kernel_size)
    cv.setTrackbarPos("Kernel Size", window_detection_name, kernel_size)
## [kernel settings for mask]

## [rescale frame to speed up computation]
def rescale_frame(frame, scale_percent):
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dsize = (width, height)
    return cv.resize(frame, dsize)
## [rescale frame to speed up computation]

## [functions for homomorphy]
def order_points(pts):
    '''Rearrange coordinates to order:
      top-left, top-right, bottom-right, bottom-left'''
    rect = np.zeros((4, 2), dtype='float32')
    pts = np.array(pts)
    # Make sure contours array is 2d
    pts = np.squeeze(pts)
    s = pts.sum(axis=1)
    # Top-left point will have the smallest sum.
    rect[0] = pts[np.argmin(s)]
    # Bottom-right point will have the largest sum.
    rect[2] = pts[np.argmax(s)]
 
    diff = np.diff(pts, axis=1)
    # Top-right point will have the smallest difference.
    rect[1] = pts[np.argmin(diff)]
    # Bottom-left will have the largest difference.
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect.astype('int')
    
def find_dest(pts):
    (tl, tr, br, bl) = pts
    # Finding the maximum width.
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
 
    # Finding the maximum height.
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # Final destination co-ordinates.
    destination_corners = [[0, 0], [maxWidth, 0], [maxWidth, maxHeight], [0, maxHeight]]
 
    return order_points(destination_corners)
## [functions for homomorphy]

def set_parking_limits(camera_device):
    global low_H, high_H, low_S, high_S, low_V, high_V, kernel_size

    ## [cap]
    cap = cv.VideoCapture(camera_device)
    ## [cap]

    ## [window]
    cv.namedWindow(window_detection_name)
    ## [window]

    ## [trackbar]
    cv.createTrackbar(low_H_name, window_detection_name , low_H, max_value_H, on_low_H_thresh_trackbar)
    cv.createTrackbar(high_H_name, window_detection_name , high_H, max_value_H, on_high_H_thresh_trackbar)
    cv.createTrackbar(low_S_name, window_detection_name , low_S, max_value, on_low_S_thresh_trackbar)
    cv.createTrackbar(high_S_name, window_detection_name , high_S, max_value, on_high_S_thresh_trackbar)
    cv.createTrackbar(low_V_name, window_detection_name , low_V, max_value, on_low_V_thresh_trackbar)
    cv.createTrackbar(high_V_name, window_detection_name , high_V, max_value, on_high_V_thresh_trackbar)
    cv.createTrackbar("Kernel Size", window_detection_name, kernel_size, 10, kernel_size_trackbar)
    ## [trackbar]

    while True:
        success, frame = cap.read()
        if frame is None:
            break

        ## [rescaling]
        scale_percent = 50
        frame = rescale_frame(frame, scale_percent)
        ## [rescaling]

        ## [HSV thresholding]
        frame_HSV = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        frame_threshold = cv.inRange(frame_HSV, (low_H, low_S, low_V), (high_H, high_S, high_V))
        ## [HSV thresholding]

        ## [Mathematical morphology erosion]
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask = cv.erode(frame_threshold, kernel)
        ## [Mathematical morphology erosion]

        ## [Contours detection and homography]
        contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        cnt = max(contours, key=cv.contourArea)
        x, y, w, h = cv.boundingRect(cnt)
        cv.rectangle(frame,(x, y),(x + w, y + h),(0, 255, 0), 2)

        corners = order_points(cnt)
        destination_corners = find_dest(corners)
        ## [Contours detection and homography]

        ## [show]
        concat = np.concatenate((frame, cv.cvtColor(frame_threshold, cv.COLOR_GRAY2BGR), cv.cvtColor(mask, cv.COLOR_GRAY2BGR)), axis=0)
        cv.imshow(window_detection_name, concat)
        ## [show]

        key = cv.waitKey(30)
        if key == ord('q') or key == 27:
            break

    cv.destroyWindow(window_detection_name)
    return corners, destination_corners

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Code for parking segmentation using inRange.')
    parser.add_argument('--camera', help='Camera device number.', default=0, type=int)
    args = parser.parse_args()
    set_parking_limits(args.camera)