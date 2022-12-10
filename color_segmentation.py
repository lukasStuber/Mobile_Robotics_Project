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
opening = 1  # boolean value to perform opening or closing

window_detection_name = 'Color Detection: '

low_H_name = 'Low H'
low_S_name = 'Low S'
low_V_name = 'Low V'
high_H_name = 'High H'
high_S_name = 'High S'
high_V_name = 'High V'

color_dict_HSV = {'yellow': [[25, 50, 70], [35, 255, 255]],
                'red': [[159, 50, 70], [180, 255, 255]],
                #'red': [[0, 50, 70], [9, 255, 255]],
                'green': [[36, 50, 70], [89, 255, 255]],
                'blue': [[90, 50, 70], [128, 255, 255]]}

## [HSV settings for color]
def on_low_H_thresh_trackbar(val):
    global low_H, high_H
    low_H = val
    low_H = min(high_H-1, low_H)
    for color in color_dict_HSV:
        cv.setTrackbarPos(low_H_name, window_detection_name + color, low_H)

def on_high_H_thresh_trackbar(val):
    global low_H, high_H
    high_H = val
    high_H = max(high_H, low_H+1)
    for color in color_dict_HSV:
        cv.setTrackbarPos(high_H_name, window_detection_name + color, high_H)

def on_low_S_thresh_trackbar(val):
    global low_S, high_S
    low_S = val
    low_S = min(high_S-1, low_S)
    for color in color_dict_HSV:
        cv.setTrackbarPos(low_S_name, window_detection_name + color, low_S)

def on_high_S_thresh_trackbar(val):
    global low_S, high_S
    high_S = val
    high_S = max(high_S, low_S+1)
    for color in color_dict_HSV:
        cv.setTrackbarPos(high_S_name, window_detection_name + color, high_S)

def on_low_V_thresh_trackbar(val):
    global low_V, high_V
    low_V = val
    low_V = min(high_V-1, low_V)
    for color in color_dict_HSV:
        cv.setTrackbarPos(low_V_name, window_detection_name + color, low_V)

def on_high_V_thresh_trackbar(val):
    global low_V, high_V
    high_V = val
    high_V = max(high_V, low_V+1)
    for color in color_dict_HSV:
        cv.setTrackbarPos(high_V_name, window_detection_name + color, high_V)
## [HSV settings for color]

## [kernel settings for color]
def kernel_size_trackbar(val):
    global kernel_size
    min_kernel_size = 1
    max_kernel_size = 10
    kernel_size = max(min(val, max_kernel_size), min_kernel_size)
    for color in color_dict_HSV:
        cv.setTrackbarPos("Kernel Size", window_detection_name + color, kernel_size)
## [kernel settings for color]

## [morphology settings for color]
def morphology_type_trackbar(val):
    global opening
    opening = val
    for color in color_dict_HSV:
        cv.setTrackbarPos("Opening boolean", window_detection_name + color, opening)
## [morphology settings for color]

## [rescale frame to speed up computation]
def rescale_frame(frame, scale_percent):
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dsize = (width, height)
    return cv.resize(frame, dsize)
## [rescale frame to speed up computation]

def get_color_mask(camera_device, corners, destination_corners, real_size=(1600, 820), HSV_preset=color_dict_HSV, kernel_preset=None, opening_preset=None):
    global low_H, high_H, low_S, high_S, low_V, high_V
    global kernel_size, opening

    color_masks = dict()
    refined_color_dict_HSV = dict()
    refined_kernel = dict()
    refined_opening = dict()

    for color in HSV_preset:
        ## [cap]
        cap = cv.VideoCapture(camera_device)
        ## [cap]

        ## [window]
        cv.namedWindow(window_detection_name + color)
        ## [window]

        ##[preset trackbar]
        low_H, low_S, low_V = HSV_preset[color][0]
        high_H, high_S, high_V = HSV_preset[color][1]
        if kernel_preset is not None:
            kernel_size = kernel_preset[color]
        if opening_preset is not None:
            opening = opening_preset[color]
        ##[preset trackbar]

        ## [trackbar]
        cv.createTrackbar(low_H_name, window_detection_name + color , HSV_preset[color][0][0], max_value_H, on_low_H_thresh_trackbar)
        cv.createTrackbar(high_H_name, window_detection_name + color , HSV_preset[color][1][0], max_value_H, on_high_H_thresh_trackbar)
        cv.createTrackbar(low_S_name, window_detection_name + color , HSV_preset[color][0][1], max_value, on_low_S_thresh_trackbar)
        cv.createTrackbar(high_S_name, window_detection_name + color , HSV_preset[color][1][1], max_value,on_high_S_thresh_trackbar)
        cv.createTrackbar(low_V_name, window_detection_name + color , HSV_preset[color][0][2], max_value, on_low_V_thresh_trackbar)
        cv.createTrackbar(high_V_name, window_detection_name + color , HSV_preset[color][1][2], max_value, on_high_V_thresh_trackbar)
        cv.createTrackbar("Kernel Size", window_detection_name + color, kernel_size, 10, kernel_size_trackbar)
        cv.createTrackbar("Opening boolean", window_detection_name + color, opening, 1, morphology_type_trackbar)
        ## [trackbar]

        while True:
            _, frame = cap.read()
            if frame is None:
                break

            ## [rescaling and homomorphy]
            scale_percent = 50
            frame = rescale_frame(frame, scale_percent)
            M = cv.getPerspectiveTransform(np.float32(corners), np.float32(destination_corners))
            cropped_frame = cv.warpPerspective(frame, M, (destination_corners[2][0], destination_corners[2][1]), flags=cv.INTER_LINEAR)
            ## [rescaling and homomorphy]

            ## [HSV thresholding]
            cropped_frame_HSV = cv.cvtColor(cropped_frame, cv.COLOR_BGR2HSV)
            color_mask = cv.inRange(cropped_frame_HSV, (low_H, low_S, low_V), (high_H, high_S, high_V))
            refined_color_dict_HSV[color] = [[low_H, low_S, low_V], [high_H, high_S, high_V]]
            ## [HSV thresholding]

            ## [Mathematical morphology]
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            if opening:
                color_mask = cv.erode(color_mask, kernel)
                color_mask = cv.dilate(color_mask, kernel)
            else:
                color_mask = cv.dilate(color_mask, kernel)
                color_mask = cv.erode(color_mask, kernel)

            ## [Enlarge obstacles]
            h, l = cropped_frame.shape[:2]
            L, H = real_size
            inflate_mm = 45  # thymio length is 110 mm
            inflate_pixel = (int((2 * inflate_mm + 1) * l/L), int((2 * inflate_mm + 1) * h/H))
            if color == "yellow":
                color_mask = cv.dilate(color_mask, np.ones(inflate_pixel, np.uint8))
            ## [Enlarge obstacles]

            color_masks[color] = color_mask
            refined_kernel[color] = kernel_size
            refined_opening[color] = opening
            ## [Mathematical morphology]

            ## [Color final mask]
            segmentation = np.zeros_like(cropped_frame)
            for c in color_masks:
                if c == "yellow":
                    segmentation[color_masks[c] > 0] = np.array([0, 255, 255])
                if c == "red":
                    segmentation[color_masks[c] > 0] = np.array([0, 0, 255])
                if c == "green":
                    segmentation[color_masks[c] > 0] = np.array([0, 255, 0])
                if c == "blue":
                    segmentation[color_masks[c] > 0] = np.array([255, 0, 0])
            ## [Color final mask]

            ## [show]
            concat = np.concatenate((cropped_frame, cv.cvtColor(color_mask, cv.COLOR_GRAY2BGR), segmentation), axis=0)
            cv.imshow(window_detection_name + color, concat)
            ## [show]

            key = cv.waitKey(30)
            if key == ord('q') or key == 27:
                break

        cv.destroyWindow(window_detection_name + color)
        segmentation = cv.resize(segmentation, real_size)
    return  segmentation, refined_color_dict_HSV, refined_kernel, refined_opening

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Code for Thresholding Operations using inRange.')
    parser.add_argument('--camera', help='Camera device number.', default=0, type=int)
    args = parser.parse_args()
    x, y, w, h = 32, 4, 284, 206
    seg, refined_color_dict = get_color_mask(args.camera, x, y, w, h)
    print(refined_color_dict)
    