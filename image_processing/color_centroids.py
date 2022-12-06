from __future__ import print_function
import cv2 as cv
import argparse
import numpy as np

## [rescale frame to speed up computation]
def rescale_frame(frame, scale_percent=20):
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dsize = (width, height)
    return cv.resize(frame, dsize)
## [rescale frame to speed up computation]

## [convert indices to mm]
def indices_to_mm(orig_frame, cropped_frame, centroid, reverse=False):
    L, H = orig_frame
    l, h = cropped_frame.shape[:2]
    x, y = centroid
    if reverse:
        return (int(H*x/h), int(L*y/l))
    return (int(h*x/H), int(l*y/L))
## [convert indices to mm]

def get_centroids(camera_device, corners, destination_corners, HSV_preset, kernel_preset, opening_preset, prev_centroids=None, orig_frame=(120, 80), real_time=False):
    color_masks = dict()
    window_name = "Thymio Localization"

     ## [cap]
    cap = cv.VideoCapture(camera_device)
    ## [cap]

    ## [window]
    if real_time:
        cv.namedWindow(window_name)
    ## [window]

    loaded_prev = False

    while True:
        _, frame = cap.read()

        ## [rescaling and homomorphy]
        scale_percent = 20
        frame = rescale_frame(frame, scale_percent)
        M = cv.getPerspectiveTransform(np.float32(corners), np.float32(destination_corners))
        cropped_frame = cv.warpPerspective(frame, M, (destination_corners[2][0], destination_corners[2][1]), flags=cv.INTER_LINEAR)
        ## [rescaling and homomorphy]

        cropped_frame_HSV = cv.cvtColor(cropped_frame, cv.COLOR_BGR2HSV)

        for color in HSV_preset:
        
            ## [HSV thresholding]
            color_mask = cv.inRange(cropped_frame_HSV, np.array(HSV_preset[color][0]), np.array(HSV_preset[color][1]))
            ## [HSV thresholding]

            ## [Mathematical morphology]
            kernel = np.ones((kernel_preset[color], kernel_preset[color]), np.uint8)
            if opening_preset[color]:
                color_mask = cv.erode(color_mask, kernel)
                color_mask = cv.dilate(color_mask, kernel)
            else:
                color_mask = cv.dilate(color_mask, kernel)
                color_mask = cv.erode(color_mask, kernel)
            color_masks[color] = color_mask
            ## [Mathematical morphology]

        ## [Use previous centroids to initialize only once]
        if not loaded_prev:
            if prev_centroids is None:
                x_goal_prev, y_goal_prev = 0, 0
                x_thymio_prev, y_thymio_prev = 0, 0
                x_green, y_green = 0, 0
                x_blue, y_blue = 0, 0
            else:
                x_goal_prev, y_goal_prev = indices_to_mm(orig_frame, cropped_frame, prev_centroids['goal'], reverse=True)
                x_thymio_prev, y_thymio_prev = indices_to_mm(orig_frame, cropped_frame, prev_centroids['thymio'], reverse=True)
                x_green, y_green = indices_to_mm(orig_frame, cropped_frame, prev_centroids['green'], reverse=True)
                x_blue, y_blue = indices_to_mm(orig_frame, cropped_frame, prev_centroids['blue'], reverse=True)
            loaded_prev = True
        ## [Use previous centroids to initialize only once]

        ## [Compute centroids]
        if np.argwhere(color_masks['red'] == 255).sum() == 0:
            y_goal, x_goal = y_goal_prev, x_goal_prev
        else:
            y_goal, x_goal = np.argwhere(color_masks['red'] == 255).mean(axis=0).astype(int)
        if (np.argwhere(color_masks['green'] == 255).sum() == 0) or (np.argwhere(color_masks['blue'] == 255).sum() == 0):
            y_thymio, x_thymio = y_thymio_prev, x_thymio_prev
        else:
            y_green, x_green = np.argwhere(color_masks['green'] == 255).mean(axis=0).astype(int)
            y_blue, x_blue = np.argwhere(color_masks['blue'] == 255).mean(axis=0).astype(int)
            y_thymio, x_thymio = int((y_green + y_blue)/2), int((x_green + x_blue)/2)
        
        arrow_len = np.sqrt((x_green - x_blue)**2 + (y_green - y_blue)**2)

        y_goal_prev, x_goal_prev = y_goal, x_goal
        y_thymio_prev, x_thymio_prev = y_thymio, x_thymio
        theta_thymio = np.arctan2(y_green - y_blue, x_green - x_blue) + np.pi/2
        x_arrow = int(x_thymio + np.cos(theta_thymio) * arrow_len)
        y_arrow = int(y_thymio + np.sin(theta_thymio) * arrow_len)
        ## [Compute centroids]

        cv.circle(cropped_frame, (x_goal, y_goal), 2, (0, 0, 255), -1)
        cv.circle(cropped_frame, (x_thymio, y_thymio), 2, (255, 255, 255), -1)
        cv.arrowedLine(cropped_frame, (x_thymio, y_thymio), ((x_arrow , y_arrow)), (255, 255, 255), 2)

        ## [convert indices to mm]
        x_goal, y_goal = indices_to_mm(orig_frame, cropped_frame, (x_goal, y_goal))
        x_thymio, y_thymio = indices_to_mm(orig_frame, cropped_frame, (x_thymio, y_thymio))
        x_green, y_green = indices_to_mm(orig_frame, cropped_frame, (x_green, y_green))
        x_blue, y_blue = indices_to_mm(orig_frame, cropped_frame, (x_blue, y_blue))
        ## [convert indices to mm]

        centroids = {'goal' : (x_goal, y_goal), 'thymio': (x_thymio, y_thymio), 'green': (x_green, y_green), 'blue': (x_blue, y_blue)}

        ## [show]
        if real_time:
            cv.imshow(window_name, cropped_frame)
        ## [show]

        key = cv.waitKey(30)
        if key == ord('q') or key == 27 or not real_time:
            break
    
    if real_time:
        cv.destroyWindow(window_name)
    return  centroids, theta_thymio, cropped_frame

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Code for Thresholding Operations using inRange.')
    parser.add_argument('--camera', help='Camera device number.', default=0, type=int)
    args = parser.parse_args()

    x, y, w, h = 70, 9, 273, 192
    HSV_dict = {'yellow': [[0, 0, 110], [53, 255, 255]], 'red': [[159, 50, 70], [180, 255, 255]], 'green': [[36, 77, 179], [89, 255, 255]], 'blue': [[90, 121, 119], [128, 255, 255]]}
    kernels_dict = {'yellow': 3, 'red': 3, 'green': 3, 'blue': 3}
    openings_dict = {'yellow': 1, 'red': 1, 'green': 1, 'blue': 1}

    get_centroids(args.camera, x, y, w, h, HSV_dict, kernels_dict, openings_dict, real_time=True)
