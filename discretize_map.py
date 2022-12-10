from create_map import *
from pathfinding import *
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2

def check_patch(patch):
    a,b,_ = patch.shape
    has_red = False
    has_green = False
    has_yellow = False
    has_blue = False
    for x in range(a):
        for y in range(b):
            if patch[x,y,0] == 0 and patch[x,y,1] == 0 and patch[x,y,2] == 255:
                has_red = True
            if patch[x,y,0] == 0 and patch[x,y,1] == 255 and patch[x,y,2] == 255:
                has_yellow = True
            if patch[x,y,0] == 0 and patch[x,y,1] == 255 and patch[x,y,2] == 0:
                has_green = True
            if patch[x,y,0] == 255 and patch[x,y,1] == 0 and patch[x,y,2] == 0:
                has_blue = True
    return (has_red, has_green, has_blue, has_yellow)

def delete_outliers(data):  # could be parallelized too
    main_data = []
    threshold = 2
    mean_x = np.mean(data[:,0])
    mean_y = np.mean(data[:,1])
    std_x = np.std(data[:,0])
    std_y = np.std(data[:,1])
    for point in data:
        z_score_x = abs(point[0] - mean_x) / std_x
        z_score_y = abs(point[1] - mean_y) / std_y
        if z_score_x < threshold and z_score_y < threshold:
            main_data.append([point[0], point[1]])
    return np.array(main_data)

def discretize_map(final_seg):
    map_arr = final_seg.copy()
    size_x, size_y, _ = map_arr.shape
    kernel = 4

    # Everything occupied at first (yellow)
    path_arr = np.zeros([size_x//(2*kernel+1), size_y//(2*kernel+1)], np.uint8)
    path_x, path_y = path_arr.shape

    end_patch = []
    start_patch_green = []
    start_patch_blue = []

    for x in range(path_x):
        for y in range(path_y):
            patch = (x,y)
            patch_pixels = map_arr[x*(2*kernel+1)-kernel:x*(2*kernel+1)+kernel+1, y*(2*kernel+1)-kernel:y*(2*kernel+1)+kernel+1, :]

            (has_red, has_green, has_blue, has_yellow) = check_patch(patch_pixels)

            # check for red
            if has_red:
                end_patch.append(list(patch))
                path_arr[patch] = 0
            # check for blue
            elif has_blue:
                start_patch_blue.append(list(patch))
                path_arr[patch] = 0
            # check for green
            elif has_green:
                start_patch_green.append(list(patch))
                path_arr[patch] = 0
            # check for yellow
            elif has_yellow:
               path_arr[patch] = 1

    end_patch = np.array(end_patch)
    end_patch = delete_outliers(end_patch)

    goal_x = (min(end_patch[:,0])+max(end_patch[:,0]))//2
    goal_y = (min(end_patch[:,1])+max(end_patch[:,1]))//2
    goal = (goal_x, goal_y)

    start_patch_green = np.array(start_patch_green)
    print(start_patch_green)
    print(start_patch_green.shape)
    start_patch_green = np.array(delete_outliers(start_patch_green))
    print(start_patch_green.shape)
    green_x = (min(start_patch_green[:,0])+max(start_patch_green[:,0]))//2
    green_y = (min(start_patch_green[:,1])+max(start_patch_green[:,1]))//2
    
    start_patch_blue = np.array(start_patch_blue)
    print(start_patch_blue)
    print(start_patch_blue.shape)
    start_patch_blue = np.array(delete_outliers(start_patch_blue))
    print(start_patch_blue.shape)
    blue_x = (min(start_patch_blue[:,0])+max(start_patch_blue[:,0]))//2
    blue_y = (min(start_patch_blue[:,1])+max(start_patch_blue[:,1]))//2

    start = ((blue_x+green_x)//2, (blue_y+green_y)//2)

    # prepare the objects for the A*
    x,y = np.mgrid[0:path_x:1, 0:path_y:1]
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x; pos[:, :, 1] = y
    pos = np.reshape(pos, (x.shape[0]*x.shape[1], 2))
    coords = list([(int(x[0]), int(x[1])) for x in pos])
    h = np.linalg.norm(pos - goal, axis=-1)
    h = dict(zip(coords, h))

    # Run A*
    path, visitedNodes = A_Star(start, goal, h, coords, path_arr, movement_type="8N", max_val=path_arr.shape)
    path = np.array(path).reshape(-1, 2)
    path[:, [1,0]] = path[:, [0,1]]
    a,b = path.shape
    path_short = path[np.arange(0,a,5)]
    path_short[-1,:] = path[-1,:]
    path_image = path_short.copy()
    path_short = path_short*(2*kernel + 1)

    # create the window of the discretized map
    window_small = "Global path coarse"
    cv2.namedWindow(window_small, cv2.WINDOW_NORMAL)
    image_arr = np.zeros((path_x, path_y, 3), np.uint8)
    image_arr[:,:,1] = path_arr.copy()*255
    image_arr[:,:,2] = path_arr.copy()*255
    for point in path_image:
        image_arr = cv2.circle(image_arr, point, 0, (255,0,255), -1)
    start_x, start_y = start
    image_arr = cv2.circle(image_arr, (start_y, start_x), 1, (255,0,0), -1)
    image_arr = cv2.circle(image_arr, (goal_y, goal_x), 1, (0,0,255), -1)
    cv2.imshow(window_small, image_arr)
    cv2.resizeWindow(window_small, path_y*kernel, path_x*kernel)

    # create the window of the large map
    window_big = "Global path fine"
    cv2.namedWindow(window_big, cv2.WINDOW_NORMAL)
    image_big_arr = final_seg.copy()
    for point in path_short:
        image_big_arr = cv2.circle(image_big_arr, point, kernel, (255,0,255), -1)
    cv2.imshow(window_big, image_big_arr)
    cv2.resizeWindow(window_big, path_y*kernel, path_x*kernel)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return path_short