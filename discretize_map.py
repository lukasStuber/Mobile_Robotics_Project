from create_map import *
from pathfinding import *
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2

def check_patch(patch):
    threshold = 200
    a,b = patch.shape
    for x in range(a):
        for y in range(b):
            if patch[x,y] > threshold:
                return True
    return False

def delete_outliers(data):
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
    return main_data

def discretize_map(final_seg, centroids):
    map_arr = final_seg
    size_x, size_y, _ = map_arr.shape
    kernel = 6

    path_arr = np.zeros([size_x//(2*kernel+1), size_y//(2*kernel+1)], np.uint8)
    path_x, path_y = path_arr.shape

    end_patch = []
    start_patch_green = []
    start_patch_blue = []

    for x in range(path_x):
        for y in range(path_y):
            patch = (x,y)

            red_patch =   map_arr[x*(2*kernel+1)-kernel:x*(2*kernel+1)+kernel+1, y*(2*kernel+1)-kernel:y*(2*kernel+1)+kernel+1, 0]
            green_patch = map_arr[x*(2*kernel+1)-kernel:x*(2*kernel+1)+kernel+1, y*(2*kernel+1)-kernel:y*(2*kernel+1)+kernel+1, 1]
            blue_patch =  map_arr[x*(2*kernel+1)-kernel:x*(2*kernel+1)+kernel+1, y*(2*kernel+1)-kernel:y*(2*kernel+1)+kernel+1, 2]

            # check for red
            if   (not check_patch(red_patch) and (not check_patch(green_patch)) and (check_patch(blue_patch))):
                end_patch.append(list(patch))
                path_arr[patch] = 0
            # check for yellow
            elif (not check_patch(red_patch) and (check_patch(green_patch)) and (check_patch(blue_patch))):
                path_arr[patch] = 1
            # check for white
            elif (check_patch(red_patch) and (check_patch(green_patch)) and (check_patch(blue_patch))):
                path_arr[patch] = 1
            # check for blue
            elif ((check_patch(red_patch)) and (not check_patch(green_patch)) and (not check_patch(blue_patch))):
                start_patch_blue.append(list(patch))
                path_arr[patch] = 0
            # check for green
            elif ((not check_patch(red_patch)) and (check_patch(green_patch)) and (not check_patch(blue_patch))):
                start_patch_green.append(list(patch))
                path_arr[patch] = 0
            # free space
            else: path_arr[patch] = 0
    #cv2.imshow("title",path_arr)
    #key = cv2.waitKey(0)

    end_patch = np.array(end_patch)
    end_patch = np.array(delete_outliers(end_patch))

    goal_x = (min(end_patch[:,0])+max(end_patch[:,0]))//2
    goal_y = (min(end_patch[:,1])+max(end_patch[:,1]))//2
    goal = (goal_x, goal_y)

    start_patch_green = np.array(start_patch_green)
    start_patch_green = np.array(delete_outliers(start_patch_green))
    green_x = (min(start_patch_green[:,0])+max(start_patch_green[:,0]))//2
    green_y = (min(start_patch_green[:,1])+max(start_patch_green[:,1]))//2
    
    start_patch_blue = np.array(start_patch_blue)
    start_patch_blue = np.array(delete_outliers(start_patch_blue))
    blue_x = (min(start_patch_blue[:,0])+max(start_patch_blue[:,0]))//2
    blue_y = (min(start_patch_blue[:,1])+max(start_patch_blue[:,1]))//2

    start = ((blue_x+green_x)//2, (blue_y+green_y)//2)
    #start = (centroids['thymio'][0], centroids['thymio'][1]) 
    #start = (centroids['goal'][0], centroids['goal'][1]) 

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

    cmap = colors.ListedColormap(['white', 'red'])
    fig_astar, ax_astar = create_empty_plot((path_y, path_x))
    ax_astar.imshow(path_arr, cmap=cmap)
    ax_astar.plot(path[:,1], path[:,0], marker="o", color = 'blue')
    ax_astar.scatter(start[1], start[0], marker="o", color = 'green', s=200)
    ax_astar.scatter(goal[1], goal[0], marker="o", color = 'purple', s=200)
    plt.show()
    path = path*(2*kernel + 1)
    path[:, [1,0]] = path[:, [0,1]]
    print(path)
    return path