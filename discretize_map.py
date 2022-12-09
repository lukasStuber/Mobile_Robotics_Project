from create_map import *
from pathfinding import *
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2

def check_patch(patch):
    threshold = 200
    return (patch > threshold).any()

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

            red_patch =   map_arr[x*(2*kernel+1)-kernel:x*(2*kernel+1)+kernel+1, y*(2*kernel+1)-kernel:y*(2*kernel+1)+kernel+1, 2]
            green_patch = map_arr[x*(2*kernel+1)-kernel:x*(2*kernel+1)+kernel+1, y*(2*kernel+1)-kernel:y*(2*kernel+1)+kernel+1, 1]
            blue_patch =  map_arr[x*(2*kernel+1)-kernel:x*(2*kernel+1)+kernel+1, y*(2*kernel+1)-kernel:y*(2*kernel+1)+kernel+1, 0]

            red_in_patch = check_patch(red_patch)
            green_in_patch = check_patch(green_patch)
            blue_in_patch = check_patch(blue_patch)

            # check for red
            if red_in_patch and (not green_in_patch) and (not blue_in_patch):
                end_patch.append(list(patch))
                path_arr[patch] = 0
            # check for yellow
            elif red_in_patch and green_in_patch and (not blue_in_patch):
                path_arr[patch] = 1
            # check for white
            elif red_in_patch and green_in_patch and blue_in_patch:  # useful ??
                path_arr[patch] = 1
            # check for blue
            elif (not red_in_patch) and (not green_in_patch) and blue_in_patch:
                start_patch_blue.append(list(patch))
                path_arr[patch] = 0
            # check for green
            elif (not red_in_patch) and green_in_patch and (not blue_in_patch):
                start_patch_green.append(list(patch))
                path_arr[patch] = 0
            # free space
            else: path_arr[patch] = 0

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
    fig_astar, ax_astar = create_empty_plot((path_y, path_x))  # this plot is in reduced shape, should plot this after expansion
    ax_astar.imshow(path_arr, cmap=cmap)
    ax_astar.plot(path[:,1], path[:,0], marker="o", color = 'blue')
    ax_astar.scatter(start[1], start[0], marker="o", color = 'green', s=200)
    ax_astar.scatter(goal[1], goal[0], marker="o", color = 'purple', s=200)
    ax_astar.invert_yaxis()  # add this just to invert y axis
    plt.show()

    # Space path points with kernel size
    path = path*(2*kernel + 1)
    # Exchange x and y coordinates of path points from A*
    # path[:, [1,0]] = path[:, [0,1]]  # we should not do that, cf imshow below already good order
    a,b = path.shape
    # Sample 1/5 of path points 
    path_short = path[np.arange(0,a,5)]
    # Reverse order of path points
    path_short[-1,:] = path[-1,:]

    return path_short

if __name__ == '__main__':
    segmentation = cv2.imread("/Users/antoineescoyez/Desktop/micro452/Project/Mobile_Robotics_Project/Astar_test.png")
    cv2.imshow("Plot", segmentation)
    k = cv2.waitKey(0)
    cv2.destroyWindow("Plot")
    path = discretize_map(segmentation, None)
    test = segmentation.copy()
    for point in path:
        test[point[0], point[1], :] = np.array([255, 255, 255])
    cv2.imshow("Plot", test)
    k = cv2.waitKey(0)
    cv2.destroyWindow("Plot")