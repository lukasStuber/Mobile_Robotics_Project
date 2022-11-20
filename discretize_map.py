from create_map import *
from pathfinding import *
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def check_patch(patch):
    threshold = 200
    a,b = patch.shape
    for x in range(a):
        for y in range(b):
            if patch[x,y] > threshold:
                return True
    return False

border = 20
line_width = 30
line_length = 200
field_width = 190

tymio_width = int(field_width*0.6)

nb_lines = 6

size_x = 800
size_y = 1600
map_size = (size_x, size_y, 3)

end = ['right', 0]
obstacles = [['left', 0],
             ['left', 1],
             ['left', 2],
             ['left', 4],
             ['left', 5],
             ['right', 3],
             ['right', 4]]


map_arr = draw_map(map_size, border, line_width, line_length, field_width, nb_lines, tymio_width, obstacles, end)

kernel = 6
thresh = 75
red_thresh = 200
black_tresh = 40

path_arr = np.zeros([size_x//(2*kernel+1), size_y//(2*kernel+1)], np.uint8)
end_patch = []

for x in np.arange(kernel, size_x-kernel, 2*kernel+1):
    for y in np.arange(kernel, size_y-kernel, 2*kernel+1):
        path_pixel = (x//(2*kernel+1), y//(2*kernel+1))
        red_patch = map_arr[x-kernel:x+kernel+1, y-kernel:y+kernel+1, 0]
        green_patch = map_arr[x-kernel:x+kernel+1, y-kernel:y+kernel+1, 1]
        blue_patch = map_arr[x-kernel:x+kernel+1, y-kernel:y+kernel+1, 2]

        # check for red
        if   (check_patch(red_patch) and (not check_patch(green_patch)) and (not check_patch(blue_patch))):
            end_patch.append(list(path_pixel))
            path_arr[path_pixel] = 0
        # check for yellow
        elif (check_patch(red_patch) and (check_patch(green_patch)) and (not check_patch(blue_patch))):
            path_arr[path_pixel] = 1
        # check for white
        elif (check_patch(red_patch) and (check_patch(green_patch)) and (check_patch(blue_patch))):
            path_arr[path_pixel] = 1
        # check for blue
        elif ((not check_patch(red_patch)) and (not check_patch(green_patch)) and (check_patch(blue_patch))):
            path_arr[path_pixel] = 1
        # free space
        else: path_arr[path_pixel] = 0

path_x, path_y = path_arr.shape

end_patch = np.array(end_patch)
goal_x = (min(end_patch[:,0])+max(end_patch[:,0]))//2
goal_y = (min(end_patch[:,1])+max(end_patch[:,1]))//2

goal = (goal_x, goal_y)
start = (5, path_y-7)


#img = np.transpose(map_arr, (1,0,2))
#img = np.transpose(path_arr)
#image = Image.fromarray(img)
#image.show()


x,y = np.mgrid[0:path_x:1, 0:path_y:1]
pos = np.empty(x.shape + (2,))
pos[:, :, 0] = x; pos[:, :, 1] = y
pos = np.reshape(pos, (x.shape[0]*x.shape[1], 2))
#pos = np.reshape(path_arr, (path_x*path_y//2, 2))
coords = list([(int(x[0]), int(x[1])) for x in pos])
h = np.linalg.norm(pos - goal, axis=-1)
h = dict(zip(coords, h))
# Run A*
path, visitedNodes = A_Star(start, goal, h, coords, path_arr, movement_type="8N", max_val=path_arr.shape)
path = np.array(path).reshape(-1, 2).transpose()
visitedNodes = np.array(visitedNodes).reshape(-1, 2).transpose()
# Display map
cmap = colors.ListedColormap(['white', 'red'])
fig_astar, ax_astar = create_empty_plot(path_arr.shape)
ax_astar.imshow(path_arr.transpose(), cmap=cmap)
ax_astar.scatter(visitedNodes[0], visitedNodes[1], marker="o", color = 'orange')
ax_astar.plot(path[0], path[1], marker="o", color = 'blue')
ax_astar.scatter(start[0], start[1], marker="o", color = 'green', s=200)
ax_astar.scatter(goal[0], goal[1], marker="o", color = 'purple', s=200)
plt.show()
