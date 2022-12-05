import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

# A* for 2D occupancy grid. Finds a path from start to goal.
# h is the heuristic function. h(n) estimates the cost to reach goal from node n.
# :param start: start node (x, y)
# :param goal_m: goal node (x, y)
# :param occupancy_grid: the grid map (0: free, 1: occupied)
# :param movement: select between 4-connectivity ('4N') and 8-connectivity ('8N', default)
# :return: a tuple that contains: (the resulting path in meters, the resulting path in data array indices)

def create_empty_plot(max_val): # create figure/grid of desired dimension
    fig, ax = plt.subplots(figsize=(7,7))
    major_ticks_x = np.arange(0, max_val[0]+1, 10)
    major_ticks_y = np.arange(0, max_val[1]+1, 10)
    minor_ticks = np.arange(0, max_val[1]+1, 1)
    ax.set_xticks(major_ticks_x); ax.set_yticks(major_ticks_y)
    ax.set_xticks(minor_ticks, minor=True); ax.set_yticks(minor_ticks, minor=True)
    ax.grid(which='minor', alpha=0.2); ax.grid(which='major', alpha=0.5)
    ax.set_xlim([-1,max_val[0]]); ax.set_ylim([-1,max_val[1]])
    ax.grid(True)
    return fig, ax

def _get_movements_4n():
    return [(1, 0, 1.0), (0, 1, 1.0), (-1, 0, 1.0), (0, -1, 1.0)]

def _get_movements_8n():
    s2 = math.sqrt(2)
    return [(1, 0, 1.0), (0, 1, 1.0), (-1, 0, 1.0), (0, -1, 1.0), (1, 1, s2), (-1, 1, s2), (-1, -1, s2), (1, -1, s2)]

def reconstruct_path(cameFrom, current): # cameFrom: map (dict) containing node immediately preceding each node on cheapest path from start to node
    total_path = [current]
    while current in cameFrom.keys():
        total_path.insert(0, cameFrom[current]) 
        current = cameFrom[current]
    return total_path

def A_Star(start, goal, h, coords, occupancy_grid, movement_type="4N", max_val=(50,50)):
    # h: heuristic function (straight-line cost to reach goal from node n)
    # returns (resulting path in meters, resulting path in data array indices)
    for point in [start, goal]:
        for i in range(len(point)):
            assert point[i] >= 0 and point[i] < max_val[i], "start or end goal not contained in the map"
    if occupancy_grid[start[0], start[1]]: raise Exception('Start node is not traversable')
    if occupancy_grid[goal[0], goal[1]]: raise Exception('Goal node is not traversable')
    if movement_type == '4N': movements = _get_movements_4n()
    elif movement_type == '8N': movements = _get_movements_8n()
    else: raise ValueError('Unknown movement')

    openSet = set(); openSet.add(start)
    closedSet = set(); cameFrom = dict()
    gScore = dict(zip(coords, [np.inf for x in range(len(coords))])); gScore[start] = 0 # cost of best path from start to node
    fScore = dict(zip(coords, [np.inf for x in range(len(coords))])); fScore[start] = h[start] # add heuristic cost

    while openSet != []:
        fScore_openSet = {key:val for (key,val) in fScore.items() if key in openSet}
        current = min(fScore_openSet, key=fScore_openSet.get) # find lowest fScore[] in openSet 
        del fScore_openSet
        if current == goal: return reconstruct_path(cameFrom, current), list(closedSet)
        openSet.remove(current); closedSet.add(current)
        for dx, dy, deltacost in movements: #for each neighbor
            neighbor = (current[0]+dx, current[1]+dy)
            if (neighbor[0] >= occupancy_grid.shape[0]) or (neighbor[1] >= occupancy_grid.shape[1]) or (neighbor[0] < 0) or (neighbor[1] < 0): continue
            if (occupancy_grid[neighbor[0], neighbor[1]]) or (neighbor in closedSet): continue
            tentative_gScore = gScore[current] + deltacost # distance from start to neighbor through current
            if neighbor not in openSet: openSet.add(neighbor)
            if tentative_gScore < gScore[neighbor]:
                cameFrom[neighbor] = current
                gScore[neighbor] = tentative_gScore
                fScore[neighbor] = gScore[neighbor] + h[neighbor]
    print("No path found to goal")
    return [], list(closedSet)

# start = (0,0)
# goal = (43,46)
# max_val = (50,50)

# # Creating the occupancy grid
# np.random.seed(0)
# data = np.random.rand(max_val[0], max_val[1]) * 20 # 50 x 50 random values between 0 and 20
# cmap = colors.ListedColormap(['white', 'red'])
# limit = 12 
# occupancy_grid = data.copy()
# occupancy_grid[data > limit] = 1; occupancy_grid[data <= limit] = 0
# # List of all coordinates in the grid
# x,y = np.mgrid[0:max_val[0]:1, 0:max_val[1]:1]
# pos = np.empty(x.shape + (2,))
# pos[:, :, 0] = x; pos[:, :, 1] = y
# pos = np.reshape(pos, (x.shape[0]*x.shape[1], 2))
# coords = list([(int(x[0]), int(x[1])) for x in pos])
# # Define the heuristic, here = distance to goal ignoring obstacles
# h = np.linalg.norm(pos - goal, axis=-1); h = dict(zip(coords, h))
# # Run A*
# path, visitedNodes = A_Star(start, goal, h, coords, occupancy_grid, movement_type="8N")
# path = np.array(path).reshape(-1, 2).transpose()
# visitedNodes = np.array(visitedNodes).reshape(-1, 2).transpose()
# # Display map
# fig_astar, ax_astar = create_empty_plot(max_val)
# ax_astar.imshow(occupancy_grid.transpose(), cmap=cmap)
# ax_astar.scatter(visitedNodes[0], visitedNodes[1], marker="o", color = 'orange')
# ax_astar.plot(path[0], path[1], marker="o", color = 'blue')
# ax_astar.scatter(start[0], start[1], marker="o", color = 'green', s=200)
# ax_astar.scatter(goal[0], goal[1], marker="o", color = 'purple', s=200)
# plt.show()