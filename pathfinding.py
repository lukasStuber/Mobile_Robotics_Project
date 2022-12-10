import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from constants import * 

# A* for 2D occupancy grid. Finds a path from start to goal.
# h is the heuristic function. h(n) estimates the cost to reach goal from node n.
# :param start: start node (x, y)
# :param goal_m: goal node (x, y)
# :param occupancy_grid: the grid map (0: free, 1: occupied)
# :param movement: select between 4-connectivity ('4N') and 8-connectivity ('8N')
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

def reconstruct_path(cameFrom, current):
    # cameFrom: map (dict) containing node immediately preceding each node on cheapest path from start to node
    total_path = [current]
    while current in cameFrom.keys():
        total_path.insert(0, cameFrom[current]) 
        current = cameFrom[current]
    return total_path

def only_corners_path(path):
    test_if_aligned = list(path.copy())
    only_corners_path = list()
    while len(test_if_aligned) > 2:
        x_a = test_if_aligned[1][0] - test_if_aligned[0][0]
        y_a = test_if_aligned[1][1] - test_if_aligned[0][1]
        x_b = test_if_aligned[2][0] - test_if_aligned[1][0]
        y_b = test_if_aligned[2][1] - test_if_aligned[1][1]
        norm_a = np.sqrt(x_a**2 + y_a**2)
        norm_b = np.sqrt(x_b**2 + y_b**2)
        normalized_dot_product = (x_a * x_b + y_a * y_b) / (norm_a * norm_b)
        if np.abs(normalized_dot_product - 1) < 1e-2:
            test_if_aligned.pop(1)
        else:
            only_corners_path.append(test_if_aligned.pop(0))
    assert len(test_if_aligned) == 2
    only_corners_path = only_corners_path + test_if_aligned
    return np.array(only_corners_path)


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