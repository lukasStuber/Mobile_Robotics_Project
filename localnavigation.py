import os
import sys
import math
from statistics import mean
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
# %matplotlib inline

### --- OBSTACLE AVOIDANCE --- ###
# (run on thymio)
speed0 = 100 #; speedGain = 2 # only if observing a ground gradient
obstSpeedGain = [6, 4, -2, -6, -8]
timer_period[0] = 10

@onevent 
def timer0():
    global prox_ground_delta, prox_horizontal, speed0, speedGain, obstSpeedGain, motor_left_target, motor_right_target
    # diffDelta = prox_ground_delta[1] - prox_ground_delta[0]
    spLeft = speed0 # - speedGain * diffDelta
    spRight = speed0 # + speedGain * diffDelta
    for i in range(5):
        spLeft += prox_horizontal[i] * obstSpeedGain[i] // 100
        spRight += prox_horizontal[i] * obstSpeedGain[4 - i] // 100
    motor_left_target = spLeft; motor_right_target = spRight

### --- LOCAL MAPPING --- ###
# needs:
# - list of displacement steps (reld_pos) (x,y,theta°)
#     before using this here, run:
#     rel_dpos[:, :2] = rel_dpos[:, :2]*20 (because 20cm grid)
#     rel_dpos[:, 2] = rel_dpos[:, 2]*math.pi/180
# - sensor values per step (map_sensor_vals)
# - sensor_distances [cm] and sensor_measurements for interpolation

# Thymio outline
center_offset = np.array([5.5,5.5])
thymio_coords = np.array([[0,0],[11,0],[11,8.5],[10.2, 9.3],[8, 10.4],[5.5,11],[3.1, 10.5],[0.9, 9.4],[0, 8.5],[0,0]])-center_offset
# Sensor positions and orientations
sensor_pos_from_center = np.array([[0.9,9.4],[3.1,10.5],[5.5,11.0],[8.0,10.4],[10.2,9.3],[8.5,0],[2.5,0]])-center_offset
sensor_angles = np.array([120, 105, 90, 75, 60, -90, -90])*math.pi/180

def sensor_val_to_cm_dist(val):
    if val == 0: return np.inf
    f = interp1d(sensor_measurements, sensor_distances)
    return f(val).item()

def obstacles_pos_from_sensor_vals(sensor_vals):
    dist_to_sensor = [sensor_val_to_cm_dist(x) for x in sensor_vals]
    dx_from_sensor = [d*math.cos(alpha) for (d, alpha) in zip(dist_to_sensor, sensor_angles)]
    dy_from_sensor = [d*math.sin(alpha) for (d, alpha) in zip(dist_to_sensor, sensor_angles)]
    obstacles_pos = [[x[0]+dx, x[1]+dy] for (x,dx,dy) in zip(sensor_pos_from_center,dx_from_sensor,dy_from_sensor )]
    return np.array(obstacles_pos)

def rotate(angle, coords):
    R = np.array(((np.cos(angle), -np.sin(angle)), (np.sin(angle),  np.cos(angle))))
    return R.dot(coords.transpose()).transpose()

abs_pos = [[0,0, math.pi/2]] # x,y,theta
for (dx,dy,dtheta) in rel_dpos[:]:
    (x,y,theta) = abs_pos[-1][0], abs_pos[-1][1], abs_pos[-1][2]
    d = np.sqrt(dx**2+dy**2)
    new_pos = [x+d*np.cos(theta+dtheta), y+d*np.sin(theta+dtheta), (theta+dtheta)%(2*math.pi)]
    abs_pos.append(new_pos)

abs_pos = np.array(abs_pos)
# Compute the local occupancy grid from the sensor values at each step
local_occupancy_grids = [obstacles_pos_from_sensor_vals(x) for x in map_sensor_vals]
# Create the global map based on the data acquired previously
global_map, overall_thymio_coords = [], []
for (local_grid, pos) in zip(local_occupancy_grids, abs_pos):
    rotated_grid = rotate(pos[2]-math.pi/2, local_grid)
    rotated_thymio_coords = rotate(pos[2]-math.pi/2, thymio_coords)
    obstacles_pos = rotated_grid+np.array([pos[0], pos[1]])
    abs_Thymio_coords = rotated_thymio_coords+np.array([pos[0], pos[1]])
    global_map.append(obstacles_pos)
    overall_thymio_coords.append(abs_Thymio_coords)

global_map = np.array(np.vstack(global_map))
# plt.figure(figsize=(10,10))
# plt.plot(abs_pos[:,0], abs_pos[:,1])
# plt.scatter(global_map[:,0],global_map[:,1], color="r", s=10)
# plt.plot(np.array(abs_pos)[:,0], np.array(abs_pos)[:,1], color="r", marker="o")
# for coords in overall_thymio_coords: plt.plot(coords[:,0], coords[:,1], color="g")
# plt.axis("equal")