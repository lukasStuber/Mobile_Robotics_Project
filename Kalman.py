# Kalman filter inspired by https://github.com/houman95/Extended-Kalman-Filter-for-Tracking-a-Two-Wheeled-Robot/blob/main/Estimator.m
# Measurements z = (x, y, theta) [mm][mm][rad]
# state x = (x, y, theta) [mm][mm][rad]
# input u = (w_l, w_r) [rad/s]

import numpy as np
from constants import *
import time

class Kalman:
    def __init__(self):
        # timestep for state propagation
        self.dt = None
        self.prev_time = None
        # state
        self.x = np.zeros((3)) # state
        self.P = 1000*np.ones((3,3)) # state covariance
        # process noise
        self.R = np.diag([NOISE_POS_XY, NOISE_POS_XY, NOISE_POS_THETA**2/6])
        # measurement noise
        self.Q = np.diag([NOISE_MEASURE_XY**2/6, NOISE_MEASURE_XY**2/6])
        # Observation Matrix H
        self.H = np.array([[1, 0, 0],
                           [0, 1, 0],
                           [0, 0, 1]])
    
    def set_state(self, x0):
        self.x = np.array(x0) # initial state
        self.x = np.reshape(x0, (3,1))

    def state_prop(self, u):
        if self.prev_time is None: # initialisation
            self.prev_time = time.time()
            return
        u = np.array(u)
        self.dt = time.time() - self.prev_time
        self.prev_time = time.time()
        # https://ocw.mit.edu/courses/6-186-mobile-autonomous-systems-laboratory-january-iap-2005/764fafce112bed6482c61f1593bd0977_odomtutorial.pdf
        (dx, dy) = self.dt*SPEED_TO_MMS*u # left and right displacements [mm]
        da = -(dy - dx)/WHEEL_DIST # rotation angle [rad]
        dc = (dx + dy)/2 # center displacement [mm]
        (vx, vy) = SPEED_TO_MMS*u # left and right wheel speeds [mm/s]
        vt = (vx + vy)/2 # translation speed [mm/s]
        vr = -(vy - vx)/WHEEL_DIST # rotation speed [rad/s]
        #print("angle is ", self.x[2])
        sin = math.sin(self.x[2])
        cos = math.cos(self.x[2])
        # state propagation
        self.x[0] = self.x[0] + dc*cos
        self.x[1] = self.x[1] + dc*sin
        self.x[2] = (self.x[2] + da) % (2*math.pi)
        # transition function (state propagation matrix)
        A = np.array([[1, 0, -self.dt*vt*sin],
                      [0, 1,  self.dt*vt*cos],
                      [0, 0, 1]])
        # input transition matrix
        L = np.array([[self.dt*vx/2*cos, -self.dt*vy/2*cos],
                      [self.dt*vx/2*cos, -self.dt*vx/2*sin],
                      [self.dt*vx/WHEEL_DIST, -self.dt*vy/WHEEL_DIST]])
        # state covariance propagation
        self.P = A@self.P@A.T + L@self.Q@L.T
    
    def state_correct(self, z):
        z = np.reshape(z, (3,1))
        K = self.P@self.H.T@np.linalg.inv(self.H@self.P@self.H.T + self.R) # kalman gain
        self.x = (self.x.T + K@(z - self.H@self.x.T)).T # state
        self.P = (np.eye(3) - K@self.H)@self.P # state covariance

if __name__ == "__main__":
    kalman = Kalman()
    kalman.set_state([0, 0, 0])
    kalman.state_prop([1, 1])
    print(kalman.x)
    kalman.state_correct([1,1,0])
    print(kalman.x)