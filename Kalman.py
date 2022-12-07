# Kalman filter inspired by https://github.com/houman95/Extended-Kalman-Filter-for-Tracking-a-Two-Wheeled-Robot/blob/main/Estimator.m
# Measurements z = (x, y, theta) [mm][mm][rad]
# state x = (x, y, theta) [mm][mm][rad]
# input u = (w_l, w_r) [rad/s]

import numpy as np
from constants import *
from time import time

class Kalman:
    def __init__(self, noisePosX, noisePosY, noiseTheta, noiseInputL, noiseInputR):
        # timestep for state propagation
        self.dt = None
        # state
        self.x = np.zeros(3) # state
        self.P = np.zeros((3,3)) # state covariance
        # process noise
        self.R = np.diag([noisePosX, noisePosY, noiseTheta**2/6])
        # measurement noise
        self.Q = np.diag([noiseInputL**2/6, noiseInputR**2/6])
        # Observation Matrix H
        self.H = np.array([[1, 0, 0],
                           [0, 1, 0],
                           [0, 0, 1]])
    
    def set_state(self, x0, P0):
        self.x = x0 # initial state
        self.P = P0 # initial state covariance

    def state_prop(self, u):
        if self.prev_time is None: # initialisation
            self.prev_time = time.time()
            return
        self.dt = time.time() - self.prev_time
        self.prev_time = time.time()
        # https://ocw.mit.edu/courses/6-186-mobile-autonomous-systems-laboratory-january-iap-2005/764fafce112bed6482c61f1593bd0977_odomtutorial.pdf
        (dx, dy) = self.dt*SPEED_TO_MMS*u # left and right displacements [mm]
        da = (dy - dx)/WHEEL_DIST # rotation angle [rad]
        dc = (dx + dy)/2 # center displacement [mm]
        (vx, vy) = u*SPEED_TO_MMS # left and right wheel speeds [mm/s]
        vt = (vx + vy)/2 # translation speed [mm/s]
        vr = (vy - vx)/WHEEL_DIST # rotation speed [rad/s]
        sin = math.sin(self.x[2])
        cos = math.cos(self.x[2])
        # state propagation
        self.x[0] = self.x[0] + dc*cos
        self.x[1] = self.x[1] + dc*sin
        self.x[2] = (self.x[2] + da) % (2*math.pi)
        # transition function (state propagation matrix)
        A = np.array([[1, 0, -self.dt*vt*sin - 0.5*self.dt**2*vt*vr*cos],
                      [0, 1,  self.dt*vt*cos - 0.5*self.dt**2*vt*vr*sin],
                      [0, 0, 1]])
        # input transition matrix
        L = np.array([[self.dt*vx/2*cos - ((self.dt**2)/(2*WHEEL_DIST))*vx**2*sin, -self.dt*vy/2*cos + ((self.dt**2)/(2*WHEEL_DIST))*vy**2*sin],
                      [self.dt*vx/2*cos + ((self.dt**2)/(2*WHEEL_DIST))*vx**2*cos, -self.dt*vx/2*sin - ((self.dt**2)/(2*WHEEL_DIST))*vx**2*cos],
                      [self.dt*vx/WHEEL_DIST, -self.dt*vy/WHEEL_DIST]])           
        # state covariance propagation
        self.P = A@self.P@A.T + L@self.Q@L.T
    
    def state_correct(self, z):
        K = self.P@self.H.T@np.linalg.inv(self.H@self.P@self.H.T + self.R) # kalman gain
        self.x = self.x + K@(z - self.H@self.x) # state
        self.P = (np.eye(3) - K@self.H)@self.P # state covariance