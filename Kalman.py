# Kalman filter inspired by https://github.com/houman95/Extended-Kalman-Filter-for-Tracking-a-Two-Wheeled-Robot/blob/main/Estimator.m
# Measurements z = (x, y, theta) [mm][mm][rad]
# state x = (x, y, theta) [mm][mm][rad]
# input u = (w_l, w_r) [rad/s]

import numpy as np
from constants import *
from time import time

class Kalman:
    def __init__(self, noisePosX, noisePosY, noiseTheta, noiseInputL, noiseInputR):
        # timestep dt for state propagation
        self.dt = 0
        self.last_t_prop = 0
        # process noise
        self.R = np.diag([noisePosX, noisePosY, noiseTheta^2/6])
        # measurement noise
        self.Q = np.diag([noiseInputL^2/6, noiseInputR^2/6])
        # Observation Matrix H
        self.H = np.array([[1, 0, 0],
                           [0, 1, 0],
                           [0, 0, 1]])
    
    def set_state(self, x0, P0):
        self.x = x0 # initial state
        self.P = P0 # initial state covariance

    def state_prop(self, u):
        self.dt = time() - self.last_t_prop
        self.last_t_prop = self.last_t_prop + self.dt
        v_l = u[0]*WHEEL_RADIUS
        v_r = u[1]*WHEEL_RADIUS
        speed_t = (v_l + v_r)/2          # translation speed
        speed_r = (v_l - v_r)/WHEEL_DIST # rotation speed
        sin = np.sin(self.x[2]) # sin(theta)
        cos = np.cos(self.x[2]) # cos(theta)
        # state propagation
        self.x[0] = self.x[0] + self.dt*speed_t*cos - 0.5*self.dt^2*speed_t*speed_r*sin
        self.x[1] = self.x[1] + self.dt*speed_t*sin + 0.5*self.dt^2*speed_t*speed_r*cos
        self.x[2] = self.x[2] + self.dt*speed_r
        # transition function (state propagation matrix)
        A = np.array([[1, 0, -self.dt*speed_t*sin - 0.5*self.dt^2*speed_t*speed_r*cos],
                      [0, 1,  self.dt*speed_t*cos - 0.5*self.dt^2*speed_t*speed_r*sin],
                      [0, 0, 1]])
        # input transition matrix
        L = np.array([[self.dt*v_l/2*cos - ((self.dt^2)/(2*WHEEL_DIST))*v_l^2*sin, -self.dt*v_r/2*cos + ((self.dt^2)/(2*WHEEL_DIST))*v_r^2*sin],
                      [self.dt*v_l/2*cos + ((self.dt^2)/(2*WHEEL_DIST))*v_l^2*cos, -self.dt*v_l/2*sin - ((self.dt^2)/(2*WHEEL_DIST))*v_l^2*cos],
                      [self.dt*v_l/WHEEL_DIST, -self.dt*v_r/WHEEL_DIST]])           
        # state covariance propagation
        self.P = A@self.P@A.T + L@self.Q@L.T
    
    def state_correct(self, z):
        # kalman gain
        K = self.P@self.H.T@np.linalg.inv(self.H@self.P@self.H.T + self.R)
        # correct state and covariance
        self.x = self.x + K@(z - self.H@self.x) 
        self.P = (np.eye(3) - K@self.H)@self.P