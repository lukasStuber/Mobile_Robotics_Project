# Kalman filter
# inspired by https://github.com/houman95/Extended-Kalman-Filter-for-Tracking-a-Two-Wheeled-Robot/blob/main/Estimator.m

# Measurements z: position (x,y) [mm] and orientation theta [rad]
#      (x    )
# z =  (y    )
#      (theta)

# state x: position (x,y) [mm], orientation (theta) [rad]
#      (x    )
# x =  (y    )
#      (theta)

# input u: rotation speed of the wheels [rad/s]
# u = (w_l)
#     (w_r)


import numpy as np
from constants import *

class Kalman:
    def __init__(self, noisePosX, noisePosY, noiseTheta, noiseInputL, noiseInputR):
        # timestep dt for state propagation
        self.dt_prop = 0
        self.last_t_prop = 0

        # timestep dt for state correction
        self.dt_correct = 0
        self.last_t_correct = 0
        
        # process noise
        self.R = np.diag([noisePosX, noisePosY, noiseTheta^2/6])

        # measurement noise
        self.Q = np.diag([noiseInputL^2/6, noiseInputR^2/6])

        # Observation Matrix H
        self.H = np.array([[1, 0, 0],
                           [0, 1, 0],
                           [0, 0, 1]])

        # distance between the wheels [mm]
        self.base = WHEEL_DIST
        # radius of the wheels [mm]
        self.radius = WHEEL_RADIUS
    
    def set_state(self, x0, P0):
        # initial state
        self.x = x0
        # initial state covariance
        self.P = P0
    
    def calc_dt_prop(self, current_t):
        self.dt_prop = current_t - self.last_t_prop
        self.last_t_prop = current_t

    def calc_dt_correct(self, current_t):
        self.dt_correct = current_t - self.last_t_correct
        self.last_t_correct = current_t

    def state_prop(self, u, current_t):
        self.calc_dt_prop(current_t)
        v_l = u[0]*self.radius
        v_r = u[1]*self.radius
        speed_t = (v_l + v_r)/2           # distance speed
        speed_r = (v_l - v_r)/self.base   # rotational speed
        
        sin = np.sin(self.x[2])
        cos = np.cos(self.x[2])

        # state propagation
        self.x[0] = self.x[0] + self.dt_prop*speed_t*cos - 0.5*self.dt_prop^2*speed_t*speed_r*sin
        self.x[1] = self.x[1] + self.dt_prop*speed_t*sin + 0.5*self.dt_prop^2*speed_t*speed_r*cos
        self.x[2] = self.x[2] + self.dt_prop*speed_r

        # transition function (state propagation matrix)
        A = np.array([[1, 0, -self.dt_prop*speed_t*sin - 0.5*self.dt_prop^2*speed_t*speed_r*cos],
                      [0, 1,  self.dt_prop*speed_t*cos - 0.5*self.dt_prop^2*speed_t*speed_r*sin],
                      [0, 0, 1]])

        # input transition matrix
        L = np.array([[self.dt_correct*v_l/2*cos - ((self.dt_correct^2)/(2*self.base))*v_l^2*sin, -self.dt_correct*v_r/2*cos + ((self.dt_correct^2)/(2*self.base))*v_r^2*sin],
                      [self.dt_correct*v_l/2*cos + ((self.dt_correct^2)/(2*self.base))*v_l^2*cos, -self.dt_correct*v_l/2*sin - ((self.dt_correct^2)/(2*self.base))*v_l^2*cos],
                      [self.dt_correct*v_l/self.base, -self.dt_correct*v_r/self.base]])
                      
        # state covariance propagation
        self.P = A@self.P@A.T + L@self.Q@L.T

    
    def correct(self, u, z, current_t):
        #self.calc_dt_correct(current_t)
        #v_l = u[0]*self.radius
        #v_r = u[1]*self.radius
        #speed_t = (v_l + v_r)/2           # distance speed
        #speed_r = (v_l - v_r)/self.base   # rotational speed
        
        #sin = np.sin(self.x[2])
        #cos = np.cos(self.x[2])

        # transition function (state propagation matrix)
        #A = np.array([[1, 0, -self.dt_correct*speed_t*sin - 0.5*self.dt_correct^2*speed_t*speed_r*cos],
        #              [0, 1,  self.dt_correct*speed_t*cos - 0.5*self.dt_correct^2*speed_t*speed_r*sin],
        #              [0, 0, 1]])
        
        # input transition matrix
        #L = np.array([[self.dt_correct*v_l/2*cos - ((self.dt_correct^2)/(2*self.base))*v_l^2*sin, -self.dt_correct*v_r/2*cos + ((self.dt_correct^2)/(2*self.base))*v_r^2*sin],
        #              [self.dt_correct*v_l/2*cos + ((self.dt_correct^2)/(2*self.base))*v_l^2*cos, -self.dt_correct*v_l/2*sin - ((self.dt_correct^2)/(2*self.base))*v_l^2*cos],
        #              [self.dt_correct*v_l/self.base, -self.dt_correct*v_r/self.base]])
        
        # state propagation
        #self.x[0] = self.x[0] + self.dt*speed_t*cos - 0.5*self.dt^2*speed_t*speed_r*sin
        #self.x[1] = self.x[1] + self.dt*speed_t*sin + 0.5*self.dt^2*speed_t*speed_r*cos
        #self.x[2] = self.x[2] + self.dt*speed_r 

        # state covariance propagation
        #self.P = A@self.P@A.T + L@self.Q@L.T

        # kalman gain
        K = self.P@self.H.T@np.linalg.inv(self.H@self.P@self.H.T + self.R)

        # correct state and covariance
        self.x = self.x + K@(z - self.H@self.x) 
        self.P = (np.eye(3) - K@self.H)@self.P