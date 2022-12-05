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

class Kalman:
    def __init__(self, dt, noisePosX, noisePosY, noiseTheta, noiseInputL, noiseInputR):
        # timestep dt
        self.dt = dt
        
        # process noise
        self.R = np.diag([noisePosX, noisePosY, noiseTheta^2/6])

        # measurement noise
        self.Q = np.diag([noiseInputL^2/6, noiseInputR^2/6])

        # Observation Matrix H
        self.H = np.array([[1, 0, 0],
                           [0, 1, 0],
                           [0, 0, 1]])

        # distance between the wheels [mm]
        self.base = 94
        # radius of the wheels [mm]
        self.radius = 22
    
    def set_state(x0, P0):
        # initial state
        self.x = x0
        # initial state covariance
        self.P = P0

    
    def estimate(self, u, z):
        v_l = u[0]*self.radius
        v_r = u[1]*self.radius
        speed_t = (v_l + v_r)/2           # distance speed
        speed_r = (v_l - v_r)/self.base   # rotational speed

        # transition function (state propagation matrix)
        A = np.array([[1, 0, -self.dt*speed_t*sin - 0.5*self.dt^2*speed_t*speed_r*cos],
                      [0, 1,  self.dt*speed_t*cos - 0.5*self.dt^2*speed_t*speed_r*sin],
                      [0, 0, 1]])
        
        sin = np.sin(self.x[2])
        cos = np.cos(self.x[2])
        # input transition matrix
        L = np.array([[self.dt*v_l/2*cos - ((self.dt^2)/(2*self.base))*v_l^2*sin, -self.dt*v_r/2*cos + ((self.dt^2)/(2*self.base))*v_r^2*sin],
                      [self.dt*v_l/2*cos + ((self.dt^2)/(2*self.base))*v_l^2*cos, -self.dt*v_l/2*sin - ((self.dt^2)/(2*self.base))*v_l^2*cos],
                      [self.dt*v_l/self.base, -self.dt*v_r/self.base]])
        
        # state propagation
        self.x[0] = self.x[0] + self.dt*speed_t*cos - 0.5*self.dt^2*speed_t*speed_r*sin
        self.x[1] = self.x[1] + self.dt*speed_t*sin + 0.5*self.dt^2*speed_t*speed_r*cos
        self.x[2] = self.x[2] + self.dt*speed_r 

        # state covariance propagation
        self.P = A@self.P@A.T + L@self.Q@L.T

        # kalman gain
        K = self.P@self.H.T@np.linalg.inv(self.H@self.P@self.H.T + self.R)

        # correct state and covariance
        self.x = self.x + K@(z - self.H@self.x) 
        self.P = (np.eye(3) - K@self.H)@self.P