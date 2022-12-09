import math
import numpy as np

class Extended_Kalman_Filter():
    def __init__(self):
        self.dt = 1     #[s]
        self.input_speed = 14 #[mm/s]
        self.scaling_factor = 1 #[mm/pxl]
        pxl_var = 0.25
        self.R = np.diag([pxl_var,  # variance of location on x-axis in pxl^2
                    pxl_var,   # variance of location on x-axis in pxl^2
                    0,         # variance of yaw angle          in rad^2
                    6.15,      # variance of velocity           in pxl^2/s^2
                    0])        # variance of angular velocity   in rad^2/s^2(yaw rate)

        self.Q = np.diag([0.04,    # variance of location on x-axis in pxl^2
                     0.04,    # variance of location on y-axis in pxl^2
                     0,       # variance of yaw angle          in rad^2
                     6.15,    # variance of velocity           in pxl^2/s^2
                     0])      # variance of angular velocity   in rad^2/s^2(yaw rate)

    def process_cov(self):
        Q = np.diag([0.04,   # variance of location on x-axis in pxl^2
                    0.04,   # variance of location on y-axis in pxl^2
                    0,       # variance of yaw angle          in rad^2
                    6.15,    # variance of velocity           in pxl^2/s^2
                    0     # variance of angular velocity   in rad^2/s^2(yaw rate)
                    ])
        return Q

    def u_input(self,speed_l,speed_r, wheel_distance, scaling_factor):
        v = self.scaling_factor * (speed_r + speed_l)/2
        theta_dot = (speed_r - speed_l)/wheel_distance
        u = np.array([v, theta_dot]).T
        return u

    def measure_state(self,x):
        C = np.eye(5)
        y = np.dot(C,x) + np.diag(self.R)
        return y

    def capture_measurements(self,ur):
        y=np.zeros(5).T
        y[0] = ur.coords[0]
        y[1] = ur.coords[1]
        y[2] = ur.angle
        y[3] = math.sqrt((y[0]-self.Mu[0])**2+(y[1]-self.Mu[1])**2)/self.dt
        y[4] = (y[2]-self.Mu[2])/self.dt
        y_current = self.measure_state(y)
        return y_current

    def thymio_state(self, u, Q, delta_t, Mu):
        theta = Mu[2]
        A = np.eye(5)*np.array([1,1,1,0,0])
        B = np.array([[delta_t*math.cos(theta),0],
                     [delta_t*math.sin(theta),0],
                     [0,delta_t],
                     [1,0],
                     [0,1]])
        Mu_pred = np.dot(A,Mu) + np.dot(B,u) + np.diag(Q)
        return Mu_pred

    def jacobian_G(self,theta, v):
        G = np.array([[1,0,-self.dt*v*math.sin(theta),self.dt*math.cos(theta),0],
                     [0,1, self.dt*v*math.cos(theta),self.dt*math.sin(theta),0],
                     [0,0,1,0,self.dt],
                     [0,0,0,1,0],
                     [0,0,0,0,1]])
        return G

    def get_Mu_pred(self, u, delta_t, ur):
        Q = self.process_cov()
        Mu_pred = self.thymio_state(u, Q, delta_t, np.array([ur.coords[0], ur.coords[1], ur.angle, 0, 0]).T)
        return Mu_pred

    def extended_kalman(self, u, y):
        '''
        Function that takes the previous mean and covariance estimations and Applies an extended Kalman Filter.
        It incorporates the camera mesurements and motor controls to find the current Kalman gain.
        Kalman gain is then used to correct system state estimations.

        Inputs: - Mu    : The previous mesurement time's mean system state values.
                - Sigma : The previous measurement time's Covariance matrix.
                          1x2 vector that holds current inputed speed and angular velocity.
                - y     : The current system camera measurements.
        Outputs: - Mu_est    : The current estimated mean system state values, after Kalman filtering.
                 - Sigma_est : The current estimated System Covariance matrix.
        '''
        #Predict values
        Q = self.process_cov()
        Mu_pred = self.thymio_state(u, Q, self.dt, self.Mu)
        G = self.jacobian_G(self.Mu[2], u[0])
        Sigma_pred = np.dot(G,np.dot(self.Sigma,G.T)) + Q
        #Calculate Kalman Gain
        H = np.eye(5)  #technically jacobian of h(x), the camera measurements function
        S = np.dot(np.dot(H,Sigma_pred),H.T) + self.R
        K = np.dot(Sigma_pred,np.dot(H.T,np.linalg.inv(S)))
        #Update Estimated values
        Mu_est = Mu_pred + np.dot(K,(y - self.measure_state(Mu_pred)))
        Sigma_est = np.dot((np.eye(5)-np.dot(K,H)),Sigma_pred)+np.eye(5)*1.00001
        # Sigma_est[Sigma_est < 1e-5] = 0
        self.Mu, self.Sigma = Mu_est, Sigma_est
