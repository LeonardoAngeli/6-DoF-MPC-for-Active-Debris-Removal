import numpy as np


class KalmanFilter:
    # Linear Kalman filter for the relative translational state [r, v]
    # Process model: x_k = Ad x_{k-1} + Bd u_{k-1} + w
    # Measurement:   y_k = C x_k + v,                   
    # R is range-dependent and provided by the sensor model at each update.

    def __init__(self, Ad, Bd, C, Q, sensor, x0, P0):
        self.Ad = Ad
        self.Bd = Bd
        self.C = C
        self.Q = Q
        self.sensor = sensor
        self.x = x0.copy()
        self.P = P0.copy()
        self.R = None                   # Measurement covariance
        self.S = None                   # Innovation covariance

    def predict(self, u):

        # State estimate a priori
        x_k = self.Ad @ self.x + self.Bd @ u

        # Covariance a priori
        P_k = self.Ad @ self.P @ self.Ad.T + self.Q

        self.x = x_k
        self.P = P_k

    def update(self, y):
        # Measurement update with the LIDAR relative-position measurement

        # Range-dependent measurement covariance
        self.R = self.sensor.get_matrix_R(self.x)

        # Innovation covariance
        S = self.C @ self.P @ self.C.T + self.R

        # Kalman gain: K = P C^T S^-1 
        K = np.linalg.solve(S, self.C @ self.P).T

        # State estimate a posteriori
        x_k = self.x + K @ (y - self.C @ self.x)

        # Covariance a posteriori 
        IKC = np.eye(self.P.shape[0]) - K @ self.C
        P_k = IKC @ self.P @ IKC.T + K @ self.R @ K.T
        P_k = 0.5 * (P_k + P_k.T)       # Forcing symmetry

        self.x = x_k
        self.P = P_k
        self.S = S

    def get_state(self):
        return self.x, self.P