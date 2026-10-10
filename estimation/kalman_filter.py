import numpy as np


class KalmanFilter:
    def __init__(self, Ad, Bd, C, Q, R, x0, P0):
        self.Ad = Ad
        self.Bd = Bd
        self.C = C
        self.Q = Q
        self.R = R
        self.x = x0.copy()
        self.P = P0.copy()
        self.S = 0

    def predict(self, u):
        # Predicting step k from k-1

        # Covariance a priori
        P_k = self.Ad @ self.P @ self.Ad.T + self.Q

        # State Estimate a priori
        x_k = self.Ad @ self.x + self.Bd @ u

        self.P = P_k
        self.x = x_k

    def update(self, y):
        # Updating using KF theory

        # Innovation Covariance
        S = self.C @ self.P @ self.C.T + self.R

        # Kalman Gain
        K = np.linalg.solve(S, self.C @ self.P).T

        # State Estimate a posteriori
        x_k = self.x + K @ (y - self.C @ self.x)

        # Covariance a posteriori (Joseph form)
        IKH = np.eye(self.P.shape[0]) - K @ self.C
        P_new = IKH @ self.P @ IKH.T + K @ self.R @ K.T
        P_new = 0.5 * (P_new + P_new.T)                               # Forcing symmetry in P


        self.P = P_new
        self.x = x_k
        self.S = S

    def get_state(self):
        return self.x, self.P

    