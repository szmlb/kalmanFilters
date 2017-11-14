import numpy as np

class linearKalmanFilter:
    def __init__(self, A, B, Bu, C, Q, R):
        self.A = A
        self.B = B
        self.Bu = B
        self.C = C
        self.Q = Q
        self.R = R
        self.P = np.zeros([A.shape[0], A.shape[0]])
        self.xvec_hat = np.zeros([A.shape[0], 1])
        self.xvec_hat_ = np.zeros([A.shape[0], 1])

    def prediction(self, u):
        self.xvec_hat_ = self.A * self.xvec_hat + self.Bu * u
        self.P_ = self.A * self.P * self.A.T + self.B * self.Q * self.B.T

    def filtering(self, y):
        self.G = self.P_ * self.C.T * (self.C * self.P_ * self.C.T + self.R).I
        self.xvec_hat = self.xvec_hat_ + self.G * (y - self.C * self.xvec_hat_)
        self.P = (np.identity(self.A.shape[0]) - self.G * self.C) * self.P_
