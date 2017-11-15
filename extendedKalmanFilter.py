import numpy as np

class extendedKalmanFilter:
    def __init__(self, f, h, A, B, Bu, C, Q, R, n):
        self.f = f
        self.h = h
        self.A = A
        self.B = B
        self.Bu = Bu
        self.C = C
        self.Q = Q
        self.R = R
        self.n = n
        self.P = np.zeros([self.n, self.n])
        self.xvec_hat = np.zeros([self.n, 1])
        self.xvec_hat_ = np.zeros([self.n, 1])

    def prediction(self, u):
        self.xvec_hat_ = self.f(self.xvec_hat[:, 0])
        self.P_ = self.A(self.xvec_hat_[:, 0]) * self.P * self.A(self.xvec_hat_[:, 0]).T + self.B * self.Q * self.B.T

    def filtering(self, y):
        self.G = self.P_ * self.C(self.xvec_hat_[:, 0]).T * (self.C(self.xvec_hat_[:, 0]) * self.P_ * self.C(self.xvec_hat_[:, 0]).T + self.R).I
        self.xvec_hat = self.xvec_hat_ + self.G * (y - self.h(self.xvec_hat_[:, 0]))
        self.P = (np.identity(self.n) - self.G * self.C(self.xvec_hat_[:, 0])) * self.P_
