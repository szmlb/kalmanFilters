import numpy as np
import linearKalmanFilter as lKF

if __name__ == '__main__':

    A = np.matrix([1])
    B = np.matrix([1])
    C = np.matrix([1])
    Q = np.matrix([1])
    R = np.matrix([10])

    N = 300

    np.random.seed(0)
    v = np.random.randn(1, N) * np.sqrt(Q[0, 0])
    w = np.random.randn(1, N) * np.sqrt(R[0, 0])

    x = np.zeros([1, N])
    y = np.zeros([1, N])

    x_hat = np.zeros([1, N])

    for i in range(N-1):
        x[:,  i+1] = A * x[:, i] + B * v[:,  i]
        y[:,  i] = C * x[:,  i] + w[:, i]
    y[:,  N-1] = C * x[:,  N-1] + w[:,  N-1]

    kf = lKF.linearKalmanFilter(A, B, 0, C, Q, R)
    for i in range(N):
        kf.prediction(0)
        kf.filtering(y[:,  i])

        x_hat[0, i] = kf.xvec_hat[0,  0]

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(y[0,  :], label='measured')
    plt.plot(x[0,  :], label='true')
    plt.plot(x_hat[0,  :],  label='estimated')
    plt.legend()
    plt.show()
