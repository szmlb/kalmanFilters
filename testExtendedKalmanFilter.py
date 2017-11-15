import numpy as np
import extendedKalmanFilter as eKF

if __name__ == '__main__':

    def f(x):
        return np.matrix(x[0] + 3.0 * np.cos(x[0]/10.0))

    def h(x):
        return np.matrix(x[0]**3)

    def A(x):
        return np.matrix( 1.0 - 3.0 / 10.0 * np.sin(x[0] / 10.0) )

    def C(x):
        return np.matrix( 3.0 * x[0]**2 )

    B = np.matrix([1])
    Q = np.matrix([1])
    R = np.matrix([100])

    N = 50

    np.random.seed(0)
    v = np.random.randn(1, N) * np.sqrt(Q[0, 0])
    w = np.random.randn(1, N) * np.sqrt(R[0, 0])

    x = np.zeros([1, N])
    y = np.zeros([1, N])

    x_hat = np.zeros([1, N])

    for i in range(N-1):
        x[:,  i+1] = f(x[:, i]) + B * v[:,  i]
        y[:,  i] = h(x[:,  i]) + w[:, i]
    y[:,  N-1] = h(x[:,  N-1]) + w[:,  N-1]

    kf = eKF.extendedKalmanFilter(f, h, A, B, 0, C, Q, R, 1)
    for i in range(N):
        kf.prediction(0)
        kf.filtering(y[:,  i])

        x_hat[0, i] = kf.xvec_hat[0,  0]

    import matplotlib.pyplot as plt
    plt.figure()
    plt.subplot(211)
    plt.plot(y[0,  :], label='measured')
    plt.legend()
    plt.subplot(212)
    plt.plot(x[0,  :], label='true')
    plt.plot(x_hat[0,  :],  label='estimated')
    plt.legend()
    plt.show()