import numpy as np

def make_circle_data(n0=1000, n1=5000, seed=0):
    np.random.seed(seed)

    theta0 = np.sort(2 * np.pi * np.random.rand(n0))
    theta1 = np.sort(2 * np.pi * np.random.rand(n1))

    r0 = 1 + 0.3 * np.random.rand(n0)
    r1 = 1 + 0.3 * np.random.rand(n1)

    # dataset 0
    x0 = 0.5 * r0 * np.cos(theta0) - 0.5
    y0 = 0.5 * r0 * np.sin(theta0) - 1
    z0 = np.zeros(n0)

    # dataset 1
    x1 = r1 * np.cos(theta1) + 3
    y1 = r1 * np.sin(theta1)
    z1 = np.random.rand(n1) - 0.5

    return np.stack([x0, y0, z0], axis=-1), np.stack([x1, y1, z1], axis=-1)
