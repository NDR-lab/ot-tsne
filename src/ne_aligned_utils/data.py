import numpy as np


def make_fig1_data(n0=1000, n1=5000, seed=0, clean=False):
    np.random.seed(seed)

    # Deformed circle data
    theta1 = np.sort(2 * np.pi * np.random.rand(n0))
    theta2 = np.sort(2 * np.pi * np.random.rand(n1))

    r1 = 0.5 * (1 + 0.3 * np.random.rand(n0))
    r2 = 1 + 0.3 * np.random.rand(n1)

    x1_clean = r1 * np.cos(theta1)
    y1_clean = r1 * np.sin(theta1)

    x2_clean = r2 * np.cos(theta2)
    y2_clean = r2 * np.sin(theta2)

    if clean:
        return np.stack([x1_clean, y1_clean], axis=-1), np.stack([x2_clean, y2_clean], axis=-1)

    else:
        x1 = x1_clean - 0.5
        y1 = y1_clean - 1
        z1 = np.zeros(n0)

        x2 = x2_clean + 3
        y2 = y2_clean
        z2 = np.random.rand(n1) - 0.5


        return np.stack([x1, y1, z1], axis=-1), np.stack([x2, y2, z2], axis=-1)