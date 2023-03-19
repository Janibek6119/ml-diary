
import numpy as np


def create_2D_clusters(distance: float, per_cluster: int):
    C1 = np.random.randn(per_cluster, 2)
    C2 = np.random.randn(per_cluster, 2) + create_2D_vector_by_magnitude(distance)
    return C1, C2

def create_2D_vector_by_magnitude(magnitude: float):
    angle = np.random.rand() * np.pi * 2
    x = np.cos(angle) * magnitude
    y = np.sin(angle) * magnitude
    return np.array([x,y])


def calculate_cost(dataset, u):
    sqr_distances_0 = np.sum((dataset - u[0])**2, axis=1)
    sqr_distances_1 = np.sum((dataset - u[1])**2, axis=1)
    K0_is_closer = sqr_distances_0 < sqr_distances_1

    examples_0 = dataset[K0_is_closer]
    examples_1 = dataset[np.invert(K0_is_closer)]

    cost_0 = np.mean(sqr_distances_0[K0_is_closer])
    cost_1 = np.mean(sqr_distances_1[np.invert(K0_is_closer)])

    return cost_0 + cost_1, examples_0, examples_1
