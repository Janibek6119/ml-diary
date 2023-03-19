
import numpy as np


def create_clusters(centrum_ranges: float, per_cluster: int, clusters=2, features=2):
    centrums = (np.random.rand(clusters, 1, features) * 2 - 1) * centrum_ranges
    clusters = np.random.randn(clusters, per_cluster, features) + centrums
    return clusters


def split_into_clusters(dataset: np.ndarray, u: np.ndarray):
    distances = dataset - u.reshape(-1, 1, u.shape[-1])
    sqr_distances = distances ** 2
    rm_distances = np.sum(sqr_distances, axis=-1)
    mins = np.argmin(rm_distances, axis=0)
    clusters = []
    for i in range(len(u)):
        mask = mins == i
        masked_set = dataset[mask]
        clusters.append(masked_set)
    return clusters


def mid_of_group(group):
    return np.mean(group, axis=0)


def mids_of_groups(groups):
    MIDS = []
    for group in groups:
        if len(group) == 0:
            continue
        MIDS.append(mid_of_group(group))
    MIDS = np.array(MIDS)
    return MIDS


def get_random_samples(dataset: np.ndarray, amount: int):
    return dataset[np.random.randint(0, len(dataset), amount)]


def cost_function(groups: list[np.ndarray], K: np.ndarray):
    cost_sum = 0
    for group, centrum in zip(groups, K):
        if len(group) == 0:
            continue
        distances: np.ndarray = group - centrum
        sqr_distances = distances ** 2
        sqr_distances_magnitude = np.sum(sqr_distances, axis=-1)
        cost_sum += np.mean(sqr_distances_magnitude)
    return cost_sum
