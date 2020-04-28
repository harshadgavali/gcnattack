import numpy as np


def get_simil(v1, v2, delta=1e-5):
    M11 = (v1 * v2).sum()
    M10 = (v1 * (1-v2)).sum()
    M01 = ((1-v1) * v2).sum()
    # print(v1.shape, v1.sum(), v2.sum())
    # print(M11, M10, M01)
    return M11 / (M10 + M01 + M11 + delta)

def defense(adj, features, alpha=0.5):
    n = adj.shape[0]
    for i in range(n):
        for j in range(n):
            if adj[i, j] and get_simil(features[i], features[j]) < alpha:
                adj[i, j] = 0
    return adj, features
