import numpy as np
# ---------------------------
# TRUE exhaustive neighbors (numpy for whole-dataset)
# ---------------------------
def compute_true_neighbors_numpy(qvec, data, N):
    """
    qvec: numpy [d]
    data: numpy [n, d]
    returns: (top_indices [N], top_dists [N])
    """
    # vectorized squared distances for speed
    diff = data - qvec  # [n, d]
    sq = np.sum(diff * diff, axis=1)
    # get smallest N
    if N >= len(sq):
        idx = np.argsort(sq)
    else:
        idx = np.argpartition(sq, N-1)[:N]
        idx = idx[np.argsort(sq[idx])]
    dists = np.sqrt(sq[idx])
    return idx, dists
