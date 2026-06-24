import numpy as np
import torch


DISTANCE_METRICS = {
    "euclidean",
    "cosine_distance",
    "manhattan",
    "chebyshev",
    "minkowski",
    "mahalanobis",
    "hybrid",
}
SIMILARITY_METRICS = {"cosine_similarity"}


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def euclidean_distance(qf, gf, squared=False):
    qf = _to_numpy(qf).astype(np.float32, copy=False)
    gf = _to_numpy(gf).astype(np.float32, copy=False)
    q_norm = np.sum(qf * qf, axis=1, keepdims=True)
    g_norm = np.sum(gf * gf, axis=1, keepdims=True).T
    dist_sq = np.maximum(q_norm + g_norm - 2.0 * np.matmul(qf, gf.T), 0.0)
    if squared:
        return dist_sq
    return np.sqrt(dist_sq)


def cosine_similarity(qf, gf):
    qf = _to_numpy(qf).astype(np.float32, copy=False)
    gf = _to_numpy(gf).astype(np.float32, copy=False)
    q_norm = np.linalg.norm(qf, axis=1, keepdims=True)
    g_norm = np.linalg.norm(gf, axis=1, keepdims=True)
    qf = qf / np.clip(q_norm, a_min=1e-12, a_max=None)
    gf = gf / np.clip(g_norm, a_min=1e-12, a_max=None)
    return np.matmul(qf, gf.T)


def cosine_distance(qf, gf):
    return 1.0 - cosine_similarity(qf, gf)


def manhattan_distance(qf, gf):
    qf = _to_numpy(qf).astype(np.float32, copy=False)
    gf = _to_numpy(gf).astype(np.float32, copy=False)
    return np.abs(qf[:, None, :] - gf[None, :, :]).sum(axis=-1)


def chebyshev_distance(qf, gf):
    qf = _to_numpy(qf).astype(np.float32, copy=False)
    gf = _to_numpy(gf).astype(np.float32, copy=False)
    return np.abs(qf[:, None, :] - gf[None, :, :]).max(axis=-1)


def minkowski_distance(qf, gf, p=3):
    qf = _to_numpy(qf).astype(np.float32, copy=False)
    gf = _to_numpy(gf).astype(np.float32, copy=False)
    return np.power(np.power(np.abs(qf[:, None, :] - gf[None, :, :]), p).sum(axis=-1), 1.0 / p)


def mahalanobis_distance(qf, gf, cov_matrix=None, regularization=1e-4):
    qf = _to_numpy(qf).astype(np.float32, copy=False)
    gf = _to_numpy(gf).astype(np.float32, copy=False)

    if cov_matrix is None:
        combined = np.vstack([qf, gf])
        cov_matrix = np.cov(combined.T) + np.eye(combined.shape[1]) * regularization

    cov_inv = np.linalg.pinv(cov_matrix)
    diff = qf[:, None, :] - gf[None, :, :]
    return np.sqrt(np.maximum(np.einsum("qgd,dd,qgd->qg", diff, cov_inv, diff), 0.0))


def cosine_euclidean_hybrid(qf, gf, alpha=0.7):
    cos_dist = cosine_distance(qf, gf)
    euc_dist = euclidean_distance(qf, gf)
    euc_dist = euc_dist / np.clip(np.max(euc_dist), a_min=1e-12, a_max=None)
    cos_dist = cos_dist / 2.0
    return alpha * cos_dist + (1.0 - alpha) * euc_dist


__distance_metrics__ = {
    "euclidean": euclidean_distance,
    "cosine_similarity": cosine_similarity,
    "cosine_distance": cosine_distance,
    "manhattan": manhattan_distance,
    "chebyshev": chebyshev_distance,
    "minkowski": minkowski_distance,
    "mahalanobis": mahalanobis_distance,
    "hybrid": cosine_euclidean_hybrid,
}


def metric_type(metric):
    if metric in SIMILARITY_METRICS:
        return "similarity"
    if metric in DISTANCE_METRICS:
        return "distance"
    raise ValueError(f"Unknown distance metric: {metric}")


def to_distance_matrix(values, metric):
    """Convert metric output to a lower-is-better matrix for ranking metrics."""
    if metric_type(metric) == "similarity":
        return 1.0 - values
    return values


def compute_distance(qf, gf, metric="euclidean", **kwargs):
    if metric not in __distance_metrics__:
        raise ValueError(
            f"Unknown distance metric: {metric}. "
            f"Available metrics: {list(__distance_metrics__.keys())}"
        )
    return __distance_metrics__[metric](qf, gf, **kwargs)
