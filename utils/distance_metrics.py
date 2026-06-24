import numpy as np
import torch


def euclidean_distance(qf, gf):
    if isinstance(qf, torch.Tensor):
        m = qf.shape[0]
        n = gf.shape[0]
        dist_mat = (
            torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n)
            + torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        )
        dist_mat.addmm_(qf, gf.t(), beta=1, alpha=-2)
        return dist_mat.cpu().numpy()
    else:
        qf = np.asarray(qf)
        gf = np.asarray(gf)
        m, n = qf.shape[0], gf.shape[0]
        dist_mat = np.zeros((m, n))
        for i in range(m):
            dist_mat[i] = np.sqrt(np.sum((qf[i] - gf) ** 2, axis=1))
        return dist_mat


def cosine_similarity(qf, gf):
    epsilon = 1e-5
    if isinstance(qf, torch.Tensor):
        dist_mat = qf.mm(gf.t())
        qf_norm = torch.norm(qf, p=2, dim=1, keepdim=True)
        gf_norm = torch.norm(gf, p=2, dim=1, keepdim=True)
        qg_normdot = qf_norm.mm(gf_norm.t())
        dist_mat = dist_mat / qg_normdot.clamp(min=epsilon)
        return dist_mat.cpu().numpy()
    else:
        qf = np.asarray(qf)
        gf = np.asarray(gf)
        qf_norm = np.linalg.norm(qf, axis=1, keepdims=True)
        gf_norm = np.linalg.norm(gf, axis=1, keepdims=True)
        qf_normalized = qf / (qf_norm + epsilon)
        gf_normalized = gf / (gf_norm + epsilon)
        similarity = qf_normalized @ gf_normalized.T
        return similarity


def cosine_distance(qf, gf):
    similarity = cosine_similarity(qf, gf)
    return 1.0 - similarity


def manhattan_distance(qf, gf):
    if isinstance(qf, torch.Tensor):
        qf = qf.unsqueeze(1)
        gf = gf.unsqueeze(0)
        dist_mat = torch.abs(qf - gf).sum(dim=-1)
        return dist_mat.cpu().numpy()
    else:
        qf = np.asarray(qf)
        gf = np.asarray(gf)
        m, n = qf.shape[0], gf.shape[0]
        dist_mat = np.zeros((m, n))
        for i in range(m):
            dist_mat[i] = np.sum(np.abs(qf[i] - gf), axis=1)
        return dist_mat


def chebyshev_distance(qf, gf):
    if isinstance(qf, torch.Tensor):
        qf = qf.unsqueeze(1)
        gf = gf.unsqueeze(0)
        dist_mat = torch.abs(qf - gf).max(dim=-1)[0]
        return dist_mat.cpu().numpy()
    else:
        qf = np.asarray(qf)
        gf = np.asarray(gf)
        m, n = qf.shape[0], gf.shape[0]
        dist_mat = np.zeros((m, n))
        for i in range(m):
            dist_mat[i] = np.max(np.abs(qf[i] - gf), axis=1)
        return dist_mat


def minkowski_distance(qf, gf, p=3):
    if isinstance(qf, torch.Tensor):
        qf = qf.unsqueeze(1)
        gf = gf.unsqueeze(0)
        dist_mat = torch.pow(torch.abs(qf - gf), p).sum(dim=-1) ** (1 / p)
        return dist_mat.cpu().numpy()
    else:
        qf = np.asarray(qf)
        gf = np.asarray(gf)
        m, n = qf.shape[0], gf.shape[0]
        dist_mat = np.zeros((m, n))
        for i in range(m):
            dist_mat[i] = np.sum(np.abs(qf[i] - gf) ** p, axis=1) ** (1 / p)
        return dist_mat


def mahalanobis_distance(qf, gf, cov_matrix=None):
    qf = np.asarray(qf)
    gf = np.asarray(gf)

    if cov_matrix is None:
        combined = np.vstack([qf, gf])
        cov_matrix = np.cov(combined.T) + np.eye(combined.shape[1]) * 1e-5

    cov_inv = np.linalg.inv(cov_matrix)
    m, n = qf.shape[0], gf.shape[0]
    dist_mat = np.zeros((m, n))

    for i in range(m):
        diff = qf[i] - gf
        dist_mat[i] = np.sqrt(np.sum(diff @ cov_inv * diff, axis=1))

    return dist_mat


def cosine_euclidean_hybrid(qf, gf, alpha=0.5):
    cos_dist = cosine_distance(qf, gf)
    euc_dist = euclidean_distance(qf, gf)

    euc_dist_normalized = euc_dist / (np.max(euc_dist) + 1e-5)
    cos_dist_normalized = cos_dist / 2.0

    hybrid_dist = alpha * cos_dist_normalized + (1 - alpha) * euc_dist_normalized
    return hybrid_dist


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


def compute_distance(qf, gf, metric="euclidean", **kwargs):
    if metric not in __distance_metrics__:
        raise ValueError(f"Unknown distance metric: {metric}. Available metrics: {list(__distance_metrics__.keys())}")

    return __distance_metrics__[metric](qf, gf, **kwargs)