"""Step 3.4: one similarity transform (rotation, scale, translation) of the four model tip edges onto the image
edges, by oriented chamfer matching: FDCM coarse search (directional integral images), then a per-sample fine
search. backend="gpu" runs the same computation in align_gpu.py."""
from __future__ import annotations

import cv2
import numpy as np

from .util import bilinear, unit
from .constants import CHAMFER_TRUNC, FINE, GRID, ORIENT_BINS, SCALES, THETAS


def directional_integral(dist_map, v, trunc):
    """Integral image of dist_map along unit direction v (FDCM)."""
    h, w = dist_map.shape
    n = np.array([-v[1], v[0]])
    corners = np.array([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]], np.float64)
    xs, ys = corners @ v, corners @ n
    xmin, ymin = xs.min() - 1, ys.min() - 1
    Wd, Hd = int(np.ceil(xs.max() - xmin)) + 2, int(np.ceil(ys.max() - ymin)) + 2
    o = xmin * v + ymin * n
    M = np.array([[v[0], n[0], o[0]], [v[1], n[1], o[1]]], np.float64)
    rot = cv2.warpAffine(dist_map, M, (Wd, Hd), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=float(trunc))
    I = np.zeros((Hd, Wd + 1), np.float32)
    np.cumsum(rot, axis=1, out=I[:, 1:])
    return I, (v, n, xmin, ymin)


def coarse_fdcm(lines, dist_maps, center, thetas, scales, grid):
    segs = []
    for k in lines:
        s, d = lines[k]["samples"], lines[k]["direction"]
        segs.append((s[0], s[-1], len(s), np.degrees(np.arctan2(d[1], d[0])) % 180, np.asarray(d)))
    total_n = sum(s[2] for s in segs)
    best = (np.inf, 0.0, 1.0, np.zeros(2))
    origin = dist_maps["origin"]
    for theta in thetas:
        c, s_ = np.cos(np.radians(theta)), np.sin(np.radians(theta))
        R = np.array([[c, s_], [-s_, c]])
        tables = []
        for p0, p1, n_s, ang, d in segs:
            b = min(int(((ang + theta) % 180) / (180.0 / ORIENT_BINS)), ORIENT_BINS - 1)
            tables.append(directional_integral(dist_maps["maps"][b], unit(d @ R), CHAMFER_TRUNC))
        costs = np.zeros((len(scales), len(grid)))
        for (p0, p1, n_s, ang, d), (I, (v, nn, xmin, ymin)) in zip(segs, tables):
            a, y, L = [], [], []
            for scale in scales:
                q0 = center + scale * (p0 - center) @ R - origin
                q1 = center + scale * (p1 - center) @ R - origin
                L.append(float((q1 - q0) @ v))
                a.append((q0[None] + grid) @ v - xmin)
                y.append((q0[None] + grid) @ nn - ymin)
            a, y, L = np.stack(a), np.stack(y), np.asarray(L)[:, None]
            vals = bilinear(I, np.stack([np.r_[a.ravel(), (a + L + 1.0).ravel()], np.r_[y.ravel(), y.ravel()]], 1), 0.0)
            lo, hi = vals[:a.size].reshape(a.shape), vals[a.size:].reshape(a.shape)
            inside = (a >= 0) & (a + L + 1 < I.shape[1] - 1) & (y >= 0) & (y < I.shape[0] - 1)
            costs += n_s * np.where(inside, (hi - lo) / (L + 1.0), CHAMFER_TRUNC)
        costs /= total_n
        for scale, cost in zip(scales, costs):
            i = int(np.argmin(cost))
            if cost[i] < best[0]:
                best = (cost[i], theta, scale, grid[i])
    return best


def cost_fn(lines, dist_maps, center):
    """Mean truncated oriented distance of all model samples under (theta, scale, offset)."""
    keys = list(lines)
    pts = np.concatenate([lines[k]["samples"] for k in keys])
    ang = np.concatenate([np.full(len(lines[k]["samples"]),
                                  np.degrees(np.arctan2(lines[k]["direction"][1], lines[k]["direction"][0])) % 180) for k in keys])
    rel = pts - center

    def cost(theta_deg, scale, offsets):
        c, s_ = np.cos(np.radians(theta_deg)), np.sin(np.radians(theta_deg))
        moved = center + scale * rel @ np.array([[c, s_], [-s_, c]])
        bins = np.minimum((((ang + theta_deg) % 180) / (180.0 / ORIENT_BINS)).astype(int), ORIENT_BINS - 1)
        total = np.zeros(len(offsets))
        for b in np.unique(bins):
            sel = moved[bins == b]
            q = sel[None, :, :] + offsets[:, None, :] - dist_maps["origin"]
            total += bilinear(dist_maps["maps"][b], q.reshape(-1, 2), CHAMFER_TRUNC).reshape(len(offsets), -1).sum(1)
        return total / len(moved)
    return cost


def fine_search(cost, best):
    _, t0, s0, o0 = best
    for theta in np.arange(t0 - 1, t0 + 1.01, 0.25):
        for scale in (s0 - 0.02, s0, s0 + 0.02):
            c = cost(theta, scale, o0 + FINE)
            i = int(np.argmin(c))
            if c[i] < best[0]:
                best = (c[i], theta, scale, o0 + FINE[i])
    return best


def align(lines, dist_maps, center, backend="cpu"):
    """(cost, theta, scale, offset) of the best similarity transform about center."""
    if backend == "gpu":      # same computation on the GPU (align_gpu.py, torch), identical result
        from . import align_gpu as G
        best = G.coarse_fdcm(lines, dist_maps, center, THETAS, SCALES, GRID, "cv")
        cost = G.original_cost_fn(lines, dist_maps, center, "cv")
        return G.fine_search(cost, (cost(best[1], best[2], best[3][None])[0],) + tuple(best[1:]))
    best = coarse_fdcm(lines, dist_maps, center, THETAS, SCALES, GRID)
    cost = cost_fn(lines, dist_maps, center)
    return fine_search(cost, (cost(best[1], best[2], best[3][None])[0],) + tuple(best[1:]))
