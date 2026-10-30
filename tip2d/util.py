"""Small numeric helpers shared by the steps."""
from __future__ import annotations

import cv2
import numpy as np


def unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def bilinear(img, pts, border=0.0):
    """Bilinear samples with a constant border (remap in chunks: it needs < 32767 rows)."""
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    n = len(pts)
    pts = np.pad(pts, ((0, -n % 1024), (0, 0))).reshape(-1, 1024, 2)
    return cv2.remap(img, pts[..., 0], pts[..., 1], cv2.INTER_LINEAR, borderValue=float(border)).ravel()[:n]


def bilinear_rep(img, pts):
    """Bilinear samples with a replicated border."""
    pts = np.asarray(pts, np.float32).reshape(1, -1, 2)
    return cv2.remap(img, pts[..., 0], pts[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)[0]


def robust_line(points, minimum_points=10, mad_scale=2.5, minimum_threshold_px=1.25, iterations=5):
    """Iteratively fit a Huber line and reject large orthogonal residuals."""
    original = np.asarray(points, dtype=np.float64)
    retained = original.copy()
    if len(retained) < minimum_points:
        raise ValueError(f"Line has only {len(retained)} points")
    for _ in range(int(iterations)):
        vx, vy, x0, y0 = cv2.fitLine(retained.astype(np.float32).reshape(-1, 1, 2), cv2.DIST_HUBER, 0, 0.01, 0.01).reshape(-1)
        direction = np.asarray([vx, vy], dtype=np.float64)
        direction /= np.linalg.norm(direction)
        origin = np.asarray([x0, y0], dtype=np.float64)
        residual = np.abs((retained - origin)[:, 0] * direction[1] - (retained - origin)[:, 1] * direction[0])
        median = float(np.median(residual))
        mad = float(np.median(np.abs(residual - median)))
        threshold = max(float(minimum_threshold_px), median + float(mad_scale) * 1.4826 * mad)
        keep = residual <= threshold
        if bool(np.all(keep)) or int(keep.sum()) < int(minimum_points):
            break
        retained = retained[keep]
    vx, vy, x0, y0 = cv2.fitLine(retained.astype(np.float32).reshape(-1, 1, 2), cv2.DIST_HUBER, 0, 0.01, 0.01).reshape(-1)
    direction = np.asarray([vx, vy], dtype=np.float64)
    direction /= np.linalg.norm(direction)
    return {"origin": np.asarray([x0, y0], dtype=np.float64), "direction": direction, "inlier_points": retained}
