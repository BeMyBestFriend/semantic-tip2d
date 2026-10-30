"""Rectified stereo camera: left intrinsics K and the R camera centre in the L frame (BASELINE).

The pair is rectified, so the R camera has the same orientation and is shifted by BASELINE: a scene point has the same
row in L and R, and the object pose in R is (R, t - BASELINE)."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

DEFAULT = Path(__file__).resolve().parents[1] / "configs/camera/zedm_query_0921_hd1080.json"
K = KINV = BASELINE = None


def load(path=DEFAULT):
    """Set the camera from a calibration json: left_rect {fx, fy, cx, cy}, t_meters (R camera centre in the L frame)."""
    global K, KINV, BASELINE
    c = json.loads(Path(path).read_text())
    left = c["left_rect"]
    K = np.array([[left["fx"], 0, left["cx"]], [0, left["fy"], left["cy"]], [0, 0, 1.0]])
    KINV = np.linalg.inv(K)
    BASELINE = np.asarray(c["t_meters"], np.float64)


load()


def project(X, R, t):
    x = K @ (R @ np.asarray(X, np.float64) + t)
    return x[:2] / x[2]


def tri(uv):
    """Triangulate one L/R pixel pair {"L": uv, "R": uv} -> 3D point in the L frame (m)."""
    P_L = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P_R = K @ np.hstack([np.eye(3), -BASELINE.reshape(3, 1)])
    X = cv2.triangulatePoints(P_L, P_R, np.float64(uv["L"]).reshape(2, 1), np.float64(uv["R"]).reshape(2, 1))
    return (X[:3] / X[3]).ravel()
