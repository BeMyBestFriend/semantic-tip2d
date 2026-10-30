"""Step 3.4 on the GPU (torch): the same computation as alignment.py, one batch per search.

  coarse_fdcm       per (theta, segment) directional integral image (warp of the bin's distance map onto the
                    segment-aligned grid, cumsum along it), two lookups per segment and hypothesis
  original_cost_fn  per-sample cost (bilinear samples of every model-edge sample in its orientation bin's map)
  fine_search       the 9 thetas x 3 scales x 81 offsets of the CPU loop as one batch; same first-occurrence argmin
The distance maps stay on the CPU (cv2.distanceTransform) and are uploaded. OpenCV's remap / warpAffine interpolate
in fixed point (1/32 px); mode "cv" reproduces it, so the result equals the CPU one.
"""
from __future__ import annotations

import numpy as np
import torch

from .constants import CHAMFER_TRUNC, FINE, ORIENT_BINS

DEV = torch.device("cuda")
NB_DX = torch.tensor([0, 1, 0, 1], device=DEV)
NB_DY = torch.tensor([0, 0, 1, 1], device=DEV)
_maps_cache = {"key": None}
_consts = {}
COMPILE = True          # torch.compile (Triton) the GPU parts; False = eager torch, same results
_fns = {"coarse": "_coarse_costs", "cost": "_cost_values", "cost_min": "_cost_min"}
_compiled = {}


def _run(name):
    fn = globals()[_fns[name]]
    if not COMPILE:
        return fn
    if name not in _compiled:
        _compiled[name] = torch.compile(fn, dynamic=True)
    return _compiled[name]


def dev_const(arr):
    """GPU copy of a constant numpy array (e.g. the coarse offset grid), uploaded once."""
    key = id(arr)
    if key not in _consts:
        _consts[key] = (arr, torch.from_numpy(np.asarray(arr, np.float64)).to(DEV))
    return _consts[key][1]


def upload_maps(dist_maps):
    """[ORIENT_BINS, h, w] float32 on the GPU; cached for the coarse -> fine calls on the same maps."""
    key = id(dist_maps)
    if _maps_cache["key"] != key:
        _maps_cache.update(key=key, maps=torch.from_numpy(np.stack(dist_maps["maps"])).to(DEV))
    return _maps_cache["maps"]


def bilinear(imgs, idx, x, y, border, mode, valid_wh=None):
    """Bilinear samples of imgs[idx] at (x, y) (float32 tensors of equal shape), constant border.

    valid_wh: optional per-sample (w, h) of the real image inside the padded stack (default: stack size).
    mode "cv": coordinates rounded to 1/32 px like cv2.remap (float32 x*32, round half to even).
    """
    _, H, W = imgs.shape
    if mode == "cv":
        x, y = torch.round(x * 32.0) / 32.0, torch.round(y * 32.0) / 32.0
    x0, y0 = torch.floor(x), torch.floor(y)
    fx, fy = (x - x0)[..., None], (y - y0)[..., None]
    xx = x0.long()[..., None] + NB_DX                       # the 4 neighbours on a last axis
    yy = y0.long()[..., None] + NB_DY
    w_lim, h_lim = (W, H) if valid_wh is None else (valid_wh[0][..., None], valid_wh[1][..., None])
    ok = (xx >= 0) & (xx < w_lim) & (yy >= 0) & (yy < h_lim)
    v = imgs.reshape(-1)[(idx[..., None] * H + yy.clamp(0, H - 1)) * W + xx.clamp(0, W - 1)]
    wx = torch.where(NB_DX.bool(), fx, 1 - fx)
    wy = torch.where(NB_DY.bool(), fy, 1 - fy)
    return border + (wx * wy * torch.where(ok, v - border, 0.0)).sum(-1)


def _segments(lines):
    segs = []
    for k in lines:
        s = lines[k]["samples"]
        d = lines[k]["direction"]
        segs.append((s[0], s[-1], len(s), np.degrees(np.arctan2(d[1], d[0])) % 180, np.asarray(d)))
    return segs


def coarse_fdcm(lines, dist_maps, center, thetas, scales, grid, mode="cv", return_table=False):
    """Same result as alignment.coarse_fdcm: (cost, theta, scale, offset); return_table: the [T, S, G] costs."""
    maps = upload_maps(dist_maps)
    h, w = maps.shape[1:]
    segs = _segments(lines)
    total_n = sum(s[2] for s in segs)
    origin = dist_maps["origin"]
    T, S, Sg = len(thetas), len(scales), len(segs)

    # per (theta, segment) table geometry as alignment.directional_integral, vectorised (float64, bitwise equal)
    th = np.radians(np.asarray(thetas, np.float64))
    c, s_ = np.cos(th)[:, None], np.sin(th)[:, None]                      # [T, 1]
    d = np.stack([g[4] for g in segs])                                    # [Sg, 2]
    vx, vy = d[:, 0] * c - d[:, 1] * s_, d[:, 0] * s_ + d[:, 1] * c       # d @ R, [T, Sg]
    nv = np.sqrt(vx ** 2 + vy ** 2)
    vx, vy = vx / nv, vy / nv
    nx, ny = -vy, vx
    cx, cy = np.array([0, w - 1, 0, w - 1], np.float64), np.array([0, 0, h - 1, h - 1], np.float64)
    xs = cx * vx[..., None] + cy * vy[..., None]                          # corners @ v, [T, Sg, 4]
    ys = cx * nx[..., None] + cy * ny[..., None]
    xmin, ymin = xs.min(-1) - 1, ys.min(-1) - 1
    Wt = (np.ceil(xs.max(-1) - xmin) + 2).astype(np.int64)
    Ht = (np.ceil(ys.max(-1) - ymin) + 2).astype(np.int64)
    ox, oy = xmin * vx + ymin * nx, xmin * vy + ymin * ny
    ang = np.array([g[3] for g in segs])
    bins = np.minimum((((ang + np.asarray(thetas)[:, None]) % 180) / (180.0 / ORIENT_BINS)).astype(int),
                      ORIENT_BINS - 1)                                  # [T, Sg]
    # segment ends per scale: center + scale * (p - center) @ R - origin, [T, Sg, S, 2]
    sc = np.asarray(scales, np.float64)[:, None]
    q = []
    for p in (np.stack([g[0] for g in segs]), np.stack([g[1] for g in segs])):
        rx, ry = sc * (p[:, 0] - center[0]), sc * (p[:, 1] - center[1])   # [S, Sg]
        qx = center[0] + (rx[None] * c[..., None] + ry[None] * (-s_[..., None])) - origin[0]
        qy = center[1] + (rx[None] * s_[..., None] + ry[None] * c[..., None]) - origin[1]
        q.append(np.stack([qx, qy], -1).transpose(0, 2, 1, 3))            # [T, Sg, S, 2]
    q0, q1 = q
    L = (q1[..., 0] - q0[..., 0]) * vx[..., None] + (q1[..., 1] - q0[..., 1]) * vy[..., None]   # [T, Sg, S]
    K = T * Sg
    Wm, Hm = int(Wt.max()), int(Ht.max())
    geo = np.concatenate([np.stack([vx, vy, nx, ny, ox, oy, xmin, ymin, Wt, Ht, bins], -1).reshape(K, 11),
                          q0.reshape(K, S * 2), L.reshape(K, S),
                          np.tile([g[2] for g in segs], T)[:, None]], 1)
    flat, best = _run("coarse")(maps, torch.from_numpy(geo).to(DEV), dev_const(grid), Wm, Hm, S, Sg, total_n, mode)
    if return_table:
        return flat.reshape(T, S, len(grid)).cpu().numpy()
    val, i = best.cpu().numpy()
    it, rest = divmod(int(i), S * len(grid))
    isc, ig = divmod(rest, len(grid))
    return (float(val), thetas[it], scales[isc], grid[ig])


def _coarse_costs(maps, geo, grid, Wm: int, Hm: int, S: int, Sg: int, total_n: int, mode: str):
    """GPU part of coarse_fdcm: tables, integral images, lookups -> (flat costs [T*S*G], [min, argmin])."""
    K = geo.shape[0]
    vx_, vy_, nx_, ny_, ox_, oy_, xmin_, ymin_ = (geo[:, i] for i in range(8))
    Wt_, Ht_, b_ = geo[:, 8].long(), geo[:, 9].long(), geo[:, 10].long()
    q0_ = geo[:, 11:11 + 2 * S].reshape(K, S, 2)
    L_ = geo[:, 11 + 2 * S:11 + 3 * S]
    n_s = geo[:, 11 + 3 * S]

    # warpAffine (WARP_INVERSE_MAP): dst (x', y') -> src o + x' v + y' n, all tables in one padded batch
    xs_ = torch.arange(Wm, dtype=torch.float64, device=geo.device)
    ys_ = torch.arange(Hm, dtype=torch.float64, device=geo.device)
    if mode == "cv":
        # OpenCV: (round(M0 x 1024) + round((M1 y + M2) 1024) + 16) >> 5, i.e. 1/32 px fixed point
        def fixed(m0, m1, m2):
            a = torch.round(m0[:, None] * xs_ * 1024).int()
            b = torch.round((m1[:, None] * ys_ + m2[:, None]) * 1024).int()
            return ((b[:, :, None] + a[:, None, :] + 16) >> 5).float() / 32.0
        sx, sy = fixed(vx_, nx_, ox_), fixed(vy_, ny_, oy_)
    else:
        sx = (vx_[:, None, None] * xs_ + nx_[:, None, None] * ys_[:, None] + ox_[:, None, None]).float()
        sy = (vy_[:, None, None] * xs_ + ny_[:, None, None] * ys_[:, None] + oy_[:, None, None]).float()
    # in "cv" mode the coordinates are already on the 1/32 grid, so plain bilinear on them = cv2
    rot = bilinear(maps, b_[:, None, None].expand(K, Hm, Wm), sx, sy, float(CHAMFER_TRUNC), "exact")
    I = torch.nn.functional.pad(torch.cumsum(rot, dim=2), (1, 0))       # [K, Hm, Wm + 1]

    # lookups: per table, scale, grid offset -> segment start a, row y
    px, py = q0_[..., 0, None] + grid[:, 0], q0_[..., 1, None] + grid[:, 1]   # [K, S, G]
    a = px * vx_[:, None, None] + py * vy_[:, None, None] - xmin_[:, None, None]
    y = px * nx_[:, None, None] + py * ny_[:, None, None] - ymin_[:, None, None]
    Lk = L_[:, :, None]
    Wk, Hk = Wt_[:, None, None], Ht_[:, None, None]
    shape = (2,) + a.shape
    kidx = torch.arange(K, device=geo.device)[:, None, None].expand(shape)
    vals = bilinear(I, kidx, torch.stack([a, a + Lk + 1.0]).float(), torch.stack([y, y]).float(),   # util.bilinear: float32
                    0.0, mode, valid_wh=((Wk + 1).expand(shape), Hk.expand(shape)))
    inside = (a >= 0) & (a + Lk + 1 < Wk) & (y >= 0) & (y < Hk - 1)
    seg_cost = torch.where(inside, (vals[1] - vals[0]).double() / (Lk + 1.0), float(CHAMFER_TRUNC))
    costs = (seg_cost * n_s[:, None, None]).reshape(K // Sg, Sg, S, -1).sum(1) / total_n   # [T, S, G]
    flat = costs.reshape(-1)
    i = torch.argmin(flat)
    return flat, torch.stack([flat[i], i.double()])


def _cost_values(maps, const, hyp, co, mode: str):
    """cost_fn's cost for hypotheses hyp [J, 4]; const [P, 3] = (rel x, rel y, angle); co = (center, origin)."""
    th, sc = hyp[:, 0, None], hyp[:, 1, None]
    r = th * (np.pi / 180.0)
    c, s_ = torch.cos(r), torch.sin(r)
    rx, ry, ang = sc * const[:, 0], sc * const[:, 1], const[:, 2]     # [J, P]
    # (center + (scale rel) @ [[c, s], [-s, c]]) + offset - origin, in the CPU's order of operations
    qx = co[0] + (rx * c + ry * (-s_)) + hyp[:, 2, None] - co[2]
    qy = co[1] + (rx * s_ + ry * c) + hyp[:, 3, None] - co[3]
    bins = torch.clamp(((ang + th) % 180 / (180.0 / ORIENT_BINS)).long(), max=ORIENT_BINS - 1)
    vals = bilinear(maps, bins, qx.float(), qy.float(), float(CHAMFER_TRUNC), mode)
    return vals.double().sum(1) / const.shape[0]


def _cost_min(maps, const, hyp, co, mode: str):
    c = _cost_values(maps, const, hyp, co, mode)
    i = torch.argmin(c)
    return torch.stack([c[i], i.double()])


class GpuCost:
    """alignment.cost_fn on the GPU; batch() evaluates many (theta, scale, offset)."""

    def __init__(self, lines, dist_maps, center, mode="cv"):
        self.maps = upload_maps(dist_maps)
        self.mode = mode
        keys = list(lines)
        pts = np.concatenate([lines[k]["samples"] for k in keys])
        ang = np.concatenate([np.full(len(lines[k]["samples"]),
                                      np.degrees(np.arctan2(lines[k]["direction"][1], lines[k]["direction"][0])) % 180)
                              for k in keys])
        rel = pts - center
        self.const = torch.from_numpy(np.concatenate([rel, ang[:, None]], 1)).to(DEV)    # [P, 3]
        self.co = torch.from_numpy(np.r_[center, dist_maps["origin"]].astype(np.float64)).to(DEV)

    def batch(self, hyp):
        """hyp: [J, 4] (theta deg, scale, offset x, offset y) -> costs [J] (float64 tensor)."""
        return _run("cost")(self.maps, self.const, torch.as_tensor(hyp, dtype=torch.float64, device=DEV),
                            self.co, self.mode)

    def best(self, hyp):
        """(min cost, first index of it) over the hypotheses, one GPU -> CPU copy."""
        val, i = _run("cost_min")(self.maps, self.const, torch.as_tensor(hyp, dtype=torch.float64, device=DEV),
                                  self.co, self.mode).cpu().numpy()
        return float(val), int(i)

    def __call__(self, theta_deg, scale, offsets):
        offsets = np.asarray(offsets, np.float64).reshape(-1, 2)
        J = len(offsets)
        return self.batch(np.column_stack([np.full(J, theta_deg), np.full(J, scale), offsets])).cpu().numpy()


def fine_search(cost, best):
    """alignment.fine_search as one batch (same hypothesis order, first-occurrence argmin, strict < vs best)."""
    _, t0, s0, o0 = best
    thetas, scales = np.arange(t0 - 1, t0 + 1.01, 0.25), (s0 - 0.02, s0, s0 + 0.02)
    nf = len(FINE)
    hyp = np.column_stack([np.repeat(thetas, len(scales) * nf),
                           np.tile(np.repeat(np.asarray(scales), nf), len(thetas)),
                           np.tile(o0 + FINE, (len(thetas) * len(scales), 1))])
    ci, i = cost.best(hyp)
    if ci < best[0]:
        return (ci, hyp[i, 0], hyp[i, 1], hyp[i, 2:])
    return best


def original_cost_fn(lines, dist_maps, center, mode="cv"):
    return GpuCost(lines, dist_maps, center, mode)
