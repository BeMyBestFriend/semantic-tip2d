"""Steps 3.1 - 3.10: everything done on one view (L or R) of the stereo pair.

3.1 project the semantic vectors, crop      3.2 alpha render, gradients, Canny, oriented distance maps
3.3 model tip edges                          3.4 alignment (alignment.py)
3.5 normal search, robust lines               3.6 Canny chain walk
3.7 end candidates (view_candidates returns here; 4.1 picks the stereo pair)
3.8 ridge extension, image border             3.9 coating boundary            3.10 upper / lower labels
"""
from __future__ import annotations

import cv2
import numpy as np

from .alignment import align
from .camera import project
from .util import bilinear, bilinear_rep, robust_line, unit
from .constants import BORDER_PX, CANNY, CHAMFER_TRUNC, CROP_MARGIN, GRAD_THR, GRID, H, MIN_FOUND_FRAC, NEIGHBOURS, ORIENT_BINS, RIDGE_GAP, RIDGE_STEP, RIDGE_THR, SAMPLE_STEP, SEARCH_PX, SIDES, TIPS, VEC, W


def project_vectors(obj, R, t):
    sv = obj.axis["semantic_vectors"]
    return {tip: {"start_pixel": project(sv[n]["start"], R, t).tolist(), "end_pixel": project(sv[n]["end"], R, t).tolist()}
            for tip, n in VEC.items()}


def crop_box(vectors):
    pts = np.array([vectors[k][p] for k in TIPS for p in ("start_pixel", "end_pixel")], np.float64)
    x0, y0 = np.maximum(np.floor(pts.min(0)).astype(int) - CROP_MARGIN, 0)
    x1, y1 = np.minimum(np.ceil(pts.max(0)).astype(int) + CROP_MARGIN, [W, H])
    return int(x0), int(y0), int(x1), int(y1)


def image_gradients(bgr):
    gray = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (5, 5), 1.0).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    return gray, gx, gy


def oriented_distance_maps(edges, gx, gy):
    """Truncated distance transform per edge-orientation bin (each bin +-1 neighbour), on the edges' bounding box."""
    ys, xs = np.nonzero(edges)
    if not len(xs):
        return {"maps": [np.full((1, 1), CHAMFER_TRUNC, np.float32)] * ORIENT_BINS, "origin": np.zeros(2)}
    pad = int(np.ceil(CHAMFER_TRUNC)) + 2
    h, w = edges.shape
    x0, y0 = max(int(xs.min()) - pad, 0), max(int(ys.min()) - pad, 0)
    x1, y1 = min(int(xs.max()) + pad + 1, w), min(int(ys.max()) + pad + 1, h)
    e, cgx, cgy = edges[y0:y1, x0:x1], gx[y0:y1, x0:x1], gy[y0:y1, x0:x1]
    orient = (np.degrees(np.arctan2(cgy, cgx)) + 90.0) % 180.0
    bins = np.minimum((orient / (180.0 / ORIENT_BINS)).astype(int), ORIENT_BINS - 1)
    maps = []
    for b in range(ORIENT_BINS):
        sel = e & np.isin(bins, [(b - 1) % ORIENT_BINS, b, (b + 1) % ORIENT_BINS])
        dist = cv2.distanceTransform((~sel).astype(np.uint8), cv2.DIST_L2, 3)
        maps.append(np.minimum(dist, CHAMFER_TRUNC).astype(np.float32))
    return {"maps": maps, "origin": np.array([x0, y0], dtype=np.float64)}


def model_tip_edges(alpha, vectors, obj):
    """Silhouette points of the tip section (SILVER_MM from the end) per (tip, side), robust line, 2 px samples."""
    contours, _ = cv2.findContours((alpha > 127).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    pts = np.concatenate([c[:, 0, :] for c in contours]).astype(np.float64)
    lines = {}
    for tip, other in (("tip1", "tip2"), ("tip2", "tip1")):
        s, e = (np.asarray(vectors[tip][k]) for k in ("start_pixel", "end_pixel"))
        length = np.linalg.norm(e - s)
        u = (e - s) / length
        n = np.array([-u[1], u[0]])
        so, eo = (np.asarray(vectors[other][k]) for k in ("start_pixel", "end_pixel"))
        uo = unit(eo - so)
        along = (pts - s) @ u / length
        across = (pts - s) @ n
        d_other = np.abs((pts - so)[:, 0] * uo[1] - (pts - so)[:, 1] * uo[0])
        silver_start = 1.0 - obj.silver_mm / (obj.axis["semantic_vectors"][VEC[tip]]["length"] * 1000)
        keep = ((along >= silver_start) & (along <= 1.05) & (np.abs(across) <= d_other)
                & (np.abs(across) <= max(8.0, 0.25 * length)))
        for sign in (-1, 1):
            side_pts = pts[keep & (np.sign(across) == sign)]
            if len(side_pts) < 10:
                continue
            fit = robust_line(side_pts)
            d = np.asarray(fit["direction"])
            if d @ u < 0:
                d = -d
            proj = (side_pts - fit["origin"]) @ d
            ts = np.arange(proj.min(), proj.max() + 1e-6, SAMPLE_STEP)
            lines[(tip, sign)] = {"direction": d, "samples": np.asarray(fit["origin"]) + ts[:, None] * d}
    return lines


def normal_search(samples, direction, gx, gy):
    n = np.array([-direction[1], direction[0]])
    offs = np.arange(-SEARCH_PX, SEARCH_PX + 1e-6, 0.5)
    q = samples[:, None, :] + offs[None, :, None] * n
    g = np.abs(bilinear(gx, q.reshape(-1, 2)) * n[0] + bilinear(gy, q.reshape(-1, 2)) * n[1]).reshape(len(samples), -1)
    idx = g.argmax(1)
    peak = g[np.arange(len(samples)), idx]
    ok = (peak >= GRAD_THR) & (idx > 0) & (idx < len(offs) - 1)
    return samples[ok] + offs[idx[ok], None] * n


def trace(canny, seed, u, allowed, max_steps):
    """Least-turn walk along Canny pixels from the pixel nearest the seed."""
    ys, xs = np.nonzero(canny[int(seed[1]) - 3:int(seed[1]) + 4, int(seed[0]) - 3:int(seed[0]) + 4])
    if not len(xs):
        return None
    cand = np.stack([xs + int(seed[0]) - 3, ys + int(seed[1]) - 3], 1)
    p = tuple(cand[np.argmin(np.linalg.norm(cand - seed, axis=1))])
    path, seen, heading = [p], {p}, np.asarray(u, np.float64)
    for _ in range(max_steps):
        best, best_cos = None, -2.0
        for dx, dy in NEIGHBOURS:
            q = (p[0] + dx, p[1] + dy)
            if q in seen or not (0 <= q[1] < canny.shape[0] and 0 <= q[0] < canny.shape[1]) or not canny[q[1], q[0]]:
                continue
            if allowed is not None and not allowed(q, path):
                continue
            c = (np.array([dx, dy]) / np.hypot(dx, dy)) @ heading
            if c > best_cos:
                best, best_cos = q, c
        if best is None:
            break
        seen.add(best)
        path.append(best)
        back = np.asarray(path[max(0, len(path) - 4)], np.float64)
        d = np.asarray(best, np.float64) - back
        heading = unit(d) if np.linalg.norm(d) > 0 else heading
        p = best
    return np.asarray(path, np.float64)


def self_line_corridor(interior_point, fit_px, out_px, in_px):
    """After fit_px points: stay within -out_px .. +in_px (toward the interior) of the line of the first points."""
    state = {}

    def allowed(q, path):
        if len(path) < fit_px:
            return True
        if "line" not in state:
            pts = np.asarray(path[:fit_px], np.float64)
            o = pts.mean(0)
            d = unit(np.linalg.svd(pts - o)[2][0])
            n = np.array([-d[1], d[0]])
            state["line"] = (o, n if (np.asarray(interior_point) - o) @ n > 0 else -n)
        o, n = state["line"]
        s = (np.asarray(q, np.float64) - o) @ n
        return -out_px <= s <= in_px
    return allowed


def is_uturn(path, u, c, k=3, thr=90.0):
    """True if the path turns by > thr deg (after uturn_min_index) and then goes back along u by uturn_back px."""
    ang = np.full(len(path), np.nan)
    for i in range(k, len(path) - k):
        a, b = path[i] - path[i - k], path[i + k] - path[i]
        ang[i] = np.degrees(np.arccos(np.clip(unit(a) @ unit(b), -1, 1)))
    along = path @ u
    m = c["uturn_min_index"]
    for i in np.nonzero(ang[m:] > thr)[0] + m:
        if along[i] - along[i:i + c["uturn_window"]].min() >= c["uturn_back"]:
            return True
    return False


def view_candidates(query, alpha_full, sam_full, vec_full, obj, backend="cpu"):
    c = obj.consts
    pts = np.array([vec_full[k][p] for k in TIPS for p in ("start_pixel", "end_pixel")], np.float64)
    h, w = query.shape[:2]
    x0, y0 = np.maximum(np.floor(pts.min(0)).astype(int) - CROP_MARGIN, 0)
    x1, y1 = np.minimum(np.ceil(pts.max(0)).astype(int) + CROP_MARGIN, [w, h])
    off = np.array([x0, y0], np.float64)
    img, alpha, sam = query[y0:y1, x0:x1], alpha_full[y0:y1, x0:x1], sam_full[y0:y1, x0:x1]
    vectors = {k: {p: (np.asarray(vec_full[k][p]) - off).tolist() for p in ("start_pixel", "end_pixel")} for k in TIPS}
    gray, gx, gy = image_gradients(img)
    canny_walk = cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), cv2.COLOR_RGB2GRAY),
                                            (5, 5), 1.0), 30, 90, L2gradient=True) > 0
    lines = model_tip_edges(alpha, vectors, obj)
    allpts = np.concatenate([v["samples"] for v in lines.values()])
    wx0, wy0 = np.maximum(allpts.min(0).astype(int) - 60, 0)
    wx1, wy1 = allpts.max(0).astype(int) + 60
    window = np.zeros(sam.shape, bool)
    window[wy0:wy1, wx0:wx1] = True
    edges = (cv2.Canny(gray.astype(np.uint8), *CANNY) > 0) & window
    edges &= cv2.dilate(sam.astype(np.uint8), np.ones((31, 31), np.uint8)) > 0
    dist_maps = oriented_distance_maps(edges, gx, gy)
    center = np.mean([vectors[k]["end_pixel"] for k in TIPS], axis=0)
    best = align(lines, dist_maps, center, backend)
    _, theta, scale, offset = best
    cs, sn = np.cos(np.radians(theta)), np.sin(np.radians(theta))
    rot = np.array([[cs, sn], [-sn, cs]])
    sides = {tip: {} for tip in TIPS}
    for (tip, sign), m in lines.items():
        samples = center + scale * (m["samples"] - center) @ rot + offset
        direction = unit(m["direction"] @ rot)
        found = normal_search(samples, direction, gx, gy)
        need = max(5, int(np.ceil(MIN_FOUND_FRAC * len(samples))))
        if len(found) < need:
            continue
        fit = robust_line(found, minimum_points=need)
        d = unit(fit["direction"])
        sides[tip][sign] = {"found": found, "dir": d if d @ direction >= 0 else -d, "origin": np.asarray(fit["origin"])}
    tips = {}
    for tip in TIPS:
        if len(sides[tip]) < 2:
            tips[tip] = {"failure": "fewer than two edges found"}
            continue
        s1, s2 = sides[tip].values()
        u = unit(s1["dir"] + s2["dir"])
        seeds = []
        for s in (s1, s2):
            a = s["found"] @ u
            seeds.append(s["found"][np.argmin(np.abs(a - (a.max() - c["seed_back"])))])
        ends, turns = [], []
        for i, s in enumerate((s1, s2)):
            path = trace(canny_walk, seeds[i], s["dir"],
                         self_line_corridor(seeds[1 - i], c["self_fit"], c["self_out"], c["self_in"]), c["max_steps"])
            if path is None or len(path) <= c["min_chain"]:
                continue
            turns.append(is_uturn(path, u, c))
            ends.append(path[np.argmax(path @ u)])
        if not ends:
            tips[tip] = {"failure": "no chain"}
            continue
        side_ends = list(ends)
        if len(ends) == 2 and np.linalg.norm(ends[0] - ends[1]) > c["side_sep_max"] and turns.count(True) == 1:
            ends = [ends[turns.index(True)]]
        default = np.mean(ends, 0)
        cands = [default]
        if len(side_ends) == 2 and np.linalg.norm(side_ends[0] - side_ends[1]) > c["side_sep_max"]:
            cands += [e for e in side_ends if np.linalg.norm(e - default) > 1e-9]
        tips[tip] = {"s1": s1, "s2": s2, "u": u, "default": default, "candidates": cands}
    at_limit = bool(np.max(np.abs(offset)) >= np.max(np.abs(GRID)) - 1e-9 + 1.0)
    return {"gray": gray, "off": off, "vectors": vectors, "wh": (w, h), "tips": tips,
            "align": {"theta": float(theta), "scale": float(scale), "offset": np.asarray(offset).tolist(), "cost": float(best[0]),
                      "at_grid_limit": at_limit}}


def ridge_extend(gray, start, u, ridge_max, ridge_half):
    """Follow a thin line on a locally uniform background beyond start along u."""
    n = np.array([-u[1], u[0]])
    offs = np.arange(-ridge_half, ridge_half + 1e-6, 0.5)
    flank_o = np.arange(ridge_half + 2, ridge_half + 5.01, 1.0)
    lat, last, miss = 0.0, None, 0
    for t in np.arange(RIDGE_STEP, ridge_max + 1e-6, RIDGE_STEP):
        c = start + t * u + lat * n
        prof = bilinear_rep(gray, c + offs[:, None] * n)
        fa = float(np.median(bilinear_rep(gray, c + flank_o[:, None] * n)))
        fb = float(np.median(bilinear_rep(gray, c - flank_o[:, None] * n)))
        dev = np.minimum(np.abs(prof - fa), np.abs(prof - fb))
        k = int(np.argmax(dev))
        if dev[k] >= RIDGE_THR and abs(fa - fb) < RIDGE_THR:
            last = c + offs[k] * n
            lat += float(np.clip(offs[k], -RIDGE_STEP, RIDGE_STEP))
            miss = 0
        else:
            miss += 1
            if miss * RIDGE_STEP > RIDGE_GAP:
                break
    return np.asarray(start, np.float64) if last is None else last


def coat_boundary(gray, origin, d, inward, end, expected, window, insets, polarity, subpix_clip):
    """Strongest coating step along the edge line within +-window of the expected distance behind the end.

    subpix_clip: limit the parabolic sub-sample offset to +-0.5 px (a peak at the window edge, with the true step
    outside the window, makes it run away)."""
    o0 = origin + ((end - origin) @ d) * d
    s = np.arange(-(expected + window + 4), -(expected - window - 4), 0.5)
    prof = np.mean([cv2.remap(gray, *(np.ascontiguousarray(c, np.float32).reshape(1, -1)
                                      for c in (o0[None] + s[:, None] * d + k * inward).T), cv2.INTER_LINEAR)[0]
                    for k in insets], axis=0)
    grad = polarity * np.gradient(prof, 0.5)
    win = np.abs(s + expected) <= window
    if not win.any():
        return None
    i = int(np.argmax(np.where(win, grad, -np.inf)))
    if grad[i] <= 0:
        return None
    sub = 0.0
    if 0 < i < len(s) - 1:
        a, b, cc = grad[i - 1], grad[i], grad[i + 1]
        den = a - 2 * b + cc
        sub = 0.5 * (a - cc) / den * 0.5 if abs(den) > 1e-9 else 0.0
        if subpix_clip:
            sub = float(np.clip(sub, -0.5, 0.5))
    return o0 + (s[i] + sub) * d


def upper_on(obj, R, t):
    """Side (facing the tip) on which the pose puts the profile's upper coat point: 'left' or 'right'."""
    out = {}
    for tip in TIPS:
        E = np.asarray(obj.profile["tips"][tip]["end"])
        seg = {s: (project(obj.profile["tips"][tip]["coat"][s], R, t), project((np.asarray(obj.profile["tips"][tip]["coat"][s]) + E) / 2, R, t))
               for s in SIDES}
        axis = 0.5 * (seg["upper"][1] + seg["lower"][1]) - 0.5 * (seg["upper"][0] + seg["lower"][0])
        sep = seg["upper"][0] - seg["lower"][0]
        out[tip] = "left" if axis[0] * sep[1] - axis[1] * sep[0] < 0 else "right"
    return out


def label_edges(edges, u, side):
    c0, c1 = (np.asarray(e["centroid"]) for e in edges)
    sep = c0 - c1
    first_left = u[0] * sep[1] - u[1] * sep[0] < 0
    first_upper = first_left == (side == "left")
    return {"upper": edges[0], "lower": edges[1]} if first_upper else {"upper": edges[1], "lower": edges[0]}


def finish_view(state, chosen, obj, R, t):
    c = obj.consts
    gray, off, vectors, (w, h) = state["gray"], state["off"], state["vectors"], state["wh"]
    sides = upper_on(obj, R, t)
    out = {}
    for tip in TIPS:
        st = state["tips"][tip]
        if "failure" in st:
            out[tip] = {"failure": st["failure"]}
            continue
        s1, s2, u = st["s1"], st["s2"], st["u"]
        end = np.asarray(chosen.get(tip, st["default"]))
        ridge = ridge_extend(gray, end, u, c["ridge_max"], c["ridge_half"])                         # 3.8
        expected = np.linalg.norm(np.subtract(vectors[tip]["end_pixel"], vectors[tip]["start_pixel"]))
        end_full = end + off
        border = bool(min(end_full[0], end_full[1], w - 1 - end_full[0], h - 1 - end_full[1]) < BORDER_PX)
        anchor = end if not border else np.asarray(vectors[tip]["start_pixel"]) + expected * u
        edges = []
        for s, other in zip((s1, s2), (s2, s1)):                                                     # 3.9
            n = np.array([-s["dir"][1], s["dir"][0]])
            inward = n if (other["origin"] - s["origin"]) @ n > 0 else -n
            rec = {"origin": (s["origin"] + off).tolist(), "dir": s["dir"].tolist(), "centroid": (s["found"].mean(0) + off).tolist()}
            coat = coat_boundary(gray, s["origin"], s["dir"], inward, anchor, expected, c["coat_window"], c["coat_inset"],
                                 c["coat_polarity"], c["coat_subpix_clip"])
            if coat is not None:
                rec["coat"] = (coat + off).tolist()
            edges.append(rec)
        out[tip] = {"walk": (end + off).tolist(), "ridge": (ridge + off).tolist(), "u": u.tolist(), "end_at_border": border,
                    "labelled": label_edges(edges, u, sides[tip])}                                     # 3.10
    return out
