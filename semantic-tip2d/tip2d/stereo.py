"""Steps 4.1 - 4.4: choices made with both views (rectified pair: a true L/R correspondence shares the row).

4.1 choose_ends      chain-end candidates -> the L/R pair on the same row
4.2 select_pair      walk end or ridge end per view
4.3 row_check_coats  coat points must share the row; outward snap otherwise
4.4 select_by_length near-horizontal tips: the pair closest to the tip length (triangulates, selection only)
pair_2d runs 4.2 - 4.4 and builds the 2D result handed to step 5 (3D, disabled: lift3d.py)."""
from __future__ import annotations

import itertools

import cv2
import numpy as np

from . import camera as cam
from .camera import tri
from .util import unit
from .constants import BORDER_PX, FAR_ROW_MAX, GRAD_THR, MIN_UY, ROW_FIX_MARGIN, ROW_GATE, ROW_MIN_DEG, SIDES, SNAP_RANGE, SNAP_REL, STEREO_MARGIN, TIPS


def at_border(p, wh):
    return bool(min(p[0], p[1], wh[0] - 1 - p[0], wh[1] - 1 - p[1]) < BORDER_PX)


def choose_ends(states):
    """Per tip, the L/R candidate pair with the smallest row difference (if it beats the default pair)."""
    chosen = {"L": {}, "R": {}}
    for tip in TIPS:
        tl, tr = states["L"]["tips"][tip], states["R"]["tips"][tip]
        if "failure" in tl or "failure" in tr or max(abs(tl["u"][1]), abs(tr["u"][1])) < MIN_UY:
            continue
        oL, oR = states["L"]["off"], states["R"]["off"]
        if at_border(tl["default"] + oL, states["L"]["wh"]) or at_border(tr["default"] + oR, states["R"]["wh"]):
            continue
        base = abs((tl["default"] + oL)[1] - (tr["default"] + oR)[1])
        best, best_d = None, base
        for cl, cr in itertools.product(tl["candidates"], tr["candidates"]):
            d = abs((cl + oL)[1] - (cr + oR)[1])
            if d <= base - STEREO_MARGIN and d < best_d and d <= ROW_GATE:
                best, best_d = (cl, cr), d
        if best is not None:
            chosen["L"][tip], chosen["R"][tip] = best
    return chosen


def select_pair(tl, tr):
    """Walk or ridge end per view: the L/R pair on the same row (switch only if it cuts the row diff by >= 1 px)."""
    wl, rl, wr, rr = (np.asarray(tl["walk"]), np.asarray(tl["ridge"]), np.asarray(tr["walk"]), np.asarray(tr["ridge"]))
    cand_l = [("w", wl)] + ([("r", rl)] if abs(tl["u"][1]) >= MIN_UY and np.linalg.norm(rl - wl) > 0.5 else [])
    cand_r = [("w", wr)] + ([("r", rr)] if abs(tr["u"][1]) >= MIN_UY and np.linalg.norm(rr - wr) > 0.5 else [])
    base = abs(wl[1] - wr[1])
    best, best_d = (wl, wr), base
    for kl, pl in cand_l:
        for kr, pr in cand_r:
            if kl == kr == "w":
                continue
            d = abs(pl[1] - pr[1])
            if d <= base - STEREO_MARGIN and d < best_d:
                best, best_d = (pl, pr), d
    return best


def outward_dir(lab, side):
    e, o = lab[side], lab["lower" if side == "upper" else "upper"]
    d = np.asarray(e["dir"])
    n = np.array([-d[1], d[0]])
    return d, (-n if (np.asarray(o["origin"]) - np.asarray(e["origin"])) @ n > 0 else n)


def snap_outward(gray, c, d, outward, rel):
    """Outermost strong gray-level change across the edge, 1-5 px tipward of the coat point."""
    s = np.arange(SNAP_RANGE[0], SNAP_RANGE[1] + 1e-6, 0.25)
    pts = np.array([[c + a * d + t * outward for t in s] for a in (1, 2, 3, 4, 5)], np.float32)
    prof = cv2.remap(gray, pts[..., 0].reshape(1, -1), pts[..., 1].reshape(1, -1), cv2.INTER_LINEAR)[0].reshape(pts.shape[:2]).mean(0)
    g = np.abs(np.gradient(prof, 0.25))
    thr = max(GRAD_THR, rel * g.max())
    peaks = [i for i in range(1, len(s) - 1) if g[i] >= thr and g[i] >= g[i - 1] and g[i] >= g[i + 1]]
    return c if not peaks else c + s[max(peaks)] * outward


def row_check_coats(labs, grays):
    """L/R row check of the coat points; an outward-snapped pair replaces them if it cuts the row diff by >= 1 px."""
    for side in SIDES:
        if any("coat" not in labs[c][side] for c in ("L", "R")):
            continue
        orig = {c: np.asarray(labs[c][side]["coat"]) for c in ("L", "R")}
        dy0 = abs(orig["L"][1] - orig["R"][1])
        if dy0 <= FAR_ROW_MAX:
            continue
        cand = {}
        for c in ("L", "R"):
            d, outward = outward_dir(labs[c], side)
            cand[c] = [("orig", orig[c])] + [("snap", snap_outward(grays[c], orig[c], d, outward, rel)) for rel in SNAP_REL]
        best, best_dy = None, dy0
        for kl, pl in cand["L"]:
            for kr, pr in cand["R"]:
                dy = abs(pl[1] - pr[1])
                if (kl, kr) != ("orig", "orig") and dy <= dy0 - ROW_FIX_MARGIN and dy < best_dy - 1e-9:
                    best, best_dy = ((kl, pl), (kr, pr)), dy
        if best is not None:
            for c, (k, pt) in zip(("L", "R"), best):
                if k != "orig":
                    labs[c][side]["coat"] = pt.tolist()


def angle_to_horizontal(d):
    a = abs(np.degrees(np.arctan2(d[1], d[0]))) % 180
    return min(a, 180 - a)


def coat_mid_3d(labs):
    """Mean of the two coat points on the L rays at the median coat depth (= step 5's F_mid)."""
    zs = []
    for side in SIDES:
        a, b = labs["L"][side], labs["R"][side]
        ca, cb = a.get("coat"), b.get("coat")
        if ca is None:
            continue
        if cb is not None and abs(ca[1] - cb[1]) <= FAR_ROW_MAX:
            zs.append(tri({"L": ca, "R": cb})[2])
        if angle_to_horizontal(a.get("coat_end_dir", a["dir"])) >= ROW_MIN_DEG:
            o, d = np.asarray(b.get("coat", b["origin"])), np.asarray(b.get("coat_end_dir", b["dir"]))
            zs.append(tri({"L": ca, "R": o + (ca[1] - o[1]) / d[1] * d})[2])
    if not zs:
        return None
    z = float(np.median(zs))
    F = [cam.KINV @ np.append(labs["L"][s]["coat"], 1.0) * (z / (cam.KINV @ np.append(labs["L"][s]["coat"], 1.0))[2])
         for s in SIDES if "coat" in labs["L"][s]]
    return np.mean(F, axis=0)


def select_by_length(tl, tr, labs, length):
    """Near-horizontal tips: the walk/ridge pair whose triangulated distance to the coat region is closest to the tip length."""
    F_mid = coat_mid_3d(labs)
    if F_mid is None:
        return None
    pairs = [(tl[kl], tr[kr]) for kl in ("walk", "ridge") for kr in ("walk", "ridge")]
    return min(pairs, key=lambda p: abs(np.linalg.norm(tri({"L": p[0], "R": p[1]}) - F_mid) - length))


def pair_2d(res, query, obj):
    """Steps 4.2 - 4.4 and the 2D result handed to step 5, per tip."""
    grays = {c: cv2.GaussianBlur(cv2.cvtColor(query[c], cv2.COLOR_BGR2GRAY), (5, 5), 1.0).astype(np.float32) for c in ("L", "R")}
    out = {}
    for tip in TIPS:
        tl, tr = res["L"][tip], res["R"][tip]
        if "walk" not in tl or "walk" not in tr:
            out[tip] = {"failure": tl.get("failure") or tr.get("failure")}
            continue
        if tl["end_at_border"] or tr["end_at_border"]:
            chosen = [None if t["end_at_border"] else np.asarray(t["walk"]) for t in (tl, tr)]
        else:
            chosen = list(select_pair(tl, tr))                                                       # 4.2
        labs = {"L": tl["labelled"], "R": tr["labelled"]}
        row_check_coats(labs, grays)                                                                 # 4.3
        horizontal = (chosen[0] is not None and chosen[1] is not None and max(abs(tl["u"][1]), abs(tr["u"][1])) < MIN_UY)
        if horizontal:                                                                               # 4.4
            sel = select_by_length(tl, tr, labs, obj.lengths[tip])
            chosen = list(sel) if sel is not None else chosen
        ends = {c: None if e is None else np.asarray(e).tolist() for c, e in zip(("L", "R"), chosen)}
        edges = {}
        for c in ("L", "R"):
            edges[c] = {}
            for side in SIDES:
                e = labs[c][side]
                r = {"line_origin_px": e["origin"], "line_dir": e["dir"], "coat_px": e.get("coat")}
                if ends[c] is not None and "coat" in e:
                    r["coat_end_dir"] = unit(np.subtract(ends[c], e["coat"])).tolist()
                edges[c][side] = r
        out[tip] = {"end_px": ends, "edges": edges, "end_at_border": {"L": tl["end_at_border"], "R": tr["end_at_border"]},
                    "horizontal": bool(horizontal),
                    "end_row_diff_px": None if any(e is None for e in chosen) else float(abs(chosen[0][1] - chosen[1][1]))}
    return out
