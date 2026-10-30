"""One stereo frame: pose + SAM masks + query pair -> per tip the 2D result (steps 3 and 4)."""
from __future__ import annotations

from . import camera as cam
from .constants import TIPS
from .stereo import choose_ends, pair_2d
from .view import crop_box, finish_view, project_vectors, view_candidates


def detect_tips_2d(query, sam, R, t_left, obj, render, backend="cpu"):
    """query / sam: {"L": image, "R": image}; R, t_left: refined pose (left camera). Returns per tip the 2D result."""
    t = {"L": t_left, "R": t_left - cam.BASELINE}
    states, boxes = {}, {}
    for s in ("L", "R"):
        vec = project_vectors(obj, R, t[s])
        boxes[s] = crop_box(vec)
        alpha = render(R, t[s], boxes[s])                                                                    # 3.2
        states[s] = view_candidates(query[s], alpha, sam[s], vec, obj, backend)                              # 3.1 - 3.7
    chosen = choose_ends(states)                                                                             # 4.1
    res = {s: finish_view(states[s], chosen[s], obj, R, t[s]) for s in ("L", "R")}                           # 3.8 - 3.10
    out = pair_2d(res, query, obj)                                                                           # 4.2 - 4.4
    for tip in TIPS:
        out[tip]["align_at_grid_limit"] = {s: states[s]["align"]["at_grid_limit"] for s in ("L", "R")}
        out[tip]["end_reselected_4_1"] = tip in chosen["L"]
    return out
