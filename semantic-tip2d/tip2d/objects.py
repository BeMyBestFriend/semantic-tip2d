"""Object presets (configs/objects/<name>.json): what the tip stage needs to know about one object.

  semantic_axis   semantic vectors per tip (GS frame): start = coating-boundary side, end = tip end
  tip_profile     6-point tip profile per tip (GS frame): end E, coat points upper / lower, tip length, coat width
  gs_ply          the object's 3DGS model, for the alpha render (data/, not in git)
  coat_polarity   +1: darker coating -> brighter tip; -1: the reverse
  coat_subpix_clip   limit the coat point's sub-pixel offset to +-0.5 px
  tip_length_mm   coat -> end length per tip, used only by step 4.4
Size constants: REF (px, at the VETUS size) * s_L (tip-length ratio) or * s_W (coat-width ratio) of the profiles."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .constants import INT_KEYS, LENGTH_KEYS, REF, REF_COAT_WIDTH_MM, REF_TIP_LENGTH_MM, SILVER_MM_REF, TIPS

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class TipObject:
    name: str
    axis: dict
    profile: dict
    consts: dict           # px constants of this object (REF scaled) + coat_polarity, coat_subpix_clip
    silver_mm: float       # model-edge window from the tip end (step 3.3)
    lengths: dict          # m, step 4.4
    gs_ply: Path


def load_object(name):
    p = Path(name) if str(name).endswith(".json") else ROOT / "configs/objects" / f"{name}.json"
    o = json.loads(p.read_text())
    axis = json.loads((ROOT / o["semantic_axis"]).read_text())
    prof = json.loads((ROOT / o["tip_profile"]).read_text())
    s_L = float(np.mean([prof["tips"][t]["tip_length_mm"] for t in TIPS])) / REF_TIP_LENGTH_MM
    s_W = float(np.mean([prof["tips"][t]["coat_width_mm"] for t in TIPS])) / REF_COAT_WIDTH_MM
    c = {}
    for k, v in REF.items():
        s = s_L if k in LENGTH_KEYS else s_W
        v = tuple(x * s for x in v) if isinstance(v, tuple) else v * s
        c[k] = max(1, int(round(v))) if k in INT_KEYS else v
    c["coat_polarity"] = o["coat_polarity"]
    c["coat_subpix_clip"] = o["coat_subpix_clip"]
    return TipObject(o["name"], axis, prof, c, SILVER_MM_REF * s_L, {t: v / 1000 for t, v in o["tip_length_mm"].items()},
                     ROOT / o["gs_ply"])
