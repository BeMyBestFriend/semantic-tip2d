"""tip2d: 2D tip detection of tweezers on a rectified stereo pair, given the object pose and SAM masks.

    from tip2d import load_object, Gaussians, AlphaRenderer, detect_tips_2d
    obj = load_object("pinktweezers")
    render = AlphaRenderer(Gaussians(obj.gs_ply))
    result = detect_tips_2d({"L": bgr_L, "R": bgr_R}, {"L": mask_L, "R": mask_R}, R, t, obj, render, backend="gpu")
"""
from .gs_render import AlphaRenderer, Gaussians
from .objects import TipObject, load_object
from .pipeline import detect_tips_2d

__all__ = ["AlphaRenderer", "Gaussians", "TipObject", "load_object", "detect_tips_2d"]
