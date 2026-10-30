"""Step 5 (3D), disabled: the 2D result -> left-camera 3D endpoint E, coat points F and directions E - F.

Kept as a reference until its error is reduced; nothing in the package calls it. The pair_2d result holds what it needs:
end_px {L, R}, edges.<L|R>.<upper|lower>.coat_px / line_origin_px / line_dir / coat_end_dir, plus the tip length.

  F     coat points: depth candidates per tip (L/R coat points on the same row -> triangulated; otherwise the L coat point
        with the R edge line at the same row), median depth, each L coat point on its ray at that depth
  E     both ends seen -> triangulated; if | |E - F_mid| - tip length | > max(11 %, 2 mm), E is put on the L end ray at the
        tip length from F_mid (the root nearest the stereo E depth); one end seen -> on that view's end ray at the tip length
  dirs  unit(E - F[side]) for upper and lower
"""

# import numpy as np
#
# from . import camera as cam
# from .camera import tri
# from .constants import SIDES
# from .stereo import coat_mid_3d
# from .util import unit
#
# LENGTH_TOL = (0.111, 0.002)            # max(fraction of the tip length, metres)
#
#
# def ray_point(cam_name, uv, depth):
#     centre = np.zeros(3) if cam_name == "L" else cam.BASELINE
#     r = cam.KINV @ np.append(uv, 1.0)
#     return centre + r * (depth / r[2])
#
#
# def endpoint_from_length(cam_name, uv, F_mid, length, z_hint):
#     """Point on the end ray of cam_name at distance length from F_mid (the root nearest z_hint)."""
#     r = unit(cam.KINV @ np.append(uv, 1.0))
#     C = np.zeros(3) if cam_name == "L" else cam.BASELINE
#     b = r @ (F_mid - C)
#     disc = b * b - np.sum((F_mid - C) ** 2) + length ** 2
#     if disc <= 0:
#         return C + b * r
#     roots = (b - np.sqrt(disc), b + np.sqrt(disc))
#     return C + min(roots, key=lambda t: abs((C + t * r)[2] - z_hint)) * r
#
#
# def tip_geometry(ends, labs, length, z_hint):
#     """ends {L, R} pixels (None = not seen), labs {L, R}{upper, lower} edges with coat points -> E, F, dirs."""
#     F_mid = coat_mid_3d(labs)
#     ...  # F per side on the L rays at the median coat depth (as coat_mid_3d)
#     tol = max(LENGTH_TOL[0] * length, LENGTH_TOL[1])
#     if ends["L"] is not None and ends["R"] is not None:
#         E = tri(ends)
#         if abs(np.linalg.norm(E - F_mid) - length) > tol:
#             E = endpoint_from_length("L", ends["L"], F_mid, length, z_hint=float(E[2]))
#     else:
#         seen = "L" if ends["L"] is not None else "R"
#         E = endpoint_from_length(seen, ends[seen], F_mid, length, z_hint)
#     return {"E": E, "dirs": {side: unit(E - F[side]) for side in SIDES}}
