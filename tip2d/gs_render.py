"""Step 3.2: alpha-only render of the object's 3D Gaussian Splatting model on a crop camera (gsplat).

The alpha mask is the model silhouette under the pose; step 3.3 takes the tip edges from it."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import camera as cam
from .constants import H, W

PLY_TYPES = {"float": "<f4", "double": "<f8", "int": "<i4", "uint": "<u4", "short": "<i2", "ushort": "<u2",
             "char": "i1", "uchar": "u1"}


def read_ply_vertices(path):
    """Vertex records of a binary little-endian PLY as a structured array (the vertex element must come first)."""
    raw = Path(path).read_bytes()
    end = raw.index(b"end_header\n") + len(b"end_header\n")
    head = raw[:end].decode("ascii").splitlines()
    assert "format binary_little_endian 1.0" in head, path
    elements, fields = [], []
    for line in head:
        p = line.split()
        if p[0] == "element":
            elements.append((p[1], int(p[2])))
        elif p[0] == "property" and elements[-1][0] == "vertex":
            fields.append((p[2], PLY_TYPES[p[1]]))
    assert elements[0][0] == "vertex", elements
    return np.frombuffer(raw, dtype=np.dtype(fields), count=elements[0][1], offset=end)


class Gaussians:
    """Activated 3DGS parameters on the GPU (as GaussianModel: exp scales, normalised quaternions, sigmoid opacity)."""

    def __init__(self, ply_path, device="cuda"):
        import torch
        v = read_ply_vertices(ply_path)
        names = v.dtype.names
        col = lambda ns: torch.tensor(np.stack([v[n] for n in ns], 1).astype(np.float32), device=device)
        by_index = lambda prefix: sorted((n for n in names if n.startswith(prefix)), key=lambda n: int(n.split("_")[-1]))
        self.means = col(("x", "y", "z"))
        self.quats = torch.nn.functional.normalize(col(by_index("rot_")))
        self.scales = torch.exp(col(by_index("scale_")))
        self.opacity = torch.sigmoid(col(("opacity",))).squeeze(-1)


class AlphaRenderer:
    """render(R, t, (x0, y0, x1, y1)) -> full-size uint8 alpha (0 outside the crop box)."""

    def __init__(self, gaussians: Gaussians):
        import torch
        self.torch, self.g = torch, gaussians
        dev = gaussians.means.device
        self.colors = torch.zeros((gaussians.means.shape[0], 1), dtype=torch.float32, device=dev)
        self.bg = torch.zeros((1, 1), dtype=torch.float32, device=dev)

    def __call__(self, R, t, xyxy):
        from gsplat import rasterization
        torch, g = self.torch, self.g
        x0, y0, x1, y1 = xyxy
        Kc = cam.K.copy()
        Kc[0, 2] -= x0
        Kc[1, 2] -= y0
        dev = g.means.device
        view = torch.eye(4, dtype=torch.float32, device=dev)
        view[:3, :3] = torch.as_tensor(R, dtype=torch.float32, device=dev)
        view[:3, 3] = torch.as_tensor(t, dtype=torch.float32, device=dev)
        with torch.inference_mode():
            _, alpha, _ = rasterization(
                means=g.means, quats=g.quats, scales=g.scales, opacities=g.opacity, colors=self.colors,
                viewmats=view[None], Ks=torch.as_tensor(Kc, dtype=torch.float32, device=dev)[None],
                width=x1 - x0, height=y1 - y0, near_plane=0.01, far_plane=100.0, backgrounds=self.bg,
                render_mode="RGB", packed=False)
            crop = (alpha[0, ..., 0].clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
        full = np.zeros((H, W), np.uint8)
        full[y0:y1, x0:x1] = crop
        return full
