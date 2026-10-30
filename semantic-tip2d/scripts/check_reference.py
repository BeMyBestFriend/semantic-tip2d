"""Compare a run (scripts/run_pose_results.py output) with the reference result in reference/.

The reference is the output of the research pipeline on the 0921 query sets (VETUS, pink: 7 stereo frames each), so an
unchanged implementation reproduces it exactly.

  python scripts/check_reference.py outputs/pinktweezers_gpu.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def same(a, b, path, bad):
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            bad.append(f"{path}: keys {sorted(set(a) ^ set(b))}")
        for k in set(a) & set(b):
            same(a[k], b[k], f"{path}.{k}", bad)
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b) and all(isinstance(x, (int, float)) for x in a + b):
        if not np.allclose(a, b, rtol=0, atol=1e-9):
            bad.append(f"{path}: {a} vs {b}")
    elif isinstance(a, float) and isinstance(b, float):
        if abs(a - b) > 1e-9:
            bad.append(f"{path}: {a} vs {b}")
    elif a != b:
        bad.append(f"{path}: {a} vs {b}")


def main():
    run = json.loads(Path(sys.argv[1]).read_text())
    ref = json.loads((ROOT / "reference" / f"{run['object']}_query0921.json").read_text())
    bad = []
    same(run["frames"], ref["frames"], "frames", bad)
    n = sum(len(f) for f in ref["frames"].values())
    print(f"{run['object']} ({run['align']}): {n} tips, " + ("identical to the reference" if not bad else f"{len(bad)} differences"))
    for b in bad[:20]:
        print("  ", b)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
