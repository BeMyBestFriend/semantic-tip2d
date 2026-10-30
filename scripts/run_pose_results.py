"""Run the 2D tip stage on stored pose results (refined_poses.json + SAM masks of the pose pipeline) and stereo queries.

  python scripts/run_pose_results.py --object pinktweezers \
      --pose_run <pose output>/result --query_dir <queries> [--align gpu] [--out outputs/pink.json]

  --pose_run   folder with refined_poses.json (one entry per stereo pair, "query" = the L image stem, R and t of the left
               camera) and masks/<stem>_sam2.png for L and R
  --query_dir  folder with <stem>.png for the L stem and its _R twin
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import tip2d  # noqa: E402


def load_frames(pose_run, query_dir):
    frames = []
    for p in json.loads((pose_run / "refined_poses.json").read_text()):
        sl = p["query"]
        sr = sl[:-1] + "R"
        frames.append({"frame": sl[:6], "R": np.asarray(p["R"], np.float64), "t": np.asarray(p["t"], np.float64).reshape(3),
                       "query": {s: cv2.imread(str(query_dir / f"{st}.png")) for s, st in (("L", sl), ("R", sr))},
                       "sam": {s: cv2.imread(str(pose_run / f"masks/{st}_sam2.png"), 0) > 0 for s, st in (("L", sl), ("R", sr))}})
    return frames


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--object", required=True, help="preset name in configs/objects")
    ap.add_argument("--pose_run", required=True, type=Path)
    ap.add_argument("--query_dir", required=True, type=Path)
    ap.add_argument("--align", default="cpu", choices=("cpu", "gpu"))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    obj = tip2d.load_object(args.object)
    render = tip2d.AlphaRenderer(tip2d.Gaussians(obj.gs_ply))
    frames = load_frames(args.pose_run, args.query_dir)
    f0 = frames[0]
    tip2d.detect_tips_2d(f0["query"], f0["sam"], f0["R"], f0["t"], obj, render, args.align)        # warm-up (JIT, compile)
    results, times = {}, []
    for f in frames:
        t0 = time.perf_counter()
        results[f["frame"]] = tip2d.detect_tips_2d(f["query"], f["sam"], f["R"], f["t"], obj, render, args.align)
        times.append((time.perf_counter() - t0) * 1000)
        print(f["frame"], {tip: (r.get("failure") or ("end at the image border" if any(r["end_at_border"].values())
                                                     else f"L/R end row diff {r['end_row_diff_px']:.1f} px"))
                           for tip, r in results[f["frame"]].items()}, f"{times[-1]:.0f} ms")
    out = args.out or ROOT / "outputs" / f"{args.object}_{args.align}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"object": args.object, "align": args.align, "ms_per_stereo_frame": float(np.median(times)),
                               "frames": results}, indent=1, default=float))
    print(f"median {np.median(times):.1f} ms per stereo frame -> {out}")


if __name__ == "__main__":
    main()
