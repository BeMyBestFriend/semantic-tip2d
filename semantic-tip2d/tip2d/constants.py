"""Constants of the tip stage. Fixed ones depend on the camera / pose error, not on the object; REF holds the
object-size constants at the VETUS size, scaled per object (objects.py)."""
import numpy as np

TIPS, SIDES = ("tip1", "tip2"), ("upper", "lower")
VEC = {"tip1": "vector1", "tip2": "vector2"}          # tip -> semantic vector name in semantic_axis.json
W, H = 1920, 1080

# fixed (pose error, image, stereo)
CROP_MARGIN = 90               # px around the projected semantic vectors
CANNY = (30, 90)
ORIENT_BINS, CHAMFER_TRUNC = 6, 8.0
THETAS = np.arange(-6, 6.1, 2.0)               # coarse alignment
SCALES = (0.96, 1.0, 1.04)
GRID = np.stack(np.meshgrid(np.arange(-40, 41, 2.0), np.arange(-40, 41, 2.0)), -1).reshape(-1, 2)
FINE = np.stack(np.meshgrid(np.arange(-2, 2.01, 0.5), np.arange(-2, 2.01, 0.5)), -1).reshape(-1, 2)
SEARCH_PX, GRAD_THR, SAMPLE_STEP = 5.0, 6.0, 2.0   # normal search, model-edge sampling
NEIGHBOURS = [(dx, dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dx or dy]
RIDGE_THR, RIDGE_STEP, RIDGE_GAP = 6.0, 0.5, 2.0
BORDER_PX = 3.0
MIN_UY, STEREO_MARGIN, ROW_GATE = 0.2, 1.0, 3.0    # stereo end choice
FAR_ROW_MAX, ROW_FIX_MARGIN = 1.5, 1.0               # coat row check
SNAP_RANGE, SNAP_REL = (-3.0, 10.0), (0.4, 0.25)
ROW_MIN_DEG = 10.0
SILVER_MM_REF = 15.0
# object-size constants at the VETUS size (px); scaled per object by s_L (length) or s_W (width)
REF = {"seed_back": 15.0, "min_chain": 12, "max_steps": 120, "self_fit": 12, "ridge_max": 20.0, "coat_window": 12.0,
       "uturn_min_index": 6, "uturn_window": 12,
       "side_sep_max": 4.0, "self_out": 2.0, "self_in": 6.0, "coat_inset": (1.5, 2.5, 3.5), "ridge_half": 2.0, "uturn_back": 3.0}
LENGTH_KEYS = ("seed_back", "min_chain", "max_steps", "self_fit", "ridge_max", "coat_window", "uturn_min_index", "uturn_window")
INT_KEYS = ("min_chain", "max_steps", "self_fit", "uturn_min_index", "uturn_window")
MIN_FOUND_FRAC = 0.3
# VETUS 6-point profile means (tip_profile_20261029/VETUStweezers/tip_profile.json): the size reference of REF
REF_TIP_LENGTH_MM = (17.887006691237534 + 17.42926706938966) / 2
REF_COAT_WIDTH_MM = (3.173232558606679 + 3.236467436099949) / 2
