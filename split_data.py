"""STEP 1 - run this first.
Splits the Indian Pines cube into a TRAIN region and a TEST region,
normalizes both with TRAIN statistics, and saves them to data/split/.

    python split_data.py                 # default: columns 0-99 train, 100-144 test
    python split_data.py --split_col 90  # different split
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/indianpinearray.npy")
ap.add_argument("--split_col", type=int, default=100)
args = ap.parse_args()

cube = np.load(args.data).astype(np.float32)        # (145, 145, 200)
H, W, B = cube.shape
c = args.split_col

# Spatial split: a vertical cut. Train and test are different ground areas,
# so the test score shows how well a codec generalizes to unseen terrain.
train_raw = cube[:, :c]
test_raw = cube[:, c:]

# Normalize with TRAIN min/max only (no information leaks from the test area).
lo, hi = float(train_raw.min()), float(train_raw.max())
train = (train_raw - lo) / (hi - lo)
test = (test_raw - lo) / (hi - lo)

os.makedirs("data/split", exist_ok=True)
np.save("data/split/train.npy", train)
np.save("data/split/test.npy", test)
info = {"source": args.data, "cube_shape": [H, W, B], "split_col": c,
        "train_columns": [0, c - 1], "test_columns": [c, W - 1],
        "train_shape": list(train.shape), "test_shape": list(test.shape),
        "train_pixels": int(H * c), "test_pixels": int(H * (W - c)),
        "norm_min": lo, "norm_max": hi,
        "test_range_after_norm": [float(test.min()), float(test.max())]}
with open("data/split/split_info.json", "w") as f:
    json.dump(info, f, indent=2)

# Figure for the report: where the split is.
rgb = cube[:, :, [50, 27, 17]]
p1, p99 = np.percentile(rgb, [1, 99])
rgb = np.clip((rgb - p1) / (p99 - p1), 0, 1)
fig, ax = plt.subplots(figsize=(5.2, 5.2))
ax.imshow(rgb)
ax.axvline(c - 0.5, color="white", linewidth=2.5)
ax.text(c / 2, 8, f"TRAIN\n{H}x{c} px", color="white", ha="center", va="top",
        fontsize=11, fontweight="bold")
ax.text(c + (W - c) / 2, 8, f"TEST\n{H}x{W - c} px", color="white", ha="center",
        va="top", fontsize=11, fontweight="bold")
ax.set_xticks([]); ax.set_yticks([])
ax.set_title("Indian Pines - spatial train/test split", fontsize=12)
os.makedirs("results/figures", exist_ok=True)
fig.savefig("results/figures/fig_split.png", dpi=200, bbox_inches="tight")

print(json.dumps(info, indent=2))
print("Saved data/split/train.npy, test.npy, split_info.json, results/figures/fig_split.png")
