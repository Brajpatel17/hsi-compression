"""Reads results/classical + results/neural (CSVs and saved reconstructions), writes figures + summary table
to results/figures/. Run after the run_*.py scripts."""
import glob, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common.data import load_split

OUT = "results/figures"
os.makedirs(OUT, exist_ok=True)

# Fixed colour per method (colour follows the method in every figure).
COLORS = {
    "PCA": "#2a78d6", "DCT": "#eb6834", "3D-DCT": "#1baf7a",
    "JPEG": "#eda100", "JPEG2000": "#e87ba4", "PCA+JPEG2000": "#008300",
    "MLP-AE": "#6250d6", "1D-CNN-AE": "#e34948", "3D-CNN-AE": "#0b0b0b",
}
MARKERS = {"PCA": "o", "DCT": "s", "3D-DCT": "D", "JPEG": "^", "JPEG2000": "v",
           "PCA+JPEG2000": "P", "MLP-AE": "o", "1D-CNN-AE": "s", "3D-CNN-AE": "D"}
NEURAL = ["MLP-AE", "1D-CNN-AE", "3D-CNN-AE"]
CLASSICAL = ["PCA", "DCT", "3D-DCT", "JPEG", "JPEG2000", "PCA+JPEG2000"]

plt.rcParams.update({
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#c3c2b7", "axes.grid": True, "grid.color": "#e1e0d9",
    "grid.linewidth": 0.6, "legend.frameon": False, "savefig.dpi": 200,
    "savefig.bbox": "tight"})

df = pd.concat([pd.read_csv(f) for f in
                sorted(glob.glob("results/classical/*.csv") + glob.glob("results/neural/*.csv"))])
df = df.sort_values(["method", "bpppb"])


def rd_figure(methods, fname, title, xmax=1.7):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for m in methods:
        d = df[(df.method == m) & (df.bpppb <= xmax)]
        if d.empty:
            continue
        neural = m in NEURAL
        kw = dict(color=COLORS[m], marker=MARKERS[m], markersize=7, linewidth=2,
                  linestyle="-" if neural else "--",
                  markerfacecolor=COLORS[m] if neural else "white",
                  markeredgewidth=1.6, label=m)
        axes[0].plot(d.bpppb, d.PSNR, **kw)
        axes[1].plot(d.bpppb, d.SAM, **kw)
    axes[0].set(xlabel="Rate (bits per pixel per band)", ylabel="PSNR (dB)  ↑ better")
    axes[1].set(xlabel="Rate (bits per pixel per band)", ylabel="SAM (degrees)  ↓ better")
    axes[1].legend(loc="upper right", ncol=1)
    fig.suptitle(title, fontsize=13)
    fig.savefig(f"{OUT}/{fname}")
    plt.close(fig)


rd_figure(CLASSICAL, "fig1_classical_rate_distortion.png",
          "Classical codecs on the held-out test region")
rd_figure(NEURAL + ["PCA", "JPEG2000", "PCA+JPEG2000"],
          "fig2_neural_vs_classical.png",
          "Neural autoencoders vs. strongest classical codecs", xmax=0.8)
rd_figure(CLASSICAL + NEURAL, "fig0_all_methods.png",
          "All methods (dashed = classical, solid = neural)")


# ---------- summary table: PSNR / SAM interpolated at fixed rates ----------
RATES = [0.1, 0.2, 0.3, 0.5]
rows = []
for m in CLASSICAL + NEURAL:
    d = df[df.method == m].sort_values("bpppb")
    if d.empty:
        continue
    row = {"method": m}
    for r in RATES:
        ok = d.bpppb.min() <= r <= d.bpppb.max()
        lr, lb = np.log(r), np.log(d.bpppb.values)
        row[f"PSNR@{r}"] = round(float(np.interp(lr, lb, d.PSNR)), 2) if ok else None
        row[f"SAM@{r}"] = round(float(np.interp(lr, lb, d.SAM)), 2) if ok else None
    rows.append(row)
table = pd.DataFrame(rows)
table.to_csv(f"{OUT}/summary_table.csv", index=False)
with open(f"{OUT}/summary_table.md", "w") as f:
    f.write("PSNR (dB) / SAM (deg) at fixed rates, interpolated in log-rate. "
            "Blank = method's tested range does not reach that rate.\n\n")
    f.write("| Method | " + " | ".join(f"{r} bpppb" for r in RATES) + " |\n")
    f.write("|---" * (len(RATES) + 1) + "|\n")
    for row in rows:
        cells = []
        for r in RATES:
            p, s = row[f"PSNR@{r}"], row[f"SAM@{r}"]
            cells.append("" if p is None else f"{p:.2f} / {s:.2f}")
        f.write(f"| {row['method']} | " + " | ".join(cells) + " |\n")
print(open(f"{OUT}/summary_table.md").read())


# ---------- visual comparison ----------
train, test, _ = load_split()
RGB = [50, 27, 17]   # common false-colour band choice for Indian Pines

def false_color(cube):
    img = cube[:, :, RGB].astype(np.float32)
    lo, hi = np.percentile(test[:, :, RGB], [1, 99])
    return np.clip((img - lo) / (hi - lo), 0, 1)

recon_files = glob.glob("results/classical/recon/*.npy") + glob.glob("results/neural/recon/*.npy")
recs = {os.path.basename(p)[:-4]: np.load(p).astype(np.float32) for p in recon_files}
rate_file = {os.path.basename(p)[:-4]: p[:-4] + ".txt" for p in recon_files}
order = [m for m in ["PCA", "3D-DCT", "JPEG", "JPEG2000", "PCA+JPEG2000",
                     "MLP-AE", "1D-CNN-AE", "3D-CNN-AE"] if m in recs]
saved_rate = {m: float(open(rate_file[m]).read()) for m in recs}

n = len(order) + 1
fig, axes = plt.subplots(2, n, figsize=(1.45 * n, 7.4))
axes[0, 0].imshow(false_color(test)); axes[0, 0].set_title("Original\n16 bpppb", fontsize=8)
axes[1, 0].axis("off")
vmax = 0.05
for i, m in enumerate(order, start=1):
    axes[0, i].imshow(false_color(recs[m]))
    axes[0, i].set_title(f"{m}\n{saved_rate[m]:.2f} bpppb", fontsize=8)
    e = np.sqrt(np.mean((recs[m] - test) ** 2, axis=2))
    im = axes[1, i].imshow(e, cmap="Blues", vmin=0, vmax=vmax)
    axes[1, i].set_title("error", fontsize=8)
for ax in axes.ravel():
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
cbar = fig.colorbar(im, ax=axes[1, 1:], orientation="horizontal", fraction=0.05, pad=0.04)
cbar.set_label("Per-pixel RMSE (normalized units)")
fig.suptitle("Test region at ~0.16 bpppb (100x compression): false colour (bands 50/27/17) and per-pixel error", fontsize=12)
fig.savefig(f"{OUT}/fig3_visual_comparison.png")
plt.close(fig)


# ---------- spectrum of one pixel ----------
y, x = 70, 20
fig, ax = plt.subplots(figsize=(9, 4.4))
ax.plot(test[y, x], color="#0b0b0b", linewidth=2.4, label="Original")
for m in ["PCA", "JPEG2000", "MLP-AE", "3D-CNN-AE"]:
    if m in recs:
        ax.plot(recs[m][y, x], color=COLORS[m], linewidth=1.6,
                linestyle="-" if m in NEURAL else "--",
                label=f"{m} ({saved_rate[m]:.2f} bpppb)")
ax.set(xlabel="Band index", ylabel="Normalized reflectance",
       title=f"Reconstructed spectrum of test pixel (row {y}, col {x})")
ax.legend(loc="upper right")
fig.savefig(f"{OUT}/fig4_spectrum.png")
plt.close(fig)
print("Figures written to", OUT)
