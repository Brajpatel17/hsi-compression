# Hyperspectral image compression — Indian Pines

Classical codecs vs. neural autoencoders on the Indian Pines cube (145 × 145 × 200, 16-bit).

## Run

```bash
pip install -r requirements.txt
python run_all.py            # everything, in order (~45-90 min on CPU)
```

Or step by step, always from the project root:

| Step | Script | What it does | Time (CPU) |
|---|---|---|---|
| 1 | `split_data.py` | train/test split + normalization → `data/split/` | 2 s |
| 2 | `classical/run_classical.py` | PCA, spectral DCT, 3D-DCT, JPEG, JPEG2000, PCA+JPEG2000 | ~15 s |
| 3 | `neural/run_mlp.py` | Model 1 – spectral MLP autoencoder | ~3 min |
| 4 | `neural/run_cnn1d.py` | Model 2 – 1D-CNN spectral autoencoder | ~10 min |
| 5 | `neural/run_cnn3d.py` | Model 3 – 3D-CNN spatial–spectral autoencoder | ~30-60 min |
| 6 | `plot_results.py` | figures + summary table | ~10 s |

Quick smoke test (runs in a minute, numbers are meaningless):
`python neural/run_mlp.py 3`, `python neural/run_cnn1d.py 3`, `python neural/run_cnn3d.py 5`.

## Outputs

- `data/split/train.npy`, `test.npy`, `split_info.json` — the split (normalized)
- `results/classical/classical.csv` — every classical setting: method, setting, bpppb, PSNR, SAM
- `results/neural/{mlp_ae,cnn1d_ae,cnn3d_ae}.csv` — same columns for each network
- `results/{classical,neural}/recon/` — one saved reconstruction per method (~0.16 bpppb)
- `results/figures/`
  - `fig_split.png` — where the train/test cut is
  - `fig0_all_methods.png` — all rate–distortion curves (PSNR and SAM vs. bpppb)
  - `fig1_classical_rate_distortion.png` — classical only
  - `fig2_neural_vs_classical.png` — neural vs. best classical
  - `fig3_visual_comparison.png` — false colour + error maps, all methods at ~0.16 bpppb
  - `fig4_spectrum.png` — one pixel's reconstructed spectrum
  - `summary_table.md` / `.csv` — PSNR/SAM at 0.1 / 0.2 / 0.3 / 0.5 bpppb

## Experimental protocol

- **Split:** spatial (`split_data.py`, change with `--split_col`). Columns 0–99 = train (145×100 px), columns 100–144 = test (145×45 px).
  Everything that is *learned* (PCA basis, DCT coefficient selection, network weights)
  uses only the train region. All numbers are reported on the test region.
- **Normalization:** min–max with train-region statistics → [0, 1].
- **Metrics:** PSNR (peak = 1 on normalized data), SAM (mean spectral angle, degrees),
  rate in **bpppb** (bits per pixel per band; the raw data is 16 bpppb).
- **Rate accounting:**
  - PCA, spectral DCT, all autoencoders: latent code uniformly quantized to 8 bits per value,
    per channel; min/max per channel counted as side information.
  - JPEG / JPEG2000: actual compressed file size (bands tiled into one mosaic image so the
    header is paid once).
  - 3D-DCT: zero-order entropy estimate of the quantized coefficients
    (what an ideal arithmetic coder would spend; slightly optimistic).
  - Decoder weights / PCA basis are **not** counted (they are trained once and reused),
    which is the usual convention for learned codecs.

## Methods

**Classical**
- **PCA** – k principal components per pixel (KLT, optimal linear spectral transform).
- **Spectral DCT** – 1D DCT along bands, keep the k highest-energy coefficients (zonal coding).
- **3D-DCT** – JPEG-like 8×8×8 block DCT, uniform quantization step, entropy-coded rate.
- **JPEG** – baseline 8-bit JPEG, band by band.
- **JPEG2000** – 16-bit, band by band (no spectral decorrelation).
- **PCA+JPEG2000** – PCA to 20 components, then JPEG2000 on the component images
  (the standard spectral + spatial pipeline).

**Neural** (all: Adam, lr 2e-3, cosine schedule, MSE loss)
- **Model 1 – MLP-AE**: 200→128→64→k→64→128→200 per pixel, ReLU, sigmoid output.
- **Model 2 – 1D-CNN-AE**: three strided Conv1d layers along the spectrum (200→100→50→25),
  linear bottleneck to k, transposed convs back up.
- **Model 3 – 3D-CNN-AE**: fully convolutional over (band, row, col); bands ↓8, space ↓4,
  c latent channels → bpppb = 0.0625·c. Trained on random 24×24 crops of the train region,
  applied to the whole test region in one pass.

## Layout

```
split_data.py                STEP 1: train/test split
run_all.py                   runs everything
plot_results.py              figures + summary table
data/
  indianpinearray.npy        original cube
  split/                     train.npy, test.npy, split_info.json
common/                      shared by both families
  data.py                    load_split()
  metrics.py                 PSNR, SAM, quantization, bpppb
  logger.py                  CSV + reconstruction saving
classical/
  codecs.py                  PCA, DCT, 3D-DCT, JPEG, JPEG2000, PCA+JPEG2000
  run_classical.py
neural/
  models/                    mlp_ae.py, cnn1d_ae.py, cnn3d_ae.py
  train.py                   training / evaluation loops
  run_mlp.py, run_cnn1d.py, run_cnn3d.py
results/
  classical/  neural/  figures/
inr_v1/                      INR paper reproduction (separate track)
```
## Contributions
- *Dharti Patel*: problem statement, Indian Pines dataset, spatial train/test split and normalisation (split_data.py), evaluation metrics PSNR, SAM and bpppb (common/metrics.py), limitations and future work
