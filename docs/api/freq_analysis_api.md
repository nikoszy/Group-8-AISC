# `src/freq_analysis/` — API Reference

> **Module 3: FFT Frequency Anomaly Pipeline**
> Version: multi-feature (4-sub-feature) implementation
> Last updated: 2026-05-26

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Module: `utils.py`](#2-module-utilspy)
3. [Module: `fft_extractor.py`](#3-module-fft_extractorpy)
4. [Module: `texture_scorer.py`](#4-module-texture_scorerpy)
5. [Module: `anomaly_scorer.py`](#5-module-anomaly_scorerpy)
6. [Module: `frequency_analyzer.py`](#6-module-frequency_analyzerpy)
7. [Data Flow Diagram](#7-data-flow-diagram)
8. [Calibration Constants](#8-calibration-constants)
9. [Known Limitations](#9-known-limitations)

---

## 1. Architecture Overview

The frequency analysis pipeline is layered: each layer depends only on the layers below it.

```
frequency_analyzer.py   ← public batch API + visualisation
        │
anomaly_scorer.py       ← single-image scoring (fft_anomaly_score, fft_spectral_features)
        │
fft_extractor.py        ← pure FFT mathematics (no I/O, no scoring)
        │
utils.py                ← low-level image helpers (resize, load, normalise)
        │
texture_scorer.py       ← Laplacian sharpness score (independent branch)
```

**Rule:** `ensemble.py` imports from `anomaly_scorer` and `texture_scorer` only.
It never imports directly from `fft_extractor` or `utils`.

---

## 2. Module: `utils.py`

Low-level image helpers. Every other module in this package imports from here.

---

### `resize_to_square(img, size=224)`

Resize any image to a fixed square.

**Why this matters for FFT:** The FFT produces a frequency grid whose scale
depends on image dimensions. Resizing to a common size ensures all frequency
scores are directly comparable — "ring 20 out of 40" always means the same
spatial frequency across every image.

| Parameter | Type | Description |
|-----------|------|-------------|
| `img` | `np.ndarray` | `(H × W × C)` BGR or `(H × W)` grayscale |
| `size` | `int` | Target side length in pixels. Default `224` (matches CNN convention) |

**Returns:** `np.ndarray` of shape `(size, size, C)` or `(size, size)`

**Implementation note:** Uses `cv2.INTER_AREA` interpolation, which averages
pixels when downsampling — the best quality method for shrinking.

---

### `load_face_image(path, target_size=224)`

Load a JPEG face crop from disk, resize to square, return BGR array.

Returns `None` on failure so callers can skip bad files without crashing.

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | `str` | Absolute or relative file path |
| `target_size` | `int` | Side length to resize to. Default `224` |

**Returns:** `np.ndarray` `(target_size × target_size × 3, uint8)` or `None`

---

### `normalize_to_uint8(arr)`

Scale a float array to `[0, 255]` and cast to `uint8`.

Used when saving or displaying intermediate results (e.g. FFT spectrum images)
as standard image files.

| Parameter | Type | Description |
|-----------|------|-------------|
| `arr` | `np.ndarray` | Float array of any range |

**Returns:** `np.ndarray` `uint8`, values in `[0, 255]`

**Edge case:** If `arr` is constant (all values equal), returns an all-zero array.

---

## 3. Module: `fft_extractor.py`

Pure FFT mathematics. No I/O, no scoring — only signal-processing primitives.
These are the building blocks for `anomaly_scorer.py`.

**Background — why 2D FFT detects deepfakes:**
Real photographs follow a "1/f" power law: low spatial frequencies carry most
energy and energy falls off predictably at higher frequencies. GAN-generated
and autoencoder-reconstructed faces violate this law — they produce either too
much or too little energy at certain frequency bands, or periodic spikes from
upsampling artifacts. This module measures those deviations.

---

### `to_grayscale(img_bgr)`

Convert a BGR colour image to grayscale.

FFT artifacts are largely colour-invariant (they affect luminance more than
chrominance), so grayscale reduces computation by 3× with minimal signal loss.

| Parameter | Type | Description |
|-----------|------|-------------|
| `img_bgr` | `np.ndarray` | `(H × W × 3)` BGR, `uint8` |

**Returns:** `np.ndarray` `(H × W)` grayscale, `uint8`

---

### `compute_log_magnitude_spectrum(gray)`

Compute the 2D FFT of a grayscale image and return the DC-centred
log-magnitude spectrum.

**Pipeline:**
1. Cast to `float32` (prevents uint8 overflow in FFT arithmetic)
2. `np.fft.fft2` — 2D discrete Fourier transform → complex output
3. `np.fft.fftshift` — move DC component from corner to centre
4. `np.abs` — magnitude = √(real² + imag²)
5. `np.log(1 + x)` — compress dynamic range so non-DC content is visible

| Parameter | Type | Description |
|-----------|------|-------------|
| `gray` | `np.ndarray` | `(H × W)` grayscale, any dtype |

**Returns:** `np.ndarray` `(H × W)` `float32` — log-magnitude spectrum.
Typical value range: `3–12` for 224×224 face images.

---

### `compute_power_spectrum_2d(gray)`

Compute the 2D power spectrum (`|FFT|²`) **without** a log transform.

> **Note:** This function is **not used** by `anomaly_scorer.py`. It was
> written to test whether raw power space gives better entropy separation
> than log-magnitude. Empirical results on 778 FF++ C23 frames showed
> log-magnitude entropy has 2.7× stronger signal (`|Δ|/std = 0.212` vs `0.079`).
> Kept as a documented building block for future experiments.

| Parameter | Type | Description |
|-----------|------|-------------|
| `gray` | `np.ndarray` | `(H × W)` grayscale, any dtype |

**Returns:** `np.ndarray` `(H × W)` `float64` — raw power spectrum, DC at centre.

---

### `make_center_mask(shape, center_fraction=0.1)`

Build a binary mask: `0` inside a circular central region (the dominant DC
neighbourhood), `1` everywhere else.

**Why circular, not square:** ISO-frequency contours in a 2D FFT are circles
(distance from DC = `√(u² + v²)`). A square mask would incorrectly include
corner pixels that are actually high-frequency.

| Parameter | Type | Description |
|-----------|------|-------------|
| `shape` | `(H, W)` tuple | Should match the spectrum shape |
| `center_fraction` | `float` | Diameter of blank circle as fraction of shorter axis. Default `0.10` |

**Returns:** `np.ndarray` `(H × W)` `float32` — `0.0` inside circle, `1.0` outside.

---

### `compute_radial_power_spectrum(log_mag_spectrum, n_bands=40)`

Bin the log-magnitude spectrum into `n_bands` concentric radial rings.

**Why 40 rings instead of a single peripheral mean:**
A single mean collapses all high-frequency information into one number.
The radial profile preserves *how* energy changes with frequency, enabling
the slope fit — a much stronger discriminator than a peripheral mean alone.

| Parameter | Type | Description |
|-----------|------|-------------|
| `log_mag_spectrum` | `np.ndarray` `(H × W)` | Output of `compute_log_magnitude_spectrum()` |
| `n_bands` | `int` | Number of radial rings. Default `40` (≈2.8 px/ring for 224×224) |

**Returns:**
- `band_centers` — `float32 (n_bands,)` — radius in pixels at each ring centre
- `band_means` — `float32 (n_bands,)` — mean log-magnitude per ring

---

### `compute_high_freq_energy_ratio(band_means, outer_fraction=0.5)`

Fraction of total radial energy that lives in the outer (high-frequency) rings.

**Interpretation:**
- Low ratio → smoother face → more suspicious (fake-like)
- High ratio → more textured → more real-like

FF++ autoencoder decoders apply bilinear upsampling + convolution (a low-pass
filter), making decoded faces smoother than originals → lower HF ratio.

| Parameter | Type | Description |
|-----------|------|-------------|
| `band_means` | `float array (n_bands,)` | Output of `compute_radial_power_spectrum()` |
| `outer_fraction` | `float` | Fraction of rings treated as "high frequency". Default `0.5` |

**Returns:** `float` in `[0.0, 1.0]`. Returns `0.5` on degenerate (all-zero) input.

**Note:** Scoring direction (low ratio = suspicious) is handled in `anomaly_scorer.py`.

---

### `compute_spectral_entropy(band_means)`

Normalised Shannon entropy of the radial power distribution.

**Formula:** `H = -Σ(p_i × log(p_i))` where `p_i = band_means[i] / total`.
Normalised by `log(n_bands)` so result is always in `[0, 1]`.

**Interpretation:**
- Low entropy → energy concentrated in low-freq rings → smoother → suspicious
- High entropy → energy spread evenly → more textured → real-like

On FF++ C23, this has the strongest real/fake signal of the four sub-features:
`|Δ|/pooled_std ≈ 0.212`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `band_means` | `float array (n_bands,)` | Output of `compute_radial_power_spectrum()` |

**Returns:** `float` in `[0.0, 1.0]`. Returns `0.5` on degenerate input.

---

### `compute_peak_excess(band_centers, band_means, skip_bands=3)`

Height of the tallest positive spike above the fitted 1/f decay line.

**How it works:**
1. Fit a least-squares line to `(log(ring_radius), ring_mean)` across usable bands
2. Compute residuals: `actual − fitted`
3. Return `max(residuals)`, clamped to 0

**Why it helps:** Upsampling artifacts (common in GAN-style deepfakes) create
periodic spikes in the frequency domain. Even FF++ autoencoder fakes can show
mild convolutional ringing that appears as local bumps above the 1/f line.

| Parameter | Type | Description |
|-----------|------|-------------|
| `band_centers` | `float array (n_bands,)` | Ring radii from `compute_radial_power_spectrum()` |
| `band_means` | `float array (n_bands,)` | Mean log-magnitude per ring |
| `skip_bands` | `int` | DC-side bands to skip before fitting. Default `3` |

**Returns:** `float ≥ 0.0` — log-magnitude units, **not** normalised to [0,1].
Returns `0.0` on degenerate input (< 4 usable bands, flat spectrum).

---

## 4. Module: `texture_scorer.py`

Laplacian-variance sharpness score for deepfake detection.

**Principle:** FF++ autoencoders apply bilinear upsampling + convolution in every
decoder layer — an implicit low-pass filter. The output face is slightly smoother
than a genuine photograph. The Laplacian operator (`d²I/dx² + d²I/dy²`) measures
second-order spatial derivatives and is large wherever pixel values change rapidly
(edges, fine skin texture). `Var(Laplacian)` captures overall texture richness.

---

### `laplacian_score(face_img, target_size=224)`

Compute the Laplacian-variance sharpness score for a single face image.

| Parameter | Type | Description |
|-----------|------|-------------|
| `face_img` | `np.ndarray (H × W × 3, BGR)` | Input face image |
| `target_size` | `int` | Resize to this square before scoring. Default `224` |

**Returns:** `float` in `[0.0, 1.0]`, rounded to 4 decimal places.

**Interpretation:**
- Low score → smoother → more suspicious (possible deepfake)
- High score → sharper → looks like a genuine photograph

**Calibration:** `_LAP_CLIP = 3000.0`. Real faces: Var(Lap) ≈ 300–4000.
Deepfakes: Var(Lap) ≈ 150–2500. Clipping at 3000 keeps 95% of samples in [0,1].

**Signal on FF++ C23:** `Δ ≈ 0.06` between real and fake (moderate — the strongest
of the three handcrafted features in the current ensemble).

---

## 5. Module: `anomaly_scorer.py`

Converts raw FFT primitives into a single 0–1 anomaly score **or** a dict of
four raw sub-features. This is the module that `ensemble.py` imports.

**Two public functions for two use cases:**
- `fft_anomaly_score()` — one pre-combined scalar (legacy path)
- `fft_spectral_features()` — four raw values for the LR to weight separately (preferred)

---

### `fft_anomaly_score(face_img, target_size=224)`

Score a single face image for FFT-based deepfake anomalies.

**Pipeline:**
1. Resize to `target_size × target_size`
2. Convert to grayscale
3. Compute 2D FFT → log-magnitude spectrum
4. Bin into 40 radial rings
5. Extract 4 sub-features (slope, HF ratio, entropy, peak excess)
6. Normalise each to [0,1] via calibrated constants
7. Return mean of the four normalised components

| Parameter | Type | Description |
|-----------|------|-------------|
| `face_img` | `np.ndarray (H × W × 3, BGR)` | Face image, any size |
| `target_size` | `int` | Resize target. Default `224` |

**Returns:** `float` in `[0.0, 1.0]`, rounded to 4 decimal places.
Higher = more suspicious. Returns `0.5` on degenerate input.

**⚠️ Calibration leakage warning:** The normalisation constants (`_*_BASELINE`,
`_*_RANGE`) were derived over all 778 FF++ C23 frames including the ~20%
validation set. Per-score shift is estimated at < 0.01, partially cancelled by
`StandardScaler` in `ensemble.py`. Treat AUC as the reliable metric.

---

### `fft_spectral_features(face_img, target_size=224)`

Return the four raw FFT sub-features as a dict — **no pre-combining**.

**Why prefer this over `fft_anomaly_score()`:**
Equal-weight averaging blunts the stronger signals. On FF++ C23, spectral
entropy has `|Δ|/std ≈ 0.212` while peak excess has only `≈ 0.081` — a 2.6×
difference. Returning raw values lets `StandardScaler + LogisticRegression`
learn the correct weights from training data. This also eliminates the
calibration leakage in `fft_anomaly_score()`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `face_img` | `np.ndarray (H × W × 3, BGR)` | Face image, any size |
| `target_size` | `int` | Resize target. Default `224` |

**Returns:** `dict` with four `float` values:

| Key | Typical range | Real-like direction | Signal strength |
|-----|--------------|---------------------|----------------|
| `"fft_slope"` | `-3` to `-1` | More negative | Weak |
| `"fft_hf_ratio"` | `[0.0, 1.0]` | Higher | Moderate (`|Δ|/std ≈ 0.182`) |
| `"fft_entropy"` | `[0.0, 1.0]` | Higher | **Strongest** (`|Δ|/std ≈ 0.212`) |
| `"fft_peak_excess"` | `[0.0, ∞)` | Lower | Weakest (`|Δ|/std ≈ 0.081`) |

Returns real-face baseline values on degenerate input (constant spectrum).

---

### Private helper: `_score_component(raw_value, baseline, range_)`

Normalise one raw sub-feature to [0, 1] where higher = more suspicious.

**Formula:** `clip( (baseline − raw) / range, 0, 1 )`

Used internally by `fft_anomaly_score()`. Not needed when using
`fft_spectral_features()` (the LR handles normalisation via `StandardScaler`).

---

### Calibration Constants

All constants are module-level. Do not hand-tune. Re-run `_calibrate.py` if the dataset changes substantially.

| Constant | Value | Meaning |
|----------|-------|---------|
| `_SLOPE_BASELINE` | `-2.163877` | Mean slope of real faces |
| `_SLOPE_RANGE` | `0.592026` | 3 × std of real-face slope distribution |
| `_HF_RATIO_BASELINE` | `0.401319` | Mean HF ratio of real faces |
| `_HF_RATIO_RANGE` | `0.032208` | 3 × std |
| `_ENTROPY_BASELINE` | `0.991984` | Mean entropy of real faces |
| `_ENTROPY_RANGE` | `0.004608` | 3 × std |
| `_PEAK_EXCESS_BASELINE` | `0.294229` | Mean peak excess of real faces |
| `_PEAK_EXCESS_RANGE` | `0.308040` | 3 × std |
| `_N_BANDS` | `40` | Number of radial frequency rings |
| `_SKIP_BANDS` | `3` | DC-side rings dropped from slope/peak fits |

---

## 6. Module: `frequency_analyzer.py`

Public batch API and visualisation. `ensemble.py` and demos import from here.

---

### `compute_fft_score_batch(image_paths, target_size=224, verbose=True)`

Score a list of image files and return `(path, score)` pairs.

| Parameter | Type | Description |
|-----------|------|-------------|
| `image_paths` | `list[str]` | File paths to score |
| `target_size` | `int` | Resize target before FFT. Default `224` |
| `verbose` | `bool` | Print progress every 20 files |

**Returns:** `list[tuple[str, float]]` — `(path, score)` in input order.
Images that fail to load are silently omitted.

---

### `visualize_spectrum(face_img, save_path=None, label=None)`

Create a 3-panel side-by-side diagnostic image.

| Panel | Content |
|-------|---------|
| 1 — ORIGINAL | The resized face photograph |
| 2 — LOG SPECTRUM | Full FFT log-magnitude (bright = high energy) |
| 3 — HIGH-FREQ ONLY | Spectrum with low-freq centre masked out |

**How to read the spectrum:**
- Centre pixel = DC (average brightness)
- Bright ring near centre = dominant low frequencies (expected for natural images)
- Bright spikes away from centre = periodic patterns (GAN upsampling artifacts)

| Parameter | Type | Description |
|-----------|------|-------------|
| `face_img` | `np.ndarray (H × W × 3, BGR)` | Input face |
| `save_path` | `str` or `None` | Write to this path, or `cv2.imshow` if None |
| `label` | `str` or `None` | Optional class label ("REAL" = green, "FAKE" = red) |

**Returns:** `np.ndarray (224 × 672 × 3)` — the combined 3-panel image.

---

## 7. Data Flow Diagram

```
Face image (BGR array)
        │
        ▼
utils.resize_to_square(img, 224)
        │
        ▼
fft_extractor.to_grayscale(img)
        │
        ▼
fft_extractor.compute_log_magnitude_spectrum(gray)
        │                    │
        ▼                    ▼
compute_radial_power_spectrum(log_spec, n_bands=40)
        │
        ├──→ compute_high_freq_energy_ratio(band_means)   → fft_hf_ratio
        ├──→ compute_spectral_entropy(band_means)          → fft_entropy
        ├──→ np.polyfit(log(centers), means, 1)[0]         → fft_slope
        └──→ compute_peak_excess(centers, means)           → fft_peak_excess
                                                    │
                                                    ▼
                          fft_spectral_features() → dict of 4 raw values
                          fft_anomaly_score()     → single 0-1 float
```

---

## 8. Calibration Constants

Calibration was performed by running `_calibrate.py` over the 778-frame FF++ C23
dataset (`data/manifest.csv`). The constants represent the **real-face distribution**:

- `BASELINE` = mean of the real-face distribution for each sub-feature
- `RANGE` = 3 × std of the real-face distribution

A face scoring 1σ below the real mean gets ≈ 0.33; 2σ below gets ≈ 0.67; 3σ below scores 1.0.

> ⚠️ **Known leakage:** These constants were computed over all 778 frames including
> the ~20% later held out for validation. Estimated per-score shift < 0.01.
> See `anomaly_scorer.py` comment block for the correct fix (compute from training split only).

---

## 9. Known Limitations

| Limitation | Impact | Mitigation |
|-----------|--------|-----------|
| FF++ C23 is heavily compressed (H.264) | Smooths GAN artifacts that FFT/Laplacian detect | Use C0 or raw if available |
| Calibration leakage in `fft_anomaly_score()` | < 0.01 per-score bias | Use AUC, not accuracy metrics |
| Equal weights in `fft_anomaly_score()` | Blunts the entropy signal (2.6× stronger than peak excess) | Use `fft_spectral_features()` + LR instead |
| Radial binning ignores spatial location | Misses quadrant-specific GAN artifacts | Could add azimuthal bins in v2 |
| Haar cascade face detection fallback | Some crops may include non-face content | Screen by `MIN_FACE_FRAC` in `inspect_dataset.py` |
