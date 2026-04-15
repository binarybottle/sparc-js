# Python vs JavaScript Feature Comparison

Comparison of EMA articulatory features produced by the Python
[Speech-Articulatory-Coding](https://github.com/cheoljun95/Speech-Articulatory-Coding)
pipeline and this JavaScript port, using `sample1.wav` (9.52 s, 16 kHz).

## Models

| Component | Python | JavaScript |
|-----------|--------|------------|
| WavLM | `microsoft/wavlm-large` (PyTorch, FP32) | `wavlm_large_layer9.onnx` (ONNX, FP32) |
| Linear model | `wavlm_large-9_cut-10_mngu_linear.pkl` (scikit-learn) | `wavlm_linear_model.json` (converted from pickle) |
| Hidden layer | Layer 9, 1024 dimensions | Layer 9, 1024 dimensions |

## Preprocessing

| Step | Python | JavaScript | Match |
|------|--------|------------|-------|
| Audio normalization | z-score (`(wav - mean) / std`) | z-score (identical formula) | Yes |
| Zero-padding | 160 samples each side | 160 samples each side | Yes |
| Hidden-state filtering | 10 Hz Butterworth filtfilt (5th order) | None (see note below) | No |
| Frame selection | Middle frame | Second-to-last frame | No |

**Filtering note:** The Python pipeline applies a 5th-order Butterworth
low-pass filter (`scipy.signal.filtfilt`) to the WavLM hidden states before
linear projection. This JavaScript port omits filtering because `filtfilt`
requires scipy's IIR implementation, and approximate substitutes (Gaussian
forward-backward) produced worse results than no filtering at all.

**Frame selection note:** Python selects the middle frame of the sequence for
offline analysis. JavaScript selects the second-to-last frame for lower
latency in real-time use. Both are valid choices for different use cases.

## Accuracy: FP32 ONNX vs PyTorch

With the FP32 (unquantized) ONNX model, the JavaScript output is numerically
identical to PyTorch within floating-point rounding:

| Metric | Value |
|--------|-------|
| Average absolute EMA difference | 0.000003 |
| Maximum absolute EMA difference | 0.000005 |
| Mean hidden-state difference | 0.00003 |
| Maximum hidden-state difference | 0.001 |

### Per-feature comparison (second-to-last frame, `sample1.wav`)

| Feature | PyTorch | FP32 ONNX | Difference |
|---------|---------|-----------|------------|
| td_x | -0.2394 | -0.2394 | 0.000005 |
| td_y | -0.7032 | -0.7032 | 0.000000 |
| tb_x | -0.1720 | -0.1720 | 0.000005 |
| tb_y | -0.2772 | -0.2772 | 0.000004 |
| tt_x | -0.7770 | -0.7770 | 0.000003 |
| tt_y | 0.1401 | 0.1401 | 0.000005 |
| li_x | 0.6314 | 0.6314 | 0.000002 |
| li_y | 0.4658 | 0.4658 | 0.000000 |
| ul_x | 1.9987 | 1.9987 | 0.000005 |
| ul_y | -0.3835 | -0.3835 | 0.000002 |
| ll_x | 0.8680 | 0.8681 | 0.000001 |
| ll_y | -0.2026 | -0.2026 | 0.000001 |

## EMA value ranges

Statistics from the full 9.52 s utterance (475 frames at 50 Hz):

| Feature | Mean | Std | Min | Max | Range |
|---------|------|-----|-----|-----|-------|
| td_x | 0.08 | 1.07 | -2.30 | 2.99 | 5.29 |
| td_y | -0.41 | 0.94 | -2.37 | 2.89 | 5.26 |
| tb_x | 0.19 | 1.02 | -2.27 | 3.02 | 5.29 |
| tb_y | -0.33 | 0.99 | -3.37 | 2.51 | 5.89 |
| tt_x | 0.08 | 1.09 | -2.61 | 3.41 | 6.02 |
| tt_y | 0.07 | 1.08 | -3.09 | 2.90 | 5.99 |
| li_x | 0.08 | 1.25 | -3.28 | 2.94 | 6.22 |
| li_y | -0.01 | 1.18 | -4.01 | 2.12 | 6.12 |
| ul_x | -0.08 | 1.13 | -2.79 | 2.31 | 5.11 |
| ul_y | -0.12 | 0.96 | -2.60 | 2.06 | 4.66 |
| ll_x | -0.03 | 1.34 | -2.55 | 3.62 | 6.16 |
| ll_y | 0.07 | 1.22 | -3.66 | 2.99 | 6.65 |

Values are in a z-scored coordinate space derived from the MNGU0 EMA dataset.
Most values fall within approximately -3 to +3.

## Other feature differences

| Feature | Python | JavaScript |
|---------|--------|------------|
| Pitch detection | CREPE or PENN (neural) | YIN (algorithmic) |
| Loudness | Amplitude pooling | RMS to dB |

These features are auxiliary (not part of the core EMA pipeline) and are used
only for visualization.

## Display note

The accuracy tables above compare **raw model output** (z-scores). In the
browser UI, tongue and LI positions are derived from these z-scores via
per-articulator-group display scales. However, **lip vertical positions (UL/LL
y) are driven by F1** (first formant frequency, estimated via LPC), not the
model's `ul_y`/`ll_y` output, because the model's lip channels do not
differentiate vowels well enough for clinical use. See
[MNGU0_COORDINATES.md](MNGU0_COORDINATES.md) for details.
