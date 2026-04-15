# MNGU0 Articulatory Coordinate System

## Background

The SPARC linear model outputs EMA (Electromagnetic Articulography) features in the **MNGU0 z-scored space**. During training, each of the 12 EMA channels was independently z-scored within each utterance (Cho et al. 2024, §III-A). This means:

- Model output for channel *i* at time *t* is a z-score: how many standard deviations the articulator is from its mean position in that utterance.
- The absolute spatial relationships between articulators are removed by per-channel z-scoring.
- Typical output range is approximately [−3, +3] with standard deviation ≈ 1.0 per channel.
- Different articulators have very different physical ranges of motion (e.g., tongue tip ~3 mm/σ, upper lip ~1 mm/σ), so a z-score of 1.0 represents a different physical distance for each channel.

## MNGU0 Coordinate Frame

- **Origin**: Approximately at the upper incisors
- **X-axis**: Anterior–posterior; **+x = anterior** (toward lips)
- **Y-axis**: Superior–inferior; **+y = superior** (toward palate/up)
- **Units**: Millimeters (mm) in the original corpus; z-scores in model output

## Display Transform

There are two versions of the app:

- **Model version** (`index.html`): Tongue/LI positions come from the model's z-scores. Lip positions are F1-driven.
- **Formant version** (`formant.html`): All positions are formant-driven (F1 for lips, F1+F2 for tongue/LI). No ML model.

Both versions map z-scores to SVG coordinates using per-articulator center positions and **per-articulator-group scales**. Different groups use different scales because their z-scores represent different physical magnitudes.

### Articulator Centers (SVG coordinates at z-score = 0)

| Sensor | SVG center x | SVG center y | Anatomical position |
|--------|-------------|-------------|---------------------|
| TD     | −4.0        | −3.2        | Far back, up        |
| TB     | −2.0        | −2.5        | Mid-back, up        |
| TT     |  0.5        | −0.3        | Mid, mid-height     |
| LI     |  2.0        |  0.0        | Front, midline      |
| UL     |  3.0        | −1.25       | Frontmost (F1-driven y) |
| LL     |  3.0        | −1.25       | Frontmost (F1-driven y) |

### Per-articulator-group display scales

| Group | x scale | y scale | Notes |
|-------|---------|---------|-------|
| Tongue (TD, TB, TT) | 0.8 | 1.2 | Used for phonetic z-score → SVG mapping |
| Lower incisor (LI) | 0.5 | 1.0 | Used for phonetic z-score → SVG mapping |
| Lips (UL, LL) | 0.0 | 0.0 | Model z-scores not used; lip y is F1-driven |

### Mapping formula

```
svg_x = center_x + z_x × x_scale
svg_y = center_y − z_y × y_scale     (flipped: MNGU0 +y = up, SVG +y = down)
```

### Formant-driven articulator positioning

The model's raw EMA z-scores don't differentiate vowels well enough for
real-time clinical display. All articulator positions are instead driven by
**formant frequencies** estimated via LPC in the web worker.

#### F1-driven lips

F1 (first formant) correlates strongly with mouth opening:

- F1 ≈ 250 Hz → lips nearly closed (high vowels like /i/, /u/)
- F1 ≈ 650 Hz → mouth wide open (low vowels like /a/)
- Linear interpolation between, mapped to a symmetric UL/LL gap around `y = −1.25`

Lip x-positions are fixed at their centers (`x = 3.0`).

#### F1+F2-driven tongue and LI

F1 correlates with tongue height and F2 (second formant) with tongue
front-back position. Together they define the vowel space. Tongue and LI
z-scores are computed by **bilinear interpolation** between three corner
vowels in (F1, F2) space:

| Corner | F1 (Hz) | F2 (Hz) | Tongue shape |
|--------|---------|---------|--------------|
| /i/ (high-front) | 270 | 2300 | TB bunched high and front |
| /a/ (low-central) | 730 | 1090 | Tongue flat and low |
| /u/ (high-back) | 300 | 870 | TD raised high in back |

The interpolation normalizes F1 to a height parameter (0 = high, 1 = low)
and F2 to a frontness parameter (0 = back, 1 = front):

```
height   = clamp01((F1 − 250) / (650 − 250))
frontness = clamp01((F2 − 800) / (2400 − 800))
```

At height = 0, tongue z-scores interpolate between /u/ (front = 0) and /i/
(front = 1). At height = 1, tongue z-scores are fixed at /a/ regardless
of frontness. Between these extremes, linear interpolation applies. The
resulting z-scores are then mapped to SVG via the standard display scales
above.

## Reference Vowel Positions

Test sound markers use formant-driven positions. `VOWEL_Z_SCORES` in `visualization.js` stores both phonetically-motivated z-scores (used as corner-vowel anchors for the bilinear interpolation) and canonical F1/F2 values:

| Vowel | F1 (Hz) | F2 (Hz) | Tongue shape |
|-------|---------|---------|--------------|
| /i/   | 270     | 2300    | TB bunched high and front |
| /e/   | 530     | 1840    | Similar to /i/ but less extreme |
| /a/   | 730     | 1090    | Tongue flat and low |
| /o/   | 570     | 880     | Tongue body raised in back |
| /u/   | 300     | 870     | TD raised high in back |

The `tests/` directory contains earlier model-output extraction scripts, retained as offline analysis tools.

## Set References

The in-app "Set References" feature captures a speaker saying each vowel. From the captured audio, **speaker-specific F1 and F2** are saved (stored in `localStorage` under `sparc-reference-positions`). These formant values drive all articulator positions for reference markers: F1 for lips, F1+F2 for tongue/LI.

## Calibration

When calibrated, a per-speaker mean z-score is subtracted from the model output before the display transform, re-centering each articulator around its calibrated average. The anatomical layout (center positions) and scales remain fixed.

## References

- Richmond, K., Hoole, P., & King, S. (2011). Announcing the electromagnetic articulography (day 1) subset of the mngu0 articulatory corpus. *Interspeech 2011*.
- Cho, C.J., Wu, P., Prabhune, T.S., Agarwal, D., & Anumanchipalli, G.K. (2024). Coding Speech through Vocal Tract Kinematics. *IEEE JSTSP*.
- Peterson, G.E. & Barney, H.L. (1952). Control methods used in a study of the vowels. *JASA*.
