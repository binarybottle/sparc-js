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

The display maps model z-scores to SVG coordinates using per-articulator center positions and **per-articulator-group scales**. Different groups use different scales because their z-scores represent different physical magnitudes.

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
| Tongue (TD, TB, TT) | 0.8 | 1.2 | Large scale — tongue has wide physical range |
| Lower incisor (LI) | 0.5 | 1.0 | Moderate — tracks jaw movement |
| Lips (UL, LL) | 0.0 | 0.0 | Model z-scores not used; lip y is F1-driven |

### Mapping formula (tongue and LI)

```
svg_x = center_x + z_x × x_scale
svg_y = center_y − z_y × y_scale     (flipped: MNGU0 +y = up, SVG +y = down)
```

### F1-driven lip positioning

The model's UL/LL z-scores don't differentiate vowels well enough for clinical use (the upper lip physically moves only ~1 mm, and the model captures very little variation). Instead, lip vertical separation is driven by **F1 (first formant frequency)**, estimated via LPC in the web worker:

- F1 ≈ 250 Hz → lips nearly closed (high vowels like /i/, /u/)
- F1 ≈ 650 Hz → mouth wide open (low vowels like /a/)
- Linear interpolation between, mapped to a symmetric UL/LL gap around `y = −1.25`

Lip x-positions are fixed at their centers (`x = 3.0`).

## Reference Vowel Positions

Test sound markers use **phonetically-motivated z-scores** (defined in `VOWEL_Z_SCORES` in `visualization.js`) that reflect known articulatory patterns from EMA literature:

| Vowel | Tongue shape | F1 (Hz) |
|-------|-------------|---------|
| /i/   | TB bunched high and front (TB is peak) | 270 |
| /e/   | Similar to /i/ but less extreme | 530 |
| /a/   | Tongue flat and low throughout | 730 |
| /o/   | Tongue body raised in back, moderate | 570 |
| /u/   | TD raised high in back (TD is peak) | 300 |

These are **not** derived from the SPARC model's output on synthetic audio. The `tests/` directory contains scripts and data from earlier model-output extraction, retained as offline analysis tools.

## Set References

The in-app "Set References" feature captures a speaker saying each vowel. From the captured audio, only the **speaker-specific F1** is saved (stored in `localStorage` under `sparc-reference-positions`). Tongue and LI reference positions always use the phonetically-motivated defaults, since the model's tongue channels don't reliably differentiate tongue shapes across vowels.

## Calibration

When calibrated, a per-speaker mean z-score is subtracted from the model output before the display transform, re-centering each articulator around its calibrated average. The anatomical layout (center positions) and scales remain fixed.

## References

- Richmond, K., Hoole, P., & King, S. (2011). Announcing the electromagnetic articulography (day 1) subset of the mngu0 articulatory corpus. *Interspeech 2011*.
- Cho, C.J., Wu, P., Prabhune, T.S., Agarwal, D., & Anumanchipalli, G.K. (2024). Coding Speech through Vocal Tract Kinematics. *IEEE JSTSP*.
- Peterson, G.E. & Barney, H.L. (1952). Control methods used in a study of the vowels. *JASA*.
