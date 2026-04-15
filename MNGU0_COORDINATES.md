# MNGU0 Articulatory Coordinate System

## Background

The SPARC linear model outputs EMA (Electromagnetic Articulography) features in the **MNGU0 z-scored space**. During training, each of the 12 EMA channels was independently z-scored within each utterance (Cho et al. 2024, §III-A). This means:

- Model output for channel *i* at time *t* is a z-score: how many standard deviations the articulator is from its mean position in that utterance.
- The absolute spatial relationships between articulators are removed by per-channel z-scoring.
- Typical output range is approximately [−3, +3] with standard deviation ≈ 1.0 per channel.

## MNGU0 Coordinate Frame

- **Origin**: Approximately at the upper incisors
- **X-axis**: Anterior–posterior; **+x = anterior** (toward lips)
- **Y-axis**: Superior–inferior; **+y = superior** (toward palate/up)
- **Units**: Millimeters (mm) in the original corpus; z-scores in model output

## Display Transform

The display maps model z-scores to SVG coordinates using anatomical center positions and **separate x/y scales**. The x-scale is kept moderate to prevent the tongue from crossing in front of the lips (an artifact of per-channel z-scoring removing physical scale differences). The y-scale is larger to make vertical differences (lip opening, tongue height) clearly visible.

### Articulator Centers (SVG coordinates at z-score = 0)

| Sensor | SVG center x | SVG center y | Anatomical position |
|--------|-------------|-------------|---------------------|
| TD     | −4.0        | −3.2        | Far back, up        |
| TB     | −2.0        | −2.5        | Mid-back, up        |
| TT     |  0.5        | −0.3        | Mid, mid-height     |
| LI     |  1.0        |  2.2        | Mid-front, low      |
| UL     |  3.0        | −2.5        | Frontmost, up       |
| LL     |  3.0        |  0.0        | Frontmost, mid-low  |

### Mapping formula

```
svg_x = center_x + z_x × x_scale
svg_y = center_y − z_y × y_scale     (flipped: MNGU0 +y = up, SVG +y = down)
```

Where:
- `x_scale = 0.3` SVG units per z-score (moderate, preserves anterior-posterior anatomy)
- `y_scale = 0.8` SVG units per z-score (larger, maximizes visibility of vertical differences)

### Why separate scales?

Different articulators have vastly different physical ranges of motion (e.g., tongue tip moves ~5mm anterior-posteriorly while upper lip moves ~1.2mm). Z-scoring normalizes each channel independently, removing these physical scale differences. A single uniform scale would cause the tongue tip (which gets large x z-scores) to appear in front of the lips (which get small x z-scores). Separate scales restore anatomical plausibility while keeping vertical differences clearly visible.

### Anatomical constraint verification

With x_scale=0.3 and the chosen centers, even at extreme real-speech z-scores (TT_x=+3.4, UL_x=−2.8), the tongue tip stays behind the lips:
- TT_x max: 0.5 + 3.4 × 0.3 = 1.52
- UL_x min: 3.0 + (−2.8) × 0.3 = 2.16

## Measured Vowel Z-Scores

Z-scores were extracted from the ONNX pipeline on synthesized vowel audio (`tests/extract_vowel_zscores.py`). These serve as default positions; the in-app "Set References" feature captures more accurate positions from a real speaker.

Full measured values are in `tests/vowel_zscores.json`.

## Calibration

When calibrated, a per-speaker mean z-score is subtracted before the display transform, re-centering each articulator around its calibrated average. The anatomical layout (center positions) and scales remain fixed.

## References

- Richmond, K., Hoole, P., & King, S. (2011). Announcing the electromagnetic articulography (day 1) subset of the mngu0 articulatory corpus. *Interspeech 2011*.
- Cho, C.J., Wu, P., Prabhune, T.S., Agarwal, D., & Anumanchipalli, G.K. (2024). Coding Speech through Vocal Tract Kinematics. *IEEE JSTSP*.
