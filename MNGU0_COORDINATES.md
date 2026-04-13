# MNGU0 Articulatory Coordinate System

## Background

The SPARC linear model outputs EMA (Electromagnetic Articulography) features in the **MNGU0 z-scored space**. During training, each of the 12 EMA channels was independently z-scored within each utterance (Cho et al. 2024, §III-A). This means:

- Model output for channel *i* at time *t* is a z-score: how many standard deviations the articulator is from its mean position in that utterance.
- The absolute spatial relationships between articulators are removed by per-channel z-scoring.

To recover anatomically meaningful display coordinates, we **un-z-score** using estimated per-channel statistics from the MNGU0 corpus (Richmond et al. 2011).

## MNGU0 Coordinate Frame

- **Origin**: Approximately at the upper incisors
- **X-axis**: Anterior–posterior; **+x = anterior** (toward lips)
- **Y-axis**: Superior–inferior; **+y = superior** (toward palate/up)
- **Units**: Millimeters (mm)

## Per-Channel Statistics (estimated)

| Sensor | mean_x (mm) | mean_y (mm) | std_x (mm) | std_y (mm) |
|--------|-------------|-------------|------------|------------|
| UL     |   4         |   4         | 1.2        | 0.8        |
| LL     |   4         |  −3         | 1.5        | 2.0        |
| LI     |   0         | −10         | 1.0        | 2.0        |
| TT     |   0         |  −3         | 5.0        | 4.5        |
| TB     | −15         |   3         | 4.0        | 4.0        |
| TD     | −28         |   5         | 3.5        | 3.0        |

*Means* represent the average sensor position in mm during speech.
*Stds* represent typical within-utterance displacement (matching the per-utterance z-scoring used in training).

## Display Transform

Given a model z-score output `(z_x, z_y)` for articulator `key`:

```
mm_x = z_x × std_x + mean_x
mm_y = z_y × std_y + mean_y

svg_x = (mm_x + 35) / 45 × 9 − 5     // mm_x ∈ [−35, 10] → svg_x ∈ [−5, 4]
svg_y = (10 − mm_y) / 25 × 9 − 5      // mm_y ∈ [−15, 10] → svg_y ∈ [−5, 4] (flipped)
```

This maps the full vocal tract from tongue dorsum (back) to lips (front) onto the SVG viewport, with anatomically correct front/back and up/down orientation.

## Calibration

When calibrated, a per-speaker mean z-score is subtracted before un-z-scoring, re-centering each articulator around its calibrated average. The anatomical scale and offsets remain data-driven from the MNGU0 statistics.

## References

- Richmond, K., Hoole, P., & King, S. (2011). Announcing the electromagnetic articulography (day 1) subset of the mngu0 articulatory corpus. *Interspeech 2011*.
- Cho, C.J., Wu, P., Prabhune, T.S., Agarwal, D., & Anumanchipalli, G.K. (2024). Coding Speech through Vocal Tract Kinematics. *IEEE JSTSP*.
