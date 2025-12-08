# MNGU0 Articulatory Coordinate System

## Background

The linear model outputs EMA (Electromagnetic Articulography) features in the **MNGU0 dataset coordinate space**. This is a real coordinate system from actual measurements of tongue and lip positions during speech.

## Coordinate System

- **Origin**: Approximately at the upper incisors (front teeth)
- **X-axis**: Horizontal, **positive = forward** (toward lips), **negative = backward** (toward throat)
- **Y-axis**: Vertical, **positive = up**, **negative = down**
- **Units**: Centimeters (cm)

## Typical Ranges (from MNGU0 dataset)

Based on the MNGU0 articulatory database, here are typical ranges during speech:

### Lips
```
Upper Lip (ul):
  x: 1.0 to 3.5 cm    (1-3.5cm forward from incisors)
  y: -2.0 to -0.5 cm  (0.5-2cm below incisor level)

Lower Lip (ll):
  x: 1.0 to 3.5 cm
  y: -4.0 to -1.5 cm  (below upper lip)

Lip Interface (li):
  x: 1.0 to 3.5 cm
  y: -3.0 to -1.0 cm  (between upper and lower lip)
```

### Tongue
```
Tongue Tip (tt):
  x: -1.0 to 3.0 cm   (can move from pharynx to lips)
  y: -4.0 to 0.0 cm   (can reach palate or drop low)

Tongue Body (tb):
  x: -2.0 to 1.0 cm   (middle of tongue)
  y: -5.0 to -1.0 cm

Tongue Dorsum (td):
  x: -4.0 to 0.0 cm   (back of tongue, always behind origin)
  y: -5.0 to -1.0 cm
```

## Visualization Mapping

To map MNGU0 coordinates to SVG visualization:

1. **Keep X as-is**: Forward (positive) → right, back (negative) → left
2. **Flip Y**: Down (negative) → down in SVG
3. **Scale to viewBox**: SVG viewBox is typically `-2, -2, 4, 3`

### Example Mapping:
```
MNGU0: ul_x=2.5, ul_y=-1.5
SVG:   x=2.5/4 → scaled, y=-1.5 → flipped

For viewBox="-2, -2, 4, 3":
- Center at (0, 0)
- X range: -2 to +2 (4 units wide)
- Y range: -2 to +1 (3 units tall)
```

## Common Speech Positions

### /i/ (ee)
- Tongue tip: forward and high (tt_x ~1-2, tt_y ~-1)
- Lips: spread (ul_x, ll_x ~2.5-3)
- Jaw: closed (small y-distance between lips)

### /a/ (ah)  
- Tongue: low and back (tt_x ~0, tt_y ~-4, td_x ~-3)
- Lips: neutral (ul_x, ll_x ~2)
- Jaw: open (large y-distance between lips)

### /u/ (oo)
- Tongue: high and back (td_x ~-2, td_y ~-2)
- Lips: rounded/protruded (ul_x, ll_x ~1-1.5)
- Jaw: closed

### /t/
- Tongue tip: touches alveolar ridge (tt_x ~0-1, tt_y ~-1 to 0)

### /k/
- Tongue dorsum: raises to velum (td_x ~-2 to -1, td_y ~-2)

## Debugging Checklist

If visualized positions don't make sense:

1. **Check raw model output**: Are values in expected ranges (cm scale)?
2. **Check coordinate system**: Is X forward/back correct? Is Y up/down correct?
3. **Check SVG mapping**: Are MNGU0 coords correctly mapped to SVG viewBox?
4. **Check constraints**: Are anatomical constraints too restrictive?
5. **Validate with Python**: Do Python and JS extract same values?

## Note on Scaling

The JavaScript implementation should **NOT** apply additional scaling to the linear model output. The model is trained on MNGU0 data and outputs in that coordinate space directly.

If you see values like `ul_x=0.8, tt_x=-0.5`, these are likely:
- **Wrong**: Values have been incorrectly normalized to [-1, 1] range
- **Should be**: Values in centimeters like `ul_x=2.5, tt_x=1.2`

The `sensitivityFactor` should be 1.0 to get raw MNGU0 coordinates.



