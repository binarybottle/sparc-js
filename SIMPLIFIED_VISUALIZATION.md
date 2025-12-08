# Simplified Visualization - Fresh Start

**Date:** November 26, 2025  
**Status:** ✅ Ready to test

---

## What Changed

I've **completely simplified** the visualization to start from scratch and understand the coordinate system properly.

### New Approach

Instead of trying to draw complex tongue/lip shapes, the visualization now shows:

1. **Reference Grid** - Shows the coordinate system clearly
2. **Origin Marker** - Red crosshair at (0,0) = upper teeth level  
3. **6 Colored Dots** - One for each articulator with labels
4. **Simple Palate** - Basic arc above to show mouth structure
5. **Upper Teeth** - Rectangle at origin

**NO complex curves, NO jaw animation, NO lip shapes yet** - just the raw data points!

---

## Coordinate System

### ViewBox: `"-2 -1 4 3"`
- **X axis:** -2 (back/pharynx) to +2 (front/lips)
- **Y axis:** -1 (above teeth) to +2 (below teeth/jaw)
- **Origin (0,0):** Upper teeth level

### EMA → SVG Mapping
- **X:** Direct mapping (EMA_X = SVG_X)
- **Y:** **Flip only** (EMA_Y → -EMA_Y for SVG)
  - EMA: Y+ = up, Y- = down
  - SVG: Y+ = down (screen), Y- = up

### No Scaling!
The features come in the right range already (~±2), so we don't scale anything.

---

## The 6 Articulators

Based on actual validation data:

| Articulator | Color | Label | EMA Coords | SVG Coords (Y-flipped) |
|-------------|-------|-------|------------|------------------------|
| Upper Lip (ul) | 🔴 Red | UL | (0.35, -0.45) | (0.35, 0.45) |
| Lower Lip (ll) | 🔵 Blue | LL | (0.41, -1.36) | (0.41, 1.36) |
| Lip Interface (li) | 🟡 Yellow | LI | (0.71, -1.31) | (0.71, 1.31) |
| Tongue Tip (tt) | 🟢 Green | TT | (0.01, -1.13) | (0.01, 1.13) |
| Tongue Body (tb) | 🟣 Purple | TB | (-1.49, 0.06) | (-1.49, -0.06) |
| Tongue Dorsum (td) | 🟠 Orange | TD | (-0.55, -0.56) | (-0.55, 0.56) |

---

## What to Look For

### ✅ Expected Layout (Static)

```
        Y = -1 (TOP)
┌─────────────────────────────────┐
│                                 │
│  [Purple TB]                    │  ← Tongue body way back, ABOVE origin
│                                 │
│  PALATE ~~~~~~~~~~~~~~~~~~~~~~~~│
│             [Red +]  □          │  ← Origin & Upper teeth
│                                 │
│  [Orange TD]                    │  ← Tongue dorsum
│  [Red UL]                       │  ← Upper lip  
│         [Green TT]              │  ← Tongue tip
│  [Yellow LI]                    │  ← Lip interface
│  [Blue LL]                      │  ← Lower lip
│                                 │
└─────────────────────────────────┘
        Y = +2 (BOTTOM)
```

### ✅ Expected Relationships

1. **UL and LL vertically aligned** - Both around X ≈ 0.4
2. **TB far to the left** - X ≈ -1.5 (back of tongue)
3. **TB ABOVE origin** - Y ≈ -0.06 in SVG (negative = up!)
4. **TT near center** - X ≈ 0
5. **TD between TB and TT** - X ≈ -0.6
6. **LL below UL** - UL at Y≈0.45, LL at Y≈1.36

### ✅ During Speech (Real-time)

- **Markers should move smoothly**
- **TB can move UP and DOWN** (arching tongue)
- **UL and LL should stay roughly aligned** (same X)
- **Markers should NOT jump all over** - movements should be small

---

## Testing Steps

### 1. Load the Page

```
http://localhost:8000/index.html
```

### 2. Observe Static State

Before clicking anything:
- Do you see the reference grid?
- Do you see 6 colored dots with labels?
- Is TB (purple) on the LEFT side, slightly ABOVE the origin line?
- Is LL (blue) below UL (red)?

### 3. Test Extreme Positions

Click **"Test Extreme Positions"** button:
- Watch the dots move through vowel positions
- Do they stay within reasonable bounds?
- Does TB move up/down (tongue arching)?

### 4. Test Real-Time

Click **"Start Recording"** and speak:
- **/i/ (ee)** - TB should move UP (tongue high)
- **/a/ (ah)** - LL should move DOWN (jaw open), tongue LOW
- **/u/ (oo)** - Lips should move FORWARD (X increases)

---

## Console Logging

Every 50 frames, the console will log current positions:

```javascript
Current positions: {
  ul: "(0.35, -0.45)",
  ll: "(0.41, -1.36)",
  tt: "(0.01, -1.13)",
  tb: "(-1.49, 0.06)",  // NOTE: positive Y = tongue reaching UP
  td: "(-0.55, -0.56)"
}
```

Watch these values - they should:
- Stay within ±2 range
- Change smoothly (not jump wildly)
- Make anatomical sense

---

## Known Behaviors (CORRECT)

### Tongue Body Above Tongue Tip
**This is CORRECT!** The tongue arches:
- For /i/ sounds, the tongue body rises toward the palate
- This makes TB have a LESS NEGATIVE (or even positive) Y value
- In SVG, this puts TB higher on screen (smaller/negative Y)

### Markers Close Together
Some markers will overlap - that's OK!
- UL, LL, LI are all near the lips (X ≈ 0.4-0.7)
- TT, TD are in the tongue region
- TB is far back in the throat

---

## Next Steps (After Validation)

Once we confirm the 6 dots are positioned correctly:

1. **Add simple tongue shape** - Connect TT → TB → TD with smooth curve
2. **Add lip shapes** - Ellipses at UL/LL positions
3. **Add jaw line** - Below LL
4. **Add lower teeth** - Attached to jaw
5. **Refine aesthetics** - Colors, shadows, etc.

But **FIRST** - let's make sure the raw coordinates make sense!

---

## Troubleshooting

### If dots are scattered randomly:
- Check console for position logs
- Verify Y-flip is working (TB should have NEGATIVE SVG Y)

### If TB is at the bottom:
- Y-flip might be wrong
- Should be: `svgY = -emaY`

### If nothing moves:
- Check if worker is processing audio
- Check console for "Current positions" logs

---

## Files Changed

- `app.js`:
  - `setupVocalTractVisualization()` - New viewBox, simpler setup
  - `createReferenceGrid()` - NEW: Shows coordinate system
  - `createSimpleStaticElements()` - Minimal static elements
  - `createSimpleDynamicElements()` - Just 6 circles
  - `updateCharts()` - Simplified to just move circles
  - `initializeDefaultPositions()` - Real validation data

- `index.html`:
  - Updated CSS for `.articulator-marker` and `.articulator-label`
  - Removed complex shape styling

---

**Ready to test!** 🔬

Open `http://localhost:8000/index.html` and let's verify the basics work before adding complexity.



