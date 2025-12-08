# Vocal Tract Visualization Improvements

**Date:** November 26, 2025  
**Status:** ✅ Complete

---

## Overview

The vocal tract visualization has been completely redesigned to show **realistic, smooth cross-sections** of the mouth, tongue, lips, and teeth based on validated SPARC articulatory features.

---

## Key Improvements

### 1. ✨ **Anatomically Accurate Shapes**

#### Tongue
- **Smooth cubic Bézier curves** create a natural tongue surface
- Three control points: Tip (tt) → Body (tb) → Dorsum (td) → Root (fixed)
- Realistic thickness with proper top and bottom surfaces
- Natural movement during speech

#### Lips
- **Upper and lower lips** with independent movement
- **Protrusion/rounding** - lips move forward for /u/ sounds
- **Spreading** - lips spread wide for /i/ sounds
- Smooth cubic Bézier curves for natural shape

#### Anatomical Context
- **Velum (soft palate)** - rounded soft tissue at back
- **Hard palate** - smooth arc of roof of mouth
- **Alveolar ridge** - bumpy ridge behind teeth
- **Pharyngeal wall** - back wall of throat
- **Upper & lower teeth** - prominent, move with jaw

### 2. 🎨 **Improved Color Scheme** (No Skin Tones)

```css
Tongue:    #ff9db3  (pink)
Upper Lip: #e88989  (light red-pink)
Lower Lip: #dd7777  (darker red-pink)
Palate:    #d4a5a5  (soft beige-pink)
Teeth:     #ffffff  (bright white)
Jaw:       #888     (neutral gray, dashed outline)
```

### 3. 📐 **Coordinate Normalization**

Features from the model come in the **±4 range** and are scaled to **±2** for the SVG viewBox:

```javascript
scale = 0.5;  // Convert from EMA ±4 to SVG ±2

// Applied to all articulators:
x_svg = x_ema * 0.5
y_svg = -y_ema * 0.5  // Also flip Y-axis
```

### 4. 🎬 **Smooth Transitions**

All moving elements have CSS transitions:
```css
transition: d 0.05s ease-out;
```

- **50ms** update rate (20fps minimum)
- **Ease-out timing** for natural deceleration
- **No jitter** on small movements

### 5. 🎯 **Updated Default Positions**

All default and vowel positions updated to match the **normalized ±4 coordinate space**:

- **Neutral position:** Tongue mid-position, lips relaxed
- **Vowel presets:** /i/, /e/, /a/, /o/, /u/ positions
- **Consistent with model output range**

---

## Technical Details

### SVG ViewBox
```html
viewBox="-2 -2 4 3"
```
- **X range:** -2 (back) to +2 (front)
- **Y range:** -2 (bottom) to +1 (top)
- **Origin:** Approximately at upper teeth

### Coordinate System

**EMA (Model Output):**
- X+: Forward (toward lips)
- X-: Backward (toward throat)
- Y+: Up (toward palate)
- Y-: Down (toward jaw)
- Range: ±4 units (normalized)

**SVG (Display):**
- X+: Right on screen
- X-: Left on screen
- Y+: Down on screen (⚠️ inverted!)
- Y-: Up on screen
- Range: ±2 units

**Mapping:**
```javascript
svg_x = ema_x * 0.5
svg_y = -ema_y * 0.5  // Negative for Y-flip
```

---

## File Changes

### `index.html`
- Updated CSS for `.tongue`, `.lips`, `.palate`, `.velum`, `.teeth`, `.jaw`
- Added smooth transitions for all dynamic elements
- Improved color scheme (no skin tones as requested)

### `app.js`

#### Functions Updated:
1. **`createStaticElements()`**
   - Added velum (soft palate)
   - Improved hard palate curve
   - Larger, more prominent teeth
   - Better pharyngeal wall

2. **`createTonguePath(tt, tb, td)`**
   - Smooth cubic Bézier curves
   - Coordinate normalization (scale × 0.5)
   - Natural thickness and shape
   - Anchored at pharyngeal wall

3. **`createLipPaths(ul, ll, li)`**
   - Natural protrusion/rounding
   - Spreading for smile sounds
   - Coordinate normalization
   - Separate upper/lower lip shapes

4. **`updateCharts()`**
   - Applied scale factor to marker positions
   - Consistent coordinate mapping

5. **`initializeDefaultPositions()`**
   - Updated to ±4 coordinate range
   - Matches model output scale

6. **`VOWEL_POSITIONS`**
   - All vowels updated to ±4 range
   - Anatomically accurate positions
   - Proper jaw openings

---

## How to Use

### 1. Start the Application

```bash
cd /Users/arno.klein/Software/sparc-js
python3 server.py
```

Open: `http://localhost:8000/index.html`

### 2. Test the Visualization

**Without Recording:**
- Click "Test Extreme Positions" to see vowel movements
- Watch smooth transitions between /i/, /a/, /u/, /e/, /o/

**With Recording:**
- Click "Start Recording"
- Speak vowel sounds clearly:
  - **/i/ (ee)** - tongue tip high, lips spread
  - **/a/ (ah)** - tongue low, jaw open wide
  - **/u/ (oo)** - tongue back raised, lips rounded
- Watch real-time vocal tract movements

### 3. Debug Mode

- ✅ **Check "Show articulator markers"** to see feature positions
- Markers appear as colored dots at tongue tip, body, dorsum, and lips
- Helps verify feature extraction accuracy

---

## Validation

The visualization uses **validated features** with:
- **Average difference:** 0.23 from Python SPARC
- **Coordinate range:** ±4 (normalized EMA space)
- **Update rate:** ~20 fps (50ms)
- **Smoothing:** Applied to reduce jitter

See `FEATURE_VALIDATION_REPORT.md` for full validation details.

---

## Visual Reference

### Anatomical Orientation

```
                BACK ← → FRONT
                
    ╔═══════ PALATE (roof) ═══════╗
    ║                              ║ TEETH (upper)
VELUM         TONGUE                 ║
    ║    /‾‾‾‾‾‾‾‾‾‾‾\              ║
    ║   /TD    TB    TT\───────────║ LIPS (upper)
PHARYNX  \_______________/          ║ 
    ║         (root)                ║ LIPS (lower)
    ║                               ║ TEETH (lower)
    ╚═════════ JAW ═════════════════╝
```

### Speech Sounds

| Sound | Tongue | Jaw | Lips |
|-------|--------|-----|------|
| /i/ (ee) | High front | Closed | Spread wide |
| /a/ (ah) | Low back | **Wide open** | Neutral |
| /u/ (oo) | High back | Closed | **Protruded** |
| /e/ (eh) | Mid front | Medium | Slight spread |
| /o/ (oh) | Mid back | Medium | Rounded |

---

## Future Enhancements

### Optional Additions:
1. **Airflow visualization** - show breath stream
2. **Constriction highlighting** - show narrow channels
3. **Slow-motion mode** - for detailed analysis
4. **Side-by-side comparison** - target vs. actual position
5. **Recording playback** - replay captured speech

### Performance Optimizations:
1. **Canvas rendering** instead of SVG (for very high frame rates)
2. **WebGL shaders** for smooth interpolation
3. **Optimized Butterworth filter** (if higher accuracy needed)

---

## Technical Notes

### Why Scale by 0.5?

The SPARC model outputs features in **normalized ±4 range** (from MNGU0 dataset). The SVG viewBox uses **±2 range** for optimal display size. Therefore:

```
scale = viewBox_range / model_range = 2 / 4 = 0.5
```

### Why Flip Y-Axis?

**EMA coordinates:** Y+ means "up" (toward palate)  
**SVG coordinates:** Y+ means "down" (screen coordinates)

To map correctly: `svg_y = -ema_y * scale`

### Why No Skin Tone?

As requested, the jaw uses a **neutral gray dashed outline** instead of filled skin color. This keeps focus on the articulators (tongue, lips) without suggesting race or ethnicity.

---

## Summary

✅ **Realistic cross-section** with smooth Bézier curves  
✅ **Anatomically accurate** shapes and movements  
✅ **Proper coordinate mapping** from model to display  
✅ **Smooth transitions** at 20fps  
✅ **No skin tones** - neutral color scheme  
✅ **Validated features** - matches Python SPARC within 0.23 units  

**The visualization is ready for production use!** 🎉

---

**Questions or Issues?**  
See `README.md` for general usage or `FEATURE_VALIDATION_REPORT.md` for validation details.



