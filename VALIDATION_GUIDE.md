# How to Validate SPARC Feature Extraction

Now that your system is running, here's how to confirm the microphone input is being correctly translated to articulatory features.

---

## Method 1: Real-Time Visual Validation (Easiest)

### Test Vowels

Speak these sounds clearly and watch the animation:

| Sound | Word Example | What to Look For |
|-------|--------------|------------------|
| **/i/** (ee) | "beet", "see" | • Tongue tip high & forward<br>• Small jaw opening<br>• Lips spread wide |
| **/a/** (ah) | "father", "spa" | • Tongue low & back<br>• **Large jaw opening**<br>• Neutral lips |
| **/u/** (oo) | "boot", "too" | • Tongue high & back<br>• Small jaw opening<br>• **Lips protruded** (forward) |
| **/ɛ/** (eh) | "bet", "pet" | • Tongue mid-front<br>• Medium jaw opening |
| **/o/** (oh) | "boat", "go" | • Tongue mid-back<br>• Lips slightly rounded |

**✅ Good signs:**
- Clear differences between vowels
- Smooth transitions
- Jaw opens wide for /a/, stays small for /i/ and /u/
- Lips move forward for /u/

**❌ Bad signs:**
- All vowels look the same
- Jerky/noisy movements
- Tongue stuck in one position
- No jaw movement

### Test Consonants

| Sound | Word Example | What to Look For |
|-------|--------------|------------------|
| **/t, d/** | "tea", "day" | • **Tongue tip** touches alveolar ridge (front top)<br>• Brief contact |
| **/k, g/** | "key", "go" | • **Tongue dorsum** (back) raises<br>• Contact at back of mouth |
| **/p, b/** | "pea", "bee" | • **Lips close** completely<br>• Brief closure |
| **/s, z/** | "sea", "zoo" | • Tongue tip near alveolar ridge<br>• Narrow channel |

### Test Sequences

Try these to see smooth transitions:

1. **"ee-ah-oo"** (`/i/-/a/-/u/`)
   - Tongue: front → down → back
   - Jaw: small → **wide** → small
   - Lips: spread → neutral → **round**

2. **"ta-ka-pa"** (`/t/-/k/-/p/`)
   - Contact: tongue tip → tongue back → lips

3. **"see-she"** (`/s/-/ʃ/`)
   - Tongue tip: forward → back

---

## Method 2: Debug Mode (Technical Validation)

Enable debug markers to see raw articulator positions:

1. **Check "Debug Mode"** in the UI
2. Colored dots appear on the vocal tract:
   - 🔴 Red (ul): Upper lip
   - 🔵 Blue (ll): Lower lip
   - 🟡 Yellow (li): Lip interface
   - 🟢 Green (tt): Tongue tip
   - 🟣 Purple (tb): Tongue body
   - 🟠 Orange (td): Tongue dorsum

3. **Watch the dots move** as you speak
4. **Open the console** (F12) to see raw coordinate values

### Expected Ranges (MNGU0 coordinate space)

```
Typical ranges for speech:
  ul_x: 0.6 to 1.0   (lips: back to front)
  ul_y: -1.2 to -0.8 (upper lip: down to up)
  
  ll_x: 0.6 to 1.0   
  ll_y: -0.8 to -0.2 (lower lip: down to up)
  
  tt_x: -0.5 to 0.9  (tongue tip: back to front)
  tt_y: -1.1 to -0.3 (tongue tip: down to up)
  
  td_x: -1.3 to 0.1  (tongue dorsum: back to front)
  td_y: -1.0 to -0.3 (tongue dorsum: down to up)
```

---

## Method 3: Quantitative Validation (Most Accurate)

Compare your live microphone input with Python SPARC output.

### Step 1: Record Test Audio

```bash
# Record a 5-second test sample
# macOS: Use QuickTime Player > File > New Audio Recording
# Save as test_recording.wav
```

### Step 2: Generate Python Ground Truth

```bash
cd tests
python validate_features.py path/to/test_recording.wav -o python_features.json
```

### Step 3: Compare with JavaScript

1. Open `tests/validate_js_features.html` in browser
2. Load `python_features.json`
3. Load the same `test_recording.wav`
4. Click **"Run Validation"**

### Step 4: Check Results

**✅ PASS Criteria:**
- Correlation > 0.9 for all 12 features
- RMSE < 0.5 for all features
- All features show green ✅

**Example good output:**
```
Feature    Python Mean  JS Mean    Diff    Status
ul_x       0.8234      0.8241     0.0007  ✅ (r=0.98)
ul_y      -0.9456     -0.9449     0.0007  ✅ (r=0.97)
tt_x       0.3421      0.3418     0.0003  ✅ (r=0.99)
...
```

---

## Method 4: Console Monitoring

Open browser DevTools (F12) and watch for these messages:

### Good Signs ✅

```
WORKER: Processing audio: 16000 samples, sensitivity: 1.0
WORKER: Applying 10Hz lowpass filter to WavLM features...
WORKER: Raw EMA output (no scaling): {
  ul: "(0.834, -0.956)"
  ll: "(0.841, -0.712)"
  tt: "(0.521, -0.743)"
  td: "(-0.234, -0.567)"
}
```

**What this means:**
- Features are being extracted every ~200ms
- Values are in reasonable ranges
- No NaN or Infinity errors

### Bad Signs ❌

```
❌ Invalid audio data received
❌ Processing error: Invalid coordinates for ul: (NaN, NaN)
❌ Worker response timeout
```

---

## Method 5: Feature History Charts

If you add chart visualization to the UI:

### Expected Patterns

**Silent periods:**
- Features should be relatively stable
- Small variation (breathing, rest position)
- Jaw slightly open

**Speech:**
- Clear movements correlated with sounds
- Smooth transitions between positions
- Larger variations during speech

**Noise:**
- Random, erratic movements
- No clear patterns
- May indicate poor feature extraction

---

## Quick Validation Checklist

Run through this checklist to confirm everything is working:

- [ ] **Vowel /i/ (ee)**: Tongue front & high, small jaw
- [ ] **Vowel /a/ (ah)**: Tongue low & back, **large jaw opening**
- [ ] **Vowel /u/ (oo)**: Tongue high & back, **lips forward**
- [ ] **/t/ sound**: **Tongue tip** goes to front/top
- [ ] **/k/ sound**: **Tongue back** raises
- [ ] **/p/ sound**: **Lips close**
- [ ] **Smooth transitions** between sounds
- [ ] **No frozen positions** (everything stuck)
- [ ] **No erratic jitter** (random noise)
- [ ] **Console shows** successful processing (no errors)
- [ ] **(Optional) Validation test passes** with correlation > 0.9

If all items are checked ✅, your feature extraction is working correctly!

---

## Common Issues and Fixes

### Issue: All articulators stay in one position

**Causes:**
- Sensitivity too low
- Microphone not working
- Worker not processing audio

**Fix:**
- Check browser console for errors
- Try increasing sensitivity slider
- Verify microphone access granted

### Issue: Movements are too jerky/noisy

**Causes:**
- Background noise
- Sensitivity too high
- Smoothing too low

**Fix:**
- Increase smoothing factor (0.4 → 0.6)
- Decrease sensitivity
- Use quieter environment

### Issue: Features don't match expected positions

**Causes:**
- Wrong model loaded (check if 768 vs 1024 error)
- Artificial scaling applied
- Filter not working

**Fix:**
- Verify console shows: `WavLM output shape: [1, XX, 1024]`
- Check sensitivity is ~1.0
- Look for "Applying 10Hz lowpass filter" message

---

## Need More Help?

1. **Enable verbose logging:**
   - Open `app.js`
   - Set debug counters visible
   - Watch feature update counts

2. **Compare with Python SPARC:**
   - Use the validation script in `tests/`
   - Compare frame-by-frame

3. **Test with known audio:**
   - Use samples from `Speech-Articulatory-Coding/sample_audio/`
   - These have been validated with the Python implementation

---

**TL;DR:** Say "ee-ah-oo" into the microphone. The tongue should move from front→down→back, jaw should open wide for "ah", and lips should round for "oo". If you see this, it's working! 🎉



