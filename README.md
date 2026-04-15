# SPARC-JS

Browser-based real-time extraction and visualization of speech articulatory
features. This is a JavaScript port of the feature extraction pipeline from
[Speech-Articulatory-Coding](https://github.com/cheoljun95/Speech-Articulatory-Coding)
(SPARC).

There are two versions of the app:

| | **Model version** (`index.html`) | **Formant version** (`formant.html`) |
|---|---|---|
| Lip positions | F1-driven | F1-driven |
| Tongue/LI positions | SPARC model z-scores | F1+F2 bilinear interpolation |
| Model download | ~483 MB WavLM ONNX | None |
| Startup time | 10–30 s | < 1 s |
| Per-frame latency | 200–800 ms (WavLM inference) | < 5 ms (LPC only) |
| Calibration | Yes (recenters model output) | No |

## Pipeline (model version)

1. **Audio capture** — Microphone input via Web Audio API (`AudioWorklet`)
   at 16 kHz, with z-score normalization and 160-sample zero-padding.
2. **WavLM inference** — Layer 9 hidden states (1024-dim) from a
   full-precision (FP32) ONNX model, run via ONNX Runtime Web (WASM backend).
3. **Linear projection** — Learned linear map from 1024-dim hidden states to
   12 EMA (electromagnetic articulography) channels in the MNGU0 coordinate
   space: tongue dorsum, tongue body, tongue tip, lower incisor, upper lip,
   and lower lip (x, y each).
4. **F1 estimation** — LPC-based first formant frequency estimation (in the
   worker), used to drive lip vertical separation.
5. **Visualization** — Six colored markers on an SVG grid. Tongue and LI
   positions come from the model's z-scores; lip vertical positions from F1.

The FP32 ONNX model produces output that is numerically identical to the
original PyTorch model (average difference < 0.00001). See
[COMPARISON.md](COMPARISON.md) for detailed accuracy data.

## Pipeline (formant version)

1. **Audio capture** — Same as model version (AudioWorklet at 16 kHz).
2. **Formant estimation** — LPC-based F1 and F2 estimation. F1 drives lip
   vertical separation; F1+F2 together drive tongue and LI positions via
   bilinear interpolation between corner vowels (/i/, /a/, /u/).
3. **Visualization** — Same six markers, same SVG grid.

No ML model is loaded. All positions are derived from acoustic features.
See [MNGU0_COORDINATES.md](MNGU0_COORDINATES.md) for the display transform.

## Deviations from the Python pipeline

- **No hidden-state filtering.** Python applies a 5th-order Butterworth
  low-pass filter (`scipy.signal.filtfilt`) to WavLM hidden states. This is
  omitted because `filtfilt` cannot be faithfully reproduced without scipy,
  and approximate substitutes worsened accuracy.
- **Pitch detection** uses YIN (Python uses CREPE or PENN).
- **Loudness** uses RMS-to-dB (Python uses amplitude pooling).
- **Frame selection** uses the second-to-last frame for lower latency
  (Python uses the middle frame for offline analysis).

## Features

- **Set References** — A normal speaker dictates individual vowels; the app
  captures speaker-specific formant values and saves them to `localStorage`
  for persistent reference targets. Model version captures F1; formant
  version captures F1 and F2.
- **Calibrate** (model version only) — Reads a passage aloud to collect
  per-speaker audio normalization statistics and per-articulator mean
  z-scores.
- **Test Sounds** — Displays reference positions for /i/, /e/, /a/, /o/, /u/.
- **Live Recording** — Real-time marker positions from the model (tongue)
  and F1 (lips) in the model version, or from F1+F2 (all articulators) in
  the formant version.

## Sound type limitations

The visualization is designed for **vowel-focused** clinical assessment:

- **Vowels (/i/, /e/, /a/, /o/, /u/)** — Best supported. Reference tongue
  shapes come from articulatory phonetics; lip opening tracks F1 reliably.
- **Diphthongs (/aɪ/, /oʊ/, /aʊ/)** — F1 tracks the vowel-to-vowel
  transition smoothly. Tongue markers show movement but there are no
  diphthong-specific reference targets.
- **Consonants** — During voiceless consonants (/p/, /t/, /k/, /f/, /s/),
  F1 estimation returns zero (no voicing), driving lips toward a closed
  position. Nasals (/m/, /n/) produce a low F1 (~250 Hz), which also shows
  as closed — approximately correct.
- **Connected speech** — The formant approach works well for sustained vowels.
  The model version may capture more coarticulation dynamics during running
  speech.

## Latency

### Model version

| Source | Default | Notes |
|--------|---------|-------|
| Audio buffer | 500 ms | `bufferDuration` |
| Extraction interval | 100 ms | `updateInterval` |
| WavLM inference | 200–800 ms | Hardware-dependent (WASM backend) |
| Display smoothing | ~10 ms | Exponential smoothing |

**Typical perceived lag: ~0.6–1.2 seconds.**

### Formant version

| Source | Default | Notes |
|--------|---------|-------|
| Audio buffer | 500 ms | `bufferDuration` |
| Extraction interval | 100 ms | `updateInterval` |
| LPC + YIN | < 5 ms | Very fast |
| Display smoothing | ~10 ms | Exponential smoothing |

**Typical perceived lag: ~0.5–0.6 seconds.** No model inference bottleneck.

## File structure

```
index.html              Model version: WavLM + F1 lips
formant.html            Formant version: F1 lips + F1/F2 tongue (no model)

app.js                  Model version: audio capture, worker management,
                        display transform, F1 lip mapping, calibration,
                        Set References
app-formant.js          Formant version: audio capture, formant-to-articulator
                        mapping, Set References (no model, no calibration)

sparc-worker.js         Model worker: WavLM + linear model inference,
                        LPC F1/F2 estimation, YIN pitch detection
formant-worker.js       Formant worker: LPC F1/F2 estimation, YIN pitch
                        detection (no ONNX, instant startup)

visualization.js        Shared: SVG markers, demo animation, vowel
                        reference positions, controls
server.py               Local HTTP server with CORS headers
validation.html         Standalone page for comparing JS output with Python

models/
  wavlm_large_layer9.onnx        WavLM Large layer 9 (FP32, ~483 MB)
  wavlm_linear_model.json        Linear projection weights (from pickle)

prep/                             Model conversion scripts
  convert_linear_model_pkl2json/  Pickle-to-JSON converter + source .pkl
  convert_pytorch2onnx/           PyTorch-to-ONNX export (Docker)
  convert_pytorch2onnx_truncate9layers_quantize.py
  convert_wavlm_large_to_onnx.py

tests/                            Offline analysis tools (not used at runtime)
  extract_vowel_zscores.py        Extract model z-scores from vowel audio
  vowel_zscores.json              Cached extraction output
  vowels/                         Synthetic vowel audio files
  validate_features.py            Python ground truth extraction
  python_features.json            Cached Python features for sample1.wav
  sample1.wav                     Test audio file
```

## Requirements

- A modern browser (Chrome, Firefox, Edge) with microphone access
- Python 3 (only for `server.py`; no Python packages needed at runtime)

## Usage

```bash
python3 server.py
```

- **Model version:** Open `http://localhost:8000` — loads the ~483 MB WavLM
  model (cached by the browser afterward).
- **Formant version:** Open `http://localhost:8000/formant.html` — no model
  download, ready instantly.

Click **Start Recording** and speak to see articulatory features in real time.

## Upstream references

- Paper: [Speech Articulatory Coding](https://arxiv.org/abs/2206.11394)
- Python code: https://github.com/cheoljun95/Speech-Articulatory-Coding
- WavLM: https://huggingface.co/microsoft/wavlm-large
- Linear model: https://huggingface.co/cheoljun95/Speech-Articulatory-Coding
