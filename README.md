# SPARC-JS

Browser-based real-time extraction and visualization of speech articulatory
features. This is a JavaScript port of the feature extraction pipeline from
[Speech-Articulatory-Coding](https://github.com/cheoljun95/Speech-Articulatory-Coding)
(SPARC).

## Pipeline

1. **Audio capture** — Microphone input via Web Audio API (`AudioWorklet`)
   at 16 kHz, with z-score normalization and 160-sample zero-padding.
2. **WavLM inference** — Layer 9 hidden states (1024-dim) from a
   full-precision (FP32) ONNX model, run via ONNX Runtime Web (WASM backend).
3. **Linear projection** — Learned linear map from 1024-dim hidden states to
   12 EMA (electromagnetic articulography) channels in the MNGU0 coordinate
   space: tongue dorsum, tongue body, tongue tip, lower incisor, upper lip,
   and lower lip (x, y each).
4. **F1 estimation** — LPC-based first formant frequency estimation (in the
   worker), used to drive lip vertical separation since the model's lip
   channels lack sufficient vowel differentiation.
5. **Visualization** — Six colored markers on an SVG grid representing
   tongue (TD, TB, TT), lower incisor (LI), and lips (UL, LL).

Tongue and LI positions are driven by the model's z-scores with
per-articulator-group display scales. Lip vertical positions (mouth
opening) are driven by F1, not the model's UL/LL output. See
[MNGU0_COORDINATES.md](MNGU0_COORDINATES.md) for the full display
transform.

The FP32 ONNX model produces output that is numerically identical to the
original PyTorch model (average difference < 0.00001). See
[COMPARISON.md](COMPARISON.md) for detailed accuracy data.

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
  captures speaker-specific F1 values and saves them to `localStorage` for
  persistent reference targets.
- **Calibrate** — Reads a passage aloud to collect per-speaker audio
  normalization statistics and per-articulator mean z-scores.
- **Test Sounds** — Displays phonetically-motivated reference positions for
  /i/, /e/, /a/, /o/, /u/ with correct tongue shapes and F1-driven lip gaps.
- **Live Recording** — Real-time marker positions from model inference
  (tongue/LI) and F1 estimation (lips).

## Sound type limitations

The visualization is designed for **vowel-focused** clinical assessment:

- **Vowels (/i/, /e/, /a/, /o/, /u/)** — Best supported. Reference tongue
  shapes come from articulatory phonetics; lip opening tracks F1 reliably.
- **Diphthongs (/aɪ/, /oʊ/, /aʊ/)** — F1 tracks the vowel-to-vowel
  transition smoothly. Tongue markers show movement but there are no
  diphthong-specific reference targets.
- **Consonants** — The model captures some tongue movement (e.g., for /t/,
  /k/, /s/), but there are no consonant reference targets. During voiceless
  consonants (/p/, /t/, /k/, /f/, /s/), F1 estimation returns zero (no
  voicing), driving lips toward a closed position. Nasals (/m/, /n/) produce
  a low F1 (~250 Hz), which also shows as closed — approximately correct.
- **Connected speech** — The SPARC model was trained on continuous speech, so
  tongue markers respond to running speech better than to isolated sustained
  sounds. Lip opening via F1 works well for any voiced segment.

## Latency

End-to-end latency from speech to marker update has four components:

| Source | Default | Notes |
|--------|---------|-------|
| Audio buffer | 500 ms | Audio accumulated before sending to worker (`bufferDuration`) |
| Extraction interval | 100 ms | Polling interval between extraction requests (`updateInterval`) |
| WavLM inference | 200–800 ms | ONNX model processing time (hardware-dependent, WASM backend) |
| Display smoothing | ~10 ms | Exponential smoothing on marker positions (`smoothingFactor`) |

**Typical perceived lag: ~0.6–1.2 seconds.** The largest factors are the
audio buffer and model inference time. To reduce latency further:

- Switch to the ONNX Runtime **WebGPU backend** (2–5x faster inference on
  machines with a GPU).
- Reduce `bufferDuration` below 0.5s (trades stability for speed; WavLM
  needs sufficient context for meaningful output).

## File structure

```
app.js                  Core: audio capture, worker management, display
                        transform, F1-to-lip mapping, calibration, Set References
visualization.js        Visualization: SVG markers, demo animation, vowel
                        reference positions, controls
sparc-worker.js         Web Worker: WavLM + linear model inference, LPC F1
                        estimation, YIN pitch detection
index.html              Main application page
validation.html         Standalone page for comparing JS output with Python
server.py               Local HTTP server with CORS headers

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

Open `http://localhost:8000` in a browser. The ~483 MB WavLM model will load
on first visit (cached by the browser afterward). Click **Start Recording**
and speak to see articulatory features in real time.

## Upstream references

- Paper: [Speech Articulatory Coding](https://arxiv.org/abs/2206.11394)
- Python code: https://github.com/cheoljun95/Speech-Articulatory-Coding
- WavLM: https://huggingface.co/microsoft/wavlm-large
- Linear model: https://huggingface.co/cheoljun95/Speech-Articulatory-Coding
