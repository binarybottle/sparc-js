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
4. **Visualization** — Six colored markers on an SVG grid, plus pitch,
   loudness, and jaw-opening bar displays.

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

## File structure

```
app.js                  Core: audio capture, worker management, feature loop
visualization.js        Visualization: SVG markers, demo animation, controls
sparc-worker.js         Web Worker: WavLM + linear model inference, YIN pitch
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

tests/
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
