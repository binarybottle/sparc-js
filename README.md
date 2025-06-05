# sparc-js
JavaScript port of Speech Articulatory Coding (SPARC)

## Speech Articulatory Coding (JavaScript version)

This project provides a browser-based real-time, JavaScript implementation 
of the Berkeley Speech Group's software Speech Articulatory Coding (SPARC), 
using a 9-layer, quantized, ONNX version of the WavLM-base model. 

This implementation enables researchers, linguists, and speech technologists to visualize articulator movements during live speech in a web browser without requiring specialized software installation.

- https://github.com/binarybottle/sparc-js.git
- Author: Arno Klein (arnoklein.info)
- License: MIT License (see LICENSE)

### Background

- **Article**: 
  - C. J. Cho, P. Wu, T. S. Prabhune, D. Agarwal and G. K. Anumanchipalli, "Coding Speech Through Vocal Tract Kinematics," in IEEE Journal of Selected Topics in Signal Processing, vol. 18, no. 8, pp. 1427-1440, Dec. 2024, doi: 10.1109/JSTSP.2024.3497655. 
    - JSTSP: https://ieeexplore.ieee.org/document/10759573
    - arXiv: https://arxiv.org/abs/2406.12998
  
- **Original Python code**: 
  - https://github.com/Berkeley-Speech-Group/Speech-Articulatory-Coding
- **Original models**:
  - wavlm-large: https://huggingface.co/microsoft/wavlm-large
  - wavlm-base: https://huggingface.co/microsoft/wavlm-base
  - wavlm_large-9_cut-10_mngu_linear.pkl: https://huggingface.co/cheoljun95/Speech-Articulatory-Coding/tree/main

### What's Included

This JavaScript port implements the core feature extraction pipeline from the original Python codebase:

1. **Audio Capture**: Real-time microphone input processing using AudioWorklet
2. **WavLM Processing**: Running a quantized WavLM model via ONNX Runtime Web
3. **Linear Projection**: Converting WavLM features to articulatory coordinates
4. **Feature Extraction**:
   - Articulator positions (Upper Lip, Lower Lip, Lower Incisor, Tongue Tip, Tongue Blade, Tongue Dorsum)
   - Source features (pitch, loudness)
5. **Visualization**:
   - X coordinates over time (front/back movement)
   - Y coordinates over time (up/down movement)
   - 2D positional visualization with trace lines
   - Numerical displays of all feature values

### Processing Pipeline

Audio Recording (microphone)
↓
processAudioData() → Stores audio in circular buffer
↓
extractFeaturesLoop() → Main feature extraction loop
↓
extractWavLMFeatures() → WavLM model processing
↓
filterWavLMFeatures() → Apply Butterworth filtering
↓
extractArticulationFeatures() → Apply linear projection to get articulatory features
↓
extractPitch() / extractPitchSmoothed() → YIN algorithm for pitch detection
↓
calculateLoudness() → Compute audio loudness
↓
updateFeatureHistory() → Adds new values to history arrays (100 frames per feature)
↓
updateFeatureUI() → Updates text displays
↓
updateCharts() → Updates the visualization charts


### What's Not Ported from Python SPARC

This browser version focuses specifically on real-time feature extraction and visualization. The following components from the Python version are not included:

1. **Speech Synthesis**: The HiFiGAN generator for speech synthesis is not implemented
2. **Speaker Encoding**: The speaker embedding functionality is not included
3. **Voice Conversion**: The voice transformation capabilities are omitted
4. **External Package Dependencies**: No need for torchcrepe, librosa, or other Python dependencies

### Technical Implementation

- **Audio Processing**: Uses Web Audio API with AudioWorklet for efficient processing
- **Model Inference**: ONNX Runtime Web for running the WavLM model
- **Pitch Detection**: Custom implementation of the YIN algorithm
- **Signal Processing**: JavaScript implementation of Butterworth filters
- **Visualization**: Chart.js for interactive visualizations

### Requirements

- A modern web browser that supports:
  - Web Audio API with AudioWorklet
  - WebAssembly (for ONNX Runtime)
- Models directory containing:
  - `wavlm_base_layer9_quantized.onnx`: Truncated and quantized WavLM model
  - `wavlm_linear_model.json`: Linear projection weights converted from the original .pkl file

### Usage

1. Host the files on a web server (HTTPS recommended for microphone access)
2. Visit the page in a compatible browser
3. Click "Start Recording" to begin capturing and visualizing speech features in real-time

Run locally:
1. python3 server.py
2. Visit [http://localhost:8000/](http://localhost:8000/) in a compatible browser
3. Click "Start Recording" to begin capturing and visualizing speech features in real-time

