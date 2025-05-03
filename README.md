# sparc-js
Speech Articulatory Coding (JavaScript version)

These web demos are in preparation for extracting voice features
in real-time using a JavaScript version of Speech Articulatory Coding 
(SPARC), and a 9-layer, quantized, onnx version of the wavml-base model.

- Article: 
  - arXiv: https://arxiv.org/abs/2406.12998
  - JSTSP: https://ieeexplore.ieee.org/document/10759573
- Original Python code: 
  - https://github.com/Berkeley-Speech-Group/Speech-Articulatory-Coding
- Original models:
  - wavlm-large: https://huggingface.co/microsoft/wavlm-large
  - wavlm-base: https://huggingface.co/microsoft/wavlm-base
  - wavlm_large-9_cut-10_mngu_linear.pkl: 
    https://huggingface.co/cheoljun95/Speech-Articulatory-Coding/tree/main


extractFeaturesLoop()
    ↓
extractWavLMFeatures() → WavLM model processing
    ↓
extractArticulationFeatures() → Simulated articulatory features
    ↓
updateFeatureHistory() → Adds new values to history arrays (100 frames per feature)
    ↓
updateFeatureUI() → Updates text displays
    ↓
updateCharts() → Updates the visualization charts

In the current implementation, the articulation features are just simulated. 
For a real SPARC implementation:
  - Take the WavLM output tensor (shape [1, 49, 768])
  - Apply a linear transformation to map from 768 dimensions to the 12 EMA channels

