/******************************************************************************
 * SPARC Feature Extraction - Web Worker
 * 
 * This worker thread handles the computationally intensive parts of SPARC:
 * - WavLM model inference
 * - Linear projection for articulation features
 * - Pitch detection using YIN algorithm
 * - Audio feature filtering and processing
 * 
 * Part of the Speech Articulatory Coding (SPARC) system that provides 
 * real-time visualization of speech articulatory features from microphone input.
******************************************************************************/

// Load ONNX runtime in the worker
importScripts('https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js');

// Worker state
let wavlmSession = null;
let linearModel = null;
let linearModelWorkingMemory = null;
let initialized = false;

// Pitch detection state
let pitchHistory = Array(5).fill(0);
let featuresFilterBank = null;

// Handle messages from main thread
self.onmessage = async function(e) {
  const message = e.data;
  
  switch(message.type) {
    case 'init':
      await initializeModels(message.onnxPath, message.linearModelPath);
      break;
      
    case 'process':
      if (!initialized) {
        self.postMessage({ type: 'error', error: 'Worker not initialized' });
        return;
      }
      
      // Process the audio data
      const audioData = new Float32Array(message.audio);
      const config = message.config;
      
      try {
        // Extract features
        const wavlmOutput = await extractWavLMFeatures(audioData, wavlmSession);
        const articulationFeatures = extractArticulationFeatures(wavlmOutput);
        
        if (!articulationFeatures) {
          self.postMessage({ 
            type: 'error', 
            error: 'Failed to extract articulation features' 
          });
          return;
        }
        
        // Extract pitch
        let pitch = 0;
        if (config.extractPitchFn === 1) {
          pitch = extractPitch(audioData);
        } else if (config.extractPitchFn === 2) {
          pitch = extractPitchSmoothed(audioData);
        }
        
        // Calculate loudness
        const loudness = calculateLoudness(audioData);
        
        // Send results back to main thread
        self.postMessage({
          type: 'features',
          articulationFeatures: articulationFeatures,
          pitch: pitch,
          loudness: loudness
        });
      }
      catch (error) {
        self.postMessage({ 
          type: 'error', 
          error: 'Processing error: ' + error.message 
        });
      }
      break;
  }
};

// Initialize the models
async function initializeModels(onnxPath, linearModelPath) {
  try {
    // Configure ONNX Runtime environment BEFORE creating session
    self.ort.env.wasm.numThreads = 1;
    self.ort.env.wasm.simd = false;
    
    // Set WASM paths to CDN
    self.ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';
    
    // Alternative: Set individual paths
    /*
    self.ort.env.wasm.wasmPaths = {
      'ort-wasm.wasm': 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort-wasm.wasm',
      'ort-wasm-simd.wasm': 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort-wasm-simd.wasm'
    };
    */
    
    const options = {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'basic',
      enableCpuMemArena: false,
      executionMode: 'sequential'
    };
    
    // Load the WavLM model
    self.postMessage({ type: 'status', message: 'Loading WavLM model...' });
    wavlmSession = await ort.InferenceSession.create(onnxPath, options);
    
    // Load the linear model
    self.postMessage({ type: 'status', message: 'Loading linear projection model...' });
    const response = await fetch(linearModelPath);
    const modelData = await response.json();
    
    linearModel = {
      weights: modelData.weights.map(w => new Float32Array(w)),
      biases: new Float32Array(modelData.biases),
      inputDim: modelData.input_dim,
      outputDim: modelData.output_dim,
      metadata: modelData.metadata
    };
    
    linearModelWorkingMemory = new Float32Array(linearModel.outputDim);
    
    initialized = true;
    self.postMessage({ type: 'initialized' });
  }
  catch (error) {
    self.postMessage({ 
      type: 'error', 
      error: 'Initialization error: ' + error.message
    });
  }
}

/******************************************************************************
* FEATURE EXTRACTION *
******************************************************************************/

// Calculate audio loudness
function calculateLoudness(audioData) {
  // Calculate RMS loudness
  let sum = 0;
  for (let i = 0; i < audioData.length; i++) {
    sum += audioData[i] * audioData[i];
  }
  const rms = Math.sqrt(sum / audioData.length);
  const dbFS = 20 * Math.log10(rms);
  
  return dbFS;
}

// ----- WavLM & Linear Projection ----- 

// Process audio through WavLM model
async function extractWavLMFeatures(audioData, session) {
  try {
    // Ensure audioData is a Float32Array of the right length
    const inputLength = 16000; // 1 second at 16kHz
    
    // Create a properly sized array
    const inputData = new Float32Array(inputLength);
    
    // Copy available data (with zero-padding if needed)
    const copyLength = Math.min(audioData.length, inputLength);
    for (let i = 0; i < copyLength; i++) {
      inputData[i] = audioData[i];
    }
    
    // Create tensor with the shape the model expects
    const inputTensor = new ort.Tensor('float32', inputData, [1, inputLength]);
    
    // Run inference
    const feeds = {};
    feeds[session.inputNames[0]] = inputTensor;
    
    const outputData = await session.run(feeds);
    
    // Extract the output tensor
    let output = outputData[session.outputNames[0]];
    
    // Apply Butterworth filtering (10Hz cutoff as in the Python version)
    output = filterWavLMFeatures(output);
    
    return output;
  } catch (error) {
    console.error("Error in WavLM feature extraction:", error);
    throw error;
  }
}
  
// Extract articulation features from WavLM output
function extractArticulationFeatures(wavlmFeatures) {
  try {
    if (!linearModel) {
      console.error("Linear model not loaded");
      return null;
    }
    // Extract the output tensor data from ONNX model output
    const features = wavlmFeatures.data;
    const dims = wavlmFeatures.dims;

    // WavLM outputs [batch_size, sequence_length, hidden_size]
    // Take the middle frame as the current representation
    const batchSize = dims[0];  // Should be 1 for real-time
    const seqLength = dims[1];  // Number of frames
    const hiddenSize = dims[2]; // 768 for WavLM base

    // Take the middle frame for real-time feedback
    const middleFrameIdx = Math.floor(seqLength / 2);
    const startIdx = middleFrameIdx * hiddenSize;
    
    // Apply the linear model with optimized matrix multiplication
    // Compute y = Wx + b where W is weights, x is features, b is bias
    const output = linearModelWorkingMemory;
    
    // First set output to bias values
    output.set(linearModel.biases);
    
    // Then add the weighted input features
    for (let i = 0; i < linearModel.outputDim; i++) {
      const weights = linearModel.weights[i];
      for (let j = 0; j < hiddenSize; j++) {
        output[i] += weights[j] * features[startIdx + j];
      }
    }
    
    // Map the output to articulators
    const articulationFeatures = {
      ul: {x: output[0], y: output[1]},
      ll: {x: output[2], y: output[3]},
      li: {x: output[4], y: output[5]},
      tt: {x: output[6], y: output[7]},
      tb: {x: output[8], y: output[9]},
      td: {x: output[10], y: output[11]}
    };

    // Apply scaling and offsets to map to SVG coordinate space
    const scaleFactorX = 1.0;
    const scaleFactorY = 1.0;
    const offsetX = 0.0;      
    const offsetY = 0.0;      

    for (const key in articulationFeatures) {
      articulationFeatures[key].x = articulationFeatures[key].x * scaleFactorX + offsetX;
      articulationFeatures[key].y = articulationFeatures[key].y * scaleFactorY + offsetY;
    }

    return articulationFeatures;
  } catch (error) {
    console.error("Error in articulation feature extraction:", error);
    return null;
  }
}
  
// ----- Pitch Detection ----- 
  
// YIN pitch detection implementation (based on algorithm by de Cheveigné and Kawahara)
class YINPitchDetector {
  constructor(options = {}) {
    this.sampleRate = options.sampleRate || 16000;
    this.threshold = options.threshold || 0.15;
    this.minFrequency = options.minFrequency || 70;
    this.maxFrequency = options.maxFrequency || 400;
    
    // Pre-calculate some values based on our constraints
    this.minPeriod = Math.floor(this.sampleRate / this.maxFrequency);
    this.maxPeriod = Math.floor(this.sampleRate / this.minFrequency);
  }
  
  detect(audioBuffer) {
    // Ensure we have a Float32Array
    const buffer = audioBuffer instanceof Float32Array ? audioBuffer : new Float32Array(audioBuffer);
    
    // Use a reasonable window size for speech
    const bufferSize = Math.min(buffer.length, 2048);
    
    // Calculate difference function (step 1 & 2 of YIN algorithm)
    const yinBuffer = new Float32Array(bufferSize / 2);
    
    // Step 1: Autocorrelation method
    for (let tau = 0; tau < yinBuffer.length; tau++) {
      yinBuffer[tau] = 0;
      
      for (let j = 0; j < yinBuffer.length; j++) {
        const delta = buffer[j] - buffer[j + tau];
        yinBuffer[tau] += delta * delta;
      }
    }
    
    // Step 2: Cumulative mean normalized difference
    yinBuffer[0] = 1;
    let runningSum = 0;
    for (let tau = 1; tau < yinBuffer.length; tau++) {
      runningSum += yinBuffer[tau];
      if (runningSum === 0) {
        yinBuffer[tau] = 1;
      } else {
        yinBuffer[tau] *= tau / runningSum;
      }
    }
    
    // Step 3: Find the first minimum below the threshold
    let minTau = 0;
    let minVal = 1000; // Initialize with a value higher than possible values
    
    // Only look for minima between our min and max period
    for (let tau = this.minPeriod; tau <= this.maxPeriod && tau < yinBuffer.length; tau++) {
      if (yinBuffer[tau] < minVal) {
        minVal = yinBuffer[tau];
        minTau = tau;
      }
      
      if (minVal < this.threshold) {
        // Found a minimum below threshold
        
        // Step 4: Interpolate for better accuracy
        const exactTau = this.parabolicInterpolation(yinBuffer, minTau);
        
        // Convert period to frequency
        const frequency = this.sampleRate / exactTau;
        
        return frequency;
      }
    }
    
    // If no minimum found below threshold, check the lowest value
    if (minTau > 0) {
      // Convert to frequency
      const frequency = this.sampleRate / minTau;
      
      // Only return if confidence is reasonable
      if (minVal < 0.3) {
        return frequency;
      }
    }
    
    // No pitch detected
    return 0;
  }

  parabolicInterpolation(array, position) {
    // Handle edge cases
    if (position === 0 || position === array.length - 1) {
      return position;
    }
    
    // Quadratic interpolation using the point and its neighbors
    const x1 = position - 1;
    const x2 = position;
    const x3 = position + 1;
    
    const y1 = array[x1];
    const y2 = array[x2];
    const y3 = array[x3];
    
    // Fit a parabola: y = a*x^2 + b*x + c
    const a = (y3 + y1 - 2 * y2) / 2;
    const b = (y3 - y1) / 2;
    
    if (a === 0) {
      // If a is zero, the parabola is actually a line
      return position;
    }
    
    // The minimum of the parabola is at: x = -b / (2*a)
    const exactPosition = x2 - b / (2 * a);
    
    return exactPosition;
  }
}
  
// Function to extract pitch using our YIN implementation
function extractPitch(audioData) {
  try {
    // Create detector if it doesn't exist yet
    if (!self.yinDetector) {
      self.yinDetector = new YINPitchDetector({
        sampleRate: 16000, // Use hardcoded value or access from config
        threshold: 0.15,
        minFrequency: 70,
        maxFrequency: 400
      });
    }
    
    // Extract a smaller window from the audio buffer for efficiency
    const bufferSize = Math.min(audioData.length, 2048);
    const startIdx = Math.floor((audioData.length - bufferSize) / 2);
    const audioSlice = audioData.slice(startIdx, startIdx + bufferSize);
    
    // Detect pitch
    const pitch = self.yinDetector.detect(audioSlice);
    
    return pitch || 0;
  } catch (error) {
    console.error("Pitch detection error:", error);
    return 0;
  }
}
  
// Smoothed version with history for stable output
function extractPitchSmoothed(audioData) {
  const rawPitch = extractPitch(audioData);
  
  // Update history
  pitchHistory.push(rawPitch);
  pitchHistory.shift();
  
  // Filter out zeros for median calculation
  const nonZeroPitches = pitchHistory.filter(p => p > 0);
  
  if (nonZeroPitches.length === 0) {
    return 0;
  }
  
  // Use median for smoothing (more robust than average)
  const sortedPitches = [...nonZeroPitches].sort((a, b) => a - b);
  const medianPitch = sortedPitches[Math.floor(sortedPitches.length / 2)];
  
  return medianPitch;
}
  
// ----- Filtering ----- 
  
// Butterworth filter implementation with pre-computed coefficients for a 10Hz cutoff at 50Hz sample rate
class LowpassFilter {
  constructor() {
    // Pre-computed coefficients for a 5th order Butterworth low-pass filter
    // with 10Hz cutoff at 50Hz sample rate
    this.b = [0.0008, 0.0039, 0.0078, 0.0078, 0.0039, 0.0008];
    this.a = [1.0000, -3.0756, 3.8289, -2.3954, 0.7475, -0.0930];
    
    // Initialize delay lines
    this.x_history = new Float32Array(this.b.length).fill(0);
    this.y_history = new Float32Array(this.a.length-1).fill(0);
  }
  
  // Process a single sample
  processSample(x) {
    // Shift x history
    for (let i = this.x_history.length - 1; i > 0; i--) {
      this.x_history[i] = this.x_history[i-1];
    }
    this.x_history[0] = x;
    
    // Apply filter
    let y = 0;
    for (let i = 0; i < this.b.length; i++) {
      y += this.b[i] * this.x_history[i];
    }
    
    for (let i = 0; i < this.y_history.length; i++) {
      y -= this.a[i+1] * this.y_history[i];
    }
    
    // Shift y history
    for (let i = this.y_history.length - 1; i > 0; i--) {
      this.y_history[i] = this.y_history[i-1];
    }
    this.y_history[0] = y;
    
    return y;
  }
  
  // Process an array of samples
  process(inputArray) {
    const outputArray = new Float32Array(inputArray.length);
    for (let i = 0; i < inputArray.length; i++) {
      outputArray[i] = this.processSample(inputArray[i]);
    }
    return outputArray;
  }
  
  // Reset the filter state
  reset() {
    this.x_history.fill(0);
    this.y_history.fill(0);
  }
}
  
// For filtering WavLM feature dimensions
function createFilterBank(numFilters) {
  const filters = [];
  for (let i = 0; i < numFilters; i++) {
    filters.push(new LowpassFilter());
  }
  return filters;
}
  
// Filter WavLM features in the time dimension
function filterWavLMFeatures(wavlmFeatures) {
  const dims = wavlmFeatures.dims;
  const data = wavlmFeatures.data;
  const batchSize = dims[0];
  const seqLength = dims[1];
  const hiddenSize = dims[2];
  
  // Only create filters when needed
  if (!self.featuresFilterBank || self.featuresFilterBank.length !== hiddenSize) {
    self.featuresFilterBank = createFilterBank(hiddenSize);
  }
  
  // Apply filtering to each feature dimension across time
  const filteredData = new Float32Array(data.length);
  
  for (let h = 0; h < hiddenSize; h++) {
    // Extract this feature dimension across all time steps for the first batch item
    const featureTimeSeries = new Float32Array(seqLength);
    for (let t = 0; t < seqLength; t++) {
      const idx = t * hiddenSize + h; // Assuming batch size 1
      featureTimeSeries[t] = data[idx];
    }
    
    // Apply filter
    const filteredTimeSeries = self.featuresFilterBank[h].process(featureTimeSeries);
    
    // Put filtered data back
    for (let t = 0; t < seqLength; t++) {
      const idx = t * hiddenSize + h; // Assuming batch size 1
      filteredData[idx] = filteredTimeSeries[t];
    }
  }
  
  return new ort.Tensor('float32', filteredData, dims);
}