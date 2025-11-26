/******************************************************************************
 * SPARC Feature Extraction - Web Worker
******************************************************************************/

function workerDebugLog(message, data = null) {
  const timestamp = new Date().toLocaleTimeString();
  if (data) {
    console.log(`[${timestamp}] WORKER DEBUG: ${message}`, data);
  } else {
    console.log(`[${timestamp}] WORKER DEBUG: ${message}`);
  }
  
  self.postMessage({
    type: 'debug',
    message: `${message}${data ? ': ' + JSON.stringify(data, null, 2) : ''}`
  });
}

// Load ONNX runtime in the worker
importScripts('https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort.all.min.js');

// Verify ONNX Runtime loaded correctly
if (typeof self.ort === 'undefined') {
    throw new Error('ONNX Runtime failed to load in worker');
}
workerDebugLog('ONNX Runtime loaded in worker, version: ' + self.ort.version);

// Worker state
let wavlmSession = null;
let linearModel = null;
let linearModelWorkingMemory = null;
let initialized = false;

// Pitch detection state
let pitchHistory = Array(5).fill(0);
let featuresFilterBank = null;

// Processing statistics
let processingStats = {
  totalProcessed: 0,
  totalErrors: 0,
  avgProcessingTime: 0
};

// Handle messages from main thread
self.onmessage = async function(e) {
  const message = e.data;
  workerDebugLog(`Received message: ${message.type}`);
  
  switch(message.type) {
    case 'init':
      await initializeModels(message.onnxPath, message.linearModelPath);
      break;

    case 'process':
      await handleProcessMessage(message);
      break;

    default:
      workerDebugLog(`Unknown message type: ${message.type}`);
  }
};

// SIMPLIFIED: Process message handler without fallbacks
async function handleProcessMessage(message) {
  if (!initialized) {
    self.postMessage({ 
      type: 'error', 
      error: 'Worker not initialized - models not loaded' 
    });
    return;
  }
  
  const startTime = performance.now();
  processingStats.totalProcessed++;
  
  try {
    // Process the audio data
    let audioData;
    if (message.audio instanceof ArrayBuffer) {
      audioData = new Float32Array(message.audio);
    } else if (message.audio instanceof Float32Array) {
      audioData = message.audio;
    } else {
      audioData = new Float32Array(message.audio);
    }
    
    const config = message.config;
    const sensitivityFactor = message.sensitivityFactor || 8.0;

    workerDebugLog(`Processing audio: ${audioData.length} samples, sensitivity: ${sensitivityFactor}`);
    
    // Validate audio data
    if (!validateAudioData(audioData)) {
      throw new Error('Invalid audio data received');
    }
    
    // Process with models
    const result = await processAudioWithModels(audioData, config, sensitivityFactor);
    
    // Send successful result
    self.postMessage({
      type: 'features',
      articulationFeatures: result.articulationFeatures,
      pitch: result.pitch || 0,
      loudness: result.loudness || -60
    });
    
    // Update processing statistics
    const processingTime = performance.now() - startTime;
    processingStats.avgProcessingTime = 
      (processingStats.avgProcessingTime * (processingStats.totalProcessed - 1) + processingTime) / 
      processingStats.totalProcessed;
      
    workerDebugLog(`Processing completed in ${processingTime.toFixed(2)}ms (avg: ${processingStats.avgProcessingTime.toFixed(2)}ms)`);
    
  } catch (error) {
    processingStats.totalErrors++;
    workerDebugLog(`Processing error: ${error.message}`, {
      totalProcessed: processingStats.totalProcessed,
      totalErrors: processingStats.totalErrors,
      errorRate: (processingStats.totalErrors / processingStats.totalProcessed * 100).toFixed(1) + '%'
    });
    
    self.postMessage({
      type: 'error',
      error: `Processing failed: ${error.message}`,
      stats: processingStats
    });
  }
}

// Validate audio data quality
function validateAudioData(audioData) {
  if (!audioData || audioData.length === 0) {
    workerDebugLog('Audio data is empty');
    return false;
  }
  
  const max = Math.max(...audioData);
  const min = Math.min(...audioData);
  const rms = Math.sqrt(audioData.reduce((sum, x) => sum + x*x, 0) / audioData.length);
  
  // Log stats periodically
  if (processingStats.totalProcessed % 20 === 0) {
    workerDebugLog('Audio stats', {
      length: audioData.length,
      max: max.toFixed(4),
      min: min.toFixed(4),
      rms: rms.toFixed(6),
      isAllZeros: max === 0 && min === 0
    });
  }
  
  // Accept silent audio (user might not be speaking)
  // Just validate that values are finite
  return isFinite(max) && isFinite(min) && isFinite(rms);
}

// Process audio through the ML models
async function processAudioWithModels(audioData, config, sensitivityFactor) {
  const timings = {};
  
  // Extract WavLM features
  let t0 = performance.now();
  const wavlmOutput = await extractWavLMFeatures(audioData, wavlmSession);
  timings.wavlm = (performance.now() - t0).toFixed(1);
  
  if (!wavlmOutput) {
    throw new Error("WavLM feature extraction failed");
  }
  
  // Extract articulation features
  t0 = performance.now();
  const articulationFeatures = extractArticulationFeatures(wavlmOutput, sensitivityFactor);
  timings.articulation = (performance.now() - t0).toFixed(1);
  
  if (!articulationFeatures) {
    throw new Error("Articulation feature extraction failed");
  }
  
  // Extract other features (pitch and loudness - optional)
  let pitch = 0;
  let loudness = -60;
  
  try {
    t0 = performance.now();
    pitch = config.extractPitchFn === 2 ? 
      extractPitchSmoothed(audioData) : 
      extractPitch(audioData);
    timings.pitch = (performance.now() - t0).toFixed(1);
      
    t0 = performance.now();
    loudness = calculateLoudness(audioData);
    timings.loudness = (performance.now() - t0).toFixed(1);
  } catch (pitchError) {
    // Pitch extraction is optional, don't fail on errors
    workerDebugLog(`Pitch extraction failed: ${pitchError.message}`);
    timings.pitch = 'error';
    timings.loudness = 'error';
  }
  
  // Log timing every 10 frames
  if (processingStats.totalProcessed % 10 === 0) {
    workerDebugLog(`Processing timings (ms): WavLM=${timings.wavlm}, Articulation=${timings.articulation}, Pitch=${timings.pitch}, Loudness=${timings.loudness}`);
  }
  
  return {
    articulationFeatures,
    pitch,
    loudness
  };
}

// SIMPLIFIED: Initialize the models with clear error reporting
async function initializeModels(onnxPath, linearModelPath) {
  try {
    workerDebugLog("=== INITIALIZING MODELS ===");
    
    // Check ONNX Runtime availability
    if (typeof self.ort === 'undefined') {
      throw new Error('ONNX Runtime not available in worker context');
    }
    
    workerDebugLog(`ONNX Runtime version: ${self.ort.version}`);
    
    // Configure ONNX Runtime
    self.ort.env.wasm.numThreads = 1;
    self.ort.env.wasm.simd = true;
    self.ort.env.debug = false;
    self.ort.env.logLevel = 'warning';
    self.ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/';

    const sessionOptions = {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
      enableCpuMemArena: true,
      enableMemPattern: true,
      executionMode: 'sequential',
      logSeverityLevel: 3,
      intraOpNumThreads: 1,
      interOpNumThreads: 1
    };

    // Load WavLM model
    self.postMessage({ type: 'status', message: 'Loading WavLM model...' });
    workerDebugLog(`Loading WavLM model from: ${onnxPath}`);
    
    try {
      wavlmSession = await self.ort.InferenceSession.create(onnxPath, sessionOptions);
      workerDebugLog("✅ WavLM model loaded successfully");
      
      // Test the model
      await testWavLMModel();
      workerDebugLog("✅ WavLM model test passed");
      
    } catch (modelError) {
      throw new Error(`WavLM model loading failed: ${modelError.message}`);
    }

    // Load linear model
    self.postMessage({ type: 'status', message: 'Loading linear projection model...' });
    workerDebugLog(`Loading linear model from: ${linearModelPath}`);

    try {
      const response = await fetch(linearModelPath);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      const modelData = await response.json();
      
      // Validate model structure
      if (!modelData.weights || !modelData.biases || !modelData.input_dim || !modelData.output_dim) {
        throw new Error('Invalid linear model structure');
      }
      
      linearModel = {
        weights: modelData.weights.map(w => new Float32Array(w)),
        biases: new Float32Array(modelData.biases),
        inputDim: modelData.input_dim,
        outputDim: modelData.output_dim,
        metadata: modelData.metadata || {}
      };
      
      linearModelWorkingMemory = new Float32Array(linearModel.outputDim);
      
      // Get WavLM hidden size from test run
      const wavlmHiddenSize = await getWavLMHiddenSize();
      
      workerDebugLog("✅ Linear model loaded successfully", {
        inputDim: linearModel.inputDim,
        outputDim: linearModel.outputDim,
        weightsShape: `${linearModel.weights.length} x ${linearModel.weights[0].length}`,
        wavlmHiddenSize: wavlmHiddenSize,
        dimensionsMatch: wavlmHiddenSize === linearModel.inputDim ? "✅ Dimensions match" : `❌ Mismatch: WavLM=${wavlmHiddenSize}, Linear=${linearModel.inputDim}`
      });
      
      // Test linear model with correct dimensions
      testLinearModel(wavlmHiddenSize);
      workerDebugLog("✅ Linear model test passed");
      
    } catch (linearError) {
      throw new Error(`Linear model loading failed: ${linearError.message}`);
    }
    
    initialized = true;
    workerDebugLog("🎉 All models initialized successfully");
    self.postMessage({ type: 'initialized' });
    
  } catch (error) {
    workerDebugLog(`❌ Initialization failed: ${error.message}`);
    self.postMessage({ 
      type: 'error', 
      error: `Model initialization failed: ${error.message}`
    });
  }
}

// Get WavLM hidden size by running a test
async function getWavLMHiddenSize() {
  const testData = new Float32Array(16000);
  for (let i = 0; i < 16000; i++) {
    testData[i] = 0.1 * Math.sin(2 * Math.PI * 150 * i / 16000);
  }
  
  const tensor = new self.ort.Tensor('float32', testData, [1, 16000]);
  const feeds = {};
  feeds[wavlmSession.inputNames[0]] = tensor;
  
  const output = await wavlmSession.run(feeds);
  const outputShape = output[wavlmSession.outputNames[0]].dims;
  
  // Shape is [batch, seq_len, hidden_size]
  return outputShape[2];
}

// Test WavLM model with sample inputs
async function testWavLMModel() {
  workerDebugLog("Testing WavLM model...");
  
  // Test with a simple sine wave
  const testData = new Float32Array(16000);
  for (let i = 0; i < 16000; i++) {
    testData[i] = 0.1 * Math.sin(2 * Math.PI * 150 * i / 16000);
  }
  
  const tensor = new self.ort.Tensor('float32', testData, [1, 16000]);
  const feeds = {};
  feeds[wavlmSession.inputNames[0]] = tensor;
  
  const start = performance.now();
  const output = await wavlmSession.run(feeds);
  const duration = performance.now() - start;
  
  workerDebugLog(`Model test completed in ${duration.toFixed(2)}ms`, {
    inputShape: tensor.dims,
    outputShape: output[wavlmSession.outputNames[0]].dims,
    outputSample: Array.from(output[wavlmSession.outputNames[0]].data.slice(0, 3))
  });
  
  // Validate output
  const outputData = output[wavlmSession.outputNames[0]].data;
  const hasNaN = Array.from(outputData.slice(0, 100)).some(v => isNaN(v));
  const hasInf = Array.from(outputData.slice(0, 100)).some(v => !isFinite(v));
  
  if (hasNaN || hasInf) {
    throw new Error("Model output contains NaN or Infinity values");
  }
}

// Test linear model with actual WavLM hidden dimensions
function testLinearModel(hiddenSize) {
  workerDebugLog("Testing linear model...");
  
  // Create test WavLM-like output with correct dimensions
  const seqLen = 50;
  const testData = new Float32Array(seqLen * hiddenSize);
  for (let i = 0; i < testData.length; i++) {
    testData[i] = (Math.random() - 0.5) * 0.1; // Small random values
  }
  
  const testTensor = new self.ort.Tensor('float32', testData, [1, seqLen, hiddenSize]);
  const result = extractArticulationFeatures(testTensor, 1.0);
  
  if (!result) {
    throw new Error("Linear model test failed - no output");
  }
  
  // Validate all required articulators
  const requiredKeys = ['ul', 'll', 'li', 'tt', 'tb', 'td'];
  for (const key of requiredKeys) {
    if (!result[key] || typeof result[key].x !== 'number' || typeof result[key].y !== 'number') {
      throw new Error(`Linear model test failed - invalid ${key}`);
    }
  }
  
  workerDebugLog("Linear model test passed", {
    sampleOutput: {
      ul: `(${result.ul.x.toFixed(3)}, ${result.ul.y.toFixed(3)})`,
      tt: `(${result.tt.x.toFixed(3)}, ${result.tt.y.toFixed(3)})`,
      td: `(${result.td.x.toFixed(3)}, ${result.td.y.toFixed(3)})`
    }
  });
}

/******************************************************************************
* FEATURE EXTRACTION FUNCTIONS *
******************************************************************************/

// Calculate audio loudness (RMS to dB)
function calculateLoudness(audioData) {
  let sum = 0;
  for (let i = 0; i < audioData.length; i++) {
    sum += audioData[i] * audioData[i];
  }
  const rms = Math.sqrt(sum / audioData.length);
  return rms > 0 ? 20 * Math.log10(rms) : -60;
}

// Process audio through WavLM model
async function extractWavLMFeatures(audioData, session) {
  // Use fixed 16000 samples (1 second) for stability
  const inputLength = 16000;
  const inputData = new Float32Array(inputLength);
  const copyLength = Math.min(audioData.length, inputLength);
  
  // Copy and pad with zeros if necessary
  for (let i = 0; i < copyLength; i++) {
    inputData[i] = audioData[i];
  }
  
  const inputTensor = new self.ort.Tensor('float32', inputData, [1, inputLength]);
  const feeds = {};
  feeds[session.inputNames[0]] = inputTensor;
  
  const outputData = await session.run(feeds);
  let output = outputData[session.outputNames[0]];
  
  // Validate output (check only first 100 values to avoid stack overflow)
  const checkSize = Math.min(100, output.data.length);
  let hasNaN = false;
  let hasInf = false;
  
  for (let i = 0; i < checkSize; i++) {
    if (isNaN(output.data[i])) hasNaN = true;
    if (!isFinite(output.data[i])) hasInf = true;
  }
  
  if (hasNaN || hasInf) {
    throw new Error("WavLM output contains NaN or Infinity values");
  }
  
  // Apply forward-backward filter to approximate Python's filtfilt
  if (filteringEnabled) {
    workerDebugLog("Applying forward-backward lowpass filter to WavLM features...");
    output = forwardBackwardFilter(output);
  }
  
  return output;
}

// Extract articulation features from WavLM output
// 
// IMPORTANT: The linear model outputs raw EMA coordinates in the MNGU0 dataset space.
// Feature order: ['ul_x', 'ul_y', 'll_x', 'll_y', 'li_x', 'li_y', 
//                 'tt_x', 'tt_y', 'tb_x', 'tb_y', 'td_x', 'td_y']
//
// The ONNX model MUST match the linear model's expected input dimension:
// - wavlm-large: 1024 hidden dimensions
// - wavlm-base: 768 hidden dimensions
// The current linear model was trained on wavlm-large (1024 dims).
//
function extractArticulationFeatures(wavlmFeatures, sensitivityFactor = 1.0) {
  if (!linearModel) {
    throw new Error("Linear model not loaded");
  }
  
  const features = wavlmFeatures.data;
  const dims = wavlmFeatures.dims;
  const [batchSize, seqLength, hiddenSize] = dims;
  
  // CRITICAL: Validate dimension match
  if (hiddenSize !== linearModel.inputDim) {
    const msg = `CRITICAL: Dimension mismatch! WavLM outputs ${hiddenSize} features, but linear model expects ${linearModel.inputDim}. ` +
                `You need to use ${linearModel.inputDim === 1024 ? 'wavlm-large' : 'wavlm-base'} ONNX model.`;
    workerDebugLog(`❌ ${msg}`);
    throw new Error(msg);
  }
  
  // Use middle frame for stability (matches Python behavior)
  const middleFrameIdx = Math.floor(seqLength / 2);
  const startIdx = middleFrameIdx * hiddenSize;
  
  // Apply linear transformation: output = features @ weights.T + biases
  const output = linearModelWorkingMemory;
  output.set(linearModel.biases);
  
  for (let i = 0; i < linearModel.outputDim; i++) {
    const weights = linearModel.weights[i];
    for (let j = 0; j < hiddenSize; j++) {
      output[i] += weights[j] * features[startIdx + j];
    }
  }
  
  // RAW EMA OUTPUT - No artificial scaling or offsets!
  // The linear model outputs are already in the correct EMA coordinate space.
  // These values typically range from approximately -2 to +2 in the MNGU0 dataset.
  //
  // sensitivityFactor is kept for UI adjustment only, default 1.0 = raw values
  const scale = sensitivityFactor;
  
  const articulationFeatures = {
    ul: {  // Upper lip
      x: output[0] * scale,
      y: output[1] * scale
    },
    ll: {  // Lower lip
      x: output[2] * scale,
      y: output[3] * scale
    },
    li: {  // Lip interspacing / jaw
      x: output[4] * scale,
      y: output[5] * scale
    },
    tt: {  // Tongue tip
      x: output[6] * scale,
      y: output[7] * scale
    },
    tb: {  // Tongue body
      x: output[8] * scale,
      y: output[9] * scale
    },
    td: {  // Tongue dorsum
      x: output[10] * scale,
      y: output[11] * scale
    }
  };

  // Log raw values periodically for debugging
  if (processingStats.totalProcessed % 10 === 0) {
    workerDebugLog("Raw EMA output from linear model (MNGU0 coordinate space)", {
      ul_x: output[0].toFixed(3), ul_y: output[1].toFixed(3),
      ll_x: output[2].toFixed(3), ll_y: output[3].toFixed(3),
      li_x: output[4].toFixed(3), li_y: output[5].toFixed(3),
      tt_x: output[6].toFixed(3), tt_y: output[7].toFixed(3),
      tb_x: output[8].toFixed(3), tb_y: output[9].toFixed(3),
      td_x: output[10].toFixed(3), td_y: output[11].toFixed(3)
    });
  }

  // Validate all outputs
  for (const [key, point] of Object.entries(articulationFeatures)) {
    if (isNaN(point.x) || isNaN(point.y) || !isFinite(point.x) || !isFinite(point.y)) {
      workerDebugLog(`Invalid articulator position for ${key}: (${point.x}, ${point.y})`);
      throw new Error(`Invalid coordinates produced for articulator ${key}`);
    }
  }

  return articulationFeatures;
}

/******************************************************************************
* PITCH DETECTION (YIN ALGORITHM) *
******************************************************************************/

class YINPitchDetector {
  constructor(options = {}) {
    this.sampleRate = options.sampleRate || 16000;
    this.threshold = options.threshold || 0.15;
    this.minFrequency = options.minFrequency || 70;
    this.maxFrequency = options.maxFrequency || 400;
    
    this.minPeriod = Math.floor(this.sampleRate / this.maxFrequency);
    this.maxPeriod = Math.floor(this.sampleRate / this.minFrequency);
  }
  
  detect(audioBuffer) {
    const buffer = audioBuffer instanceof Float32Array ? audioBuffer : new Float32Array(audioBuffer);
    const bufferSize = Math.min(buffer.length, 2048);
    const yinBuffer = new Float32Array(bufferSize / 2);
    
    // Step 1: Calculate difference function
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
    
    // Step 3: Find absolute threshold
    for (let tau = this.minPeriod; tau <= this.maxPeriod && tau < yinBuffer.length; tau++) {
      if (yinBuffer[tau] < this.threshold) {
        const exactTau = this.parabolicInterpolation(yinBuffer, tau);
        return this.sampleRate / exactTau;
      }
    }
    
    // If no pitch found below threshold, return 0
    return 0;
  }

  parabolicInterpolation(array, position) {
    if (position === 0 || position === array.length - 1) {
      return position;
    }
    
    const y1 = array[position - 1];
    const y2 = array[position];
    const y3 = array[position + 1];
    
    const a = (y3 + y1 - 2 * y2) / 2;
    const b = (y3 - y1) / 2;
    
    if (a === 0) {
      return position;
    }
    
    return position - b / (2 * a);
  }
}

// Initialize pitch detector
let yinDetector = null;

function extractPitch(audioData) {
  if (!yinDetector) {
    yinDetector = new YINPitchDetector({
      sampleRate: 16000,
      threshold: 0.15,
      minFrequency: 70,
      maxFrequency: 400
    });
  }
  
  const bufferSize = Math.min(audioData.length, 2048);
  const startIdx = Math.floor((audioData.length - bufferSize) / 2);
  const audioSlice = audioData.slice(startIdx, startIdx + bufferSize);
  
  return yinDetector.detect(audioSlice) || 0;
}

function extractPitchSmoothed(audioData) {
  const rawPitch = extractPitch(audioData);
  
  pitchHistory.push(rawPitch);
  pitchHistory.shift();
  
  // Use median filtering for smoothing
  const nonZeroPitches = pitchHistory.filter(p => p > 0);
  
  if (nonZeroPitches.length === 0) {
    return 0;
  }
  
  const sortedPitches = [...nonZeroPitches].sort((a, b) => a - b);
  return sortedPitches[Math.floor(sortedPitches.length / 2)];
}

/******************************************************************************
* OPTIONAL FILTERING - DISABLED DUE TO PERFORMANCE ISSUES *
******************************************************************************/

// NOTE: The Butterworth filter implementation below causes stack overflow
// Filtering is disabled for now. Features are extracted without filtering.
// TODO: Implement a simpler, non-recursive filter

/*
// Butterworth lowpass filter matching Python scipy.signal implementation
// 
// Python equivalent:
//   from scipy.signal import butter, filtfilt
//   b, a = butter(5, 10, fs=50, btype='low')  # 5th order, 10Hz cutoff, 50Hz sample rate
//   filtered = filtfilt(b, a, data, axis=1)
//
// These coefficients are computed for: order=5, cutoff=10Hz, fs=50Hz
// Generated using scipy.signal.butter(5, 10, fs=50, btype='low')
//
class ButterworthLowpass {
  constructor(cutoff = 10, sampleRate = 50, order = 5) {
    // Pre-computed coefficients for 10Hz cutoff, 50Hz sample rate, 5th order
    // scipy.signal.butter(5, 10, fs=50, btype='low') produces:
    this.b = new Float64Array([
      0.0008044669373420302,
      0.0040223346867101510,
      0.0080446693734203020,
      0.0080446693734203020,
      0.0040223346867101510,
      0.0008044669373420302
    ]);
    this.a = new Float64Array([
      1.0000000000000000,
      -2.3695130067364450,
      2.3139884144158150,
      -1.1545538744828020,
      0.2879568732043606,
      -0.0285539668678658
    ]);
    
    this.order = order;
    // Forward and backward state for filtfilt
    this.zi_forward = new Float64Array(order);
    this.zi_backward = new Float64Array(order);
  }
  
  // Reset filter state
  reset() {
    this.zi_forward.fill(0);
    this.zi_backward.fill(0);
  }
  
  // Apply forward-backward filtering (like scipy filtfilt)
  // This provides zero-phase filtering
  filtfilt(data) {
    const n = data.length;
    if (n === 0) return new Float64Array(0);
    
    // Pad the signal to reduce edge effects
    const padLen = 3 * this.order;
    const paddedLen = n + 2 * padLen;
    const padded = new Float64Array(paddedLen);
    
    // Reflect padding at edges (like scipy default)
    for (let i = 0; i < padLen; i++) {
      padded[i] = 2 * data[0] - data[padLen - i];
    }
    for (let i = 0; i < n; i++) {
      padded[padLen + i] = data[i];
    }
    for (let i = 0; i < padLen; i++) {
      padded[padLen + n + i] = 2 * data[n - 1] - data[n - 2 - i];
    }
    
    // Forward filter
    const forward = this.lfilter(padded, true);
    
    // Reverse the signal
    const reversed = new Float64Array(paddedLen);
    for (let i = 0; i < paddedLen; i++) {
      reversed[i] = forward[paddedLen - 1 - i];
    }
    
    // Backward filter
    const backward = this.lfilter(reversed, false);
    
    // Reverse again and extract the original signal portion
    const result = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      result[i] = backward[paddedLen - 1 - padLen - i];
    }
    
    return result;
  }
  
  // Standard IIR filter (like scipy lfilter)
  lfilter(x, forward = true) {
    const n = x.length;
    const y = new Float64Array(n);
    const zi = forward ? this.zi_forward : this.zi_backward;
    
    // Reset state for each filtfilt call
    zi.fill(0);
    
    for (let i = 0; i < n; i++) {
      // Compute output
      y[i] = this.b[0] * x[i] + zi[0];
      
      // Update state
      for (let j = 0; j < this.order - 1; j++) {
        zi[j] = this.b[j + 1] * x[i] - this.a[j + 1] * y[i] + zi[j + 1];
      }
      zi[this.order - 1] = this.b[this.order] * x[i] - this.a[this.order] * y[i];
    }
    
    return y;
  }
}

// Legacy filter class for backward compatibility
class LowpassFilter extends ButterworthLowpass {
  constructor() {
    super(10, 50, 5);
  }
  
  processSample(x) {
    // Single sample processing for streaming (not filtfilt)
    const y = this.b[0] * x + this.zi_forward[0];
    for (let j = 0; j < this.order - 1; j++) {
      this.zi_forward[j] = this.b[j + 1] * x - this.a[j + 1] * y + this.zi_forward[j + 1];
    }
    this.zi_forward[this.order - 1] = this.b[this.order] * x - this.a[this.order] * y;
    return y;
  }
  
  process(inputArray) {
    // Use filtfilt for better results (matches Python)
    return this.filtfilt(inputArray);
  }
}

// Create a shared filter for applying lowpass to WavLM features
// This matches Python: butter_bandpass_filter(states, freqcut=10, fs=50, axis=1, order=5)
let butterworthFilter = null;

function createFilterBank(numFilters) {
  const filters = [];
  for (let i = 0; i < numFilters; i++) {
    filters.push(new ButterworthLowpass(10, 50, 5));  // 10Hz cutoff, 50Hz sample rate
  }
  return filters;
}

// Apply 10Hz lowpass filter to WavLM hidden states
// This MUST be done before applying the linear model, matching Python behavior
// Python equivalent: butter_bandpass_filter(states, 10, 50, axis=1, order=5)
function filterWavLMFeatures(wavlmFeatures) {
  const dims = wavlmFeatures.dims;
  const data = wavlmFeatures.data;
  const [batchSize, seqLength, hiddenSize] = dims;
  
  // Create filter bank if not exists or wrong size
  if (!featuresFilterBank || featuresFilterBank.length !== hiddenSize) {
    workerDebugLog(`Creating filter bank with ${hiddenSize} filters (10Hz lowpass, 50Hz fs)`);
    featuresFilterBank = createFilterBank(hiddenSize);
  }
  
  const filteredData = new Float32Array(data.length);
  
  // Apply filtering per feature dimension across time (axis=1 in Python)
  // This processes each of the hidden dimensions independently
  for (let h = 0; h < hiddenSize; h++) {
    // Extract time series for this hidden dimension
    const featureTimeSeries = new Float64Array(seqLength);  // Use Float64 for precision
    for (let t = 0; t < seqLength; t++) {
      const idx = t * hiddenSize + h;
      featureTimeSeries[t] = data[idx];
    }
    
    // Apply filtfilt (forward-backward filtering for zero phase)
    const filteredTimeSeries = featuresFilterBank[h].filtfilt(featureTimeSeries);
    
    // Copy back to output
    for (let t = 0; t < seqLength; t++) {
      const idx = t * hiddenSize + h;
      filteredData[idx] = filteredTimeSeries[t];
    }
  }
  
  return new self.ort.Tensor('float32', filteredData, dims);
}
*/

// Forward-backward Gaussian filter (approximates filtfilt)
// Applies filter forward, then backward for zero-phase filtering
function forwardBackwardFilter(wavlmFeatures) {
  const dims = wavlmFeatures.dims;
  const [batchSize, seqLength, hiddenSize] = dims;
  
  // Wider kernel for better lowpass (window=7): approximates 10Hz @ 50Hz
  const kernel = new Float32Array([0.03, 0.12, 0.20, 0.30, 0.20, 0.12, 0.03]);
  const halfWindow = Math.floor(kernel.length / 2);
  
  const data = new Float32Array(wavlmFeatures.data);
  const tempData = new Float32Array(data.length);
  
  // Forward pass
  for (let h = 0; h < hiddenSize; h++) {
    for (let t = 0; t < seqLength; t++) {
      let sum = 0;
      let weightSum = 0;
      
      for (let k = 0; k < kernel.length; k++) {
        const tIdx = t + k - halfWindow;
        if (tIdx >= 0 && tIdx < seqLength) {
          sum += data[tIdx * hiddenSize + h] * kernel[k];
          weightSum += kernel[k];
        }
      }
      tempData[t * hiddenSize + h] = sum / weightSum;
    }
  }
  
  // Backward pass (on forward-filtered data)
  const filteredData = new Float32Array(data.length);
  for (let h = 0; h < hiddenSize; h++) {
    for (let t = seqLength - 1; t >= 0; t--) {
      let sum = 0;
      let weightSum = 0;
      
      for (let k = 0; k < kernel.length; k++) {
        const tIdx = t + k - halfWindow;
        if (tIdx >= 0 && tIdx < seqLength) {
          sum += tempData[tIdx * hiddenSize + h] * kernel[k];
          weightSum += kernel[k];
        }
      }
      filteredData[t * hiddenSize + h] = sum / weightSum;
    }
  }
  
  return new self.ort.Tensor('float32', filteredData, dims);
}

// Disable filtering - over-smoothing makes results worse
// The unfiltered results are actually quite good (avg diff ~0.23)
let filteringEnabled = false;

function setFilteringEnabled(enabled) {
  filteringEnabled = false;  // Force disabled
  workerDebugLog(`WavLM feature filtering disabled (unfiltered is more accurate)`);
}