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
  
  workerDebugLog('Audio stats', {
    length: audioData.length,
    max: max.toFixed(4),
    min: min.toFixed(4),
    rms: rms.toFixed(6),
    isAllZeros: max === 0 && min === 0
  });
  
  // Audio is valid if it's not all zeros and has reasonable values
  return !(max === 0 && min === 0) && isFinite(max) && isFinite(min) && isFinite(rms);
}

// Process audio through the ML models
async function processAudioWithModels(audioData, config, sensitivityFactor) {
  // Extract WavLM features
  workerDebugLog("Starting WavLM feature extraction...");
  const wavlmOutput = await extractWavLMFeatures(audioData, wavlmSession);
  
  if (!wavlmOutput) {
    throw new Error("WavLM feature extraction failed");
  }
  
  // Extract articulation features
  workerDebugLog("Starting articulation feature extraction...");
  const articulationFeatures = extractArticulationFeatures(wavlmOutput, sensitivityFactor);
  
  if (!articulationFeatures) {
    throw new Error("Articulation feature extraction failed");
  }
  
  // Extract other features
  const pitch = config.extractPitchFn === 2 ? 
    extractPitchSmoothed(audioData) : 
    extractPitch(audioData);
    
  const loudness = calculateLoudness(audioData);
  
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
      workerDebugLog("✅ Linear model loaded successfully", {
        inputDim: linearModel.inputDim,
        outputDim: linearModel.outputDim,
        weightsShape: `${linearModel.weights.length} x ${linearModel.weights[0].length}`,
        note: linearModel.inputDim !== 768 ? `⚠️ Expected 768 features but model expects ${linearModel.inputDim}` : "✅ Dimensions match WavLM-Base"
      });
      
      // Test linear model
      testLinearModel();
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

// Test linear model
function testLinearModel() {
  workerDebugLog("Testing linear model...");
  
  // Create test WavLM-like output
  const testData = new Float32Array(50 * 768); // 50 frames, 768 features
  for (let i = 0; i < testData.length; i++) {
    testData[i] = (Math.random() - 0.5) * 0.1; // Small random values
  }
  
  const testTensor = new self.ort.Tensor('float32', testData, [1, 50, 768]);
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
  // Prepare input data (exactly 16000 samples)
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
  
  // Validate output
  const outputArray = Array.from(output.data);
  const hasNaN = outputArray.some(v => isNaN(v));
  const hasInf = outputArray.some(v => !isFinite(v));
  
  if (hasNaN || hasInf) {
    throw new Error("WavLM output contains NaN or Infinity values");
  }
  
  // Apply optional filtering
  if (featuresFilterBank) {
    output = filterWavLMFeatures(output);
  }
  
  return output;
}

// Extract articulation features from WavLM output
function extractArticulationFeatures(wavlmFeatures, sensitivityFactor = 8.0) {
  if (!linearModel) {
    throw new Error("Linear model not loaded");
  }
  
  const features = wavlmFeatures.data;
  const dims = wavlmFeatures.dims;
  const [batchSize, seqLength, hiddenSize] = dims;
  
  // Handle dimension mismatch by truncating or padding features
  let effectiveHiddenSize = hiddenSize;
  if (hiddenSize !== linearModel.inputDim) {
    workerDebugLog(`⚠️ Dimension mismatch: WavLM=${hiddenSize}, Linear=${linearModel.inputDim}. Adapting...`);
    effectiveHiddenSize = Math.min(hiddenSize, linearModel.inputDim);
  }
  
  // Use middle frame for stability
  const middleFrameIdx = Math.floor(seqLength / 2);
  const startIdx = middleFrameIdx * hiddenSize;
  
  // Apply linear transformation with dimension adaptation
  const output = linearModelWorkingMemory;
  output.set(linearModel.biases);
  
  for (let i = 0; i < linearModel.outputDim; i++) {
    const weights = linearModel.weights[i];
    
    // Use available features up to the minimum of model expectation and actual features
    for (let j = 0; j < effectiveHiddenSize; j++) {
      if (j < weights.length && (startIdx + j) < features.length) {
        output[i] += weights[j] * features[startIdx + j];
      }
    }
  }
  
  // Apply scaling and offsets
  const scaleFactorX = sensitivityFactor * 0.1;  // Adjust scaling
  const scaleFactorY = sensitivityFactor * 0.1;
  
  const articulationFeatures = {
    ul: {
      x: output[0] * scaleFactorX + 0.9,   // Upper lip
      y: output[1] * scaleFactorY - 1.0
    },
    ll: {
      x: output[2] * scaleFactorX + 0.9,   // Lower lip
      y: output[3] * scaleFactorY - 0.7
    },
    li: {
      x: output[4] * scaleFactorX + 0.9,   // Lip interface
      y: output[5] * scaleFactorY - 0.85
    },
    tt: {
      x: output[6] * scaleFactorX + 0.5,   // Tongue tip
      y: output[7] * scaleFactorY - 0.7
    },
    tb: {
      x: output[8] * scaleFactorX + 0.0,   // Tongue body
      y: output[9] * scaleFactorY - 0.6
    },
    td: {
      x: output[10] * scaleFactorX - 0.5,  // Tongue dorsum
      y: output[11] * scaleFactorY - 0.5
    }
  };

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
* OPTIONAL FILTERING *
******************************************************************************/

// Butterworth lowpass filter for feature smoothing
class LowpassFilter {
  constructor() {
    // 5th order Butterworth filter coefficients (cutoff ~10Hz for smoothing)
    this.b = [0.0008, 0.0039, 0.0078, 0.0078, 0.0039, 0.0008];
    this.a = [1.0000, -3.0756, 3.8289, -2.3954, 0.7475, -0.0930];
    this.x_history = new Float32Array(this.b.length).fill(0);
    this.y_history = new Float32Array(this.a.length-1).fill(0);
  }
  
  processSample(x) {
    // Shift input history
    for (let i = this.x_history.length - 1; i > 0; i--) {
      this.x_history[i] = this.x_history[i-1];
    }
    this.x_history[0] = x;
    
    // Apply filter equation
    let y = 0;
    for (let i = 0; i < this.b.length; i++) {
      y += this.b[i] * this.x_history[i];
    }
    
    for (let i = 0; i < this.y_history.length; i++) {
      y -= this.a[i+1] * this.y_history[i];
    }
    
    // Shift output history
    for (let i = this.y_history.length - 1; i > 0; i--) {
      this.y_history[i] = this.y_history[i-1];
    }
    this.y_history[0] = y;
    
    return y;
  }
  
  process(inputArray) {
    const outputArray = new Float32Array(inputArray.length);
    for (let i = 0; i < inputArray.length; i++) {
      outputArray[i] = this.processSample(inputArray[i]);
    }
    return outputArray;
  }
}

function createFilterBank(numFilters) {
  const filters = [];
  for (let i = 0; i < numFilters; i++) {
    filters.push(new LowpassFilter());
  }
  return filters;
}

function filterWavLMFeatures(wavlmFeatures) {
  const dims = wavlmFeatures.dims;
  const data = wavlmFeatures.data;
  const [batchSize, seqLength, hiddenSize] = dims;
  
  if (!featuresFilterBank || featuresFilterBank.length !== hiddenSize) {
    featuresFilterBank = createFilterBank(hiddenSize);
  }
  
  const filteredData = new Float32Array(data.length);
  
  // Apply filtering per feature dimension across time
  for (let h = 0; h < hiddenSize; h++) {
    const featureTimeSeries = new Float32Array(seqLength);
    for (let t = 0; t < seqLength; t++) {
      const idx = t * hiddenSize + h;
      featureTimeSeries[t] = data[idx];
    }
    
    const filteredTimeSeries = featuresFilterBank[h].process(featureTimeSeries);
    
    for (let t = 0; t < seqLength; t++) {
      const idx = t * hiddenSize + h;
      filteredData[idx] = filteredTimeSeries[t];
    }
  }
  
  return new self.ort.Tensor('float32', filteredData, dims);
}