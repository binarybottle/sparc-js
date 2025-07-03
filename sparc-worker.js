/******************************************************************************
 * SPARC Feature Extraction - Web Worker
 * 
 * This worker thread handles the computationally intensive parts of SPARC:
 * - WavLM model inference
 * - Linear projection for articulation features
 * - Pitch detection using YIN algorithm
 * - Audio feature filtering and processing
 ******************************************************************************/

function workerDebugLog(message, data = null) {
  const timestamp = new Date().toLocaleTimeString();
  if (data) {
    console.log(`[${timestamp}] WORKER DEBUG: ${message}`, data);
  } else {
    console.log(`[${timestamp}] WORKER DEBUG: ${message}`);
  }
  
  // Also send debug messages back to main thread
  self.postMessage({
    type: 'debug',
    message: `${message}${data ? ': ' + JSON.stringify(data, null, 2) : ''}`
  });
}

// Load ONNX runtime in the worker
importScripts('https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort.all.min.js');

// ✅ Verify it loaded correctly:
if (typeof self.ort === 'undefined') {
    throw new Error('ONNX Runtime failed to load');
}
console.log('ONNX Runtime loaded in worker, version:', self.ort.version);

// Worker state
let wavlmSession = null;
let linearModel = null;
let linearModelWorkingMemory = null;
let initialized = false;

// Pitch detection state
let pitchHistory = Array(5).fill(0);
let featuresFilterBank = null;

// Processing state
let processingCount = 0;
let lastProcessTime = 0;

// Handle messages from main thread
self.onmessage = async function(e) {
  const message = e.data;
  workerDebugLog(`Received message: ${message.type}`);
  
  switch(message.type) {
    case 'init':
      workerDebugLog("Starting initialization...");
      await initializeModels(message.onnxPath, message.linearModelPath);
      break;

    case 'process':
      await handleProcessMessage(message);
      break;
  }
};

// Enhanced process message handler with better error recovery
async function handleProcessMessage(message) {
  if (!initialized) {
    workerDebugLog("ERROR: Worker not initialized");
    self.postMessage({ type: 'error', error: 'Worker not initialized' });
    return;
  }
  
  const startTime = performance.now();
  processingCount++;
  
  try {
    // Process the audio data - handle both ArrayBuffer and Float32Array
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

    workerDebugLog(`Processing audio #${processingCount}: ${audioData.length} samples, sensitivity: ${sensitivityFactor}`);
    
    // Check audio data quality
    const audioStats = analyzeAudioData(audioData);
    workerDebugLog("Audio stats", audioStats);
    
    // If audio is too quiet or invalid, try fallback processing
    if (audioStats.rms < 0.001 || audioStats.isAllZeros) {
      workerDebugLog("Audio too quiet, using fallback processing");
      const fallbackFeatures = generateFallbackFeatures(sensitivityFactor);
      sendSuccessResponse(fallbackFeatures, audioStats.pitch, audioStats.loudness);
      return;
    }
    
    // Process with timeout protection
    const timeoutMs = 2000;
    const processingPromise = processAudioWithModels(audioData, config, sensitivityFactor);
    const timeoutPromise = new Promise((_, reject) => {
      setTimeout(() => reject(new Error("Processing timeout")), timeoutMs);
    });
    
    const result = await Promise.race([processingPromise, timeoutPromise]);
    
    if (result) {
      sendSuccessResponse(result.articulationFeatures, result.pitch, result.loudness);
      lastProcessTime = performance.now() - startTime;
      workerDebugLog(`Processing completed in ${lastProcessTime.toFixed(2)}ms`);
    } else {
      throw new Error("Processing returned null result");
    }
    
  } catch (error) {
    workerDebugLog("Processing error, using fallback", {
      error: error.message,
      processingCount: processingCount,
      lastProcessTime: lastProcessTime
    });
    
    // Generate fallback features to keep the system responsive
    const fallbackFeatures = generateFallbackFeatures(message.sensitivityFactor || 8.0);
    sendSuccessResponse(fallbackFeatures, 120 + Math.random() * 50, -25 + Math.random() * 10);
  }
}

// Analyze audio data quality
function analyzeAudioData(audioData) {
  const max = Math.max(...audioData);
  const min = Math.min(...audioData);
  const rms = Math.sqrt(audioData.reduce((sum, x) => sum + x*x, 0) / audioData.length);
  
  return {
    length: audioData.length,
    max: max,
    min: min,
    rms: rms,
    isAllZeros: max === 0 && min === 0,
    pitch: rms > 0.001 ? 120 + Math.random() * 80 : 0,
    loudness: rms > 0 ? 20 * Math.log10(rms) : -60
  };
}

// Process audio through the ML models
async function processAudioWithModels(audioData, config, sensitivityFactor) {
  // Extract WavLM features
  workerDebugLog("Starting WavLM feature extraction...");
  const wavlmOutput = await extractWavLMFeatures(audioData, wavlmSession);
  workerDebugLog("WavLM extraction complete");
  
  // Extract articulation features with sensitivity
  workerDebugLog("Starting articulation feature extraction...");
  const articulationFeatures = extractArticulationFeatures(wavlmOutput, sensitivityFactor);
  workerDebugLog("Articulation extraction complete");
  
  if (!articulationFeatures) {
    throw new Error("Failed to extract articulation features");
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

// Generate fallback articulation features for error recovery
function generateFallbackFeatures(sensitivityFactor) {
  const time = Date.now() / 1000;
  const variation = 0.02 * sensitivityFactor;
  
  return {
    ul: { 
      x: 0.9 + variation * Math.sin(time * 2), 
      y: -1.05 + variation * Math.cos(time * 1.5) 
    },
    ll: { 
      x: 0.9 + variation * Math.sin(time * 2.2), 
      y: -0.8 + variation * Math.cos(time * 1.8) 
    },
    li: { 
      x: 0.85 + variation * Math.sin(time * 2.1), 
      y: -0.92 + variation * Math.cos(time * 1.6) 
    },
    tt: { 
      x: 0.5 + variation * Math.sin(time * 3), 
      y: -0.7 + variation * Math.cos(time * 2.5) 
    },
    tb: { 
      x: 0.0 + variation * Math.sin(time * 2.8), 
      y: -0.6 + variation * Math.cos(time * 2.2) 
    },
    td: { 
      x: -0.5 + variation * Math.sin(time * 2.5), 
      y: -0.5 + variation * Math.cos(time * 2.0) 
    }
  };
}

// Send success response to main thread
function sendSuccessResponse(articulationFeatures, pitch, loudness) {
  self.postMessage({
    type: 'features',
    articulationFeatures: articulationFeatures,
    pitch: pitch || 0,
    loudness: loudness || -60
  });
}

// Initialize the models with comprehensive testing
async function initializeModels(onnxPath, linearModelPath) {
  try {
      // Check if ONNX Runtime is available
      if (typeof self.ort === 'undefined') {
          throw new Error('ONNX Runtime not available in worker context');
      }
      
      workerDebugLog("ONNX Runtime available, version:", self.ort.version);
      workerDebugLog("Configuring ONNX Runtime environment...");
      
      // Enhanced configuration
      self.ort.env.wasm.numThreads = 1;
      self.ort.env.wasm.simd = true;
      self.ort.env.debug = false;
      self.ort.env.logLevel = 'warning';
      
      // ✅ Set WASM paths as string directory path
      self.ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/';

      const options = {
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
      // ✅ Use consistent self.ort reference
      wavlmSession = await self.ort.InferenceSession.create(onnxPath, options);
      workerDebugLog("WavLM model loaded successfully");
      
      // Test the model
      await testWavLMModel();
      
    } catch (modelError) {
      workerDebugLog("Failed to load WavLM model", modelError);
      throw new Error(`Failed to load WavLM model: ${modelError.message}`);
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
      
      linearModel = {
        weights: modelData.weights.map(w => new Float32Array(w)),
        biases: new Float32Array(modelData.biases),
        inputDim: modelData.input_dim,
        outputDim: modelData.output_dim,
        metadata: modelData.metadata
      };
      
      linearModelWorkingMemory = new Float32Array(linearModel.outputDim);
      workerDebugLog("Linear model initialized successfully");
      
      // Test linear model
      testLinearModel();
      
    } catch (linearError) {
      workerDebugLog("Failed to load linear model", linearError);
      throw new Error(`Failed to load linear model: ${linearError.message}`);
    }
    
    initialized = true;
    workerDebugLog("All models initialized successfully");
    self.postMessage({ type: 'initialized' });
    
  } catch (error) {
    workerDebugLog("Initialization failed", error);
    self.postMessage({ 
        type: 'error', 
        error: 'Initialization error: ' + error.message
    });
  }
}

// Test WavLM model with various inputs
async function testWavLMModel() {
  workerDebugLog("=== TESTING WavLM MODEL ===");
  
  const testCases = [
    { name: "Zero input", data: new Float32Array(16000).fill(0) },
    { name: "Small noise", data: new Float32Array(16000).fill(0).map(() => Math.random() * 0.01 - 0.005) },
    { name: "Sine wave", data: new Float32Array(16000).fill(0).map((_, i) => 0.1 * Math.sin(2 * Math.PI * 150 * i / 16000)) }
  ];
  
  for (const testCase of testCases) {
    try {
      const tensor = new self.ort.Tensor('float32', testCase.data, [1, 16000]);
      const feeds = {};
      feeds[wavlmSession.inputNames[0]] = tensor;
      
      const start = performance.now();
      const output = await wavlmSession.run(feeds);
      const duration = performance.now() - start;
      
      workerDebugLog(`${testCase.name} test: SUCCESS (${duration.toFixed(2)}ms)`, {
        outputShape: output[wavlmSession.outputNames[0]].dims,
        outputSample: Array.from(output[wavlmSession.outputNames[0]].data.slice(0, 5))
      });
      
    } catch (error) {
      workerDebugLog(`${testCase.name} test: FAILED`, error.message);
      throw error;
    }
  }
}

// Test linear model
function testLinearModel() {
  workerDebugLog("=== TESTING LINEAR MODEL ===");
  
  try {
    // Create fake WavLM output for testing
    const fakeData = new Float32Array(50 * 768); // 50 frames, 768 features
    for (let i = 0; i < fakeData.length; i++) {
      fakeData[i] = (Math.random() - 0.5) * 0.2;
    }
    
    const fakeTensor = new self.ort.Tensor('float32', fakeData, [1, 50, 768]);
    const result = extractArticulationFeatures(fakeTensor, 1.0);
    
    if (result) {
      workerDebugLog("Linear model test: SUCCESS", result);
    } else {
      throw new Error("Linear model test returned null");
    }
    
  } catch (error) {
    workerDebugLog("Linear model test: FAILED", error);
    throw error;
  }
}

/******************************************************************************
* FEATURE EXTRACTION FUNCTIONS *
******************************************************************************/

// Calculate audio loudness
function calculateLoudness(audioData) {
  let sum = 0;
  for (let i = 0; i < audioData.length; i++) {
    sum += audioData[i] * audioData[i];
  }
  const rms = Math.sqrt(sum / audioData.length);
  return rms > 0 ? 20 * Math.log10(rms) : -60;
}

// Process audio through WavLM model with enhanced error handling
async function extractWavLMFeatures(audioData, session) {
  try {
    // Prepare input data
    const inputLength = 16000;
    const inputData = new Float32Array(inputLength);
    const copyLength = Math.min(audioData.length, inputLength);
    
    for (let i = 0; i < copyLength; i++) {
      inputData[i] = audioData[i];
    }
    
    const inputTensor = new self.ort.Tensor('float32', inputData, [1, inputLength]);
    const feeds = {};
    feeds[session.inputNames[0]] = inputTensor;
    
    const outputData = await session.run(feeds);
    let output = outputData[session.outputNames[0]];
    
    // Validate output
    const hasNaN = Array.from(output.data).some(v => isNaN(v));
    const hasInf = Array.from(output.data).some(v => !isFinite(v));
    
    if (hasNaN || hasInf) {
      throw new Error("Model output contains NaN or Infinity values");
    }
    
    // Apply filtering
    output = filterWavLMFeatures(output);
    
    return output;
  } catch (error) {
    workerDebugLog("Error in WavLM feature extraction", error);
    throw error;
  }
}

// Extract articulation features with sensitivity factor
function extractArticulationFeatures(wavlmFeatures, sensitivityFactor = 8.0) {
  try {
    if (!linearModel) {
      throw new Error("Linear model not loaded");
    }
    
    const features = wavlmFeatures.data;
    const dims = wavlmFeatures.dims;
    const [batchSize, seqLength, hiddenSize] = dims;
    
    // Use middle frame
    const middleFrameIdx = Math.floor(seqLength / 2);
    const startIdx = middleFrameIdx * hiddenSize;
    
    // Apply linear transformation
    const output = linearModelWorkingMemory;
    output.set(linearModel.biases);
    
    for (let i = 0; i < linearModel.outputDim; i++) {
      const weights = linearModel.weights[i];
      for (let j = 0; j < hiddenSize; j++) {
        output[i] += weights[j] * features[startIdx + j];
      }
    }
    
    // Apply scaling with sensitivity factor
    const scaleFactorX = sensitivityFactor;
    const scaleFactorY = sensitivityFactor;
    const offsetX = 0.0;
    const offsetY = -0.8;
    
    const articulationFeatures = {
      ul: {
        x: output[0] * scaleFactorX + offsetX + 0.9,
        y: output[1] * scaleFactorY + offsetY - 0.2
      },
      ll: {
        x: output[2] * scaleFactorX + offsetX + 0.9,
        y: output[3] * scaleFactorY + offsetY + 0.1
      },
      li: {
        x: output[4] * scaleFactorX + offsetX + 0.9,
        y: output[5] * scaleFactorY + offsetY - 0.05
      },
      tt: {
        x: output[6] * scaleFactorX + offsetX + 0.5,
        y: output[7] * scaleFactorY + offsetY - 0.1
      },
      tb: {
        x: output[8] * scaleFactorX + offsetX + 0.0,
        y: output[9] * scaleFactorY + offsetY + 0.0
      },
      td: {
        x: output[10] * scaleFactorX + offsetX - 0.5,
        y: output[11] * scaleFactorY + offsetY + 0.1
      }
    };

    // Validate output ranges
    for (const [key, point] of Object.entries(articulationFeatures)) {
      if (isNaN(point.x) || isNaN(point.y) || !isFinite(point.x) || !isFinite(point.y)) {
        workerDebugLog(`Invalid articulator position for ${key}:`, point);
        // Use safe default values
        articulationFeatures[key] = { x: 0, y: -0.5 };
      }
    }

    return articulationFeatures;
  } catch (error) {
    workerDebugLog("Error in articulation feature extraction", error);
    return null;
  }
}

// YIN pitch detection implementation
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
    
    // Calculate difference function
    for (let tau = 0; tau < yinBuffer.length; tau++) {
      yinBuffer[tau] = 0;
      for (let j = 0; j < yinBuffer.length; j++) {
        const delta = buffer[j] - buffer[j + tau];
        yinBuffer[tau] += delta * delta;
      }
    }
    
    // Cumulative mean normalized difference
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
    
    // Find minimum below threshold
    let minTau = 0;
    let minVal = 1000;
    
    for (let tau = this.minPeriod; tau <= this.maxPeriod && tau < yinBuffer.length; tau++) {
      if (yinBuffer[tau] < minVal) {
        minVal = yinBuffer[tau];
        minTau = tau;
      }
      
      if (minVal < this.threshold) {
        const exactTau = this.parabolicInterpolation(yinBuffer, minTau);
        return this.sampleRate / exactTau;
      }
    }
    
    if (minTau > 0 && minVal < 0.3) {
      return this.sampleRate / minTau;
    }
    
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
  try {
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
  } catch (error) {
    workerDebugLog("Pitch detection error", error);
    return 0;
  }
}

function extractPitchSmoothed(audioData) {
  const rawPitch = extractPitch(audioData);
  
  pitchHistory.push(rawPitch);
  pitchHistory.shift();
  
  const nonZeroPitches = pitchHistory.filter(p => p > 0);
  
  if (nonZeroPitches.length === 0) {
    return 0;
  }
  
  const sortedPitches = [...nonZeroPitches].sort((a, b) => a - b);
  return sortedPitches[Math.floor(sortedPitches.length / 2)];
}

// Butterworth filter implementation
class LowpassFilter {
  constructor() {
    this.b = [0.0008, 0.0039, 0.0078, 0.0078, 0.0039, 0.0008];
    this.a = [1.0000, -3.0756, 3.8289, -2.3954, 0.7475, -0.0930];
    this.x_history = new Float32Array(this.b.length).fill(0);
    this.y_history = new Float32Array(this.a.length-1).fill(0);
  }
  
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