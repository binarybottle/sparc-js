/******************************************************************************
 * SPARC Feature Extraction - Web Worker
 *
 * Extracts speech articulatory features from audio using:
 *   1. WavLM (layer 9, FP32 ONNX) for frame-wise hidden states (1024-dim)
 *   2. A learned linear projection to 12 EMA channels (MNGU0 coordinate space)
 *   3. LPC-based F1 (first formant) estimation for lip positioning
 *   4. YIN pitch detection and RMS loudness
 *
 * Returns to the main thread: articulationFeatures (6 x,y z-score pairs),
 * pitch, loudness, and f1 (Hz).
 *
 * Python equivalent (Speech-Articulatory-Coding):
 *   - Audio z-score normalization and 160-sample zero-padding
 *   - WavLM hidden states at layer 9
 *   - Linear model: wavlm_large-9_cut-10_mngu_linear.pkl
 *   - 10 Hz Butterworth filtfilt on hidden states (not implemented here)
 *
 * Deviation from Python pipeline:
 *   - No lowpass filtering on hidden states (Butterworth filtfilt is not
 *     faithfully reproducible in JS without scipy; omitting it yields closer
 *     results than an approximate Gaussian filter)
 *   - Pitch via YIN (Python uses CREPE or PENN)
 *   - Loudness via RMS-to-dB (Python uses amplitude pooling)
 *   - F1 estimation via LPC (not in original Python pipeline; used by the
 *     client to drive lip vertical separation)
 ******************************************************************************/

function workerDebugLog(message, data = null) {
  const timestamp = new Date().toLocaleTimeString();
  if (data) {
    console.log(`[${timestamp}] WORKER: ${message}`, data);
  } else {
    console.log(`[${timestamp}] WORKER: ${message}`);
  }
  self.postMessage({
    type: 'debug',
    message: `${message}${data ? ': ' + JSON.stringify(data, null, 2) : ''}`
  });
}

importScripts('https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort.all.min.js');

if (typeof self.ort === 'undefined') {
  throw new Error('ONNX Runtime failed to load in worker');
}
workerDebugLog('ONNX Runtime loaded, version: ' + self.ort.version);

let wavlmSession = null;
let linearModel = null;
let linearModelWorkingMemory = null;
let initialized = false;

let pitchHistory = Array(5).fill(0);

let processingStats = {
  totalProcessed: 0,
  totalErrors: 0,
  avgProcessingTime: 0
};

// Running audio statistics for z-score normalization.
// The Python pipeline normalizes the entire utterance at once. For real-time,
// we accumulate mean and variance across the recording session so the
// normalization converges toward full-utterance statistics.
let audioStats = { count: 0, mean: 0, m2: 0 };

function resetAudioStats() {
  audioStats = { count: 0, mean: 0, m2: 0 };
}

function updateAudioStats(samples) {
  for (let i = 0; i < samples.length; i++) {
    audioStats.count++;
    const delta = samples[i] - audioStats.mean;
    audioStats.mean += delta / audioStats.count;
    audioStats.m2 += delta * (samples[i] - audioStats.mean);
  }
}

function getAudioStd() {
  if (audioStats.count < 2) return 0;
  return Math.sqrt(audioStats.m2 / audioStats.count);
}

/******************************************************************************
 * MESSAGE HANDLING
 ******************************************************************************/

// Calibration accumulator: per-articulator EMA sums and frame count.
let calibrationEma = null;

function resetCalibrationEma() {
  calibrationEma = {
    count: 0,
    sums: { td: {x:0,y:0}, tb: {x:0,y:0}, tt: {x:0,y:0},
            li: {x:0,y:0}, ul: {x:0,y:0}, ll: {x:0,y:0} }
  };
}

self.onmessage = async function(e) {
  const message = e.data;
  workerDebugLog(`Received message: ${message.type}`);

  switch (message.type) {
    case 'init':
      await initializeModels(message.onnxPath, message.linearModelPath);
      break;
    case 'process':
      await handleProcessMessage(message);
      break;
    case 'reset_stats':
      resetAudioStats();
      break;
    case 'calibrate':
      await handleCalibrateMessage(message);
      break;
    case 'calibrate_start':
      resetAudioStats();
      resetCalibrationEma();
      workerDebugLog('Calibration started: stats and EMA accumulator reset');
      break;
    case 'calibrate_finish': {
      const result = {
        audioStats: { count: audioStats.count, mean: audioStats.mean,
                      std: getAudioStd() },
        emaMeans: null
      };
      if (calibrationEma && calibrationEma.count > 0) {
        const n = calibrationEma.count;
        result.emaMeans = {};
        for (const key of Object.keys(calibrationEma.sums)) {
          result.emaMeans[key] = {
            x: calibrationEma.sums[key].x / n,
            y: calibrationEma.sums[key].y / n
          };
        }
      }
      workerDebugLog('Calibration finished', {
        audioSamples: audioStats.count,
        emaFrames: calibrationEma ? calibrationEma.count : 0
      });
      self.postMessage({ type: 'calibration_result', ...result });
      break;
    }
    default:
      workerDebugLog(`Unknown message type: ${message.type}`);
  }
};

async function handleProcessMessage(message) {
  if (!initialized) {
    self.postMessage({ type: 'error', error: 'Worker not initialized' });
    return;
  }

  const startTime = performance.now();
  processingStats.totalProcessed++;

  try {
    let audioData;
    if (message.audio instanceof Float32Array) {
      audioData = message.audio;
    } else {
      audioData = new Float32Array(message.audio);
    }

    const config = message.config;
    const deviceSampleRate = message.deviceSampleRate || 16000;

    if (deviceSampleRate !== 16000) {
      audioData = resampleTo16k(audioData, deviceSampleRate);
    }

    if (!validateAudioData(audioData)) {
      throw new Error('Invalid audio data received');
    }

    const result = await processAudioWithModels(audioData, config);

    self.postMessage({
      type: 'features',
      articulationFeatures: result.articulationFeatures,
      pitch: result.pitch || 0,
      loudness: result.loudness || -60,
      f1: result.f1 || 0
    });

    const processingTime = performance.now() - startTime;
    processingStats.avgProcessingTime =
      (processingStats.avgProcessingTime * (processingStats.totalProcessed - 1) + processingTime) /
      processingStats.totalProcessed;

    if (processingStats.totalProcessed % 10 === 0) {
      workerDebugLog(`Processing: ${processingTime.toFixed(1)}ms (avg ${processingStats.avgProcessingTime.toFixed(1)}ms)`);
    }

  } catch (error) {
    processingStats.totalErrors++;
    workerDebugLog(`Processing error: ${error.message}`);
    self.postMessage({
      type: 'error',
      error: `Processing failed: ${error.message}`,
      stats: processingStats
    });
  }
}

async function handleCalibrateMessage(message) {
  if (!initialized) {
    self.postMessage({ type: 'error', error: 'Worker not initialized' });
    return;
  }

  try {
    let audioData;
    if (message.audio instanceof Float32Array) {
      audioData = message.audio;
    } else {
      audioData = new Float32Array(message.audio);
    }

    const deviceSampleRate = message.deviceSampleRate || 16000;
    if (deviceSampleRate !== 16000) {
      audioData = resampleTo16k(audioData, deviceSampleRate);
    }

    if (!validateAudioData(audioData)) return;

    const wavlmOutput = await extractWavLMFeatures(audioData, wavlmSession);
    if (!wavlmOutput) return;

    const emaFeatures = extractArticulationFeatures(wavlmOutput);
    if (!emaFeatures || !calibrationEma) return;

    calibrationEma.count++;
    for (const key of Object.keys(calibrationEma.sums)) {
      calibrationEma.sums[key].x += emaFeatures[key].x;
      calibrationEma.sums[key].y += emaFeatures[key].y;
    }

    self.postMessage({
      type: 'calibration_progress',
      frames: calibrationEma.count,
      audioSamples: audioStats.count
    });
  } catch (error) {
    workerDebugLog(`Calibration processing error: ${error.message}`);
  }
}

function validateAudioData(audioData) {
  if (!audioData || audioData.length === 0) {
    return false;
  }
  let max = -Infinity, min = Infinity, sumSq = 0;
  for (let i = 0; i < audioData.length; i++) {
    const v = audioData[i];
    if (v > max) max = v;
    if (v < min) min = v;
    sumSq += v * v;
  }
  const rms = Math.sqrt(sumSq / audioData.length);

  if (processingStats.totalProcessed % 20 === 0) {
    workerDebugLog('Audio stats', {
      length: audioData.length,
      max: max.toFixed(4),
      min: min.toFixed(4),
      rms: rms.toFixed(6)
    });
  }

  return isFinite(max) && isFinite(min) && isFinite(rms);
}

/**
 * Resample audio from deviceRate to 16 kHz using linear interpolation.
 * WavLM expects 16 kHz; browsers often provide 44100 or 48000 Hz.
 */
function resampleTo16k(audioData, fromRate) {
  const ratio = fromRate / 16000;
  const outLength = Math.round(audioData.length / ratio);
  const out = new Float32Array(outLength);
  for (let i = 0; i < outLength; i++) {
    const srcIdx = i * ratio;
    const lo = Math.floor(srcIdx);
    const hi = Math.min(lo + 1, audioData.length - 1);
    const frac = srcIdx - lo;
    out[i] = audioData[lo] * (1 - frac) + audioData[hi] * frac;
  }
  return out;
}

/******************************************************************************
 * MODEL INITIALIZATION
 ******************************************************************************/

async function initializeModels(onnxPath, linearModelPath) {
  try {
    workerDebugLog('Initializing models...');

    if (typeof self.ort === 'undefined') {
      throw new Error('ONNX Runtime not available');
    }

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

    self.postMessage({ type: 'status', message: 'Loading WavLM model...' });
    workerDebugLog(`Loading WavLM model from: ${onnxPath}`);

    wavlmSession = await self.ort.InferenceSession.create(onnxPath, sessionOptions);
    await testWavLMModel();

    self.postMessage({ type: 'status', message: 'Loading linear projection model...' });
    workerDebugLog(`Loading linear model from: ${linearModelPath}`);

    const response = await fetch(linearModelPath);
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    const modelData = await response.json();
    if (!modelData.weights || !modelData.biases || !modelData.input_dim || !modelData.output_dim) {
      throw new Error('Invalid linear model structure');
    }

    linearModel = {
      weights: modelData.weights.map(w => new Float32Array(w)),
      biases: new Float32Array(modelData.biases),
      inputDim: modelData.input_dim,
      outputDim: modelData.output_dim
    };

    linearModelWorkingMemory = new Float32Array(linearModel.outputDim);

    const wavlmHiddenSize = await getWavLMHiddenSize();
    workerDebugLog('Linear model loaded', {
      inputDim: linearModel.inputDim,
      outputDim: linearModel.outputDim,
      wavlmHiddenSize,
      dimensionsMatch: wavlmHiddenSize === linearModel.inputDim
    });

    testLinearModel(wavlmHiddenSize);

    initialized = true;
    workerDebugLog('All models initialized');
    self.postMessage({ type: 'initialized' });

  } catch (error) {
    workerDebugLog(`Initialization failed: ${error.message}`);
    self.postMessage({
      type: 'error',
      error: `Model initialization failed: ${error.message}`
    });
  }
}

async function getWavLMHiddenSize() {
  const testData = new Float32Array(16000);
  for (let i = 0; i < 16000; i++) {
    testData[i] = 0.1 * Math.sin(2 * Math.PI * 150 * i / 16000);
  }
  const tensor = new self.ort.Tensor('float32', testData, [1, 16000]);
  const feeds = {};
  feeds[wavlmSession.inputNames[0]] = tensor;
  const output = await wavlmSession.run(feeds);
  return output[wavlmSession.outputNames[0]].dims[2];
}

async function testWavLMModel() {
  const testData = new Float32Array(16000);
  for (let i = 0; i < 16000; i++) {
    testData[i] = 0.1 * Math.sin(2 * Math.PI * 150 * i / 16000);
  }
  const tensor = new self.ort.Tensor('float32', testData, [1, 16000]);
  const feeds = {};
  feeds[wavlmSession.inputNames[0]] = tensor;

  const start = performance.now();
  const output = await wavlmSession.run(feeds);
  workerDebugLog(`WavLM test: ${(performance.now() - start).toFixed(1)}ms`, {
    outputShape: output[wavlmSession.outputNames[0]].dims
  });

  const outputData = output[wavlmSession.outputNames[0]].data;
  for (let i = 0; i < Math.min(100, outputData.length); i++) {
    if (!isFinite(outputData[i])) {
      throw new Error('WavLM test output contains NaN or Infinity');
    }
  }
}

function testLinearModel(hiddenSize) {
  const seqLen = 50;
  const testData = new Float32Array(seqLen * hiddenSize);
  for (let i = 0; i < testData.length; i++) {
    testData[i] = (Math.random() - 0.5) * 0.1;
  }
  const testTensor = new self.ort.Tensor('float32', testData, [1, seqLen, hiddenSize]);
  const result = extractArticulationFeatures(testTensor, 1.0);
  if (!result) throw new Error('Linear model test failed');

  for (const key of ['ul', 'll', 'li', 'tt', 'tb', 'td']) {
    if (!result[key] || typeof result[key].x !== 'number' || typeof result[key].y !== 'number') {
      throw new Error(`Linear model test: invalid ${key}`);
    }
  }
  workerDebugLog('Linear model test passed');
}

/******************************************************************************
 * FEATURE EXTRACTION
 *
 * Pipeline (matches Python Speech-Articulatory-Coding/inversion.py):
 *   1. Z-score normalize audio
 *   2. Zero-pad 160 samples each side
 *   3. Run WavLM layer 9 -> hidden states [batch, seq, 1024]
 *   4. Select second-to-last frame
 *   5. Linear projection -> 12 EMA channels
 *
 * EMA output order (per SPARC README):
 *   [td_x, td_y, tb_x, tb_y, tt_x, tt_y,
 *    li_x, li_y, ul_x, ul_y, ll_x, ll_y]
 ******************************************************************************/

async function processAudioWithModels(audioData, config) {
  const wavlmOutput = await extractWavLMFeatures(audioData, wavlmSession);
  if (!wavlmOutput) throw new Error('WavLM feature extraction failed');

  const articulationFeatures = extractArticulationFeatures(wavlmOutput);
  if (!articulationFeatures) throw new Error('Articulation feature extraction failed');

  let pitch = 0;
  let loudness = -60;
  let f1 = 0;
  try {
    pitch = config.extractPitchFn === 2
      ? extractPitchSmoothed(audioData)
      : extractPitch(audioData);
    loudness = calculateLoudness(audioData);
    f1 = estimateF1Smoothed(audioData, 16000);
  } catch (e) {
    workerDebugLog(`Pitch/loudness/F1 error: ${e.message}`);
  }

  return { articulationFeatures, pitch, loudness, f1 };
}

function calculateLoudness(audioData) {
  let sum = 0;
  for (let i = 0; i < audioData.length; i++) {
    sum += audioData[i] * audioData[i];
  }
  const rms = Math.sqrt(sum / audioData.length);
  return rms > 0 ? 20 * Math.log10(rms) : -60;
}

async function extractWavLMFeatures(audioData, session) {
  const baseLength = 16000;
  const zeroPad = 160;
  const copyLength = Math.min(audioData.length, baseLength);

  const rawData = new Float32Array(baseLength);
  for (let i = 0; i < copyLength; i++) {
    rawData[i] = audioData[i];
  }

  // Update running statistics with this window's audio
  updateAudioStats(rawData);

  // Z-score normalization using running mean/std accumulated across the
  // recording session (matches Python's full-utterance normalization).
  const mean = audioStats.mean;
  const std = getAudioStd();

  if (std > 1e-8) {
    for (let i = 0; i < baseLength; i++) {
      rawData[i] = (rawData[i] - mean) / std;
    }
  }

  // Zero-padding: 160 samples each side (matches Python inversion.py zero_pad=true)
  const inputLength = baseLength + 2 * zeroPad;
  const inputData = new Float32Array(inputLength);
  for (let i = 0; i < baseLength; i++) {
    inputData[zeroPad + i] = rawData[i];
  }

  const inputTensor = new self.ort.Tensor('float32', inputData, [1, inputLength]);
  const feeds = {};
  feeds[session.inputNames[0]] = inputTensor;

  const outputData = await session.run(feeds);
  const output = outputData[session.outputNames[0]];

  const checkSize = Math.min(100, output.data.length);
  for (let i = 0; i < checkSize; i++) {
    if (!isFinite(output.data[i])) {
      throw new Error('WavLM output contains NaN or Infinity');
    }
  }

  return output;
}

function extractArticulationFeatures(wavlmFeatures) {
  if (!linearModel) throw new Error('Linear model not loaded');

  const features = wavlmFeatures.data;
  const [batchSize, seqLength, hiddenSize] = wavlmFeatures.dims;

  if (hiddenSize !== linearModel.inputDim) {
    throw new Error(
      `Dimension mismatch: WavLM=${hiddenSize}, linear model expects ${linearModel.inputDim}. ` +
      `Use ${linearModel.inputDim === 1024 ? 'wavlm-large' : 'wavlm-base'}.`
    );
  }

  // Second-to-last frame: the last frame may have edge effects
  const frameIdx = Math.max(0, seqLength - 2);
  const startIdx = frameIdx * hiddenSize;

  // Linear projection: output = hidden @ weights.T + biases
  const output = linearModelWorkingMemory;
  output.set(linearModel.biases);
  for (let i = 0; i < linearModel.outputDim; i++) {
    const weights = linearModel.weights[i];
    for (let j = 0; j < hiddenSize; j++) {
      output[i] += weights[j] * features[startIdx + j];
    }
  }

  const articulationFeatures = {
    td: { x: output[0], y: output[1] },
    tb: { x: output[2], y: output[3] },
    tt: { x: output[4], y: output[5] },
    li: { x: output[6], y: output[7] },
    ul: { x: output[8], y: output[9] },
    ll: { x: output[10], y: output[11] }
  };

  for (const [key, point] of Object.entries(articulationFeatures)) {
    if (!isFinite(point.x) || !isFinite(point.y)) {
      throw new Error(`Invalid coordinates for articulator ${key}`);
    }
  }

  return articulationFeatures;
}

/******************************************************************************
 * F1 (FIRST FORMANT) ESTIMATION
 *
 * Uses LPC (Linear Predictive Coding) to estimate the spectral envelope,
 * then picks the first peak in the formant range (200-1000 Hz).
 * F1 correlates strongly with jaw/mouth opening:
 *   /i/ ≈ 270 Hz (closed), /a/ ≈ 730 Hz (open), /u/ ≈ 300 Hz (closed).
 ******************************************************************************/

// Lower LPC order produces a smoother spectral envelope where true formant
// peaks stand out (order 18 was creating spurious peaks that masked F1).
const F1_LPC_ORDER = 10;
const F1_WINDOW_SEC = 0.050;   // 50 ms window — captures more pitch periods
const F1_NFFT = 1024;          // finer frequency resolution (≈15.6 Hz/bin)
const F1_MIN_HZ = 200;
const F1_MAX_HZ = 1000;
const F1_ENERGY_FLOOR = 1e-4;  // skip silence / very quiet frames

let f1History = [];
const F1_HISTORY_LEN = 3;

function estimateF1(audioData, sampleRate) {
  const windowSize = Math.min(Math.floor(sampleRate * F1_WINDOW_SEC), audioData.length);
  if (windowSize < 128) return 0;
  const start = Math.floor((audioData.length - windowSize) / 2);

  // Energy gate — don't estimate on silence
  let energy = 0;
  for (let i = 0; i < windowSize; i++) {
    energy += audioData[start + i] * audioData[start + i];
  }
  energy /= windowSize;
  if (energy < F1_ENERGY_FLOOR) return 0;

  // Pre-emphasis + Hamming window
  const windowed = new Float64Array(windowSize);
  windowed[0] = audioData[start] * 0.08;
  for (let i = 1; i < windowSize; i++) {
    const preEmph = audioData[start + i] - 0.97 * audioData[start + i - 1];
    windowed[i] = preEmph * (0.54 - 0.46 * Math.cos(2 * Math.PI * i / (windowSize - 1)));
  }

  // Autocorrelation
  const order = Math.min(F1_LPC_ORDER, windowSize - 1);
  const r = new Float64Array(order + 1);
  for (let k = 0; k <= order; k++) {
    let sum = 0;
    for (let i = 0; i < windowSize - k; i++) sum += windowed[i] * windowed[i + k];
    r[k] = sum;
  }
  if (r[0] < 1e-10) return 0;

  // Levinson-Durbin recursion → LPC coefficients
  const a = new Float64Array(order + 1);
  const prev = new Float64Array(order + 1);
  a[0] = 1;
  let err = r[0];
  for (let i = 1; i <= order; i++) {
    let lambda = 0;
    for (let j = 0; j < i; j++) lambda -= a[j] * r[i - j];
    lambda /= err;
    prev.set(a);
    for (let j = 0; j <= i; j++) a[j] = prev[j] + lambda * prev[i - j];
    err *= (1 - lambda * lambda);
    if (err <= 0) return 0;
  }

  // LPC power spectrum (evaluate 1/|A(f)|^2)
  const halfFFT = F1_NFFT / 2;
  const spectrum = new Float64Array(halfFFT);
  for (let k = 0; k < halfFFT; k++) {
    let re = 1, im = 0;
    for (let i = 1; i <= order; i++) {
      const angle = -2 * Math.PI * i * k / F1_NFFT;
      re += a[i] * Math.cos(angle);
      im += a[i] * Math.sin(angle);
    }
    spectrum[k] = 1.0 / (re * re + im * im + 1e-12);
  }

  // Find all local peaks in [F1_MIN_HZ, F1_MAX_HZ], pick the strongest
  const minBin = Math.ceil(F1_MIN_HZ * F1_NFFT / sampleRate);
  const maxBin = Math.min(Math.floor(F1_MAX_HZ * F1_NFFT / sampleRate), halfFFT - 2);

  let bestBin = -1;
  let bestVal = -Infinity;
  for (let k = minBin + 1; k <= maxBin; k++) {
    if (spectrum[k] > spectrum[k - 1] && spectrum[k] > spectrum[k + 1] && spectrum[k] > bestVal) {
      bestVal = spectrum[k];
      bestBin = k;
    }
  }
  if (bestBin < 0) return 0;

  // Parabolic interpolation for sub-bin precision
  if (bestBin > 0 && bestBin < halfFFT - 1) {
    const y0 = spectrum[bestBin - 1], y1 = spectrum[bestBin], y2 = spectrum[bestBin + 1];
    const denom = 2 * (2 * y1 - y0 - y2);
    if (Math.abs(denom) > 1e-12) {
      const delta = (y0 - y2) / denom;
      return (bestBin + delta) * sampleRate / F1_NFFT;
    }
  }
  return bestBin * sampleRate / F1_NFFT;
}

function estimateF1Smoothed(audioData, sampleRate) {
  const raw = estimateF1(audioData, sampleRate);
  f1History.push(raw);
  if (f1History.length > F1_HISTORY_LEN) f1History.shift();
  // Include zeros so silence/transitions flush old values quickly
  const sorted = [...f1History].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
}

/******************************************************************************
 * PITCH DETECTION (YIN ALGORITHM)
 *
 * Python uses CREPE or PENN; this JS implementation uses YIN for simplicity
 * and to avoid an additional neural network dependency.
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

    for (let tau = 0; tau < yinBuffer.length; tau++) {
      yinBuffer[tau] = 0;
      for (let j = 0; j < yinBuffer.length; j++) {
        const delta = buffer[j] - buffer[j + tau];
        yinBuffer[tau] += delta * delta;
      }
    }

    yinBuffer[0] = 1;
    let runningSum = 0;
    for (let tau = 1; tau < yinBuffer.length; tau++) {
      runningSum += yinBuffer[tau];
      yinBuffer[tau] = runningSum === 0 ? 1 : yinBuffer[tau] * tau / runningSum;
    }

    for (let tau = this.minPeriod; tau <= this.maxPeriod && tau < yinBuffer.length; tau++) {
      if (yinBuffer[tau] < this.threshold) {
        return this.sampleRate / this.parabolicInterpolation(yinBuffer, tau);
      }
    }
    return 0;
  }

  parabolicInterpolation(array, position) {
    if (position === 0 || position === array.length - 1) return position;
    const y1 = array[position - 1];
    const y2 = array[position];
    const y3 = array[position + 1];
    const a = (y3 + y1 - 2 * y2) / 2;
    if (a === 0) return position;
    return position - (y3 - y1) / (4 * a);
  }
}

let yinDetector = null;

function extractPitch(audioData) {
  if (!yinDetector) {
    yinDetector = new YINPitchDetector({ sampleRate: 16000 });
  }
  const bufferSize = Math.min(audioData.length, 2048);
  const startIdx = Math.floor((audioData.length - bufferSize) / 2);
  return yinDetector.detect(audioData.slice(startIdx, startIdx + bufferSize)) || 0;
}

function extractPitchSmoothed(audioData) {
  const rawPitch = extractPitch(audioData);
  pitchHistory.push(rawPitch);
  pitchHistory.shift();

  const nonZero = pitchHistory.filter(p => p > 0);
  if (nonZero.length === 0) return 0;
  const sorted = [...nonZero].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
}
