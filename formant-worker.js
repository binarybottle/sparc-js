/******************************************************************************
 * Formant-Only Worker
 *
 * Lightweight audio analysis without any ML model:
 *   1. LPC-based F1 and F2 estimation
 *   2. YIN pitch detection
 *   3. RMS loudness
 *
 * Returns to the main thread: f1, f2, pitch, loudness.
 * No ONNX runtime, no WavLM, no linear model — instant startup.
 ******************************************************************************/

function workerDebugLog(message, data = null) {
  const timestamp = new Date().toLocaleTimeString();
  if (data) {
    console.log(`[${timestamp}] FORMANT-WORKER: ${message}`, data);
  } else {
    console.log(`[${timestamp}] FORMANT-WORKER: ${message}`);
  }
  self.postMessage({
    type: 'debug',
    message: `${message}${data ? ': ' + JSON.stringify(data, null, 2) : ''}`
  });
}

let processingStats = { totalProcessed: 0, totalErrors: 0, avgProcessingTime: 0 };
let pitchHistory = Array(5).fill(0);

/******************************************************************************
 * MESSAGE HANDLING
 ******************************************************************************/

self.onmessage = async function(e) {
  const message = e.data;

  switch (message.type) {
    case 'init':
      workerDebugLog('Formant worker ready (no model to load)');
      self.postMessage({ type: 'initialized' });
      break;
    case 'process':
      handleProcessMessage(message);
      break;
    case 'reset_stats':
      break;
    default:
      workerDebugLog(`Unknown message type: ${message.type}`);
  }
};

function handleProcessMessage(message) {
  const startTime = performance.now();
  processingStats.totalProcessed++;

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

    if (!validateAudioData(audioData)) {
      throw new Error('Invalid audio data received');
    }

    const formants = estimateFormants(audioData, 16000);
    const pitch = extractPitchSmoothed(audioData);
    const loudness = calculateLoudness(audioData);

    self.postMessage({
      type: 'features',
      f1: formants.f1,
      f2: formants.f2,
      pitch: pitch,
      loudness: loudness
    });

    const processingTime = performance.now() - startTime;
    processingStats.avgProcessingTime =
      (processingStats.avgProcessingTime * (processingStats.totalProcessed - 1) + processingTime) /
      processingStats.totalProcessed;

    if (processingStats.totalProcessed % 50 === 0) {
      workerDebugLog(`Processing: ${processingTime.toFixed(1)}ms (avg ${processingStats.avgProcessingTime.toFixed(1)}ms)`);
    }

  } catch (error) {
    processingStats.totalErrors++;
    workerDebugLog(`Processing error: ${error.message}`);
    self.postMessage({ type: 'error', error: `Processing failed: ${error.message}` });
  }
}

function validateAudioData(audioData) {
  if (!audioData || audioData.length === 0) return false;
  let max = -Infinity, min = Infinity, sumSq = 0;
  for (let i = 0; i < audioData.length; i++) {
    const v = audioData[i];
    if (v > max) max = v;
    if (v < min) min = v;
    sumSq += v * v;
  }
  return isFinite(max) && isFinite(min) && isFinite(Math.sqrt(sumSq / audioData.length));
}

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
 * FORMANT ESTIMATION (LPC)
 *
 * LPC order 14 (sample_rate_kHz - 2) resolves F1 and F2 reliably.
 * F1 = strongest peak in 200-1000 Hz; F2 = strongest peak above F1+200 Hz gap.
 ******************************************************************************/

const FORMANT_LPC_ORDER = 18;
const FORMANT_WINDOW_SEC = 0.050;
const FORMANT_NFFT = 1024;
const FORMANT_MIN_HZ = 200;
const FORMANT_F1_MAX_HZ = 1000;
const FORMANT_F2_MAX_HZ = 2500;
const FORMANT_ENERGY_FLOOR = 1e-4;

function estimateFormants(audioData, sampleRate) {
  const windowSize = Math.min(Math.floor(sampleRate * FORMANT_WINDOW_SEC), audioData.length);
  if (windowSize < 128) return { f1: 0, f2: 0 };
  const start = Math.floor((audioData.length - windowSize) / 2);

  let energy = 0;
  for (let i = 0; i < windowSize; i++) {
    energy += audioData[start + i] * audioData[start + i];
  }
  energy /= windowSize;
  if (energy < FORMANT_ENERGY_FLOOR) return { f1: 0, f2: 0 };

  const windowed = new Float64Array(windowSize);
  windowed[0] = audioData[start] * 0.08;
  for (let i = 1; i < windowSize; i++) {
    const preEmph = audioData[start + i] - 0.97 * audioData[start + i - 1];
    windowed[i] = preEmph * (0.54 - 0.46 * Math.cos(2 * Math.PI * i / (windowSize - 1)));
  }

  const order = Math.min(FORMANT_LPC_ORDER, windowSize - 1);
  const r = new Float64Array(order + 1);
  for (let k = 0; k <= order; k++) {
    let sum = 0;
    for (let i = 0; i < windowSize - k; i++) sum += windowed[i] * windowed[i + k];
    r[k] = sum;
  }
  if (r[0] < 1e-10) return { f1: 0, f2: 0 };

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
    if (err <= 0) return { f1: 0, f2: 0 };
  }

  const halfFFT = FORMANT_NFFT / 2;
  const spectrum = new Float64Array(halfFFT);
  for (let k = 0; k < halfFFT; k++) {
    let re = 1, im = 0;
    for (let i = 1; i <= order; i++) {
      const angle = -2 * Math.PI * i * k / FORMANT_NFFT;
      re += a[i] * Math.cos(angle);
      im += a[i] * Math.sin(angle);
    }
    spectrum[k] = 1.0 / (re * re + im * im + 1e-12);
  }

  const minBin = Math.ceil(FORMANT_MIN_HZ * FORMANT_NFFT / sampleRate);
  const maxBin = Math.min(Math.floor(FORMANT_F2_MAX_HZ * FORMANT_NFFT / sampleRate), halfFFT - 2);
  const f1MaxBin = Math.floor(FORMANT_F1_MAX_HZ * FORMANT_NFFT / sampleRate);

  const peaks = [];
  for (let k = minBin + 1; k <= maxBin; k++) {
    if (spectrum[k] > spectrum[k - 1] && spectrum[k] > spectrum[k + 1]) {
      peaks.push({ bin: k, val: spectrum[k] });
    }
  }
  if (peaks.length === 0) return { f1: 0, f2: 0 };

  let f1Bin = -1, f1Val = -Infinity;
  for (const p of peaks) {
    if (p.bin <= f1MaxBin && p.val > f1Val) { f1Val = p.val; f1Bin = p.bin; }
  }

  let f2Bin = -1, f2Val = -Infinity;
  const gapBins = Math.ceil(200 * FORMANT_NFFT / sampleRate);
  const f2MinBin = f1Bin > 0 ? f1Bin + gapBins : Math.ceil(800 * FORMANT_NFFT / sampleRate);
  for (const p of peaks) {
    if (p.bin >= f2MinBin && p.val > f2Val) { f2Val = p.val; f2Bin = p.bin; }
  }

  function binToHz(bin) {
    if (bin <= 0 || bin >= halfFFT - 1) return bin > 0 ? bin * sampleRate / FORMANT_NFFT : 0;
    const y0 = spectrum[bin - 1], y1 = spectrum[bin], y2 = spectrum[bin + 1];
    const denom = 2 * (2 * y1 - y0 - y2);
    if (Math.abs(denom) > 1e-12) {
      return (bin + (y0 - y2) / denom) * sampleRate / FORMANT_NFFT;
    }
    return bin * sampleRate / FORMANT_NFFT;
  }

  return {
    f1: f1Bin > 0 ? binToHz(f1Bin) : 0,
    f2: f2Bin > 0 ? binToHz(f2Bin) : 0
  };
}

/******************************************************************************
 * PITCH DETECTION (YIN ALGORITHM)
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
  if (!yinDetector) yinDetector = new YINPitchDetector({ sampleRate: 16000 });
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

/******************************************************************************
 * LOUDNESS
 ******************************************************************************/

function calculateLoudness(audioData) {
  let sum = 0;
  for (let i = 0; i < audioData.length; i++) {
    sum += audioData[i] * audioData[i];
  }
  const rms = Math.sqrt(sum / audioData.length);
  return rms > 0 ? 20 * Math.log10(rms) : -60;
}
