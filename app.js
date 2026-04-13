/******************************************************************************
 * SPARC Feature Extraction - Web Client
 *
 * Captures microphone audio, sends it to the SPARC web worker for feature
 * extraction, and exposes the resulting articulatory features for visualization.
 *
 * Architecture:
 *   AudioWorklet (or ScriptProcessor fallback) -> circular buffer
 *   -> periodic extraction loop sends 1 s of audio to sparc-worker.js
 *   -> worker returns 6 articulator (x,y) pairs, pitch, and loudness
 *   -> features are smoothed and stored in featureHistory for display
 ******************************************************************************/

/******************************************************************************
 * CONFIGURATION
 ******************************************************************************/
const config = {
  sampleRate: 16000,
  frameSize: 512,
  bufferSize: 16000,       // 1 second circular buffer
  updateInterval: 200,     // ms between extraction requests
  extractPitchFn: 2        // 1 = raw YIN, 2 = median-smoothed YIN
};

/******************************************************************************
 * GLOBAL STATE
 ******************************************************************************/

// Audio capture
let audioContext;
let audioStream;
let workletNode;
let isRecording = false;
let audioBuffer = new Float32Array(config.bufferSize);
let audioBufferIndex = 0;

// Feature state (EMA coordinates in MNGU0 space, typically -4 to +4)
const DISPLAY_MIN = -5.0;
const DISPLAY_MAX = 4.0;

let smoothedFeatures = {
  ul_x: 0.5, ul_y: -0.3,
  ll_x: 0.5, ll_y: 0.2,
  li_x: 0.3, li_y: 0.4,
  tt_x: 0.2, tt_y: 0.0,
  tb_x: -0.3, tb_y: -0.2,
  td_x: -0.8, td_y: -0.1,
  jaw_opening: 0.3
};

let smoothingFactor = 0.4;
let featureHistory = {};

// Worker management
let SparcWorker = null;
let workerInitialized = false;
let pendingWorkerResponses = 0;
let workerResponseTimeouts = new Set();

// Debug counters
let debugCounters = {
  audioDataReceived: 0,
  workerMessagesSent: 0,
  workerResponsesReceived: 0,
  featuresUpdated: 0,
  chartsUpdated: 0,
  errors: 0
};

// Animation state (used by visualization.js demo)
let animationRunning = false;
let animationFrame = null;

/******************************************************************************
 * UTILITY FUNCTIONS
 ******************************************************************************/

function debugLog(message, data = null) {
  const timestamp = new Date().toLocaleTimeString();
  if (data) {
    console.log(`[${timestamp}] SPARC: ${message}`, data);
  } else {
    console.log(`[${timestamp}] SPARC: ${message}`);
  }
}

function clampToDisplay(value) {
  return Math.max(DISPLAY_MIN, Math.min(DISPLAY_MAX, value));
}

function scaleToDisplay(value) {
  return clampToDisplay(value);
}

function updateStatus(message) {
  const el = document.getElementById('status');
  if (!el) return;
  el.textContent = message;

  if (message.includes('ERROR') || message.includes('CRITICAL')) {
    el.style.backgroundColor = '#ffebee';
    el.style.color = '#c62828';
    el.style.fontWeight = 'bold';
  } else if (message.includes('WARNING')) {
    el.style.backgroundColor = '#fff3e0';
    el.style.color = '#ef6c00';
  } else {
    el.style.backgroundColor = '#e8f5e8';
    el.style.color = '#2e7d32';
    el.style.fontWeight = 'normal';
  }
}

function initializeFeatureHistory() {
  const keys = [
    'ul_x', 'ul_y', 'll_x', 'll_y', 'li_x', 'li_y',
    'tt_x', 'tt_y', 'tb_x', 'tb_y', 'td_x', 'td_y',
    'jaw_opening', 'pitch', 'loudness'
  ];
  featureHistory = {};
  keys.forEach(key => {
    featureHistory[key] = Array(100).fill(key === 'jaw_opening' ? 0.2 : 0);
  });
}

/**
 * Jaw opening derived from upper-lip / lower-lip vertical distance.
 * Not a direct model output; used for visualization only.
 */
function calculateJawOpening(ul_y, ll_y) {
  const lipDistance = Math.abs(ll_y - ul_y);
  return Math.min(Math.max((lipDistance - 0.3) / 1.2, 0), 1);
}

/******************************************************************************
 * AUDIO BUFFER
 ******************************************************************************/

function getRecentAudioBuffer() {
  try {
    const recentAudio = new Float32Array(config.bufferSize);
    for (let i = 0; i < config.bufferSize; i++) {
      const index = (audioBufferIndex - config.bufferSize + i + config.bufferSize) % config.bufferSize;
      recentAudio[i] = audioBuffer[index];
    }
    return recentAudio;
  } catch (error) {
    debugLog('Error getting audio buffer', error);
    return new Float32Array(config.bufferSize);
  }
}

function processAudioData(audioData) {
  try {
    debugCounters.audioDataReceived++;
    if (!audioData || audioData.length === 0) return;

    if (debugCounters.audioDataReceived % 50 === 1) {
      let maxSample = 0;
      for (let k = 0; k < audioData.length; k++) {
        const abs = Math.abs(audioData[k]);
        if (abs > maxSample) maxSample = abs;
      }
      debugLog(`Audio chunk: ${audioData.length} samples, peak=${maxSample.toFixed(4)}`);
    }

    for (let i = 0; i < audioData.length; i++) {
      const value = audioData[i];
      audioBuffer[audioBufferIndex] = isFinite(value) ? value : 0;
      audioBufferIndex = (audioBufferIndex + 1) % config.bufferSize;
    }
  } catch (error) {
    debugLog('Error processing audio data', error);
    debugCounters.errors++;
  }
}

/******************************************************************************
 * WEB WORKER MANAGEMENT
 ******************************************************************************/

async function initSparcWorker() {
  if (SparcWorker) return Promise.resolve();

  return new Promise((resolve, reject) => {
    debugLog('Initializing ML worker...');
    SparcWorker = new Worker('sparc-worker.js');

    const initTimeout = setTimeout(() => {
      reject(new Error('Worker initialization timeout'));
    }, 15000);

    SparcWorker.onmessage = function(e) {
      const message = e.data;

      workerResponseTimeouts.forEach(id => {
        clearTimeout(id);
        workerResponseTimeouts.delete(id);
      });

      switch (message.type) {
        case 'initialized':
          clearTimeout(initTimeout);
          workerInitialized = true;
          resolve();
          break;
        case 'debug':
          console.log('WORKER:', message.message);
          break;
        case 'features':
          handleWorkerFeatures(message);
          break;
        case 'status':
          updateStatus(message.message);
          break;
        case 'error':
          clearTimeout(initTimeout);
          debugLog('Worker error', message);
          reject(new Error(message.error || 'Unknown worker error'));
          break;
      }
    };

    SparcWorker.onerror = function(error) {
      clearTimeout(initTimeout);
      reject(new Error(`Worker creation failed: ${error.message}`));
    };

    const modelVersion = 'v3';
    SparcWorker.postMessage({
      type: 'init',
      onnxPath: `models/wavlm_large_layer9.onnx?v=${modelVersion}`,
      linearModelPath: `models/wavlm_linear_model.json?v=${modelVersion}`
    });
  });
}

function handleWorkerFeatures(message) {
  pendingWorkerResponses = Math.max(0, pendingWorkerResponses - 1);
  debugCounters.workerResponsesReceived++;

  try {
    if (!message.articulationFeatures) {
      throw new Error('No articulation features in message');
    }

    const { articulationFeatures, pitch, loudness } = message;

    for (const key of ['ul', 'll', 'li', 'tt', 'tb', 'td']) {
      if (!articulationFeatures[key] ||
          typeof articulationFeatures[key].x !== 'number' ||
          typeof articulationFeatures[key].y !== 'number') {
        throw new Error(`Invalid articulation feature: ${key}`);
      }
    }

    updateFeatureHistory(articulationFeatures, pitch || 0, loudness || -60);
    debugCounters.featuresUpdated++;

    requestAnimationFrame(() => {
      if (typeof updateCharts === 'function') {
        updateCharts();
      }
      debugCounters.chartsUpdated++;
    });
  } catch (error) {
    debugLog('Error processing worker features', error);
    debugCounters.errors++;
  }
}

/******************************************************************************
 * FEATURE EXTRACTION LOOP
 ******************************************************************************/

async function extractFeaturesLoop() {
  if (!isRecording) return;
  setTimeout(extractFeaturesLoop, config.updateInterval);

  if (!workerInitialized) {
    updateStatus('ERROR: ML models not loaded');
    return;
  }

  if (pendingWorkerResponses >= 1) return;

  try {
    const recentAudio = getRecentAudioBuffer();
    if (!recentAudio || recentAudio.length === 0) return;

    const timeoutId = setTimeout(() => {
      if (workerResponseTimeouts.has(timeoutId)) {
        pendingWorkerResponses = Math.max(0, pendingWorkerResponses - 1);
        workerResponseTimeouts.delete(timeoutId);
        updateStatus('ERROR: ML processing timeout');
      }
    }, 1000);

    workerResponseTimeouts.add(timeoutId);

    SparcWorker.postMessage({
      type: 'process',
      audio: new Float32Array(recentAudio),
      config: config
    });

    pendingWorkerResponses++;
    debugCounters.workerMessagesSent++;
  } catch (error) {
    debugLog('Feature extraction error', error);
    debugCounters.errors++;
    updateStatus(`ERROR: ${error.message}`);
  }
}

/******************************************************************************
 * FEATURE HISTORY & SMOOTHING
 ******************************************************************************/

function updateFeatureHistory(articulationFeatures, pitch, loudness) {
  try {
    if (!articulationFeatures || typeof pitch !== 'number' || typeof loudness !== 'number') {
      throw new Error('Invalid feature data');
    }

    const alpha = isRecording ? smoothingFactor : 0.3;

    for (const art of ['ul', 'll', 'li', 'tt', 'tb', 'td']) {
      if (!articulationFeatures[art]) continue;

      let newX = articulationFeatures[art].x;
      let newY = articulationFeatures[art].y;
      if (!isFinite(newX) || !isFinite(newY)) continue;

      newX = scaleToDisplay(newX);
      newY = scaleToDisplay(newY);

      const oldX = smoothedFeatures[art + '_x'] || 0;
      const oldY = smoothedFeatures[art + '_y'] || 0;

      smoothedFeatures[art + '_x'] = clampToDisplay(alpha * newX + (1 - alpha) * oldX);
      smoothedFeatures[art + '_y'] = clampToDisplay(alpha * newY + (1 - alpha) * oldY);
    }

    const jawOpening = calculateJawOpening(smoothedFeatures.ul_y, smoothedFeatures.ll_y);
    smoothedFeatures.jaw_opening = alpha * jawOpening + (1 - alpha) * smoothedFeatures.jaw_opening;

    for (const key of Object.keys(featureHistory)) {
      featureHistory[key].shift();
      if (key === 'pitch') {
        featureHistory[key].push(isNaN(pitch) ? 0 : pitch);
      } else if (key === 'loudness') {
        featureHistory[key].push(isNaN(loudness) ? -60 : loudness);
      } else if (key === 'jaw_opening') {
        featureHistory[key].push(smoothedFeatures.jaw_opening);
      } else {
        const value = smoothedFeatures[key];
        featureHistory[key].push(isNaN(value) ? 0 : value);
      }
    }
  } catch (error) {
    debugLog('Error updating feature history', error);
    debugCounters.errors++;
  }
}

/******************************************************************************
 * AUDIO RECORDING
 ******************************************************************************/

const audioProcessorCode = `
class AudioProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.bufferSize = 512;
    this.buffer = new Float32Array(this.bufferSize);
    this.bufferIndex = 0;
  }
  process(inputs, outputs, parameters) {
    const input = inputs[0][0];
    if (input && input.length > 0) {
      this.port.postMessage({ audio: input.slice() });
    }
    return true;
  }
}
registerProcessor('audio-processor', AudioProcessor);
`;

async function startRecording() {
  try {
    debugLog('Starting recording...');

    animationRunning = false;
    if (animationFrame) {
      clearTimeout(animationFrame);
      animationFrame = null;
    }

    audioStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        sampleRate: config.sampleRate,
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true
      }
    });

    audioContext = new (window.AudioContext || window.webkitAudioContext)({
      sampleRate: config.sampleRate
    });
    debugLog(`Audio context: ${audioContext.sampleRate} Hz`);

    if (audioContext.audioWorklet) {
      const blob = new Blob([audioProcessorCode], { type: 'application/javascript' });
      await audioContext.audioWorklet.addModule(URL.createObjectURL(blob));

      workletNode = new AudioWorkletNode(audioContext, 'audio-processor');
      workletNode.port.onmessage = (event) => {
        if (event.data.audio) processAudioData(event.data.audio);
      };

      const source = audioContext.createMediaStreamSource(audioStream);
      source.connect(workletNode);
    } else {
      const source = audioContext.createMediaStreamSource(audioStream);
      const processor = audioContext.createScriptProcessor(config.frameSize, 1, 1);
      processor.onaudioprocess = (event) => {
        processAudioData(event.inputBuffer.getChannelData(0));
      };
      source.connect(processor);
      const silentGain = audioContext.createGain();
      silentGain.gain.value = 0;
      processor.connect(silentGain);
      silentGain.connect(audioContext.destination);
      workletNode = processor;
    }

    isRecording = true;

    const startBtn = document.getElementById('startButton');
    const stopBtn = document.getElementById('stopButton');
    if (startBtn) startBtn.disabled = true;
    if (stopBtn) stopBtn.disabled = false;

    updateStatus('Recording...');
    extractFeaturesLoop();
  } catch (error) {
    debugLog('Error starting recording', error);
    updateStatus('Error: ' + error.message);
  }
}

function stopRecording() {
  if (!audioStream) return;

  audioStream.getTracks().forEach(track => track.stop());
  if (workletNode) { workletNode.disconnect(); workletNode = null; }
  if (audioContext) { audioContext.close(); audioContext = null; }

  isRecording = false;

  const startBtn = document.getElementById('startButton');
  const stopBtn = document.getElementById('stopButton');
  if (startBtn) startBtn.disabled = false;
  if (stopBtn) stopBtn.disabled = true;

  updateStatus('Recording stopped.');

  if (!animationRunning && typeof testArticulatorAnimation === 'function') {
    testArticulatorAnimation();
  }
}

/******************************************************************************
 * INITIALIZATION
 ******************************************************************************/

async function init() {
  try {
    updateStatus('Loading models...');

    initializeFeatureHistory();
    await initSparcWorker();

    if (typeof setupCharts === 'function') setupCharts();
    if (typeof setupSensitivityControls === 'function') setupSensitivityControls();

    const startBtn = document.getElementById('startButton');
    const stopBtn = document.getElementById('stopButton');
    if (startBtn) {
      startBtn.disabled = false;
      startBtn.addEventListener('click', startRecording);
    }
    if (stopBtn) {
      stopBtn.addEventListener('click', stopRecording);
    }

    updateStatus('Models loaded. Ready to start.');
  } catch (error) {
    updateStatus(`CRITICAL ERROR: ${error.message}`);
    debugLog('Model loading failed', error);
    debugCounters.errors++;

    const startBtn = document.getElementById('startButton');
    if (startBtn) startBtn.disabled = true;

    const errorMsg = document.createElement('div');
    errorMsg.style.cssText = `
      position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%);
      background: #ffcdd2; border: 2px solid #f44336; border-radius: 10px;
      padding: 20px; font-size: 16px; font-weight: bold; color: #c62828;
      z-index: 10000; text-align: center; min-width: 400px;
    `;
    errorMsg.innerHTML = `
      <h3>SPARC Initialization Failed</h3>
      <p>The ML models could not be loaded.</p>
      <p><strong>Error:</strong> ${error.message}</p>
      <button onclick="this.parentElement.remove()" style="padding: 10px 20px; margin-top: 10px;">Close</button>
    `;
    document.body.appendChild(errorMsg);
  }
}

document.addEventListener('DOMContentLoaded', function() {
  init().catch(error => {
    console.error('Initialization error:', error);
    updateStatus('Initialization error: ' + error.message);
    debugCounters.errors++;
  });
});
