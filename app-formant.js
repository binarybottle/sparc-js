/******************************************************************************
 * SPARC Formant-Only Client
 *
 * Captures microphone audio and maps formant frequencies to articulator
 * positions — no ML model required.
 *
 * Architecture:
 *   AudioWorklet (or ScriptProcessor fallback) -> circular buffer
 *   -> periodic extraction loop sends audio to formant-worker.js
 *   -> worker returns F1, F2, pitch, loudness (via LPC + YIN)
 *   -> lip y-positions driven by F1
 *   -> tongue/LI positions driven by F1+F2 via bilinear interpolation
 *      between corner vowels /i/, /a/, /u/
 *   -> features smoothed and stored in featureHistory for display
 *
 * Also manages: Set References (speaker-specific F1/F2 capture) and
 * test sound selection. No calibration (no model z-scores to calibrate).
 ******************************************************************************/

/******************************************************************************
 * CONFIGURATION
 ******************************************************************************/
const config = {
  targetSampleRate: 16000,
  deviceSampleRate: null,
  frameSize: 512,
  bufferDuration: 0.5,
  bufferSize: 8000,
  updateInterval: 100
};

/******************************************************************************
 * GLOBAL STATE
 ******************************************************************************/

let audioContext;
let audioStream;
let workletNode;
let isRecording = false;
let audioBuffer = new Float32Array(config.bufferSize);
let audioBufferIndex = 0;

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

let FormantWorker = null;
let workerInitialized = false;
let pendingWorkerResponses = 0;
let workerResponseTimeouts = new Set();

let debugCounters = {
  audioDataReceived: 0,
  workerMessagesSent: 0,
  workerResponsesReceived: 0,
  featuresUpdated: 0,
  chartsUpdated: 0,
  errors: 0
};

let animationRunning = false;
let animationFrame = null;

// Set-references state
let isSettingRefs = false;
let setRefAudioContext = null;
let setRefAudioStream = null;
let setRefWorkletNode = null;
let setRefTimer = null;
let setRefFrames = [];
let setRefResults = {};
const SET_REF_VOWELS = [
  { id: 'i', label: '/i/', desc: 'Say "ee" as in "see"' },
  { id: 'e', label: '/e/', desc: 'Say "eh" as in "bed"' },
  { id: 'a', label: '/a/', desc: 'Say "ah" as in "father"' },
  { id: 'o', label: '/o/', desc: 'Say "oh" as in "go"' },
  { id: 'u', label: '/u/', desc: 'Say "oo" as in "food"' }
];
let setRefVowelIndex = 0;
let setRefCapturing = false;
const SET_REF_STORAGE_KEY = 'sparc-formant-reference-positions';

/******************************************************************************
 * UTILITY FUNCTIONS
 ******************************************************************************/

function debugLog(message, data = null) {
  const timestamp = new Date().toLocaleTimeString();
  if (data) {
    console.log(`[${timestamp}] SPARC-F: ${message}`, data);
  } else {
    console.log(`[${timestamp}] SPARC-F: ${message}`);
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
 * DISPLAY TRANSFORM
 ******************************************************************************/

const ARTICULATOR_CENTERS = {
  td: { x: -4.0, y: -3.2 },
  tb: { x: -2.0, y: -2.5 },
  tt: { x:  0.5, y: -0.3 },
  li: { x:  2.0, y:  0.0 },
  ul: { x:  3.0, y: -1.25 },
  ll: { x:  3.0, y: -1.25 }
};

const DISPLAY_SCALES = {
  td: { x: 0.8, y: 1.2 },
  tb: { x: 0.8, y: 1.2 },
  tt: { x: 0.8, y: 1.2 },
  li: { x: 0.5, y: 1.0 },
  ul: { x: 0.0, y: 0.0 },
  ll: { x: 0.0, y: 0.0 }
};

function emaToDisplay(key, z_x, z_y) {
  const c = ARTICULATOR_CENTERS[key];
  const s = DISPLAY_SCALES[key];
  return {
    x: c.x + z_x * s.x,
    y: c.y - z_y * s.y
  };
}

/******************************************************************************
 * F1-DRIVEN LIP POSITIONING
 ******************************************************************************/

const F1_CLOSED_HZ = 250;
const F1_OPEN_HZ   = 650;
const LIP_CENTER_Y = -1.25;
const LIP_HALF_GAP_CLOSED = 0.3;
const LIP_HALF_GAP_OPEN   = 1.8;

function f1ToLipPositions(f1Hz) {
  const t = Math.max(0, Math.min(1, (f1Hz - F1_CLOSED_HZ) / (F1_OPEN_HZ - F1_CLOSED_HZ)));
  const halfGap = LIP_HALF_GAP_CLOSED + t * (LIP_HALF_GAP_OPEN - LIP_HALF_GAP_CLOSED);
  return {
    ulY: LIP_CENTER_Y - halfGap,
    llY: LIP_CENTER_Y + halfGap
  };
}

/******************************************************************************
 * F1+F2 DRIVEN TONGUE / LI POSITIONING
 *
 * Bilinear interpolation between three corner vowels in (F1,F2) space:
 *   /i/ (high-front):  low F1, high F2
 *   /a/ (low-central): high F1, mid F2
 *   /u/ (high-back):   low F1, low F2
 ******************************************************************************/

const FORMANT_F1_MIN = 250;
const FORMANT_F1_MAX = 650;
const FORMANT_F2_MIN = 800;
const FORMANT_F2_MAX = 2400;

const CORNER_I = { td:{x:-0.3,y:0.5}, tb:{x:0.5,y:1.5}, tt:{x:0.8,y:0.8}, li:{x:0,y:1.0} };
const CORNER_A = { td:{x:-0.2,y:-1.2}, tb:{x:0,y:-1.0}, tt:{x:0.2,y:-0.5}, li:{x:0,y:-1.0} };
const CORNER_U = { td:{x:-0.8,y:1.0}, tb:{x:-0.5,y:0.8}, tt:{x:-0.2,y:0.2}, li:{x:0,y:0.8} };

function formantsToTongueZScores(f1Hz, f2Hz) {
  const height = Math.max(0, Math.min(1,
    (f1Hz - FORMANT_F1_MIN) / (FORMANT_F1_MAX - FORMANT_F1_MIN)));
  const front = Math.max(0, Math.min(1,
    (f2Hz - FORMANT_F2_MIN) / (FORMANT_F2_MAX - FORMANT_F2_MIN)));

  const result = {};
  for (const key of ['td', 'tb', 'tt', 'li']) {
    const highX = CORNER_U[key].x + front * (CORNER_I[key].x - CORNER_U[key].x);
    const highY = CORNER_U[key].y + front * (CORNER_I[key].y - CORNER_U[key].y);
    result[key] = {
      x: highX + height * (CORNER_A[key].x - highX),
      y: highY + height * (CORNER_A[key].y - highY)
    };
  }
  return result;
}

/******************************************************************************
 * WORKER MANAGEMENT
 ******************************************************************************/

async function initFormantWorker() {
  if (FormantWorker) return Promise.resolve();

  return new Promise((resolve, reject) => {
    debugLog('Initializing formant worker...');
    FormantWorker = new Worker('formant-worker.js');

    const initTimeout = setTimeout(() => {
      reject(new Error('Worker initialization timeout'));
    }, 5000);

    FormantWorker.onmessage = function(e) {
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
          console.log('FORMANT-WORKER:', message.message);
          break;
        case 'features':
          handleWorkerFeatures(message);
          break;
        case 'error':
          clearTimeout(initTimeout);
          debugLog('Worker error', message);
          reject(new Error(message.error || 'Unknown worker error'));
          break;
      }
    };

    FormantWorker.onerror = function(error) {
      clearTimeout(initTimeout);
      reject(new Error(`Worker creation failed: ${error.message}`));
    };

    FormantWorker.postMessage({ type: 'init' });
  });
}

/******************************************************************************
 * HANDLE WORKER FEATURES
 ******************************************************************************/

function handleWorkerFeatures(message) {
  pendingWorkerResponses = Math.max(0, pendingWorkerResponses - 1);
  debugCounters.workerResponsesReceived++;

  try {
    const { f1, f2, pitch, loudness } = message;

    debugLog(`Formants: F1=${(f1||0).toFixed(0)}Hz F2=${(f2||0).toFixed(0)}Hz`);

    if (setRefCapturing) {
      setRefFrames.push({ f1: f1 || 0, f2: f2 || 0 });
    }

    const articulationFeatures = {};

    // Lips: F1-driven
    const lip = (f1 > 0) ? f1ToLipPositions(f1) : { ulY: LIP_CENTER_Y - 0.3, llY: LIP_CENTER_Y + 0.3 };
    articulationFeatures.ul = { x: ARTICULATOR_CENTERS.ul.x, y: lip.ulY };
    articulationFeatures.ll = { x: ARTICULATOR_CENTERS.ll.x, y: lip.llY };

    // Tongue + LI: F1+F2 driven
    if (f1 > 0 && f2 > 0) {
      const tongueZ = formantsToTongueZScores(f1, f2);
      for (const key of ['td', 'tb', 'tt', 'li']) {
        articulationFeatures[key] = emaToDisplay(key, tongueZ[key].x, tongueZ[key].y);
      }
    } else {
      for (const key of ['td', 'tb', 'tt', 'li']) {
        articulationFeatures[key] = emaToDisplay(key, 0, 0);
      }
    }

    updateFeatureHistory(articulationFeatures, pitch || 0, loudness || -60);
    updateStatus('Recording...');
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

const SPEECH_ENERGY_THRESHOLD = 0.01;

function audioHasSpeechEnergy(audio) {
  let sumSq = 0;
  for (let i = 0; i < audio.length; i++) {
    sumSq += audio[i] * audio[i];
  }
  return Math.sqrt(sumSq / audio.length) >= SPEECH_ENERGY_THRESHOLD;
}

async function extractFeaturesLoop() {
  if (!isRecording) return;
  setTimeout(extractFeaturesLoop, config.updateInterval);

  if (!workerInitialized) {
    updateStatus('ERROR: Worker not initialized');
    return;
  }

  if (pendingWorkerResponses >= 1) return;

  try {
    const recentAudio = getRecentAudioBuffer();
    if (!recentAudio || recentAudio.length === 0) return;

    if (!audioHasSpeechEnergy(recentAudio)) {
      updateStatus('Listening... (speak into microphone)');
      return;
    }

    const timeoutId = setTimeout(() => {
      if (workerResponseTimeouts.has(timeoutId)) {
        pendingWorkerResponses = Math.max(0, pendingWorkerResponses - 1);
        workerResponseTimeouts.delete(timeoutId);
        updateStatus('ERROR: Processing timeout');
      }
    }, 1000);

    workerResponseTimeouts.add(timeoutId);

    FormantWorker.postMessage({
      type: 'process',
      audio: new Float32Array(recentAudio),
      deviceSampleRate: config.deviceSampleRate || config.targetSampleRate
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
 * SET REFERENCE SOUNDS
 ******************************************************************************/

function showSetRefOverlay() {
  const overlay = document.getElementById('setref-overlay');
  if (!overlay) return;
  setRefVowelIndex = 0;
  setRefResults = {};
  updateSetRefUI();
  overlay.style.display = 'block';
  document.getElementById('setref-start').style.display = '';
  document.getElementById('setref-next').style.display = 'none';
  document.getElementById('setref-progress-bar-container').style.display = 'none';
  document.getElementById('setref-status').textContent = '';
}

function hideSetRefOverlay() {
  const overlay = document.getElementById('setref-overlay');
  if (overlay) overlay.style.display = 'none';
}

function updateSetRefUI() {
  const v = SET_REF_VOWELS[setRefVowelIndex];
  const label = document.getElementById('setref-vowel-label');
  const desc = document.getElementById('setref-vowel-desc');
  if (label) label.textContent = v ? v.label : '';
  if (desc) desc.textContent = v ? v.desc : '';
}

async function startSetRefCapture() {
  if (!workerInitialized) { alert('Worker not ready yet.'); return; }

  try {
    animationRunning = false;
    if (animationFrame) { clearTimeout(animationFrame); animationFrame = null; }

    setRefAudioStream = await navigator.mediaDevices.getUserMedia({
      audio: { sampleRate: config.targetSampleRate, channelCount: 1,
               echoCancellation: true, noiseSuppression: true }
    });

    setRefAudioContext = new (window.AudioContext || window.webkitAudioContext)({
      sampleRate: config.targetSampleRate
    });

    config.deviceSampleRate = setRefAudioContext.sampleRate;
    config.bufferSize = Math.floor(setRefAudioContext.sampleRate * config.bufferDuration);
    audioBuffer = new Float32Array(config.bufferSize);
    audioBufferIndex = 0;

    if (setRefAudioContext.audioWorklet) {
      const blob = new Blob([audioProcessorCode], { type: 'application/javascript' });
      await setRefAudioContext.audioWorklet.addModule(URL.createObjectURL(blob));
      setRefWorkletNode = new AudioWorkletNode(setRefAudioContext, 'audio-processor');
      setRefWorkletNode.port.onmessage = (event) => {
        if (event.data.audio) processAudioData(event.data.audio);
      };
      const source = setRefAudioContext.createMediaStreamSource(setRefAudioStream);
      source.connect(setRefWorkletNode);
    } else {
      const source = setRefAudioContext.createMediaStreamSource(setRefAudioStream);
      const processor = setRefAudioContext.createScriptProcessor(config.frameSize, 1, 1);
      processor.onaudioprocess = (event) => {
        processAudioData(event.inputBuffer.getChannelData(0));
      };
      source.connect(processor);
      const silentGain = setRefAudioContext.createGain();
      silentGain.gain.value = 0;
      processor.connect(silentGain);
      silentGain.connect(setRefAudioContext.destination);
      setRefWorkletNode = processor;
    }

    isSettingRefs = true;
    setRefCapturing = true;
    setRefFrames = [];

    document.getElementById('setref-start').style.display = 'none';
    document.getElementById('setref-next').style.display = '';
    document.getElementById('setref-progress-bar-container').style.display = '';
    document.getElementById('setref-status').textContent =
      `Speak now: ${SET_REF_VOWELS[setRefVowelIndex].desc}`;

    setRefTimer = setInterval(sendSetRefAudio, config.updateInterval);
  } catch (error) {
    debugLog('Set-ref start error', error);
    document.getElementById('setref-status').textContent = 'Error: ' + error.message;
  }
}

function sendSetRefAudio() {
  if (!isSettingRefs || !workerInitialized) return;
  if (pendingWorkerResponses >= 1) return;

  const recentAudio = getRecentAudioBuffer();
  if (!recentAudio || recentAudio.length === 0) return;
  if (!audioHasSpeechEnergy(recentAudio)) return;

  FormantWorker.postMessage({
    type: 'process',
    audio: new Float32Array(recentAudio),
    deviceSampleRate: config.deviceSampleRate || config.targetSampleRate
  });
  pendingWorkerResponses++;
}

function finishCurrentVowel() {
  const vowel = SET_REF_VOWELS[setRefVowelIndex];

  if (setRefFrames.length < 3) {
    document.getElementById('setref-status').textContent =
      'Not enough speech detected. Keep speaking and try again.';
    return;
  }

  let f1Sum = 0, f1Count = 0, f2Sum = 0, f2Count = 0;
  for (const frame of setRefFrames) {
    if (frame.f1 && frame.f1 > 0) { f1Sum += frame.f1; f1Count++; }
    if (frame.f2 && frame.f2 > 0) { f2Sum += frame.f2; f2Count++; }
  }
  const result = {
    _f1: f1Count > 0 ? f1Sum / f1Count : 0,
    _f2: f2Count > 0 ? f2Sum / f2Count : 0
  };

  setRefResults[vowel.id] = result;
  debugLog(`Reference captured for /${vowel.id}/`, { frames: setRefFrames.length, f1: result._f1, f2: result._f2 });

  setRefVowelIndex++;
  setRefFrames = [];

  if (setRefVowelIndex >= SET_REF_VOWELS.length) {
    finishSetRef();
    return;
  }

  updateSetRefUI();
  const pct = (setRefVowelIndex / SET_REF_VOWELS.length) * 100;
  document.getElementById('setref-progress-bar').style.width = pct + '%';
  document.getElementById('setref-status').textContent =
    `Speak now: ${SET_REF_VOWELS[setRefVowelIndex].desc}`;
}

function finishSetRef() {
  stopSetRefAudio();

  localStorage.setItem(SET_REF_STORAGE_KEY, JSON.stringify(setRefResults));
  debugLog('Reference positions saved', setRefResults);

  if (typeof applyLearnedReferences === 'function') {
    applyLearnedReferences(setRefResults);
  }

  hideSetRefOverlay();
  updateStatus('References set. Ready.');

  const btn = document.getElementById('setRefButton');
  if (btn) {
    btn.textContent = 'Re-set References';
    btn.classList.replace('btn-info', 'btn-outline-info');
  }
}

function stopSetRefAudio() {
  setRefCapturing = false;
  isSettingRefs = false;
  if (setRefTimer) { clearInterval(setRefTimer); setRefTimer = null; }
  if (setRefAudioStream) {
    setRefAudioStream.getTracks().forEach(t => t.stop());
    setRefAudioStream = null;
  }
  if (setRefWorkletNode) { setRefWorkletNode.disconnect(); setRefWorkletNode = null; }
  if (setRefAudioContext) { setRefAudioContext.close(); setRefAudioContext = null; }
}

function cancelSetRef() {
  stopSetRefAudio();
  hideSetRefOverlay();
}

function loadSavedReferences() {
  try {
    const saved = localStorage.getItem(SET_REF_STORAGE_KEY);
    if (!saved) return false;
    const refs = JSON.parse(saved);
    if (refs && typeof refs === 'object' && Object.keys(refs).length > 0) {
      if (typeof applyLearnedReferences === 'function') {
        applyLearnedReferences(refs);
      }
      debugLog('Loaded saved reference positions', refs);

      const btn = document.getElementById('setRefButton');
      if (btn) {
        btn.textContent = 'Re-set References';
        btn.classList.replace('btn-info', 'btn-outline-info');
      }
      return true;
    }
  } catch (e) {
    debugLog('Error loading saved references', e);
  }
  return false;
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
        sampleRate: config.targetSampleRate,
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true
      }
    });

    audioContext = new (window.AudioContext || window.webkitAudioContext)({
      sampleRate: config.targetSampleRate
    });

    config.deviceSampleRate = audioContext.sampleRate;
    config.bufferSize = Math.floor(audioContext.sampleRate * config.bufferDuration);
    audioBuffer = new Float32Array(config.bufferSize);
    audioBufferIndex = 0;
    debugLog(`Audio context: ${audioContext.sampleRate} Hz` +
      (audioContext.sampleRate !== config.targetSampleRate
        ? ` (will resample to ${config.targetSampleRate} Hz)` : ''));

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
    updateStatus('Initializing...');

    initializeFeatureHistory();
    await initFormantWorker();

    if (typeof setupCharts === 'function') setupCharts();
    if (typeof setupSensitivityControls === 'function') setupSensitivityControls();

    loadSavedReferences();

    const startBtn = document.getElementById('startButton');
    const stopBtn = document.getElementById('stopButton');
    if (startBtn) {
      startBtn.disabled = false;
      startBtn.addEventListener('click', startRecording);
    }
    if (stopBtn) {
      stopBtn.addEventListener('click', stopRecording);
    }

    // Set References UI wiring
    const setRefBtn = document.getElementById('setRefButton');
    if (setRefBtn) {
      setRefBtn.disabled = false;
      setRefBtn.addEventListener('click', showSetRefOverlay);
    }
    const srStart = document.getElementById('setref-start');
    if (srStart) srStart.addEventListener('click', startSetRefCapture);
    const srNext = document.getElementById('setref-next');
    if (srNext) srNext.addEventListener('click', finishCurrentVowel);
    const srCancel = document.getElementById('setref-cancel');
    if (srCancel) srCancel.addEventListener('click', cancelSetRef);

    updateStatus('Ready to start.');
  } catch (error) {
    updateStatus(`ERROR: ${error.message}`);
    debugLog('Initialization failed', error);
    debugCounters.errors++;
  }
}

document.addEventListener('DOMContentLoaded', function() {
  init().catch(error => {
    console.error('Initialization error:', error);
    updateStatus('Initialization error: ' + error.message);
    debugCounters.errors++;
  });
});
