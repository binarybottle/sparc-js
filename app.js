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
  targetSampleRate: 16000, // WavLM expects 16 kHz
  deviceSampleRate: null,  // set at recording time from AudioContext
  frameSize: 512,
  bufferSize: 16000,       // 1 second circular buffer (resized at recording time)
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

// Calibration state
let isCalibrating = false;
let isCalibrated = false;
let calibrationEmaMeans = null; // per-articulator mean raw EMA from calibration
let calibrationAudioContext = null;
let calibrationAudioStream = null;
let calibrationWorkletNode = null;
let calibrationTimer = null;
let calibrationStartTime = 0;

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
        case 'calibration_result':
          handleCalibrationResult(message);
          break;
        case 'calibration_progress':
          handleCalibrationProgress(message);
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

// MNGU0 EMA anatomical statistics (approximate, in mm).
// The SPARC model outputs z-scored EMA: each channel is z-scored within
// utterances during MNGU0 training (Cho et al. 2024, §III-A). To recover
// anatomical positions we un-z-score using per-channel means (average
// sensor position in mm) and stds (typical within-utterance displacement).
//
// Coordinate frame: origin ≈ upper incisors, +x = anterior, +y = superior.
// Values estimated from the MNGU0 corpus (Richmond et al. 2011) and
// consistent with published EMA analyses of the dataset.
const MNGU0_STATS = {
  td: { mx: -28, my:  5,  sx: 3.5, sy: 3.0 },
  tb: { mx: -15, my:  3,  sx: 4.0, sy: 4.0 },
  tt: { mx:   0, my: -3,  sx: 5.0, sy: 4.5 },
  li: { mx:   0, my: -10, sx: 1.0, sy: 2.0 },
  ul: { mx:   4, my:  4,  sx: 1.2, sy: 0.8 },
  ll: { mx:   4, my: -3,  sx: 1.5, sy: 2.0 }
};

// Map model z-scores → real mm → SVG display coordinates.
// mm_x ∈ [−35, 10] → svg_x ∈ [−5, 4]  (BACK → FRONT)
// mm_y ∈ [−15, 10] → svg_y ∈ [ 4,−5]  (DOWN → UP, flipped for SVG)
function emaToDisplay(key, z_x, z_y) {
  const s = MNGU0_STATS[key];
  const mm_x = z_x * s.sx + s.mx;
  const mm_y = z_y * s.sy + s.my;
  return {
    x: (mm_x + 35) / 45 * 9 - 5,
    y: (10 - mm_y) / 25 * 9 - 5
  };
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
      let zx = articulationFeatures[key].x;
      let zy = articulationFeatures[key].y;
      if (calibrationEmaMeans && calibrationEmaMeans[key]) {
        zx -= calibrationEmaMeans[key].x;
        zy -= calibrationEmaMeans[key].y;
      }
      const disp = emaToDisplay(key, zx, zy);
      articulationFeatures[key].x = disp.x;
      articulationFeatures[key].y = disp.y;
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

// Minimum RMS energy (linear) to consider audio as speech.
// Silence / background noise is typically below -40 dB ≈ 0.01 linear RMS.
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
    updateStatus('ERROR: ML models not loaded');
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
        updateStatus('ERROR: ML processing timeout');
      }
    }, 1000);

    workerResponseTimeouts.add(timeoutId);

    SparcWorker.postMessage({
      type: 'process',
      audio: new Float32Array(recentAudio),
      config: config,
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
 * CALIBRATION PASSAGES
 ******************************************************************************/

const CALIBRATION_PASSAGES = {
  peggy: {
    title: 'Peggy Babcock',
    text: `It was the first day of school. It was a tough day for all the kids. One girl had a really hard time because nobody could say her name. Her name was Peggy Babcock. Go ahead. Try and say it three times quickly. "Peggy Babcock Peggy Babcock Peggy Babcock." Not easy going, right? She was afraid to say hello to any of the other kids on the playground. One boy walked up to her and asked what her name was. She said "When you hear my name it sounds simple but no one can say it. It is Peggy Babcock."

He laughed and said "Your name is tricky but mine is better. It sounds simple but no one can remember it. It is Jonas Norvin Sven Arthur Schwinn Bart Winston Ulysses M." Peggy laughed and said "Easy. Your name sounds like 'Joan is nervous when others win. But you win some, you lose some.' How do you like my version?" Jonas was so happy that he said "Let's be friends. I will call you PB." The pair of them stuck so close to each other that everyone at school called them "PB and J."`
  },
  kingdom: {
    title: 'Phonetic Kingdom',
    text: `Some time ago, in a place neither near nor far, there lived a king who didn't know how to count, not even to zero. Some say this is the reason he would always wish for more \u2014 more food, more gold, more land. He simply didn't realize how much he already owned. Everyone in his kingdom could do the math and tally bushels of corn, loaves of bread, and urns of gold. But how would they measure the height of his castle or the stretch of his kingdom? You might think "Aaah, ooh, easy \u2014 just measure it in meters!" But in those days, the useless unit of measure was based on stains splattered along the king's cloak while drinking shrub juice. The kingdom needed a new way of counting distance. "A kingdom without a proper ruler," proclaimed the king, "is like riches without measure." He launched a challenge amid trumpets, drums, flags and cannons. "The person who creates a unit of measure fit for a ruler will be rewarded beyond measure!" A tall order indeed!

The first person to come forward was a bulky locksmith with a stiff jaw. He approached the king with an air of secrecy and whispered, "I have the key to measure the kingdom, but only I can wield it." He then rubbed his beard and pulled the key from his locks of oily hair. The key turned out to be a hair itself! "Judge the reach of my vast kingdom with a hair's width?" laughed the king. "What a poor idea. That would take forever or longer!"

The second person eager for the prize was a fidgety boy who knew all numbers (including zero). He produced a curious object from one of his many pockets. It was a complex shape that seemed to change proportions depending on which direction you gazed upon it. The boy said in a measured voice, "This polyhedron has many edges, with each edge of a different length. Only a king could be counted on to use it justly." He gave the king an awful earful of an explanation that went on and on. The long and the short of it was that the king could make no more use of it than of a puddle of spilled oatmeal.

Finally, a little girl with a big idea tugged on the mismeasured cloak of the king. The king sized up the little girl with the big idea and said "I don't have time for this, and for that matter, I have no concept of space, either." The girl looked up, then down, then spun around and blurted out: "Aren't you able to solve the puzzle yourself? Why must you break up your kingdom into tiny pieces when everything around you is Humpty Dumpty together again? Your kingdom IS a unit and you are the ruler." The king \u2014 startled, befuddled, and bemused \u2014 found the words wise. He aimed to be satisfied with all around him, big or small or somewhere in between.`
  }
};

/******************************************************************************
 * CALIBRATION
 *
 * During calibration, the user reads a passage aloud. Audio is captured and
 * sent to the worker, which accumulates:
 *   1. Running audio mean/std (for z-score normalization during recording)
 *   2. Per-articulator EMA means (for re-centering the display)
 *
 * Only speech-active chunks (above SPEECH_ENERGY_THRESHOLD) are processed,
 * preventing silence/noise from biasing the statistics.
 ******************************************************************************/

function showCalibrationOverlay() {
  const overlay = document.getElementById('calibration-overlay');
  if (!overlay) return;

  const passageEl = document.getElementById('calibration-passage');
  const selector = document.getElementById('passage-selector');
  if (passageEl && selector) {
    const passage = CALIBRATION_PASSAGES[selector.value];
    passageEl.textContent = passage ? passage.text : '';
  }

  overlay.style.display = 'block';
  document.getElementById('calibration-start').style.display = '';
  document.getElementById('calibration-done').style.display = 'none';
  document.getElementById('calibration-progress').style.display = 'none';
  document.getElementById('calibration-status').textContent = '';
}

function hideCalibrationOverlay() {
  const overlay = document.getElementById('calibration-overlay');
  if (overlay) overlay.style.display = 'none';
}

async function startCalibration() {
  if (!workerInitialized) {
    alert('Models not loaded yet. Please wait.');
    return;
  }

  try {
    animationRunning = false;
    if (animationFrame) { clearTimeout(animationFrame); animationFrame = null; }

    SparcWorker.postMessage({ type: 'calibrate_start' });

    calibrationAudioStream = await navigator.mediaDevices.getUserMedia({
      audio: { sampleRate: config.targetSampleRate, channelCount: 1,
               echoCancellation: true, noiseSuppression: true }
    });

    calibrationAudioContext = new (window.AudioContext || window.webkitAudioContext)({
      sampleRate: config.targetSampleRate
    });

    config.deviceSampleRate = calibrationAudioContext.sampleRate;
    config.bufferSize = calibrationAudioContext.sampleRate;
    audioBuffer = new Float32Array(config.bufferSize);
    audioBufferIndex = 0;

    if (calibrationAudioContext.audioWorklet) {
      const blob = new Blob([audioProcessorCode], { type: 'application/javascript' });
      await calibrationAudioContext.audioWorklet.addModule(URL.createObjectURL(blob));
      calibrationWorkletNode = new AudioWorkletNode(calibrationAudioContext, 'audio-processor');
      calibrationWorkletNode.port.onmessage = (event) => {
        if (event.data.audio) processAudioData(event.data.audio);
      };
      const source = calibrationAudioContext.createMediaStreamSource(calibrationAudioStream);
      source.connect(calibrationWorkletNode);
    } else {
      const source = calibrationAudioContext.createMediaStreamSource(calibrationAudioStream);
      const processor = calibrationAudioContext.createScriptProcessor(config.frameSize, 1, 1);
      processor.onaudioprocess = (event) => {
        processAudioData(event.inputBuffer.getChannelData(0));
      };
      source.connect(processor);
      const silentGain = calibrationAudioContext.createGain();
      silentGain.gain.value = 0;
      processor.connect(silentGain);
      silentGain.connect(calibrationAudioContext.destination);
      calibrationWorkletNode = processor;
    }

    isCalibrating = true;
    calibrationStartTime = Date.now();

    document.getElementById('calibration-start').style.display = 'none';
    document.getElementById('calibration-done').style.display = '';
    document.getElementById('calibration-progress').style.display = '';
    document.getElementById('calibration-status').textContent = 'Reading... speak clearly into the microphone.';

    calibrationTimer = setInterval(sendCalibrationAudio, config.updateInterval);
  } catch (error) {
    debugLog('Calibration start error', error);
    document.getElementById('calibration-status').textContent = 'Error: ' + error.message;
  }
}

function sendCalibrationAudio() {
  if (!isCalibrating || !workerInitialized) return;

  const recentAudio = getRecentAudioBuffer();
  if (!recentAudio || recentAudio.length === 0) return;

  if (!audioHasSpeechEnergy(recentAudio)) return;

  SparcWorker.postMessage({
    type: 'calibrate',
    audio: new Float32Array(recentAudio),
    deviceSampleRate: config.deviceSampleRate || config.targetSampleRate
  });
}

function stopCalibrationAudio() {
  if (calibrationTimer) { clearInterval(calibrationTimer); calibrationTimer = null; }
  if (calibrationAudioStream) {
    calibrationAudioStream.getTracks().forEach(t => t.stop());
    calibrationAudioStream = null;
  }
  if (calibrationWorkletNode) { calibrationWorkletNode.disconnect(); calibrationWorkletNode = null; }
  if (calibrationAudioContext) { calibrationAudioContext.close(); calibrationAudioContext = null; }
  isCalibrating = false;
}

function finishCalibration() {
  stopCalibrationAudio();
  document.getElementById('calibration-status').textContent = 'Processing calibration data...';
  document.getElementById('calibration-done').disabled = true;
  SparcWorker.postMessage({ type: 'calibrate_finish' });
}

function cancelCalibration() {
  stopCalibrationAudio();
  SparcWorker.postMessage({ type: 'reset_stats' });
  hideCalibrationOverlay();
}

function handleCalibrationResult(message) {
  const { audioStats, emaMeans } = message;

  if (!emaMeans || !audioStats || audioStats.count < 1000) {
    document.getElementById('calibration-status').textContent =
      'Not enough speech detected. Please try again and speak louder.';
    document.getElementById('calibration-start').style.display = '';
    document.getElementById('calibration-done').style.display = 'none';
    document.getElementById('calibration-done').disabled = false;
    return;
  }

  calibrationEmaMeans = emaMeans;
  isCalibrated = true;

  debugLog('Calibration complete', {
    audioSamples: audioStats.count,
    audioMean: audioStats.mean.toFixed(6),
    audioStd: audioStats.std.toFixed(6),
    emaMeans
  });

  hideCalibrationOverlay();
  updateStatus('Calibrated. Ready to record.');

  const calibBtn = document.getElementById('calibrateButton');
  if (calibBtn) {
    calibBtn.textContent = 'Recalibrate';
    calibBtn.classList.replace('btn-primary', 'btn-outline-primary');
  }
}

function handleCalibrationProgress(message) {
  const bar = document.getElementById('calibration-progress-bar');
  const statusEl = document.getElementById('calibration-status');
  if (!bar || !statusEl) return;

  const elapsed = (Date.now() - calibrationStartTime) / 1000;
  const frames = message.frames || 0;
  statusEl.textContent = `Reading... ${Math.round(elapsed)}s \u2014 ${frames} speech frames captured`;

  const pct = Math.min(100, (frames / 20) * 100);
  bar.style.width = pct + '%';
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

    // If calibrated, keep the calibration audio stats for stable normalization.
    // Otherwise reset so normalization starts fresh.
    if (SparcWorker && !isCalibrated) {
      SparcWorker.postMessage({ type: 'reset_stats' });
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
    config.bufferSize = audioContext.sampleRate; // 1 second at device rate
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

    // Calibration UI wiring
    const calibBtn = document.getElementById('calibrateButton');
    if (calibBtn) {
      calibBtn.disabled = false;
      calibBtn.addEventListener('click', showCalibrationOverlay);
    }
    const calStart = document.getElementById('calibration-start');
    if (calStart) calStart.addEventListener('click', startCalibration);
    const calDone = document.getElementById('calibration-done');
    if (calDone) calDone.addEventListener('click', finishCalibration);
    const calCancel = document.getElementById('calibration-cancel');
    if (calCancel) calCancel.addEventListener('click', cancelCalibration);

    const passageSelector = document.getElementById('passage-selector');
    if (passageSelector) {
      passageSelector.addEventListener('change', () => {
        const passageEl = document.getElementById('calibration-passage');
        if (passageEl) {
          const p = CALIBRATION_PASSAGES[passageSelector.value];
          passageEl.textContent = p ? p.text : '';
        }
      });
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
