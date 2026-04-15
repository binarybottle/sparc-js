/******************************************************************************
 * SPARC Feature Extraction - Web Client
 *
 * Captures microphone audio, sends it to the SPARC web worker for feature
 * extraction, and maps articulatory features to display coordinates.
 *
 * Architecture:
 *   AudioWorklet (or ScriptProcessor fallback) -> circular buffer
 *   -> periodic extraction loop sends 1 s of audio to sparc-worker.js
 *   -> worker returns 6 articulator (x,y) z-scores, pitch, loudness, and F1
 *   -> tongue/LI z-scores are mapped to SVG via per-group DISPLAY_SCALES
 *   -> lip y-positions are driven by F1 (first formant) instead of z-scores
 *   -> features are smoothed and stored in featureHistory for display
 *
 * Also manages: calibration, Set References (speaker-specific F1 capture),
 * and test sound selection.
 ******************************************************************************/

/******************************************************************************
 * CONFIGURATION
 ******************************************************************************/
const config = {
  targetSampleRate: 16000, // WavLM expects 16 kHz
  deviceSampleRate: null,  // set at recording time from AudioContext
  frameSize: 512,
  bufferDuration: 0.5,     // seconds of audio to buffer before sending to worker
  bufferSize: 8000,        // = targetSampleRate * bufferDuration (resized at recording time)
  updateInterval: 100,     // ms between extraction requests
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

// Set-references state
let isSettingRefs = false;
let setRefAudioContext = null;
let setRefAudioStream = null;
let setRefWorkletNode = null;
let setRefTimer = null;
let setRefFrames = [];         // collected raw z-score frames for current vowel
let setRefResults = {};        // accumulated { vowel: { art: {x,y} } }
const SET_REF_VOWELS = [
  { id: 'i', label: '/i/', desc: 'Say "ee" as in "see"' },
  { id: 'e', label: '/e/', desc: 'Say "eh" as in "bed"' },
  { id: 'a', label: '/a/', desc: 'Say "ah" as in "father"' },
  { id: 'o', label: '/o/', desc: 'Say "oh" as in "go"' },
  { id: 'u', label: '/u/', desc: 'Say "oo" as in "food"' }
];
let setRefVowelIndex = 0;
let setRefCapturing = false;   // true while mic is live for current vowel
const SET_REF_STORAGE_KEY = 'sparc-reference-positions';

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

// Anatomical center of each articulator in SVG coordinates at z-score = 0.
const ARTICULATOR_CENTERS = {
  td: { x: -4.0, y: -3.2 },   // tongue dorsum: far back, up
  tb: { x: -2.0, y: -2.5 },   // tongue body: mid-back, up
  tt: { x:  0.5, y: -0.3 },   // tongue tip: mid, mid-height
  li: { x:  2.0, y:  0.0 },   // lower incisor: front, midline (tracks jaw)
  ul: { x:  3.0, y: -1.25 },  // upper lip: frontmost (y set by F1, center unused)
  ll: { x:  3.0, y: -1.25 }   // lower lip: frontmost (y set by F1, center unused)
};

// Per-articulator-group display scales (SVG units per z-score unit).
// Tongue uses large scales now that lips are F1-driven and won't overlap.
// Lips use zero scale — their y is F1-driven, x is fixed at center.
const DISPLAY_SCALES = {
  td: { x: 0.8, y: 1.2 },
  tb: { x: 0.8, y: 1.2 },
  tt: { x: 0.8, y: 1.2 },
  li: { x: 0.5, y: 1.0 },
  ul: { x: 0.0, y: 0.0 },
  ll: { x: 0.0, y: 0.0 }
};

// Map model z-scores to SVG display coordinates.
function emaToDisplay(key, z_x, z_y) {
  const c = ARTICULATOR_CENTERS[key];
  const s = DISPLAY_SCALES[key];
  return {
    x: c.x + z_x * s.x,
    y: c.y - z_y * s.y   // flip: MNGU0 +y = superior, SVG +y = down
  };
}

/******************************************************************************
 * F1-DRIVEN LIP POSITIONING
 *
 * The SPARC model's UL/LL channels don't differentiate vowels well.
 * F1 (first formant frequency) correlates strongly with mouth opening:
 *   /i/ ≈ 270 Hz (closed) → /a/ ≈ 730 Hz (open)
 *
 * We use F1 to drive the vertical gap between UL and LL, keeping their
 * x-positions fixed at the lip center.
 ******************************************************************************/

const F1_CLOSED_HZ = 250;    // F1 at lips-nearly-touching (high vowels)
const F1_OPEN_HZ   = 650;    // F1 at mouth wide open (accounts for male speakers)
const LIP_CENTER_Y = -1.25;  // midpoint between original UL/LL centers
const LIP_HALF_GAP_CLOSED = 0.3;   // SVG half-gap when mouth closed
const LIP_HALF_GAP_OPEN   = 1.8;   // SVG half-gap when mouth wide open

function f1ToLipPositions(f1Hz) {
  const t = Math.max(0, Math.min(1, (f1Hz - F1_CLOSED_HZ) / (F1_OPEN_HZ - F1_CLOSED_HZ)));
  const halfGap = LIP_HALF_GAP_CLOSED + t * (LIP_HALF_GAP_OPEN - LIP_HALF_GAP_CLOSED);
  return {
    ulY: LIP_CENTER_Y - halfGap,   // upper lip moves up (more negative SVG y)
    llY: LIP_CENTER_Y + halfGap    // lower lip moves down (more positive SVG y)
  };
}

function handleWorkerFeatures(message) {
  pendingWorkerResponses = Math.max(0, pendingWorkerResponses - 1);
  debugCounters.workerResponsesReceived++;

  try {
    if (!message.articulationFeatures) {
      throw new Error('No articulation features in message');
    }

    const { articulationFeatures, pitch, loudness, f1 } = message;

    // Collect F1 for set-references mode (only F1 is used from captures)
    if (setRefCapturing) {
      setRefFrames.push({ f1: f1 || 0 });
    }

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

    // Override UL/LL vertical positions with F1-driven mouth opening
    if (f1 > 0) {
      const lip = f1ToLipPositions(f1);
      articulationFeatures.ul.y = lip.ulY;
      articulationFeatures.ll.y = lip.llY;
    }

    // Periodic F1 diagnostic (every 10th frame)
    if (debugCounters.featuresUpdated % 10 === 0 && f1 > 0) {
      const t = Math.max(0, Math.min(1, (f1 - F1_CLOSED_HZ) / (F1_OPEN_HZ - F1_CLOSED_HZ)));
      debugLog(`F1: ${f1.toFixed(0)} Hz → opening ${(t * 100).toFixed(0)}%`);
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
    config.bufferSize = Math.floor(calibrationAudioContext.sampleRate * config.bufferDuration);
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
 * SET REFERENCE SOUNDS
 *
 * Steps through each vowel. For each, the (normal) speaker says the sound
 * for a few seconds while the model captures raw z-score output. The average
 * z-scores become the reference positions for that vowel, persisted in
 * localStorage so they survive page refreshes.
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
  if (!workerInitialized) { alert('Models not loaded yet.'); return; }

  try {
    animationRunning = false;
    if (animationFrame) { clearTimeout(animationFrame); animationFrame = null; }

    SparcWorker.postMessage({ type: 'calibrate_start' });

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

  SparcWorker.postMessage({
    type: 'process',
    audio: new Float32Array(recentAudio),
    config: config,
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

  // Only F1 is used from captured references (tongue/LI use phonetic defaults)
  let f1Sum = 0, f1Count = 0;
  for (const frame of setRefFrames) {
    if (frame.f1 && frame.f1 > 0) { f1Sum += frame.f1; f1Count++; }
  }
  const result = { _f1: f1Count > 0 ? f1Sum / f1Count : 0 };

  setRefResults[vowel.id] = result;
  debugLog(`Reference captured for /${vowel.id}/`, { frames: setRefFrames.length, f1: result._f1 });

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
  SparcWorker.postMessage({ type: 'reset_stats' });
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
    updateStatus('Loading models...');

    initializeFeatureHistory();
    await initSparcWorker();

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
