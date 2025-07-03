/******************************************************************************
 * SPARC Feature Extraction - Web Client
 ******************************************************************************/

/******************************************************************************
* CONFIGURATION & GLOBAL VARIABLES *
******************************************************************************/
const config = {
  sampleRate: 16000,
  frameSize: 512,
  bufferSize: 16000,
  updateInterval: 200,
  extractPitchFn: 2,
  maxPendingResponses: 2,
  processingTimeout: 3000
};

// Enhanced debug logging
function debugLog(message, data = null) {
  const timestamp = new Date().toLocaleTimeString();
  if (data) {
    console.log(`[${timestamp}] SPARC DEBUG: ${message}`, data);
  } else {
    console.log(`[${timestamp}] SPARC DEBUG: ${message}`);
  }
}

// Enhanced debug counters with error tracking
let debugCounters = {
  audioDataReceived: 0,
  workerMessagesSent: 0,
  workerResponsesReceived: 0,
  featuresUpdated: 0,
  chartsUpdated: 0,
  errors: 0,
  timeouts: 0,
  fallbacksUsed: 0
};

// Enhanced debug status display
function updateDebugStatus() {
  const debugInfo = `
    Audio: ${debugCounters.audioDataReceived}
    Sent: ${debugCounters.workerMessagesSent}
    Received: ${debugCounters.workerResponsesReceived}
    Features: ${debugCounters.featuresUpdated}
    Charts: ${debugCounters.chartsUpdated}
    Pending: ${pendingWorkerResponses}
    Errors: ${debugCounters.errors}
    Timeouts: ${debugCounters.timeouts}
    Fallbacks: ${debugCounters.fallbacksUsed}
    Status: ${isRecording ? 'RECORDING' : 'IDLE'}
  `;
  
  let debugDisplay = document.getElementById('debug-status');
  if (!debugDisplay) {
    debugDisplay = document.createElement('div');
    debugDisplay.id = 'debug-status';
    debugDisplay.style.cssText = `
      position: fixed; left: 10px; bottom: 10px;
      background: rgba(0,0,0,0.8); color: white;
      padding: 10px; border-radius: 5px;
      font-size: 12px; font-family: monospace;
      z-index: 1001; white-space: pre-line;
      max-width: 200px;
    `;
    document.body.appendChild(debugDisplay);
  }
  debugDisplay.textContent = debugInfo;
}

setInterval(updateDebugStatus, 500);

// Global variables
let audioContext;
let audioStream;
let workletNode;
let waveformHistory = Array(500).fill(0);
let animationRunning = false;
let animationFrame = null;
let isRecording = false;
let audioBuffer = new Float32Array(config.bufferSize);
let audioBufferIndex = 0;

// Enhanced smoothed features with validation
let smoothedFeatures = {
  ul_x: 0.9, ul_y: -1.05,
  ll_x: 0.9, ll_y: -0.8,
  li_x: 0.85, li_y: -0.92,
  tt_x: 0.5, tt_y: -0.7,
  tb_x: 0.0, tb_y: -0.6,
  td_x: -0.5, td_y: -0.5
};

let sensitivityFactor = 8.0;
let smoothingFactor = 0.4;

// Feature history with improved initialization
let featureHistory = {};

function initializeFeatureHistory() {
  const articulators = ['ul_x', 'ul_y', 'll_x', 'll_y', 'li_x', 'li_y', 
                       'tt_x', 'tt_y', 'tb_x', 'tb_y', 'td_x', 'td_y', 
                       'pitch', 'loudness'];
  
  featureHistory = {};
  articulators.forEach(key => {
    featureHistory[key] = Array(100).fill(0);
  });
}

// Enhanced worker management
let SparcWorker = null;
let workerInitialized = false;
let pendingWorkerResponses = 0;
let lastWorkerMessageTime = 0;
let workerResponseTimeouts = new Set();

/******************************************************************************
* UTILITY FUNCTIONS *
******************************************************************************/

function updateStatus(message) {
  const statusElement = document.getElementById('status');
  if (statusElement) {
    statusElement.textContent = "Status: " + message;
  }
}

function getRecentAudioBuffer() {
  try {
    const recentAudio = new Float32Array(config.bufferSize);
    
    for (let i = 0; i < config.bufferSize; i++) {
      const index = (audioBufferIndex + i) % config.bufferSize;
      recentAudio[i] = audioBuffer[index];
    }
    
    return recentAudio;
  } catch (error) {
    debugLog("Error getting audio buffer", error);
    return new Float32Array(config.bufferSize);
  }
}

// Helper function to ensure points have valid coordinates
function sanitizePoint(point, defaultX = 0, defaultY = 0) {
  if (!point || typeof point.x !== 'number' || typeof point.y !== 'number' || 
      isNaN(point.x) || isNaN(point.y) || !isFinite(point.x) || !isFinite(point.y)) {
      return { x: defaultX, y: defaultY };
  }
  
  return {
      x: Math.min(Math.max(point.x, -2), 2),
      y: Math.min(Math.max(point.y, -2), 1)
  };
}

// Apply anatomical constraints
function applyAnatomicalConstraints(tt, tb, td) {
  tt.x = Math.min(Math.max(tt.x, -1.5), 1.5);
  tt.y = Math.min(Math.max(tt.y, -1.5), 0);
  
  tb.x = Math.min(Math.max(tb.x, -1.5), 1.2);
  tb.y = Math.min(Math.max(tb.y, -1.5), 0);
  
  td.x = Math.min(Math.max(td.x, -1.5), 0.8);
  td.y = Math.min(Math.max(td.y, -1.5), 0);
  
  if (td.x > tb.x - 0.1) {
    td.x = tb.x - 0.1;
  }
  if (tb.x > tt.x - 0.1) {
    tb.x = tt.x - 0.1;
  }
}

/******************************************************************************
* ENHANCED WORKER MANAGEMENT *
******************************************************************************/

// Initialize the ML worker with better error handling
function initSparcWorker() {
  if (SparcWorker) return Promise.resolve();
  
  return new Promise((resolve, reject) => {
    try {
      debugLog("Initializing ML worker...");
      SparcWorker = new Worker('sparc-worker.js');
      
      SparcWorker.onmessage = function(e) {
        const message = e.data;
        
        workerResponseTimeouts.forEach(timeoutId => {
          clearTimeout(timeoutId);
          workerResponseTimeouts.delete(timeoutId);
        });
        
        debugLog(`Worker message received: ${message.type}`);
        
        switch(message.type) {
          case 'initialized':
            debugLog("Worker initialization complete");
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
          case 'timeout':
            handleWorkerError(message);
            break;
        }
      };
      
      SparcWorker.onerror = function(error) {
        debugLog("Worker error event", error);
        debugCounters.errors++;
        if (!workerInitialized) {
          debugLog("Worker failed to initialize, switching to demo mode");
          workerInitialized = true;
          resolve();
        }
      };
      
      const initTimeout = setTimeout(() => {
        debugLog("Worker initialization timeout - switching to demo mode");
        workerInitialized = true;
        resolve();
      }, 5000);
      
      SparcWorker.postMessage({
        type: 'init',
        onnxPath: 'models/wavlm_base_layer9_quantized.onnx',
        linearModelPath: 'models/wavlm_linear_model.json'
      });
      
      resolve = ((originalResolve) => {
        return () => {
          clearTimeout(initTimeout);
          originalResolve();
        };
      })(resolve);
      
    } catch (error) {
      debugLog("Error creating worker", error);
      debugLog("Switching to demo mode due to worker creation error");
      workerInitialized = true;
      resolve();
    }
  });
}

// ✅ ADD: Main thread fallback function
async function initSparcWithFallback() {
  try {
      // Try worker initialization first
      await initSparcWorker();
  } catch (error) {
      console.warn('Worker failed, switching to main thread processing:', error);
      
      // Check if ONNX Runtime is available in main thread
      if (typeof ort === 'undefined') {
          throw new Error('ONNX Runtime not available in main thread either');
      }
      
      console.log('Using main thread ONNX Runtime, version:', ort.version);
      
      // Configure main thread ONNX Runtime
      ort.env.wasm.numThreads = 1;
      ort.env.wasm.simd = true;
      ort.env.debug = false;
      
      // Load models in main thread
      await loadModelsInMainThread();
      
      updateStatus("Running in main thread mode (worker fallback)");
  }
}

async function loadModelsInMainThread() {
    try {
        const session = await ort.InferenceSession.create('models/wavlm_base_layer9_quantized.onnx');
        console.log('ONNX model loaded successfully in main thread');
        
        const response = await fetch('models/wavlm_linear_model.json');
        const linearModel = await response.json();
        console.log('Linear model loaded successfully in main thread');
        
        return { session, linearModel };
    } catch (error) {
        console.error('Failed to load models in main thread:', error);
        throw error;
    }
}

// Handle worker feature responses with improved validation
function handleWorkerFeatures(message) {
  pendingWorkerResponses = Math.max(0, pendingWorkerResponses - 1);
  debugCounters.workerResponsesReceived++;
  
  try {
    if (!message.articulationFeatures) {
      throw new Error("No articulation features in message");
    }
    
    const { articulationFeatures, pitch, loudness } = message;
    
    const requiredKeys = ['ul', 'll', 'li', 'tt', 'tb', 'td'];
    for (const key of requiredKeys) {
      if (!articulationFeatures[key] || 
          typeof articulationFeatures[key].x !== 'number' ||
          typeof articulationFeatures[key].y !== 'number') {
        throw new Error(`Invalid articulation feature: ${key}`);
      }
    }
    
    updateFeatureHistory(articulationFeatures, pitch || 0, loudness || -60);
    debugCounters.featuresUpdated++;
    
    requestAnimationFrame(() => {
      updateCharts();
      debugCounters.chartsUpdated++;
    });
    
  } catch (error) {
    debugLog("Error processing worker features", error);
    debugCounters.errors++;
    
    const fallbackFeatures = generateLocalFallbackFeatures();
    updateFeatureHistory(fallbackFeatures, 120, -25);
    updateCharts();
    debugCounters.fallbacksUsed++;
  }
}

// Handle worker errors with recovery
function handleWorkerError(message) {
  debugLog("Worker error/timeout", message);
  debugCounters.errors++;
  
  if (message.type === 'timeout') {
    debugCounters.timeouts++;
  }
  
  pendingWorkerResponses = Math.max(0, pendingWorkerResponses - 1);
  
  if (isRecording) {
    const fallbackFeatures = generateLocalFallbackFeatures();
    updateFeatureHistory(fallbackFeatures, 120 + Math.random() * 50, -25 + Math.random() * 10);
    updateCharts();
    debugCounters.fallbacksUsed++;
  }
  
  if (!workerInitialized) {
    updateStatus("Worker initialization failed: " + message.error);
  }
}

// Generate local fallback features
function generateLocalFallbackFeatures() {
  const time = Date.now() / 1000;
  const baseFreq = 0.5;
  const speechFreq = 2.0;
  
  return {
    ul: { 
      x: 1.0 + 0.05 * Math.sin(time * baseFreq), 
      y: -0.95 + 0.03 * Math.cos(time * speechFreq) 
    },
    ll: { 
      x: 1.0 + 0.05 * Math.sin(time * baseFreq + 0.1), 
      y: -0.7 + 0.04 * Math.cos(time * speechFreq + 0.2) 
    },
    li: { 
      x: 0.95 + 0.03 * Math.sin(time * baseFreq + 0.05), 
      y: -0.82 + 0.02 * Math.cos(time * speechFreq + 0.1) 
    },
    tt: { 
      x: 0.6 + 0.15 * Math.sin(time * speechFreq), 
      y: -0.7 + 0.1 * Math.cos(time * speechFreq * 1.3) 
    },
    tb: { 
      x: 0.0 + 0.1 * Math.sin(time * speechFreq + 0.5), 
      y: -0.6 + 0.08 * Math.cos(time * speechFreq + 0.3) 
    },
    td: { 
      x: -0.6 + 0.08 * Math.sin(time * speechFreq + 1.0), 
      y: -0.5 + 0.06 * Math.cos(time * speechFreq + 0.7) 
    }
  };
}

/******************************************************************************
* ENHANCED FEATURE EXTRACTION LOOP *
******************************************************************************/

async function extractFeaturesLoop() {
  if (!isRecording) {
    return;
  }
  
  setTimeout(extractFeaturesLoop, config.updateInterval);
  
  if (!workerInitialized) {
    const fallbackFeatures = generateLocalFallbackFeatures();
    updateFeatureHistory(fallbackFeatures, 120 + Math.random() * 50, -25 + Math.random() * 10);
    updateCharts();
    debugCounters.fallbacksUsed++;
    return;
  }
  
  if (pendingWorkerResponses >= 1) {
    debugLog(`Skipping frame - pending response: ${pendingWorkerResponses}`);
    const fallbackFeatures = generateLocalFallbackFeatures();
    updateFeatureHistory(fallbackFeatures, 120, -25);
    updateCharts();
    debugCounters.fallbacksUsed++;
    return;
  }
  
  try {
    const recentAudio = getRecentAudioBuffer();
    if (!recentAudio || recentAudio.length === 0) {
      return;
    }
    
    const timeoutId = setTimeout(() => {
      if (workerResponseTimeouts.has(timeoutId)) {
        debugLog("Worker response timeout (1s)");
        debugCounters.timeouts++;
        pendingWorkerResponses = Math.max(0, pendingWorkerResponses - 1);
        workerResponseTimeouts.delete(timeoutId);
        
        const fallbackFeatures = generateLocalFallbackFeatures();
        updateFeatureHistory(fallbackFeatures, 120 + Math.random() * 50, -25 + Math.random() * 10);
        updateCharts();
        debugCounters.fallbacksUsed++;
      }
    }, 1000);
    
    workerResponseTimeouts.add(timeoutId);
    
    SparcWorker.postMessage({
      type: 'process',
      audio: new Float32Array(recentAudio),
      config: config,
      sensitivityFactor: sensitivityFactor
    });
    
    pendingWorkerResponses++;
    debugCounters.workerMessagesSent++;
    
  } catch (error) {
    debugLog("Feature extraction error", error);
    debugCounters.errors++;
    
    const fallbackFeatures = generateLocalFallbackFeatures();
    updateFeatureHistory(fallbackFeatures, 120, -25);
    updateCharts();
    debugCounters.fallbacksUsed++;
  }
}

// Enhanced feature history update with validation
function updateFeatureHistory(articulationFeatures, pitch, loudness) {
  try {
    if (!articulationFeatures || typeof pitch !== 'number' || typeof loudness !== 'number') {
      throw new Error("Invalid feature data");
    }
    
    const alpha = isRecording ? smoothingFactor : 0.3;
    const articulators = ['ul', 'll', 'li', 'tt', 'tb', 'td'];
    
    for (const art of articulators) {
      if (articulationFeatures[art]) {
        const newX = articulationFeatures[art].x;
        const newY = articulationFeatures[art].y;
        
        if (isNaN(newX) || isNaN(newY) || !isFinite(newX) || !isFinite(newY)) {
          debugLog(`Invalid coordinates for ${art}: (${newX}, ${newY})`);
          continue;
        }
        
        const oldX = smoothedFeatures[art + '_x'];
        const oldY = smoothedFeatures[art + '_y'];
        
        smoothedFeatures[art + '_x'] = alpha * newX + (1 - alpha) * oldX;
        smoothedFeatures[art + '_y'] = alpha * newY + (1 - alpha) * oldY;
      }
    }
    
    const keys = Object.keys(featureHistory);
    for (const key of keys) {
      featureHistory[key].shift();
      
      if (key === 'pitch') {
        featureHistory[key].push(isNaN(pitch) ? 0 : pitch);
      } else if (key === 'loudness') {
        featureHistory[key].push(isNaN(loudness) ? -60 : loudness);
      } else {
        const value = smoothedFeatures[key];
        featureHistory[key].push(isNaN(value) ? 0 : value);
      }
    }
    
  } catch (error) {
    debugLog("Error updating feature history", error);
    debugCounters.errors++;
  }
}

/******************************************************************************
* INITIALIZATION & SETUP *
******************************************************************************/

function initializeDefaultPositions() {
  const defaultPositions = {
    ul: { x: 1.0, y: -0.95 },
    ll: { x: 1.0, y: -0.7 },
    li: { x: 0.95, y: -0.82 },
    tt: { x: 0.6, y: -0.7 },
    tb: { x: 0.0, y: -0.6 },
    td: { x: -0.6, y: -0.5 }
  };
  
  Object.keys(defaultPositions).forEach(art => {
    smoothedFeatures[art + '_x'] = defaultPositions[art].x;
    smoothedFeatures[art + '_y'] = defaultPositions[art].y;
  });
  
  if (featureHistory && Object.keys(featureHistory).length > 0) {
    for (let i = 0; i < featureHistory.ul_x.length; i++) {
      Object.keys(defaultPositions).forEach(art => {
        const xKey = art + '_x';
        const yKey = art + '_y';
        if (featureHistory[xKey] && featureHistory[yKey]) {
          featureHistory[xKey][i] = defaultPositions[art].x;
          featureHistory[yKey][i] = defaultPositions[art].y;
        }
      });
    }
    updateCharts();
  }
  
  debugLog("Default positions initialized", defaultPositions);
}

// ✅ CORRECTED: Initialize application with fallback
async function init() {
  try {
    updateStatus("Loading models...");
    
    initializeFeatureHistory();
    await initSparcWithFallback(); // ✅ Use fallback instead of just initSparcWorker
    
    setupCharts();
    setupSensitivityControls();
    setupTestControls();
    initializeDefaultPositions();
    
    document.getElementById('startButton').disabled = false;
    updateStatus("Models loaded. Ready to start.");
    
    document.getElementById('startButton').addEventListener('click', startRecording);
    document.getElementById('stopButton').addEventListener('click', stopRecording);
    
    // ✅ ADD: Setup debug mode toggle
    const debugMode = document.getElementById('debug-mode');
    if (debugMode) {
      debugMode.checked = true;
      debugMode.addEventListener('change', function() {
        toggleDebugMarkers(this.checked);
      });
      toggleDebugMarkers(true); // Show markers initially
    }

    if (!isRecording) {
      testArticulatorAnimation();
    }

  } catch (error) {
    updateStatus("Error loading models: " + error.message);
    debugLog("Model loading error", error);
    debugCounters.errors++;
  }
}

/******************************************************************************
* SETUP FUNCTIONS *
******************************************************************************/

function setupVocalTractVisualization() {
  const svg = document.getElementById('vocal-tract-svg');
  
  if (!svg) {
    console.error("SVG element 'vocal-tract-svg' not found!");
    return;
  }
  
  svg.setAttribute('viewBox', '-2 -2 4 3');
  svg.setAttribute('width', '600');
  svg.setAttribute('height', '400');
  
  while (svg.firstChild) {
      svg.removeChild(svg.firstChild);
  }
  
  createStaticElements(svg);
  createDynamicElements(svg);
  
  debugLog("SVG visualization setup complete");
}

// ✅ FIXED: Single createStaticElements function
function createStaticElements(svg) {
  // Pharynx wall
  const pharynxWall = document.createElementNS('http://www.w3.org/2000/svg', 'path');
  pharynxWall.setAttribute('class', 'pharynx');
  pharynxWall.setAttribute('d', 'M-1.2,-0.2 C-1.2,-0.3 -1.15,-0.45 -1.05,-0.6 C-0.95,-0.75 -0.85,-0.9 -0.75,-1.0 C-0.6,-1.1 -0.45,-1.2 -0.3,-1.25 C-0.25,-1.3 -0.2,-1.25 -0.2,-1.2 L-0.25,-1.0 L-0.35,-0.8 L-0.5,-0.6 L-0.65,-0.4 L-0.85,-0.25 Z');
  pharynxWall.setAttribute('fill', 'none');
  pharynxWall.setAttribute('stroke', '#666');
  pharynxWall.setAttribute('stroke-width', '0.02');
  svg.appendChild(pharynxWall);
  
  // Hard palate
  const palate = document.createElementNS('http://www.w3.org/2000/svg', 'path');
  palate.setAttribute('class', 'palate');
  palate.setAttribute('id', 'palate');
  palate.setAttribute('d', 'M0.9,-0.9 C0.7,-1.0 0.4,-0.95 0.1,-0.85 C-0.25,-0.75 -0.5,-0.6 -0.75,-0.4');
  palate.setAttribute('fill', 'none');
  palate.setAttribute('stroke', '#333');
  palate.setAttribute('stroke-width', '0.03');
  svg.appendChild(palate);
  
  // Jaw outline
  const jaw = document.createElementNS('http://www.w3.org/2000/svg', 'path');
  jaw.setAttribute('class', 'jaw');
  jaw.setAttribute('id', 'jaw');
  jaw.setAttribute('d', 'M0.9,-0.1 C0.7,0.0 0.5,0.05 0.3,0.07 C0.1,0.08 -0.1,0.09 -0.3,0.07 C-0.5,0.05 -0.7,0.0 -0.85,-0.1 C-0.95,-0.15 -1.05,-0.2 -1.1,-0.25');
  jaw.setAttribute('fill', 'none');
  jaw.setAttribute('stroke', '#333');
  jaw.setAttribute('stroke-width', '0.03');
  svg.appendChild(jaw);
  
  // Upper teeth
  const upperTeeth = document.createElementNS('http://www.w3.org/2000/svg', 'path');
  upperTeeth.setAttribute('class', 'teeth');
  upperTeeth.setAttribute('d', 'M0.85,-0.8 L0.85,-0.7 L0.75,-0.7 L0.75,-0.8 Z');
  upperTeeth.setAttribute('fill', 'white');
  upperTeeth.setAttribute('stroke', '#333');
  upperTeeth.setAttribute('stroke-width', '0.01');
  svg.appendChild(upperTeeth);
  
  // Lower teeth
  const lowerTeeth = document.createElementNS('http://www.w3.org/2000/svg', 'path');
  lowerTeeth.setAttribute('class', 'teeth');
  lowerTeeth.setAttribute('d', 'M0.85,-0.2 L0.85,-0.1 L0.75,-0.1 L0.75,-0.2 Z');
  lowerTeeth.setAttribute('fill', 'white');
  lowerTeeth.setAttribute('stroke', '#333');
  lowerTeeth.setAttribute('stroke-width', '0.01');
  svg.appendChild(lowerTeeth);
  
  // Labels
  addLabel(svg, "FRONT", 0.75, 0.35);
  addLabel(svg, "BACK", -0.75, 0.35);
}

function createDynamicElements(svg) {
  // Upper lip
  const upperLip = document.createElementNS('http://www.w3.org/2000/svg', 'path');
  upperLip.setAttribute('class', 'lips');
  upperLip.setAttribute('id', 'upper-lip');
  upperLip.setAttribute('fill', '#ff9999');
  upperLip.setAttribute('stroke', '#cc6666');
  upperLip.setAttribute('stroke-width', '0.01');
  svg.appendChild(upperLip);
  
  // Lower lip
  const lowerLip = document.createElementNS('http://www.w3.org/2000/svg', 'path');
  lowerLip.setAttribute('class', 'lips');
  lowerLip.setAttribute('id', 'lower-lip');
  lowerLip.setAttribute('fill', '#ff9999');
  lowerLip.setAttribute('stroke', '#cc6666');
  lowerLip.setAttribute('stroke-width', '0.01');
  svg.appendChild(lowerLip);
  
  // Tongue
  const tongue = document.createElementNS('http://www.w3.org/2000/svg', 'path');
  tongue.setAttribute('class', 'tongue');
  tongue.setAttribute('id', 'tongue');
  tongue.setAttribute('fill', '#ffb3ba');
  tongue.setAttribute('stroke', '#ff8a9b');
  tongue.setAttribute('stroke-width', '0.02');
  svg.appendChild(tongue);
  
  // Debug markers
  const articulators = ['ul', 'll', 'li', 'tt', 'tb', 'td'];
  const colors = ['#e74c3c', '#3498db', '#f1c40f', '#2ecc71', '#9b59b6', '#e67e22'];
  
  articulators.forEach((art, i) => {
      const marker = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      marker.setAttribute('id', `${art}-marker`);
      marker.setAttribute('r', '0.03');
      marker.setAttribute('fill', colors[i]);
      marker.setAttribute('stroke', '#fff');
      marker.setAttribute('stroke-width', '0.005');
      marker.setAttribute('class', 'debug-marker');
      marker.style.display = 'none';
      svg.appendChild(marker);
  });
}

function addLabel(svg, text, x, y) {
  const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
  label.setAttribute('x', x);
  label.setAttribute('y', y);
  label.setAttribute('font-size', '0.12');
  label.setAttribute('text-anchor', 'middle');
  label.setAttribute('fill', '#888');
  label.textContent = text;
  svg.appendChild(label);
}

function setupCharts() {
  setupVocalTractVisualization();
  
  debugLog("Charts setup complete", {
    tongueExists: !!document.getElementById('tongue'),
    upperLipExists: !!document.getElementById('upper-lip'),
    lowerLipExists: !!document.getElementById('lower-lip'),
    debugMarkerCount: document.querySelectorAll('.debug-marker').length
  });

  initializeDefaultPositions();

  if (!isRecording) {
    testArticulatorAnimation();
  }
}

function setupSensitivityControls() {
  const sensitivitySlider = document.getElementById('sensitivity-slider');
  const sensitivityValue = document.getElementById('sensitivity-value');
  
  if (sensitivitySlider) {
      sensitivitySlider.addEventListener('input', function() {
          sensitivityFactor = parseFloat(this.value);
          sensitivityValue.textContent = sensitivityFactor.toFixed(1);
          debugLog(`Sensitivity changed to: ${sensitivityFactor}`);
      });
  }
  
  const smoothingSlider = document.getElementById('smoothing-slider');
  const smoothingValue = document.getElementById('smoothing-value');
  
  if (smoothingSlider) {
      smoothingSlider.addEventListener('input', function() {
          smoothingFactor = parseFloat(this.value);
          smoothingValue.textContent = smoothingFactor.toFixed(1);
          debugLog(`Smoothing changed to: ${smoothingFactor}`);
      });
  }
  
  const resetButton = document.getElementById('reset-positions');
  if (resetButton) {
      resetButton.addEventListener('click', function() {
          initializeDefaultPositions();
          debugLog("Positions reset to neutral");
      });
  }
  
  const testButton = document.getElementById('test-extremes');
  if (testButton) {
      testButton.addEventListener('click', function() {
          if (!isRecording) {
              testExtremePositions();
          } else {
              alert("Stop recording first to test extreme positions");
          }
      });
  }
}

function setupTestControls() {
  const controlPanel = document.querySelector('.controls');
  
  if (controlPanel) {
    const testButton = document.createElement('button');
    testButton.textContent = 'Test Audio Patterns';
    testButton.id = 'test-audio-patterns';
    testButton.style.margin = '5px';
    testButton.addEventListener('click', testAudioPatterns);
    
    controlPanel.appendChild(testButton);
  }
}

function testExtremePositions() {
  const extremePositions = [
      { ul: {x: 1.5, y: -1.0}, ll: {x: 1.5, y: -0.5}, li: {x: 1.5, y: -0.75}, 
        tt: {x: 0.8, y: -0.8}, tb: {x: 0.2, y: -0.6}, td: {x: -0.4, y: -0.5} },
      { ul: {x: 0.2, y: -1.0}, ll: {x: 0.2, y: -0.7}, li: {x: 0.2, y: -0.85}, 
        tt: {x: -0.2, y: -0.7}, tb: {x: -0.6, y: -0.8}, td: {x: -1.2, y: -0.7} },
      { ul: {x: 0.9, y: -1.05}, ll: {x: 0.9, y: -0.9}, li: {x: 0.9, y: -0.95}, 
        tt: {x: 0.8, y: -1.3}, tb: {x: 0.3, y: -0.9}, td: {x: -0.2, y: -0.7} },
      { ul: {x: 0.9, y: -1.05}, ll: {x: 0.9, y: -0.9}, li: {x: 0.9, y: -0.95}, 
        tt: {x: -0.2, y: -0.5}, tb: {x: -0.8, y: -0.6}, td: {x: -1.5, y: -0.4} }
  ];
  
  let posIndex = 0;
  const interval = setInterval(() => {
      if (posIndex >= extremePositions.length) {
          clearInterval(interval);
          initializeDefaultPositions();
          return;
      }
      
      const pos = extremePositions[posIndex];
      updateFeatureHistory(pos, 150, -20);
      updateCharts();
      posIndex++;
  }, 1000);
}

function testAudioPatterns() {
  if (!SparcWorker || !workerInitialized) {
    alert('Worker not initialized yet');
    return;
  }
  
  debugLog("=== STARTING AUDIO PATTERN TESTS ===");
  
  const testAudio1 = new Float32Array(16000);
  for (let i = 0; i < 16000; i++) {
    testAudio1[i] = 0.1 * Math.sin(2 * Math.PI * 150 * i / 16000) + 
                    0.05 * Math.sin(2 * Math.PI * 300 * i / 16000) +
                    0.03 * Math.sin(2 * Math.PI * 450 * i / 16000);
  }
  
  debugLog("Sending vowel-like test pattern...");
  SparcWorker.postMessage({
    type: 'process',
    audio: testAudio1,
    config: config,
    sensitivityFactor: sensitivityFactor
  });
  
  setTimeout(() => {
    const testAudio2 = new Float32Array(16000);
    for (let i = 0; i < 16000; i++) {
      testAudio2[i] = 0.05 * (Math.random() - 0.5) * 
                     Math.sin(2 * Math.PI * 4000 * i / 16000);
    }
    
    debugLog("Sending fricative-like test pattern...");
    SparcWorker.postMessage({
      type: 'process',
      audio: testAudio2,
      config: config,
      sensitivityFactor: sensitivityFactor
    });
  }, 2000);
}

function testArticulatorAnimation() {
  const speechPositions = [
    {
      name: '/i/ (see)',
      ul: { x: 0.9, y: -1.05 }, ll: { x: 0.9, y: -0.9 }, li: { x: 0.9, y: -0.95 },
      tt: { x: 0.6, y: -1.0 }, tb: { x: 0.2, y: -1.0 }, td: { x: -0.2, y: -0.8 }
    },
    {
      name: '/e/ (bet)',
      ul: { x: 0.9, y: -1.0 }, ll: { x: 0.9, y: -0.85 }, li: { x: 0.9, y: -0.9 },
      tt: { x: 0.5, y: -0.8 }, tb: { x: 0.1, y: -0.8 }, td: { x: -0.3, y: -0.7 }
    },
    {
      name: '/æ/ (cat)',
      ul: { x: 0.9, y: -0.95 }, ll: { x: 0.9, y: -0.6 }, li: { x: 0.9, y: -0.75 },
      tt: { x: 0.4, y: -0.4 }, tb: { x: -0.1, y: -0.5 }, td: { x: -0.5, y: -0.4 }
    },
    {
      name: '/a/ (father)',
      ul: { x: 0.9, y: -0.9 }, ll: { x: 0.9, y: -0.5 }, li: { x: 0.9, y: -0.7 },
      tt: { x: 0.2, y: -0.3 }, tb: { x: -0.2, y: -0.4 }, td: { x: -0.6, y: -0.3 }
    },
    {
      name: '/u/ (boot)',
      ul: { x: 0.5, y: -1.0 }, ll: { x: 0.5, y: -0.85 }, li: { x: 0.5, y: -0.9 },
      tt: { x: -0.2, y: -0.8 }, tb: { x: -0.6, y: -0.9 }, td: { x: -1.0, y: -0.8 }
    }
  ];
  
  let frame = 0;
  const frameDuration = 800;
  const frameTransitions = 30;
  
  animationRunning = true;

  function animateFrame() {
    if (!document.getElementById('tongue') || isRecording || !animationRunning) {
      animationRunning = false;
      return;
    }

    const currentPosIdx = Math.floor(frame / frameTransitions) % speechPositions.length;
    const nextPosIdx = (currentPosIdx + 1) % speechPositions.length;
    const transitionProgress = (frame % frameTransitions) / frameTransitions;
    
    const currentPos = speechPositions[currentPosIdx];
    const nextPos = speechPositions[nextPosIdx];
    
    const features = {};
    const articulators = ['ul', 'll', 'li', 'tt', 'tb', 'td'];
    
    articulators.forEach(art => {
      features[art] = {
        x: currentPos[art].x + (nextPos[art].x - currentPos[art].x) * transitionProgress,
        y: currentPos[art].y + (nextPos[art].y - currentPos[art].y) * transitionProgress
      };
    });
    
    updateFeatureHistory(features, 120 + Math.sin(frame/15)*80, -25 + Math.sin(frame/10)*25);
    updateCharts();
    
    if (frame % frameTransitions === 0) {
      updateStatus(`Demo: ${currentPos.name} → ${nextPos.name}`);
    }
    
    frame++;
    animationFrame = setTimeout(animateFrame, frameDuration / frameTransitions);
  }
  
  updateStatus("Demo: Showing full articulator range...");
  animateFrame();
}

/******************************************************************************
* VISUALIZATION FUNCTIONS *
******************************************************************************/

function createTonguePath(tt, tb, td) {
  tt = sanitizePoint(tt, 0.6, -0.7);
  tb = sanitizePoint(tb, 0.0, -0.6);
  td = sanitizePoint(td, -0.6, -0.5);
  
  applyAnatomicalConstraints(tt, tb, td);
  
  const tongueRoot = { x: -1.3, y: -0.3 };
  
  const tonguePath = `
    M ${tongueRoot.x} ${tongueRoot.y}
    Q ${td.x} ${td.y} ${tb.x} ${tb.y}
    Q ${tt.x} ${tt.y} ${tt.x + 0.1} ${tt.y - 0.05}
    Q ${tt.x + 0.05} ${tt.y + 0.1} ${tt.x - 0.05} ${tt.y + 0.15}
    Q ${tb.x - 0.1} ${tb.y + 0.2} ${td.x - 0.1} ${td.y + 0.15}
    Q ${tongueRoot.x + 0.2} ${tongueRoot.y + 0.1} ${tongueRoot.x} ${tongueRoot.y}
    Z
  `;
  
  return tonguePath;
}

function createLipPaths(ul, ll, li) {
  ul = sanitizePoint(ul, 0.9, -0.95);
  ll = sanitizePoint(ll, 0.9, -0.7);
  li = sanitizePoint(li, 0.85, -0.8);
  
  const lipCornerLeft = { x: li.x - 0.3, y: (ul.y + ll.y) / 2 };
  const lipCornerRight = { x: li.x + 0.1, y: (ul.y + ll.y) / 2 };
  
  const upperLipPath = `
    M ${lipCornerLeft.x} ${lipCornerLeft.y}
    Q ${ul.x} ${ul.y} ${lipCornerRight.x} ${lipCornerRight.y}
    L ${lipCornerRight.x} ${(ul.y + ll.y) / 2}
    L ${lipCornerLeft.x} ${(ul.y + ll.y) / 2}
    Z
  `;
  
  const lowerLipPath = `
    M ${lipCornerLeft.x} ${(ul.y + ll.y) / 2}
    L ${lipCornerRight.x} ${(ul.y + ll.y) / 2}
    Q ${ll.x} ${ll.y} ${lipCornerLeft.x} ${lipCornerLeft.y}
    Z
  `;
  
  return {
    upperLip: upperLipPath,
    lowerLip: lowerLipPath
  };
}

function updateSourceFeatures(pitch, loudness) {
  const normalizedPitch = Math.min(100, Math.max(0, ((pitch - 75) / 225) * 100));
  const normalizedLoudness = Math.min(100, Math.max(0, ((loudness + 60) / 60) * 100));
  
  const pitchBar = document.getElementById('pitch-bar');
  const loudnessBar = document.getElementById('loudness-bar');
  
  if (pitchBar) pitchBar.style.height = normalizedPitch + '%';
  if (loudnessBar) loudnessBar.style.height = normalizedLoudness + '%';
}

function updateCharts() {
  try {
    if (!featureHistory || Object.keys(featureHistory).length === 0) {
      debugLog("No feature history available for chart update");
      return;
    }
    
    const articulators = ['ul', 'll', 'li', 'tt', 'tb', 'td'];
    const latestFeatures = {};

    for (const art of articulators) {
      const xKey = art + '_x';
      const yKey = art + '_y';
      
      if (featureHistory[xKey] && featureHistory[yKey]) {
        const latestX = featureHistory[xKey][featureHistory[xKey].length - 1];
        const latestY = featureHistory[yKey][featureHistory[yKey].length - 1];
        
        latestFeatures[art] = sanitizePoint(
          { x: latestX, y: latestY }, 
          0, -0.5
        );
      } else {
        latestFeatures[art] = { x: 0, y: -0.5 };
      }
      
      const marker = document.getElementById(`${art}-marker`);
      if (marker) {
        marker.setAttribute('cx', latestFeatures[art].x);
        marker.setAttribute('cy', latestFeatures[art].y);
      }
    }

    if (debugCounters.chartsUpdated % 20 === 0) {
      debugLog("Current articulator positions", {
        tt: `(${latestFeatures.tt.x.toFixed(2)}, ${latestFeatures.tt.y.toFixed(2)})`,
        tb: `(${latestFeatures.tb.x.toFixed(2)}, ${latestFeatures.tb.y.toFixed(2)})`,
        td: `(${latestFeatures.td.x.toFixed(2)}, ${latestFeatures.td.y.toFixed(2)})`
      });
    }

    const tongue = document.getElementById('tongue');
    if (tongue) {
      try {
        const tonguePath = createTonguePath(
          latestFeatures.tt, 
          latestFeatures.tb, 
          latestFeatures.td
        );
        tongue.setAttribute('d', tonguePath);
      } catch (error) {
        debugLog("Error updating tongue path", error);
        tongue.setAttribute('d', 'M-1.3,-0.3 Q-0.6,-0.5 0,-0.6 Q0.6,-0.7 0.7,-0.65 Q0.6,-0.5 0,-0.4 Q-0.6,-0.3 -1.3,-0.3 Z');
      }
    }

    try {
      const lipPaths = createLipPaths(
        latestFeatures.ul, 
        latestFeatures.ll, 
        latestFeatures.li
      );
      
      const upperLip = document.getElementById('upper-lip');
      const lowerLip = document.getElementById('lower-lip');
      
      if (upperLip) upperLip.setAttribute('d', lipPaths.upperLip);
      if (lowerLip) lowerLip.setAttribute('d', lipPaths.lowerLip);
    } catch (error) {
      debugLog("Error updating lip paths", error);
    }

    if (featureHistory.pitch && featureHistory.loudness) {
      const latestPitch = featureHistory.pitch[featureHistory.pitch.length - 1];
      const latestLoudness = featureHistory.loudness[featureHistory.loudness.length - 1];
      updateSourceFeatures(latestPitch, latestLoudness);
    }
    
  } catch (error) {
    debugLog("Error in updateCharts", error);
    debugCounters.errors++;
  }
}

// ✅ ADD: Toggle debug markers function
function toggleDebugMarkers(show) {
  const debugMarkers = document.querySelectorAll('.debug-marker');
  debugMarkers.forEach(marker => {
    marker.style.display = show ? 'block' : 'none';
  });
}

/******************************************************************************
* AUDIO RECORDING & PROCESSING *
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
    this.port.postMessage({
      audio: input.slice()
    });
  }
  
  return true;
}
}
registerProcessor('audio-processor', AudioProcessor);
`;

function processAudioData(audioData) {
  try {
    debugCounters.audioDataReceived++;
    
    if (!audioData || audioData.length === 0) {
      debugLog("Empty audio data received");
      return;
    }
    
    for (let i = 0; i < audioData.length; i++) {
      const value = audioData[i];
      if (isNaN(value) || !isFinite(value)) {
        audioBuffer[audioBufferIndex] = 0;
      } else {
        audioBuffer[audioBufferIndex] = value;
      }
      audioBufferIndex = (audioBufferIndex + 1) % config.bufferSize;
    }
    
  } catch (error) {
    debugLog("Error processing audio data", error);
    debugCounters.errors++;
  }
}

async function startRecording() {
  try {
    debugLog("Starting recording...");
    
    animationRunning = false;
    if (animationFrame) {
      clearTimeout(animationFrame);
      animationFrame = null;
    }
    debugLog("Test animation stopped");
      
    debugLog("Requesting microphone access...");
    audioStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        sampleRate: config.sampleRate,
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true
      }
    });
    debugLog("Microphone access granted");
      
    debugLog("Creating audio context...");
    audioContext = new (window.AudioContext || window.webkitAudioContext)({
      sampleRate: config.sampleRate
    });
    debugLog(`Audio context created - Sample rate: ${audioContext.sampleRate}`);
      
    if (audioContext.audioWorklet) {
      debugLog("Using AudioWorklet");
      const blob = new Blob([audioProcessorCode], { type: 'application/javascript' });
      const processorUrl = URL.createObjectURL(blob);
          
      await audioContext.audioWorklet.addModule(processorUrl);
      debugLog("AudioWorklet module added");
          
      workletNode = new AudioWorkletNode(audioContext, 'audio-processor');
      debugLog("AudioWorklet node created");
          
      workletNode.port.onmessage = (event) => {
        if (event.data.audio) {
          processAudioData(event.data.audio);
        }
      };
          
      const source = audioContext.createMediaStreamSource(audioStream);
      source.connect(workletNode);
      workletNode.connect(audioContext.destination);
      debugLog("Audio nodes connected");
          
    } else {
      debugLog("Using ScriptProcessorNode fallback");
      const source = audioContext.createMediaStreamSource(audioStream);
      const processor = audioContext.createScriptProcessor(config.frameSize, 1, 1);
      processor.onaudioprocess = (event) => {
        const input = event.inputBuffer.getChannelData(0);
        processAudioData(input);
      };
      source.connect(processor);
      processor.connect(audioContext.destination);
      workletNode = processor;
      debugLog("ScriptProcessor nodes connected");
    }
      
    isRecording = true;
    document.getElementById('startButton').disabled = true;
    document.getElementById('stopButton').disabled = false;
    updateStatus("Recording...");
    debugLog("UI updated, starting feature extraction loop");
      
    extractFeaturesLoop();
    debugLog("Feature extraction loop started");
    
  } catch (error) {
    debugLog("Error starting recording", error);
    updateStatus("Error starting recording: " + error.message);
    console.error("Recording error:", error);
  }
}

function stopRecording() {
  if (audioStream) {
    audioStream.getTracks().forEach(track => track.stop());
      
    if (workletNode) {
      workletNode.disconnect();
      workletNode = null;
    }
      
    if (audioContext) {
      audioContext.close();
      audioContext = null;
    }
      
    isRecording = false;
    document.getElementById('startButton').disabled = false;
    document.getElementById('stopButton').disabled = true;
    updateStatus("Recording stopped.");

    if (!animationRunning) {
      testArticulatorAnimation();
    }
  }
}

/******************************************************************************
* EVENT LISTENERS & INITIALIZATION *
******************************************************************************/

document.addEventListener('DOMContentLoaded', function() {
  init().catch(error => {
    console.error("Error during initialization:", error);
    updateStatus("Initialization error: " + error.message);
    debugCounters.errors++;
  });
});