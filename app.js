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

// Debug logging
function debugLog(message, data = null) {
  const timestamp = new Date().toLocaleTimeString();
  if (data) {
    console.log(`[${timestamp}] SPARC DEBUG: ${message}`, data);
  } else {
    console.log(`[${timestamp}] SPARC DEBUG: ${message}`);
  }
}

// Debug counters with error tracking
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

// Debug status display
function updateDebugStatus() {
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

// Smoothed features with validation
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

// Feature history
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

// Worker management
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
    
    // Add visual indicators for errors
    if (message.includes('ERROR') || message.includes('CRITICAL')) {
      statusElement.style.backgroundColor = '#ffebee';
      statusElement.style.color = '#c62828';
      statusElement.style.fontWeight = 'bold';
    } else if (message.includes('WARNING')) {
      statusElement.style.backgroundColor = '#fff3e0';
      statusElement.style.color = '#ef6c00';
    } else {
      statusElement.style.backgroundColor = '#e8f5e8';
      statusElement.style.color = '#2e7d32';
      statusElement.style.fontWeight = 'normal';
    }
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
// Initialize the ML worker with error handling
async function initSparcWorker() {
  if (SparcWorker) return Promise.resolve();
  
  return new Promise((resolve, reject) => {
    try {
      debugLog("Initializing ML worker...");
      SparcWorker = new Worker('sparc-worker.js');
      
      const initTimeout = setTimeout(() => {
        debugLog("❌ Worker initialization timeout - CRITICAL FAILURE");
        reject(new Error("Worker initialization timeout - models not loading"));
      }, 10000); // Longer timeout, but still fail hard
      
      SparcWorker.onmessage = function(e) {
        const message = e.data;
        
        workerResponseTimeouts.forEach(timeoutId => {
          clearTimeout(timeoutId);
          workerResponseTimeouts.delete(timeoutId);
        });
        
        debugLog(`Worker message received: ${message.type}`);
        
        switch(message.type) {
          case 'initialized':
            debugLog("✅ Worker initialization complete");
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
          case 'timeout':
            clearTimeout(initTimeout);
            debugLog("❌ Worker error during initialization", message);
            reject(new Error(`Worker error: ${message.error || 'Unknown error'}`));
            break;
        }
      };
      
      SparcWorker.onerror = function(error) {
        clearTimeout(initTimeout);
        debugLog("❌ Worker error event", error);
        reject(new Error(`Worker creation failed: ${error.message}`));
      };
      
      SparcWorker.postMessage({
        type: 'init',
        onnxPath: 'models/wavlm_base_layer9_quantized.onnx',
        linearModelPath: 'models/wavlm_linear_model.json'
      });
      
    } catch (error) {
      debugLog("❌ Error creating worker", error);
      reject(new Error(`Worker creation failed: ${error.message}`));
    }
  });
}

// Main thread fallback function
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

// Handle worker feature responses with validation
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
  debugLog("❌ Worker error - REAL PROBLEM!", message);
  debugCounters.errors++;
  
  if (message.type === 'timeout') {
    debugCounters.timeouts++;
  }
  
  pendingWorkerResponses = Math.max(0, pendingWorkerResponses - 1);
  
  updateStatus(`ERROR: Worker error: ${message.error || 'Unknown error'}`);
  
  if (!workerInitialized) {
    updateStatus("CRITICAL: Worker initialization failed: " + (message.error || 'Unknown error'));
  }
}

/******************************************************************************
* FEATURE EXTRACTION LOOP *
******************************************************************************/
async function extractFeaturesLoop() {
  if (!isRecording) {
    return;
  }
  
  setTimeout(extractFeaturesLoop, config.updateInterval);
  
  if (!workerInitialized) {
    debugLog("❌ Worker not initialized - stopping extraction");
    updateStatus("ERROR: ML models not loaded. Please wait for initialization or refresh.");
    return;
  }
  
  if (pendingWorkerResponses >= 1) {
    debugLog(`⏳ Skipping frame - pending response: ${pendingWorkerResponses}`);
    return;
  }
  
  try {
    const recentAudio = getRecentAudioBuffer();
    if (!recentAudio || recentAudio.length === 0) {
      debugLog("❌ No audio data available");
      return;
    }
    
    const timeoutId = setTimeout(() => {
      if (workerResponseTimeouts.has(timeoutId)) {
        debugLog("❌ Worker response timeout (1s) - REAL PROBLEM!");
        debugCounters.timeouts++;
        pendingWorkerResponses = Math.max(0, pendingWorkerResponses - 1);
        workerResponseTimeouts.delete(timeoutId);
        
        updateStatus("ERROR: ML processing timeout. Check worker performance.");
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
    debugLog("❌ Feature extraction error - REAL PROBLEM!", error);
    debugCounters.errors++;
    updateStatus(`ERROR: Feature extraction failed: ${error.message}`);
    // ❌ REMOVED: Fallback on error
  }
}

// Feature history update with validation
function updateFeatureHistory(articulationFeatures, pitch, loudness) {
  try {
    if (!articulationFeatures || typeof pitch !== 'number' || typeof loudness !== 'number') {
      throw new Error("Invalid feature data");
    }
    
    const alpha = isRecording ? smoothingFactor : 0.3;
    const articulators = ['ul', 'll', 'li', 'tt', 'tb', 'td'];
    
    for (const art of articulators) {
      if (articulationFeatures[art]) {
        let newX = articulationFeatures[art].x;
        let newY = articulationFeatures[art].y;
        
        if (isNaN(newX) || isNaN(newY) || !isFinite(newX) || !isFinite(newY)) {
          debugLog(`Invalid coordinates for ${art}: (${newX}, ${newY})`);
          continue;
        }
        
        if (art === 'ul') {
          // Allow lip protrusion for /u/ sounds
          newX = Math.min(Math.max(newX, 0.7), 1.0); // Allow lips to move forward/back
          newY = Math.min(Math.max(newY, -1.1), -0.8);
        } else if (art === 'll') {
          // Allow lip protrusion
          newX = Math.min(Math.max(newX, 0.7), 1.0); // Allow lips to move forward/back
          newY = Math.min(Math.max(newY, -0.6), -0.3);
        } else if (art === 'li') {
          // Allow lip interface movement
          newX = Math.min(Math.max(newX, 0.7), 1.0); // Allow lips to move forward/back
          newY = Math.min(Math.max(newY, -1.0), -0.4);
        } else if (art === 'tt') {
          // Allow tongue tip to reach teeth/alveolar ridge
          newX = Math.min(Math.max(newX, -0.5), 0.9); // Extended from 0.8 to 0.9
          newY = Math.min(Math.max(newY, -1.1), -0.25); // Extended range
        } else if (art === 'tb') {
          // Vowel space coverage
          newX = Math.min(Math.max(newX, -0.9), 0.6); // Extended range
          newY = Math.min(Math.max(newY, -1.15), -0.2); // Extended range
        } else if (art === 'td') {
          // Allow tongue dorsum to reach soft palate
          newX = Math.min(Math.max(newX, -1.3), 0.1); // Extended back range
          newY = Math.min(Math.max(newY, -1.05), -0.25); // Extended range
        }
        
        const oldX = smoothedFeatures[art + '_x'];
        const oldY = smoothedFeatures[art + '_y'];
        
        smoothedFeatures[art + '_x'] = alpha * newX + (1 - alpha) * oldX;
        smoothedFeatures[art + '_y'] = alpha * newY + (1 - alpha) * oldY;
      }
    }
    
    // Ensure lip ordering but don't over-constrain
    if (smoothedFeatures.ul_y >= smoothedFeatures.ll_y - 0.05) {
      smoothedFeatures.ll_y = smoothedFeatures.ul_y + 0.05;
    }
    
    // Log constraint application for debugging
    if (debugCounters.featuresUpdated % 50 === 0) {
      debugLog("Feature constraint check", {
        ul_x: smoothedFeatures.ul_x.toFixed(3),
        ll_x: smoothedFeatures.ll_x.toFixed(3),
        tt_x: smoothedFeatures.tt_x.toFixed(3),
        td_x: smoothedFeatures.td_x.toFixed(3)
      });
    }
    
    // Update history
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
    ul: { x: 0.95, y: -1.0 },   // Upper lip at front, high
    ll: { x: 0.95, y: -0.7 },   // Lower lip at front, lower  
    li: { x: 0.95, y: -0.85 },  // Lip interface at center
    tt: { x: 0.6, y: -0.7 },    // Tongue tip
    tb: { x: 0.0, y: -0.6 },    // Tongue body
    td: { x: -0.6, y: -0.5 }    // Tongue dorsum
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

// Initialize application with fallback
async function init() {
  try {
    updateStatus("Loading models...");
    
    initializeFeatureHistory();
    
    await initSparcWorker(); // This MUST succeed or app fails
    
    setupCharts();
    setupSensitivityControls();
    initializeDefaultPositions();
    
    document.getElementById('startButton').disabled = false;
    updateStatus("✅ Models loaded successfully. Ready to start.");
    
    document.getElementById('startButton').addEventListener('click', startRecording);
    document.getElementById('stopButton').addEventListener('click', stopRecording);
    
    const debugMode = document.getElementById('debug-mode');
    if (debugMode) {
      debugMode.checked = true;
      debugMode.addEventListener('change', function() {
        toggleDebugMarkers(this.checked);
      });
      toggleDebugMarkers(true);
    }

    // Only show test animation if models actually loaded
    if (!isRecording) {
      testArticulatorAnimation();
    }

  } catch (error) {
    // ❌ HARD FAILURE instead of graceful degradation
    updateStatus(`CRITICAL ERROR: ${error.message}`);
    debugLog("❌ Model loading failed - app cannot function", error);
    debugCounters.errors++;
    
    // Disable the app
    document.getElementById('startButton').disabled = true;
    document.getElementById('testButton').disabled = true;
    
    // Show clear error message
    const errorMsg = document.createElement('div');
    errorMsg.style.cssText = `
      position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%);
      background: #ffcdd2; border: 2px solid #f44336; border-radius: 10px;
      padding: 20px; font-size: 16px; font-weight: bold; color: #c62828;
      z-index: 10000; text-align: center; min-width: 400px;
    `;
    errorMsg.innerHTML = `
      <h3>🚨 SPARC Initialization Failed</h3>
      <p>The ML models could not be loaded.</p>
      <p><strong>Possible causes:</strong></p>
      <ul style="text-align: left; margin: 10px 0;">
        <li>Missing model files in /models/ directory</li>
        <li>Network connectivity issues</li>
        <li>Browser compatibility problems</li>
        <li>ONNX Runtime not working</li>
      </ul>
      <p><strong>Error:</strong> ${error.message}</p>
      <button onclick="this.parentElement.remove()" style="padding: 10px 20px; margin-top: 10px;">Close</button>
    `;
    document.body.appendChild(errorMsg);
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

// ANATOMICALLY ACCURATE VOCAL TRACT MAPPING
function createStaticElements(svg) {
  // COORDINATE SYSTEM based on anatomical images:
  // X: -1.5 (back of pharynx) to +1.0 (front of lips)  
  // Y: -1.2 (palate/roof) to -0.2 (jaw/floor)
  // This matches the oval oral cavity shape from the images
  
  // 1. PALATE (roof of mouth) - matches the curved roof in Image 1
  const palate = document.createElementNS('http://www.w3.org/2000/svg', 'path');
  palate.setAttribute('class', 'palate');
  palate.setAttribute('id', 'palate');
  // Anatomically correct curve: steep at back (soft palate), flatter at front (hard palate)
  palate.setAttribute('d', 'M-1.3,-0.8 Q-1.0,-1.15 -0.5,-1.2 Q0.0,-1.2 0.5,-1.15 Q0.8,-1.1 1.0,-1.0');
  palate.setAttribute('fill', 'none');
  palate.setAttribute('stroke', '#555');
  palate.setAttribute('stroke-width', '0.03');
  svg.appendChild(palate);
  
  // 2. JAW/MANDIBLE (floor of mouth) - follows the lower boundary from Image 1
  const jaw = document.createElementNS('http://www.w3.org/2000/svg', 'path');
  jaw.setAttribute('class', 'jaw');
  jaw.setAttribute('id', 'jaw');
  // Parallel curve but lower, creating the oral cavity space
  jaw.setAttribute('d', 'M-1.2,-0.3 Q-0.8,-0.25 -0.3,-0.2 Q0.2,-0.2 0.6,-0.25 Q0.9,-0.3 1.0,-0.4');
  jaw.setAttribute('fill', 'none');
  jaw.setAttribute('stroke', '#555');
  jaw.setAttribute('stroke-width', '0.03');
  svg.appendChild(jaw);
  
  // 3. PHARYNGEAL WALL (back of throat) - connects palate to jaw at back
  const pharynxWall = document.createElementNS('http://www.w3.org/2000/svg', 'path');
  pharynxWall.setAttribute('class', 'pharynx');
  // Vertical connection matching the pharyngeal cavity from Image 1
  pharynxWall.setAttribute('d', 'M-1.3,-0.8 Q-1.35,-0.55 -1.2,-0.3');
  pharynxWall.setAttribute('fill', 'none');
  pharynxWall.setAttribute('stroke', '#555');
  pharynxWall.setAttribute('stroke-width', '0.02');
  svg.appendChild(pharynxWall);
  
  // 4. ALVEOLAR RIDGE (gum ridge behind teeth) - critical for tongue contact
  const alveolarRidge = document.createElementNS('http://www.w3.org/2000/svg', 'path');
  alveolarRidge.setAttribute('class', 'alveolar');
  alveolarRidge.setAttribute('d', 'M0.7,-1.05 Q0.5,-1.08 0.3,-1.05');
  alveolarRidge.setAttribute('fill', 'none');
  alveolarRidge.setAttribute('stroke', '#777');
  alveolarRidge.setAttribute('stroke-width', '0.015');
  svg.appendChild(alveolarRidge);
  
  // 5. UPPER TEETH - positioned at front, on the palate line
  const upperTeeth = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
  upperTeeth.setAttribute('class', 'teeth');
  upperTeeth.setAttribute('x', '0.8');
  upperTeeth.setAttribute('y', '-1.1');
  upperTeeth.setAttribute('width', '0.12');
  upperTeeth.setAttribute('height', '0.08');
  upperTeeth.setAttribute('fill', 'white');
  upperTeeth.setAttribute('stroke', '#333');
  upperTeeth.setAttribute('stroke-width', '0.008');
  upperTeeth.setAttribute('rx', '0.01');
  svg.appendChild(upperTeeth);
  
  // 6. LOWER TEETH - positioned at front, on the jaw line
  const lowerTeeth = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
  lowerTeeth.setAttribute('class', 'teeth');
  lowerTeeth.setAttribute('x', '0.8');
  lowerTeeth.setAttribute('y', '-0.35');
  lowerTeeth.setAttribute('width', '0.12');
  lowerTeeth.setAttribute('height', '0.08');
  lowerTeeth.setAttribute('fill', 'white');
  lowerTeeth.setAttribute('stroke', '#333');
  lowerTeeth.setAttribute('stroke-width', '0.008');
  lowerTeeth.setAttribute('rx', '0.01');
  svg.appendChild(lowerTeeth);
  
  // Labels positioned anatomically
  addLabel(svg, "PHARYNX", -1.1, -0.1);
  addLabel(svg, "ORAL CAVITY", 0.0, 0.1);
  addLabel(svg, "FRONT", 0.8, 0.1);
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

// VOWEL POSITION PRESETS
const VOWEL_POSITIONS = {
  'i': { // High front - lips slightly spread
    ul: {x: 0.95, y: -1.05}, ll: {x: 0.95, y: -0.95}, li: {x: 0.95, y: -1.0},
    tt: {x: 0.6, y: -1.0}, tb: {x: 0.3, y: -1.05}, td: {x: -0.4, y: -0.8}
  },
  'e': { // Mid front - medium opening
    ul: {x: 0.95, y: -1.0}, ll: {x: 0.95, y: -0.8}, li: {x: 0.95, y: -0.9},
    tt: {x: 0.5, y: -0.8}, tb: {x: 0.1, y: -0.8}, td: {x: -0.6, y: -0.7}
  },
  'a': { // Low central - mouth wide open
    ul: {x: 0.95, y: -0.9}, ll: {x: 0.95, y: -0.4}, li: {x: 0.95, y: -0.65},
    tt: {x: 0.2, y: -0.4}, tb: {x: -0.2, y: -0.35}, td: {x: -0.8, y: -0.4}
  },
  'u': { // High back - small rounded opening
    ul: {x: 0.95, y: -1.0}, ll: {x: 0.95, y: -0.9}, li: {x: 0.95, y: -0.95},
    tt: {x: 0.3, y: -0.6}, tb: {x: -0.3, y: -1.0}, td: {x: -0.9, y: -0.9}
  },
  'o': { // Mid back - medium rounded opening
    ul: {x: 0.95, y: -0.95}, ll: {x: 0.95, y: -0.75}, li: {x: 0.95, y: -0.85},
    tt: {x: 0.2, y: -0.5}, tb: {x: -0.4, y: -0.8}, td: {x: -0.9, y: -0.8}
  }
};

// TESTING FUNCTION: Cycle through vowel positions (replaces testExtremePositions)
function testExtremePositions() {
  const vowelSequence = [
    { name: '/i/ - high front', pos: VOWEL_POSITIONS['i'] },
    { name: '/a/ - low central', pos: VOWEL_POSITIONS['a'] },  
    { name: '/u/ - high back', pos: VOWEL_POSITIONS['u'] },
    { name: '/e/ - mid front', pos: VOWEL_POSITIONS['e'] },
    { name: '/o/ - mid back', pos: VOWEL_POSITIONS['o'] }
  ];
  
  let index = 0;
  console.log("Testing anatomical vowel positions...");
  updateStatus("Testing vowel positions...");
  
  const interval = setInterval(() => {
    if (index >= vowelSequence.length) {
      clearInterval(interval);
      initializeDefaultPositions();
      console.log("Vowel position test complete");
      updateStatus("Ready to start.");
      return;
    }
    
    const current = vowelSequence[index];
    console.log(`Testing ${current.name}`, current.pos);
    updateStatus(`Demo: ${current.name}`);
    
    // Update with realistic pitch and loudness for each vowel
    updateFeatureHistory(current.pos, 140 + Math.random() * 40, -20 + Math.random() * 10);
    updateCharts();
    
    index++;
  }, 2500); // 2.5 seconds per vowel for clear observation
}

// COORDINATE VERIFICATION AND DEBUGGING
function verifyAnatomicalBounds() {
  console.log("=== ANATOMICAL COORDINATE SYSTEM ===");
  console.log("X-axis: -1.5 (pharynx) → +1.0 (lips)");
  console.log("Y-axis: -1.2 (palate) → -0.2 (jaw)"); 
  console.log("Oral cavity bounds: X[-1.2, 0.9], Y[-1.1, -0.3]");
  console.log("Key landmarks:");
  console.log("  Alveolar ridge: X[0.3, 0.7], Y[-1.05, -1.08]");
  console.log("  Teeth: X[0.8, 0.92], Y[-1.1, -0.35]");
  console.log("  Soft palate: X[-1.3, -0.8], Y[-0.8, -1.0]");
  console.log("  Tongue space: X[-1.2, 0.8], Y[-1.05, -0.25]");
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
// ANATOMICALLY CONSTRAINED TONGUE PATH
function createTonguePath(tt, tb, td) {
  // Map articulator coordinates to anatomical space
  // tt (tongue tip): can reach from alveolar ridge to mid-palate
  // tb (tongue body): central oral cavity, main vowel articulator  
  // td (tongue dorsum): back of mouth, can approach soft palate
  
  // Sanitize inputs first
  tt = sanitizePoint(tt, 0.4, -0.7);   // Default: mid-front, mid-height
  tb = sanitizePoint(tb, -0.2, -0.6);  // Default: central, mid-height
  td = sanitizePoint(td, -0.8, -0.5);  // Default: back, mid-height
  
  // Apply ANATOMICAL constraints based on the oral cavity boundaries
  
  // Tongue tip: front of mouth, can reach alveolar ridge
  tt.x = Math.min(Math.max(tt.x, -0.5), 0.8);   // From mid-mouth to alveolar ridge
  tt.y = Math.min(Math.max(tt.y, -1.05), -0.3); // From alveolar ridge to jaw
  
  // Tongue body: central oral cavity, main vowel space (see Image 2)
  tb.x = Math.min(Math.max(tb.x, -0.8), 0.5);   // Front-back vowel dimension
  tb.y = Math.min(Math.max(tb.y, -1.1), -0.25); // High-low vowel dimension
  
  // Tongue dorsum: back of mouth, can approach soft palate  
  td.x = Math.min(Math.max(td.x, -1.2), 0.0);   // Back region only
  td.y = Math.min(Math.max(td.y, -1.0), -0.3);  // Can approach soft palate
  
  // Ensure anatomical ordering: back to front progression
  if (td.x > tb.x - 0.1) td.x = tb.x - 0.1;
  if (tb.x > tt.x - 0.1) tb.x = tt.x - 0.1;
  
  // Tongue root: attached at base of mouth, back region
  const tongueRoot = { x: -1.15, y: -0.35 };
  
  // Create realistic tongue surface based on Image 1 and 2 tongue shapes
  const tonguePath = `
    M ${tongueRoot.x} ${tongueRoot.y}
    Q ${td.x - 0.05} ${td.y - 0.02} ${td.x} ${td.y}
    Q ${(td.x + tb.x)/2} ${Math.min(td.y, tb.y) - 0.03} ${tb.x} ${tb.y}
    Q ${(tb.x + tt.x)/2} ${Math.min(tb.y, tt.y) - 0.02} ${tt.x} ${tt.y}
    L ${tt.x + 0.02} ${tt.y + 0.04}
    Q ${tb.x + 0.02} ${tb.y + 0.06} ${td.x + 0.02} ${td.y + 0.05}
    Q ${tongueRoot.x + 0.05} ${tongueRoot.y + 0.02} ${tongueRoot.x} ${tongueRoot.y}
    Z
  `;
  
  return tonguePath;
}

// ANATOMICALLY POSITIONED LIPS
function createLipPaths(ul, ll, li) {
  // Ensure we have valid inputs
  ul = sanitizePoint(ul, 0.95, -1.0);   
  ll = sanitizePoint(ll, 0.95, -0.4);   
  li = sanitizePoint(li, 0.95, -0.7);   
  
  // Force lips to be at the front of the mouth
  ul.x = ll.x = li.x = 0.95;
  
  // Ensure proper ordering
  ul.y = Math.min(Math.max(ul.y, -1.1), -0.8);
  ll.y = Math.min(Math.max(ll.y, -0.6), -0.3);
  
  // Force upper lip to be above lower lip
  if (ul.y >= ll.y - 0.05) {
    ll.y = ul.y + 0.05;
  }
  
  const lipWidth = 0.06;
  const leftX = 0.95 - lipWidth;
  const rightX = 0.95 + lipWidth;
  const centerX = 0.95;
  
  // SIMPLE UPPER LIP - horizontal oval
  const upperLipPath = `
    M ${leftX} ${ul.y + 0.01}
    Q ${centerX} ${ul.y - 0.015} ${rightX} ${ul.y + 0.01}
    Q ${centerX} ${ul.y + 0.025} ${leftX} ${ul.y + 0.01}
    Z
  `;
  
  // SIMPLE LOWER LIP - horizontal oval
  const lowerLipPath = `
    M ${leftX} ${ll.y - 0.01}
    Q ${centerX} ${ll.y + 0.015} ${rightX} ${ll.y - 0.01}
    Q ${centerX} ${ll.y - 0.025} ${leftX} ${ll.y - 0.01}
    Z
  `;
  
  return {
    upperLip: upperLipPath,
    lowerLip: lowerLipPath
  };
}

// DEBUGGING FUNCTION to check lip positioning
function debugLipPositions() {
  console.log("=== DEBUGGING LIP POSITIONS ===");
  
  const testPositions = ['i', 'a', 'u'];
  testPositions.forEach(vowel => {
    const pos = VOWEL_POSITIONS[vowel];
    console.log(`Vowel /${vowel}/:`);
    console.log(`  UL: (${pos.ul.x}, ${pos.ul.y})`);
    console.log(`  LL: (${pos.ll.x}, ${pos.ll.y})`);
    console.log(`  LI: (${pos.li.x}, ${pos.li.y})`);
    console.log(`  Opening: ${Math.abs(pos.ul.y - pos.ll.y).toFixed(3)}`);
  });
  
  // Test current lip paths
  const testLips = createLipPaths(
    {x: 0.95, y: -1.0},
    {x: 0.95, y: -0.7}, 
    {x: 0.95, y: -0.85}
  );
  
  console.log("Sample lip paths:", testLips);
}

// TEST FUNCTION: Check lip opening range
function testLipOpening() {
  console.log("=== TESTING LIP OPENING RANGE ===");
  
  // Test different mouth openings
  const testCases = [
    { name: "Closed (sleep)", ul: -1.0, ll: -0.95 },
    { name: "Slightly open (rest)", ul: -1.0, ll: -0.8 },
    { name: "Medium open (/e/)", ul: -1.0, ll: -0.7 },
    { name: "Wide open (/a/)", ul: -0.9, ll: -0.4 }
  ];
  
  testCases.forEach(test => {
    const opening = test.ll - test.ul;
    console.log(`${test.name}: UL=${test.ul}, LL=${test.ll}, Opening=${opening.toFixed(3)}`);
    
    // Test the lip path generation
    const lipPaths = createLipPaths(
      {x: 0.95, y: test.ul},
      {x: 0.95, y: test.ll},
      {x: 0.95, y: (test.ul + test.ll)/2}
    );
    console.log(`  Generated paths: ${lipPaths.upperLip.length + lipPaths.lowerLip.length} chars`);
  });
}

function debugSVGElements() {
  console.log("=== SVG ELEMENTS DEBUG ===");
  
  const upperLip = document.getElementById('upper-lip');
  const lowerLip = document.getElementById('lower-lip');
  
  if (upperLip) {
    console.log("Upper lip element:", upperLip);
    console.log("Upper lip path:", upperLip.getAttribute('d'));
    console.log("Upper lip style:", window.getComputedStyle(upperLip));
  } else {
    console.log("❌ Upper lip element not found!");
  }
  
  if (lowerLip) {
    console.log("Lower lip element:", lowerLip);
    console.log("Lower lip path:", lowerLip.getAttribute('d'));
    console.log("Lower lip style:", window.getComputedStyle(lowerLip));
  } else {
    console.log("❌ Lower lip element not found!");
  }
  
  // Check if they're being rendered
  const svg = document.getElementById('vocal-tract-svg');
  if (svg) {
    console.log("SVG viewBox:", svg.getAttribute('viewBox'));
    console.log("All SVG children:", svg.children.length);
    Array.from(svg.children).forEach((child, i) => {
      console.log(`  ${i}: ${child.tagName} id="${child.id}" class="${child.className}"`);
    });
  }
}

// FORCE LIP RESET to fix stuck positions
function resetLipPositions() {
  console.log("🔧 Resetting lip positions...");
  
  // Force neutral lip positions
  smoothedFeatures.ul_x = 0.95;
  smoothedFeatures.ul_y = -1.0;
  smoothedFeatures.ll_x = 0.95;
  smoothedFeatures.ll_y = -0.7;
  smoothedFeatures.li_x = 0.95;
  smoothedFeatures.li_y = -0.85;
  
  // Update history
  if (featureHistory.ul_x) {
    featureHistory.ul_x.fill(0.95);
    featureHistory.ul_y.fill(-1.0);
    featureHistory.ll_x.fill(0.95);
    featureHistory.ll_y.fill(-0.7);
    featureHistory.li_x.fill(0.95);
    featureHistory.li_y.fill(-0.85);
  }
  
  updateCharts();
  console.log("✅ Lip positions reset");
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

    // Get latest positions (these should already be constrained)
    for (const art of articulators) {
      const xKey = art + '_x';
      const yKey = art + '_y';
      
      if (featureHistory[xKey] && featureHistory[yKey]) {
        latestFeatures[art] = {
          x: featureHistory[xKey][featureHistory[xKey].length - 1],
          y: featureHistory[yKey][featureHistory[yKey].length - 1]
        };
      } else {
        latestFeatures[art] = { x: 0, y: -0.5 };
      }
    }

    // Update ALL markers to their constrained positions
    for (const art of articulators) {
      const marker = document.getElementById(`${art}-marker`);
      if (marker && latestFeatures[art]) {
        marker.setAttribute('cx', latestFeatures[art].x);
        marker.setAttribute('cy', latestFeatures[art].y);
      }
    }

    // Debug current positions
    if (debugCounters.chartsUpdated % 20 === 0) {
      debugLog("Lip positions", {
        ul: `(${latestFeatures.ul.x.toFixed(2)}, ${latestFeatures.ul.y.toFixed(2)})`,
        ll: `(${latestFeatures.ll.x.toFixed(2)}, ${latestFeatures.ll.y.toFixed(2)})`,
        opening: `${(latestFeatures.ll.y - latestFeatures.ul.y).toFixed(3)}`
      });
    }

    // Update tongue
    const tongue = document.getElementById('tongue');
    if (tongue) {
      const tonguePath = createTonguePath(
        latestFeatures.tt, 
        latestFeatures.tb, 
        latestFeatures.td
      );
      tongue.setAttribute('d', tonguePath);
    }

    // Update lips
    const lipPaths = createLipPaths(
      latestFeatures.ul, 
      latestFeatures.ll, 
      latestFeatures.li
    );
    
    const upperLip = document.getElementById('upper-lip');
    const lowerLip = document.getElementById('lower-lip');
    
    if (upperLip) {
      upperLip.setAttribute('d', lipPaths.upperLip);
    }
    if (lowerLip) {
      lowerLip.setAttribute('d', lipPaths.lowerLip);
    }

    // Update pitch/loudness
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

// Toggle debug markers function
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
