/******************************************************************************
 * SPARC Feature Extraction - Web Client (Complete with Jaw Movement)
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

// Debug counters
let debugCounters = {
  audioDataReceived: 0,
  workerMessagesSent: 0,
  workerResponsesReceived: 0,
  featuresUpdated: 0,
  chartsUpdated: 0,
  errors: 0
};

// Update debug status display
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
  
  debugDisplay.innerHTML = `
Audio: ${debugCounters.audioDataReceived}
Worker msgs: ${debugCounters.workerMessagesSent}
Responses: ${debugCounters.workerResponsesReceived}
Features: ${debugCounters.featuresUpdated}
Charts: ${debugCounters.chartsUpdated}
Errors: ${debugCounters.errors}
  `.trim();
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

// IMPROVED: Smoothed features including JAW
let smoothedFeatures = {
  ul_x: 0.9, ul_y: -1.05,
  ll_x: 0.9, ll_y: -0.8,
  li_x: 0.85, li_y: -0.92,
  tt_x: 0.5, tt_y: -0.7,
  tb_x: 0.0, tb_y: -0.6,
  td_x: -0.5, td_y: -0.5,
  jaw_opening: 0.2  // NEW: Jaw opening (0 = closed, 1 = wide open)
};

// Sensitivity factor for EMA values (1.0 = raw values from model)
// Note: Set to 1.0 for accurate feature extraction matching Python SPARC
let sensitivityFactor = 1.0;
let smoothingFactor = 0.4;

// Feature history including jaw
let featureHistory = {};

function initializeFeatureHistory() {
  const articulators = ['ul_x', 'ul_y', 'll_x', 'll_y', 'li_x', 'li_y', 
                       'tt_x', 'tt_y', 'tb_x', 'tb_y', 'td_x', 'td_y', 
                       'jaw_opening', 'pitch', 'loudness'];
  
  featureHistory = {};
  articulators.forEach(key => {
    featureHistory[key] = Array(100).fill(key === 'jaw_opening' ? 0.2 : 0);
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
    
    // Read backwards from current write position to get most recent audio
    for (let i = 0; i < config.bufferSize; i++) {
      // Calculate index: go back from current position
      const index = (audioBufferIndex - config.bufferSize + i + config.bufferSize) % config.bufferSize;
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

// IMPROVED: Relaxed constraints - allow realistic movements
function applyAnatomicalConstraints(features) {
  // Allow lip protrusion for sounds like /u/
  features.ul.x = Math.min(Math.max(features.ul.x, 0.6), 1.0);  // Relaxed from forced 0.95
  features.ul.y = Math.min(Math.max(features.ul.y, -1.2), -0.8);
  
  features.ll.x = Math.min(Math.max(features.ll.x, 0.6), 1.0);  // Relaxed from forced 0.95
  features.ll.y = Math.min(Math.max(features.ll.y, -0.6), -0.2);
  
  features.li.x = Math.min(Math.max(features.li.x, 0.6), 1.0);  // Relaxed from forced 0.95
  features.li.y = Math.min(Math.max(features.li.y, -1.0), -0.4);
  
  // Extended tongue ranges for better speech sound coverage
  features.tt.x = Math.min(Math.max(features.tt.x, -0.5), 0.9);
  features.tt.y = Math.min(Math.max(features.tt.y, -1.1), -0.25);
  
  features.tb.x = Math.min(Math.max(features.tb.x, -0.9), 0.6);
  features.tb.y = Math.min(Math.max(features.tb.y, -1.15), -0.2);
  
  features.td.x = Math.min(Math.max(features.td.x, -1.3), 0.1);
  features.td.y = Math.min(Math.max(features.td.y, -1.05), -0.25);
  
  // Ensure anatomical ordering
  if (features.ul.y >= features.ll.y - 0.05) {
    features.ll.y = features.ul.y + 0.05;
  }
}

// NEW: Calculate jaw opening based on lip positions
// NOTE: In EMA coords, ul_y and ll_y are in original space (not yet flipped)
function calculateJawOpening(ul_y, ll_y) {
  // The lips have negative Y values in EMA space
  // Lower lip is more negative (more down) than upper lip
  const lipOpening = Math.abs(ll_y - ul_y);
  // Convert lip opening to jaw opening (0 = closed, 1 = wide open)
  const jawOpening = Math.min(Math.max((lipOpening - 0.05) / 0.5, 0), 1);
  return jawOpening;
}

/******************************************************************************
* WORKER MANAGEMENT - SIMPLIFIED *
******************************************************************************/
async function initSparcWorker() {
  if (SparcWorker) return Promise.resolve();
  
  return new Promise((resolve, reject) => {
    debugLog("Initializing ML worker...");
    SparcWorker = new Worker('sparc-worker.js');
    
    const initTimeout = setTimeout(() => {
      debugLog("❌ Worker initialization timeout");
      reject(new Error("Worker initialization timeout"));
    }, 15000);
    
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
          clearTimeout(initTimeout);
          debugLog("❌ Worker error", message);
          reject(new Error(`Worker error: ${message.error || 'Unknown error'}`));
          break;
      }
    };
    
    SparcWorker.onerror = function(error) {
      clearTimeout(initTimeout);
      debugLog("❌ Worker error event", error);
      reject(new Error(`Worker creation failed: ${error.message}`));
    };
    
    // IMPORTANT: Model compatibility requirement
    // The linear model was trained on WavLM Large (1024 hidden dimensions).
    // Using the correct WavLM Large ONNX model for accurate feature extraction.
    // Cache busting version appended to force reload of new model
    const modelVersion = 'v2';  // Increment when models change
    SparcWorker.postMessage({
      type: 'init',
      onnxPath: `models/wavlm_large_layer9_quantized.onnx?v=${modelVersion}`,  // ✅ Correct model (1024 dims)
      linearModelPath: `models/wavlm_linear_model.json?v=${modelVersion}`
    });
  });
}

// SIMPLIFIED: Handle worker feature responses without fallbacks
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
    // NO FALLBACK - just log the error
  }
}

/******************************************************************************
* FEATURE EXTRACTION LOOP - SIMPLIFIED *
******************************************************************************/
async function extractFeaturesLoop() {
  if (!isRecording) return;
  
  setTimeout(extractFeaturesLoop, config.updateInterval);
  
  if (!workerInitialized) {
    debugLog("❌ Worker not initialized");
    updateStatus("ERROR: ML models not loaded");
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
        debugLog("❌ Worker response timeout (1s)");
        pendingWorkerResponses = Math.max(0, pendingWorkerResponses - 1);
        workerResponseTimeouts.delete(timeoutId);
        updateStatus("ERROR: ML processing timeout");
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
    debugLog("❌ Feature extraction error", error);
    debugCounters.errors++;
    updateStatus(`ERROR: Feature extraction failed: ${error.message}`);
  }
}

// IMPROVED: Feature history update with jaw movement
function updateFeatureHistory(articulationFeatures, pitch, loudness) {
  try {
    if (!articulationFeatures || typeof pitch !== 'number' || typeof loudness !== 'number') {
      throw new Error("Invalid feature data");
    }
    
    const alpha = isRecording ? smoothingFactor : 0.3;
    const articulators = ['ul', 'll', 'li', 'tt', 'tb', 'td'];
    
    // TEMPORARILY DISABLED: Apply constraints to incoming features
    // Let's see raw model output first to understand the actual coordinate space
    // applyAnatomicalConstraints(articulationFeatures);
    
    // Update articulator positions
    for (const art of articulators) {
      if (articulationFeatures[art]) {
        let newX = articulationFeatures[art].x;
        let newY = articulationFeatures[art].y;
        
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
    
    // NEW: Calculate and update jaw opening
    const jawOpening = calculateJawOpening(smoothedFeatures.ul_y, smoothedFeatures.ll_y);
    smoothedFeatures.jaw_opening = alpha * jawOpening + (1 - alpha) * smoothedFeatures.jaw_opening;
    
    // Log constraint application for debugging
    if (debugCounters.featuresUpdated % 50 === 0) {
      debugLog("Feature constraint check", {
        ul_x: smoothedFeatures.ul_x.toFixed(3),
        ll_x: smoothedFeatures.ll_x.toFixed(3),
        tt_x: smoothedFeatures.tt_x.toFixed(3),
        td_x: smoothedFeatures.td_x.toFixed(3),
        jaw_opening: smoothedFeatures.jaw_opening.toFixed(3)
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
      } else if (key === 'jaw_opening') {
        featureHistory[key].push(smoothedFeatures.jaw_opening);
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
    ul: { x: 0.9, y: -1.0 },
    ll: { x: 0.9, y: -0.7 },
    li: { x: 0.9, y: -0.85 },
    tt: { x: 0.6, y: -0.7 },
    tb: { x: 0.0, y: -0.6 },
    td: { x: -0.6, y: -0.5 }
  };
  
  Object.keys(defaultPositions).forEach(art => {
    smoothedFeatures[art + '_x'] = defaultPositions[art].x;
    smoothedFeatures[art + '_y'] = defaultPositions[art].y;
  });
  
  // Set default jaw opening
  smoothedFeatures.jaw_opening = calculateJawOpening(
    defaultPositions.ul.y, 
    defaultPositions.ll.y
  );
  
  if (featureHistory && Object.keys(featureHistory).length > 0) {
    for (let i = 0; i < 100; i++) {
      Object.keys(defaultPositions).forEach(art => {
        const xKey = art + '_x';
        const yKey = art + '_y';
        if (featureHistory[xKey] && featureHistory[yKey]) {
          featureHistory[xKey][i] = defaultPositions[art].x;
          featureHistory[yKey][i] = defaultPositions[art].y;
        }
      });
      if (featureHistory.jaw_opening) {
        featureHistory.jaw_opening[i] = smoothedFeatures.jaw_opening;
      }
    }
    updateCharts();
  }
  
  debugLog("Default positions initialized", defaultPositions);
}

// SIMPLIFIED: Initialize application without fallbacks
async function init() {
  try {
    updateStatus("Loading models...");
    
    initializeFeatureHistory();
    await initSparcWorker(); // This MUST succeed or app fails
    
    setupCharts();
    setupSensitivityControls();
    initializeDefaultPositions();
    
    const startButton = document.getElementById('startButton');
    const stopButton = document.getElementById('stopButton');
    
    if (startButton) {
      startButton.disabled = false;
      startButton.addEventListener('click', startRecording);
    }
    
    if (stopButton) {
      stopButton.addEventListener('click', stopRecording);
    }
    
    updateStatus("✅ Models loaded successfully. Ready to start.");
    
    const debugMode = document.getElementById('debug-mode');
    if (debugMode) {
      debugMode.checked = true;
      debugMode.addEventListener('change', function() {
        toggleDebugMarkers(this.checked);
      });
      toggleDebugMarkers(true);
    }

    if (!isRecording) {
      testArticulatorAnimation();
    }

  } catch (error) {
    // HARD FAILURE - no fallbacks
    updateStatus(`CRITICAL ERROR: ${error.message}`);
    debugLog("❌ Model loading failed - app cannot function", error);
    debugCounters.errors++;
    
    const startButton = document.getElementById('startButton');
    if (startButton) {
      startButton.disabled = true;
    }
    
    // Only disable test button if it exists
    const testButton = document.getElementById('testButton');
    if (testButton) {
      testButton.disabled = true;
    }
    
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
      <p><strong>Error:</strong> ${error.message}</p>
      <button onclick="this.parentElement.remove()" style="padding: 10px 20px; margin-top: 10px;">Close</button>
    `;
    document.body.appendChild(errorMsg);
  }
}

/******************************************************************************
* VISUALIZATION FUNCTIONS WITH JAW MOVEMENT *
******************************************************************************/
function setupVocalTractVisualization() {
  const svg = document.getElementById('vocal-tract-svg');
  
  if (!svg) {
    console.error("SVG element 'vocal-tract-svg' not found!");
    return;
  }
  
  // ViewBox: Standard SVG coordinates (Y+ = down, origin at top-left)
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

function createStaticElements(svg) {
  // PALATE (roof of mouth) - static, at TOP of mouth
  const palate = document.createElementNS('http://www.w3.org/2000/svg', 'path');
  palate.setAttribute('class', 'palate');
  palate.setAttribute('id', 'palate');
  palate.setAttribute('d', 'M-1.3,-0.8 Q-1.0,-1.15 -0.5,-1.2 Q0.0,-1.2 0.5,-1.15 Q0.8,-1.1 1.0,-1.0');
  palate.setAttribute('fill', 'none');
  palate.setAttribute('stroke', '#555');
  palate.setAttribute('stroke-width', '0.03');
  svg.appendChild(palate);
  
  // PHARYNGEAL WALL (back wall of throat, from palate down to jaw area)
  const pharynxWall = document.createElementNS('http://www.w3.org/2000/svg', 'path');
  pharynxWall.setAttribute('class', 'pharynx');
  pharynxWall.setAttribute('d', 'M-1.3,-0.8 Q-1.35,0.0 -1.2,0.3');
  pharynxWall.setAttribute('fill', 'none');
  pharynxWall.setAttribute('stroke', '#555');
  pharynxWall.setAttribute('stroke-width', '0.02');
  svg.appendChild(pharynxWall);
  
  // ALVEOLAR RIDGE
  const alveolarRidge = document.createElementNS('http://www.w3.org/2000/svg', 'path');
  alveolarRidge.setAttribute('class', 'alveolar');
  alveolarRidge.setAttribute('d', 'M0.7,-1.05 Q0.5,-1.08 0.3,-1.05');
  alveolarRidge.setAttribute('fill', 'none');
  alveolarRidge.setAttribute('stroke', '#777');
  alveolarRidge.setAttribute('stroke-width', '0.015');
  svg.appendChild(alveolarRidge);
  
  // UPPER TEETH - fixed to palate
  const upperTeeth = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
  upperTeeth.setAttribute('class', 'teeth');
  upperTeeth.setAttribute('id', 'upper-teeth');
  upperTeeth.setAttribute('x', '0.8');
  upperTeeth.setAttribute('y', '-1.1');
  upperTeeth.setAttribute('width', '0.12');
  upperTeeth.setAttribute('height', '0.08');
  upperTeeth.setAttribute('fill', 'white');
  upperTeeth.setAttribute('stroke', '#333');
  upperTeeth.setAttribute('stroke-width', '0.008');
  upperTeeth.setAttribute('rx', '0.01');
  svg.appendChild(upperTeeth);
  
  // Labels
  addLabel(svg, "PHARYNX", -1.1, -0.1);
  addLabel(svg, "ORAL CAVITY", 0.0, 0.1);
  addLabel(svg, "FRONT", 0.8, 0.1);
}

function createDynamicElements(svg) {
  // JAW - NEW: Dynamic jaw that moves with mouth opening
  const jaw = document.createElementNS('http://www.w3.org/2000/svg', 'path');
  jaw.setAttribute('class', 'jaw');
  jaw.setAttribute('id', 'jaw');
  jaw.setAttribute('fill', 'none');
  jaw.setAttribute('stroke', '#333');
  jaw.setAttribute('stroke-width', '0.03');
  svg.appendChild(jaw);
  
  // LOWER TEETH - NEW: Attached to jaw, moves with jaw
  const lowerTeeth = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
  lowerTeeth.setAttribute('class', 'teeth');
  lowerTeeth.setAttribute('id', 'lower-teeth');
  lowerTeeth.setAttribute('width', '0.12');
  lowerTeeth.setAttribute('height', '0.08');
  lowerTeeth.setAttribute('fill', 'white');
  lowerTeeth.setAttribute('stroke', '#333');
  lowerTeeth.setAttribute('stroke-width', '0.008');
  lowerTeeth.setAttribute('rx', '0.01');
  svg.appendChild(lowerTeeth);
  
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

// Create dynamic jaw path based on opening
// Jaw at bottom of mouth, drops down when opening
function createJawPath(jawOpening) {
  // Base jaw position (positive Y = bottom of screen in SVG)
  const baseY = 0.3;
  // Lower jaw drops MORE as mouth opens (more positive Y = further down)
  const jawY = baseY + (jawOpening * 0.4);
  
  return `M-1.2,${jawY} Q-0.8,${jawY + 0.05} -0.3,${jawY} Q0.2,${jawY} 0.6,${jawY + 0.05} Q0.9,${jawY} 1.0,${jawY + 0.1}`;
}

// Position lower teeth with jaw
function positionLowerTeeth(jawOpening) {
  const baseY = 0.25;
  // Teeth move down with jaw opening  
  const teethY = baseY + (jawOpening * 0.4);
  return {
    x: 0.8,
    y: teethY
  };
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

// Create tongue path from EMA coordinates
// EMA: X+ = forward (lips), X- = back (pharynx); Y+ = up (palate), Y- = down
// SVG: X+ = right, X- = left; Y+ = down, Y- = up
// Mapping: EMA_X → SVG_X (same), EMA_Y → SVG_(-Y) (flip)
function createTonguePath(tt, tb, td) {
  // Flip Y-axis for SVG: EMA Y+ (up) becomes SVG Y- (up in screen)
  tt = sanitizePoint({ x: tt.x, y: -tt.y }, 0.4, -0.7);   // tongue tip
  tb = sanitizePoint({ x: tb.x, y: -tb.y }, -0.2, -0.6);  // tongue body  
  td = sanitizePoint({ x: td.x, y: -td.y }, -0.8, -0.5);  // tongue dorsum
  
  // Tongue root at back/bottom of mouth (negative X = back, slightly positive Y = below center)
  const tongueRoot = { x: -1.15, y: 0.1 };
  
  // Draw tongue from root → dorsum → body → tip, then back along bottom
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

// Create lip paths from EMA coordinates
// EMA: Y+ = up, Y- = down; SVG: Y+ = down, Y- = up
// Mapping: EMA_Y → SVG_(-Y)
function createLipPaths(ul, ll, li) {
  // Flip Y-axis: EMA positive Y (up) → SVG negative Y (up on screen)
  ul = sanitizePoint({ x: ul.x, y: -ul.y }, 0.9, -1.0);   
  ll = sanitizePoint({ x: ll.x, y: -ll.y }, 0.9, -0.7);   
  li = sanitizePoint({ x: li.x, y: -li.y }, 0.9, -0.85);   
  
  // Allow lip movement
  const lipWidth = 0.06;
  const leftX = ul.x - lipWidth;
  const rightX = ul.x + lipWidth;
  const centerX = ul.x;
  
  const upperLipPath = `
    M ${leftX} ${ul.y + 0.01}
    Q ${centerX} ${ul.y - 0.015} ${rightX} ${ul.y + 0.01}
    Q ${centerX} ${ul.y + 0.025} ${leftX} ${ul.y + 0.01}
    Z
  `;
  
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

function setupCharts() {
  setupVocalTractVisualization();
  debugLog("Charts setup complete");
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

// VOWEL POSITION PRESETS WITH JAW OPENING (Updated for dramatic movement)
const VOWEL_POSITIONS = {
  'i': { // High front - very small jaw opening
    ul: {x: 0.9, y: -1.05}, ll: {x: 0.9, y: -1.0}, li: {x: 0.9, y: -1.025},
    tt: {x: 0.6, y: -1.0}, tb: {x: 0.3, y: -1.05}, td: {x: -0.4, y: -0.8},
    jaw_opening: 0.0  // Nearly closed
  },
  'e': { // Mid front - medium jaw opening
    ul: {x: 0.9, y: -1.0}, ll: {x: 0.9, y: -0.75}, li: {x: 0.9, y: -0.875},
    tt: {x: 0.5, y: -0.8}, tb: {x: 0.1, y: -0.8}, td: {x: -0.6, y: -0.7},
    jaw_opening: 0.4  // Medium opening
  },
  'a': { // Low central - very large jaw opening
    ul: {x: 0.9, y: -0.95}, ll: {x: 0.9, y: -0.3}, li: {x: 0.9, y: -0.625},
    tt: {x: 0.2, y: -0.4}, tb: {x: -0.2, y: -0.35}, td: {x: -0.8, y: -0.4},
    jaw_opening: 1.0  // Wide open
  },
  'u': { // High back - small jaw opening, lip protrusion
    ul: {x: 0.7, y: -1.0}, ll: {x: 0.7, y: -0.92}, li: {x: 0.7, y: -0.96},
    tt: {x: 0.3, y: -0.6}, tb: {x: -0.3, y: -1.0}, td: {x: -0.9, y: -0.9},
    jaw_opening: 0.1  // Nearly closed
  },
  'o': { // Mid back - medium jaw opening, slight lip protrusion
    ul: {x: 0.8, y: -0.95}, ll: {x: 0.8, y: -0.7}, li: {x: 0.8, y: -0.825},
    tt: {x: 0.2, y: -0.5}, tb: {x: -0.4, y: -0.8}, td: {x: -0.9, y: -0.8},
    jaw_opening: 0.4  // Medium opening
  }
};

function testExtremePositions() {
  const vowelSequence = [
    { name: '/i/ - high front', pos: VOWEL_POSITIONS['i'] },
    { name: '/a/ - low central', pos: VOWEL_POSITIONS['a'] },  
    { name: '/u/ - high back', pos: VOWEL_POSITIONS['u'] },
    { name: '/e/ - mid front', pos: VOWEL_POSITIONS['e'] },
    { name: '/o/ - mid back', pos: VOWEL_POSITIONS['o'] }
  ];
  
  let index = 0;
  console.log("Testing anatomical vowel positions with jaw movement...");
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
    
    // Extract articulator positions (exclude jaw_opening from articulationFeatures)
    const articulationFeatures = {};
    Object.keys(current.pos).forEach(key => {
      if (key !== 'jaw_opening') {
        articulationFeatures[key] = current.pos[key];
      }
    });
    
    // Set jaw opening directly
    smoothedFeatures.jaw_opening = current.pos.jaw_opening;
    
    updateFeatureHistory(articulationFeatures, 140 + Math.random() * 40, -20 + Math.random() * 10);
    updateCharts();
    
    index++;
  }, 2500);
}

function testArticulatorAnimation() {
  const speechPositions = [
    {
      name: '/i/ (see)',
      ul: { x: 0.9, y: -1.05 }, ll: { x: 0.9, y: -0.9 }, li: { x: 0.9, y: -0.95 },
      tt: { x: 0.6, y: -1.0 }, tb: { x: 0.2, y: -1.0 }, td: { x: -0.2, y: -0.8 },
      jaw_opening: 0.1
    },
    {
      name: '/a/ (father)',
      ul: { x: 0.9, y: -0.9 }, ll: { x: 0.9, y: -0.5 }, li: { x: 0.9, y: -0.7 },
      tt: { x: 0.2, y: -0.3 }, tb: { x: -0.2, y: -0.4 }, td: { x: -0.6, y: -0.3 },
      jaw_opening: 0.8
    },
    {
      name: '/u/ (boot)',
      ul: { x: 0.6, y: -1.0 }, ll: { x: 0.6, y: -0.85 }, li: { x: 0.6, y: -0.9 },
      tt: { x: -0.2, y: -0.8 }, tb: { x: -0.6, y: -0.9 }, td: { x: -1.0, y: -0.8 },
      jaw_opening: 0.15
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
    
    // Interpolate jaw opening
    const jawOpening = currentPos.jaw_opening + (nextPos.jaw_opening - currentPos.jaw_opening) * transitionProgress;
    smoothedFeatures.jaw_opening = jawOpening;
    
    updateFeatureHistory(features, 120 + Math.sin(frame/15)*80, -25 + Math.sin(frame/10)*25);
    updateCharts();
    
    if (frame % frameTransitions === 0) {
      updateStatus(`Demo: ${currentPos.name} → ${nextPos.name}`);
    }
    
    frame++;
    animationFrame = setTimeout(animateFrame, frameDuration / frameTransitions);
  }
  
  updateStatus("Demo: Showing full articulator range with jaw movement...");
  animateFrame();
}

function updateSourceFeatures(pitch, loudness) {
  const normalizedPitch = Math.min(100, Math.max(0, ((pitch - 75) / 225) * 100));
  const normalizedLoudness = Math.min(100, Math.max(0, ((loudness + 60) / 60) * 100));
  
  const pitchBar = document.getElementById('pitch-bar');
  const loudnessBar = document.getElementById('loudness-bar');
  
  if (pitchBar) pitchBar.style.height = normalizedPitch + '%';
  if (loudnessBar) loudnessBar.style.height = normalizedLoudness + '%';
}

// IMPROVED: Update charts with jaw movement
function updateCharts() {
  try {
    if (!featureHistory || Object.keys(featureHistory).length === 0) {
      debugLog("No feature history available for chart update");
      return;
    }
    
    const articulators = ['ul', 'll', 'li', 'tt', 'tb', 'td'];
    const latestFeatures = {};

    // Get latest positions
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

    // Get latest jaw opening
    const latestJawOpening = featureHistory.jaw_opening ? 
      featureHistory.jaw_opening[featureHistory.jaw_opening.length - 1] : 0.2;

    // NEW: Update jaw position
    const jaw = document.getElementById('jaw');
    if (jaw) {
      const jawPath = createJawPath(latestJawOpening);
      jaw.setAttribute('d', jawPath);
    }
    
    // NEW: Update lower teeth position
    const lowerTeeth = document.getElementById('lower-teeth');
    if (lowerTeeth) {
      const teethPos = positionLowerTeeth(latestJawOpening);
      lowerTeeth.setAttribute('x', teethPos.x);
      lowerTeeth.setAttribute('y', teethPos.y);
    }

    // Update markers (flip Y-axis to match SVG coordinates)
    for (const art of articulators) {
      const marker = document.getElementById(`${art}-marker`);
      if (marker && latestFeatures[art]) {
        marker.setAttribute('cx', latestFeatures[art].x);
        marker.setAttribute('cy', -latestFeatures[art].y);  // Flip Y: EMA→SVG
      }
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

    // NEW: Update jaw opening display
    if (typeof window.updateJawOpeningDisplay === 'function') {
      window.updateJawOpeningDisplay(latestJawOpening);
    }
    
  } catch (error) {
    debugLog("Error in updateCharts", error);
    debugCounters.errors++;
  }
}

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
    
    // Debug: Log first few samples periodically to verify audio is coming in
    if (debugCounters.audioDataReceived % 50 === 1) {
      const maxSample = Math.max(...Array.from(audioData).map(Math.abs));
      debugLog(`Audio chunk: ${audioData.length} samples, max amplitude: ${maxSample.toFixed(4)}`);
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
    
    const startButton = document.getElementById('startButton');
    const stopButton = document.getElementById('stopButton');
    
    if (startButton) startButton.disabled = true;
    if (stopButton) stopButton.disabled = false;
    
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
    
    const startButton = document.getElementById('startButton');
    const stopButton = document.getElementById('stopButton');
    
    if (startButton) startButton.disabled = false;
    if (stopButton) stopButton.disabled = true;
    
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