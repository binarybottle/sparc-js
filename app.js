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

// DISPLAY BOUNDS - markers must stay well within viewBox edges
const DISPLAY_MIN = -2.0;
const DISPLAY_MAX = 2.0;

// Clamp function to ensure all values stay in bounds
function clampToDisplay(value) {
  return Math.max(DISPLAY_MIN, Math.min(DISPLAY_MAX, value));
}

// Scale and clamp: maps model output (-4 to +4) to display (-2.0 to +2.0)
function scaleToDisplay(value) {
  // Model range is roughly -4 to +4, display is -2.0 to +2.0
  const scaled = value * 0.5;  // Simple scaling factor (2.0/4 = 0.5)
  return clampToDisplay(scaled);
}

// IMPROVED: Smoothed features including JAW - initialized within bounds
let smoothedFeatures = {
  ul_x: 0.5, ul_y: -0.3,
  ll_x: 0.5, ll_y: 0.2,
  li_x: 0.3, li_y: 0.4,
  tt_x: 0.2, tt_y: 0.0,
  tb_x: -0.3, tb_y: -0.2,
  td_x: -0.8, td_y: -0.1,
  jaw_opening: 0.3
};

// Sensitivity factor for EMA values (1.0 = raw values from model)
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
  
  // Expanded Y range to accommodate full vocal tract (jaw at Y≈1.5)
  return {
      x: Math.min(Math.max(point.x, -2), 2),
      y: Math.min(Math.max(point.y, -2), 2.5)
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

// Calculate jaw opening based on lip positions
// NOTE: In EMA coords, ul_y and ll_y are in original space (not yet flipped)
function calculateJawOpening(ul_y, ll_y) {
  // The lips have negative Y values in EMA space
  // Lower lip is more negative (more down) than upper lip
  const lipDistance = Math.abs(ll_y - ul_y);
  
  // Recalibrated for ±2 coordinate system:
  // - Small opening (closed): ~0.05-0.5 units → jaw_opening ≈ 0-0.2
  // - Medium opening: ~0.5-1.0 units → jaw_opening ≈ 0.2-0.6  
  // - Large opening (/a/): ~1.0-1.5+ units → jaw_opening ≈ 0.6-1.0
  const jawOpening = Math.min(Math.max((lipDistance - 0.3) / 1.2, 0), 1);
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
    
    // Update articulator positions with AGGRESSIVE clamping
    for (const art of articulators) {
      if (articulationFeatures[art]) {
        let newX = articulationFeatures[art].x;
        let newY = articulationFeatures[art].y;
        
        if (isNaN(newX) || isNaN(newY) || !isFinite(newX) || !isFinite(newY)) {
          debugLog(`Invalid coordinates for ${art}: (${newX}, ${newY})`);
          continue;
        }
        
        // SCALE model output to display range
        newX = scaleToDisplay(newX);
        newY = scaleToDisplay(newY);
        
        const oldX = smoothedFeatures[art + '_x'] || 0;
        const oldY = smoothedFeatures[art + '_y'] || 0;
        
        // Apply smoothing
        let smoothedX = alpha * newX + (1 - alpha) * oldX;
        let smoothedY = alpha * newY + (1 - alpha) * oldY;
        
        // AGGRESSIVE final clamp to absolutely ensure within bounds
        smoothedFeatures[art + '_x'] = Math.max(-2.0, Math.min(2.0, smoothedX));
        smoothedFeatures[art + '_y'] = Math.max(-2.0, Math.min(2.0, smoothedY));
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
  // Neutral rest position - values in MODEL RANGE, will be scaled
  const defaultPositions = {
    ul: { x: 1.5, y: -0.5 },    // Upper lip - front, up
    ll: { x: 1.5, y: 0.5 },     // Lower lip - front, slightly down
    li: { x: 0.8, y: 1.0 },     // Lower incisor - behind lips
    tt: { x: 0.5, y: 0.0 },     // Tongue tip - middle
    tb: { x: -1.0, y: -0.5 },   // Tongue body - back of center
    td: { x: -2.0, y: -0.3 }    // Tongue dorsum - further back
  };
  
  // Apply scaling to default positions
  Object.keys(defaultPositions).forEach(art => {
    smoothedFeatures[art + '_x'] = scaleToDisplay(defaultPositions[art].x);
    smoothedFeatures[art + '_y'] = scaleToDisplay(defaultPositions[art].y);
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
          // IMPORTANT: Use SCALED values from smoothedFeatures, not raw defaultPositions
          featureHistory[xKey][i] = smoothedFeatures[xKey];
          featureHistory[yKey][i] = smoothedFeatures[yKey];
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
  
  // SIMPLIFIED ViewBox: X: -2 to +2 (4 units), Y: -1 to +2 (3 units)
  // Origin at (0,0) = upper teeth level
  svg.setAttribute('viewBox', '-2 -1 4 3');
  svg.setAttribute('width', '600');
  svg.setAttribute('height', '450');
  
  while (svg.firstChild) {
    svg.removeChild(svg.firstChild);
  }
  
  createReferenceGrid(svg);
  createSimpleStaticElements(svg);
  createSimpleDynamicElements(svg);
  
  debugLog("SVG visualization setup complete (SIMPLIFIED)");
}

// Reference grid showing the coordinate system (viewBox: -3.2 to +3.2)
function createReferenceGrid(svg) {
  const grid = document.createElementNS('http://www.w3.org/2000/svg', 'g');
  grid.setAttribute('id', 'reference-grid');
  grid.setAttribute('opacity', '0.15');
  
  // Horizontal lines
  for (let y = -3; y <= 3; y += 1) {
    const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    line.setAttribute('x1', '-3.2');
    line.setAttribute('y1', y);
    line.setAttribute('x2', '3.2');
    line.setAttribute('y2', y);
    line.setAttribute('stroke', '#999');
    line.setAttribute('stroke-width', '0.02');
    if (y === 0) line.setAttribute('stroke', '#666');
    grid.appendChild(line);
  }
  
  // Vertical lines
  for (let x = -3; x <= 3; x += 1) {
    const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    line.setAttribute('x1', x);
    line.setAttribute('y1', '-3.2');
    line.setAttribute('x2', x);
    line.setAttribute('y2', '3.2');
    line.setAttribute('stroke', '#999');
    line.setAttribute('stroke-width', '0.02');
    if (x === 0) line.setAttribute('stroke', '#666');
    grid.appendChild(line);
  }
  
  svg.appendChild(grid);
  
  // Axis labels
  addLabel(svg, "FRONT", 2.5, 0.2);
  addLabel(svg, "BACK", -2.8, 0.2);
  addLabel(svg, "UP", 0.1, -2.8);
  addLabel(svg, "DOWN", 0.1, 2.9);
}

function createSimpleStaticElements(svg) {
  // NO static anatomical elements - we don't know where they should be
  // in the standardized coordinate space. Just show raw model output.
  // The grid provides coordinate reference.
  
  // Add a legend for articulator colors
  const legend = document.createElementNS('http://www.w3.org/2000/svg', 'g');
  legend.setAttribute('id', 'legend');
  
  const legendItems = [
    { color: '#e74c3c', label: 'UL (upper lip)' },
    { color: '#3498db', label: 'LL (lower lip)' },
    { color: '#f1c40f', label: 'LI (lower incisor)' },
    { color: '#2ecc71', label: 'TT (tongue tip)' },
    { color: '#9b59b6', label: 'TB (tongue body)' },
    { color: '#e67e22', label: 'TD (tongue dorsum)' }
  ];
  
  legendItems.forEach((item, i) => {
    const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    circle.setAttribute('cx', -2.9);
    circle.setAttribute('cy', -2.8 + i * 0.3);
    circle.setAttribute('r', '0.06');
    circle.setAttribute('fill', item.color);
    legend.appendChild(circle);
    
    const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    text.setAttribute('x', -2.75);
    text.setAttribute('y', -2.75 + i * 0.3);
    text.setAttribute('font-size', '0.15');
    text.setAttribute('fill', '#333');
    text.textContent = item.label;
    legend.appendChild(text);
  });
  
  svg.appendChild(legend);
}

function createSimpleDynamicElements(svg) {
  // SIMPLIFIED: Just create 6 colored circles for the articulator positions
  // Note: Model outputs standardized values, NOT anatomical positions
  const articulators = [
    { id: 'ul', color: '#e74c3c', label: 'UL' },  // Upper lip - red
    { id: 'll', color: '#3498db', label: 'LL' },  // Lower lip - blue
    { id: 'li', color: '#f1c40f', label: 'LI' },  // Lower incisor - yellow
    { id: 'tt', color: '#2ecc71', label: 'TT' },  // Tongue tip - green
    { id: 'tb', color: '#9b59b6', label: 'TB' },  // Tongue body - purple
    { id: 'td', color: '#e67e22', label: 'TD' }   // Tongue dorsum - orange
  ];
  
  articulators.forEach(art => {
    // Circle marker - small enough to distinguish positions
    const marker = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    marker.setAttribute('id', `${art.id}-marker`);
    marker.setAttribute('r', '0.08');  // Small markers
    marker.setAttribute('fill', art.color);
    marker.setAttribute('stroke', '#fff');
    marker.setAttribute('stroke-width', '0.02');
    marker.setAttribute('class', 'articulator-marker');
    svg.appendChild(marker);
    
    // Label
    const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    label.setAttribute('id', `${art.id}-label`);
    label.setAttribute('class', 'articulator-label');
    label.setAttribute('fill', '#333');
    label.setAttribute('font-size', '0.18');
    label.setAttribute('font-weight', 'bold');
    label.setAttribute('text-anchor', 'middle');
    label.textContent = art.label;
    svg.appendChild(label);
  });
}

// Create dynamic jaw path based on opening
// Jaw at bottom of mouth, drops down when opening
function createJawPath(jawOpening) {
  // Base jaw position (positive Y = bottom of screen in SVG)
  // Lower lip typically around Y=1.3, so jaw starts at Y=1.5
  const baseY = 1.5;
  // Lower jaw drops MORE as mouth opens (more positive Y = further down)
  const jawY = baseY + (jawOpening * 0.3);
  
  return `M-1.2,${jawY} Q-0.8,${jawY + 0.05} -0.3,${jawY} Q0.2,${jawY} 0.6,${jawY + 0.05} Q0.9,${jawY} 1.0,${jawY + 0.1}`;
}

// Position lower teeth with jaw
function positionLowerTeeth(jawOpening) {
  // Position teeth just above jaw line
  const baseY = 1.38;
  // Teeth move down with jaw opening  
  const teethY = baseY + (jawOpening * 0.3);
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

// Create smooth, realistic tongue surface from EMA coordinates
// EMA: X+ = forward (lips), X- = back (pharynx); Y+ = up (palate), Y- = down
// SVG: X+ = right, X- = left; Y+ = down, Y- = up
// Mapping: EMA_X → SVG_X (same), EMA_Y → SVG_(-Y) (flip)
function createTonguePath(tt, tb, td) {
  // Flip Y-axis: EMA Y+ (up) becomes SVG Y- (up in screen)
  // Features come from model in ±2 range, matching our viewBox
  tt = sanitizePoint({ x: tt.x, y: -tt.y });   // tongue tip
  tb = sanitizePoint({ x: tb.x, y: -tb.y });   // tongue body
  td = sanitizePoint({ x: td.x, y: -td.y });   // tongue dorsum
  
  // Tongue root - anchored at pharyngeal wall
  const root = { x: -1.2, y: 0.15 };
  
  // Create smooth tongue surface using cubic Bézier curves
  // Top surface: root → dorsum → body → tip (smooth arc)
  const topCurve = `
    M ${root.x},${root.y}
    C ${root.x + 0.05},${root.y - 0.15} ${td.x - 0.15},${td.y - 0.05} ${td.x},${td.y}
    C ${td.x + 0.15},${td.y - 0.08} ${tb.x - 0.12},${tb.y - 0.05} ${tb.x},${tb.y}
    C ${tb.x + 0.18},${tb.y - 0.06} ${tt.x - 0.08},${tt.y - 0.03} ${tt.x},${tt.y}
  `;
  
  // Bottom surface: tip → body → dorsum → root (flatter, thicker base)
  const bottomCurve = `
    L ${tt.x},${tt.y + 0.15}
    C ${tb.x + 0.05},${tb.y + 0.13} ${td.x + 0.05},${td.y + 0.11} ${td.x - 0.05},${td.y + 0.08}
    C ${td.x - 0.12},${td.y + 0.06} ${root.x + 0.08},${root.y + 0.03} ${root.x},${root.y}
    Z
  `;
  
  return topCurve + bottomCurve;
}

// Create natural, expressive lip shapes from EMA coordinates
// Lips can spread (smile), round (protrude), and open (jaw)
// EMA: Y+ = up, Y- = down; SVG: Y+ = down, Y- = up
// Mapping: EMA_Y → SVG_(-Y)
function createLipPaths(ul, ll, li) {
  // Flip Y-axis: EMA Y+ (up) becomes SVG Y- (up in screen)
  // Features come from model in ±2 range, matching our viewBox
  ul = sanitizePoint({ x: ul.x, y: -ul.y });    // upper lip
  ll = sanitizePoint({ x: ll.x, y: -ll.y });    // lower lip
  li = sanitizePoint({ x: li.x, y: -li.y });    // lip interface
  
  // Calculate lip protrusion (rounding) - forward movement
  const protrusion = Math.max(0, (ul.x - 0.85) * 0.5);
  
  // Calculate lip spreading - horizontal width
  const baseWidth = 0.08;
  const spreading = Math.max(0, 0.9 - ul.x) * 0.3; // Lips spread when pulled back
  const lipWidth = baseWidth + spreading;
  
  // Lip positions
  const leftX = ul.x - lipWidth;
  const rightX = ul.x + lipWidth;
  const centerX = ul.x;
  
  // UPPER LIP - curves down and can protrude forward
  const upperLipPath = `
    M ${leftX},${ul.y}
    C ${leftX + lipWidth * 0.3},${ul.y - 0.025 - protrusion * 0.08} 
      ${rightX - lipWidth * 0.3},${ul.y - 0.025 - protrusion * 0.08} 
      ${rightX},${ul.y}
    C ${rightX - lipWidth * 0.2},${ul.y + 0.02} 
      ${leftX + lipWidth * 0.2},${ul.y + 0.02} 
      ${leftX},${ul.y}
    Z
  `;
  
  // LOWER LIP - curves up to meet upper lip
  const lowerLipPath = `
    M ${leftX},${ll.y}
    C ${leftX + lipWidth * 0.3},${ll.y + 0.025 + protrusion * 0.08} 
      ${rightX - lipWidth * 0.3},${ll.y + 0.025 + protrusion * 0.08} 
      ${rightX},${ll.y}
    C ${rightX - lipWidth * 0.2},${ll.y - 0.02} 
      ${leftX + lipWidth * 0.2},${ll.y - 0.02} 
      ${leftX},${ll.y}
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
  
  const testCenterButton = document.getElementById('test-center');
  if (testCenterButton) {
    testCenterButton.addEventListener('click', function() {
      console.log("🎯 Testing: Moving all markers to center (0,0)");
      const articulators = ['ul', 'll', 'li', 'tt', 'tb', 'td'];
      for (const art of articulators) {
        smoothedFeatures[art + '_x'] = 0;
        smoothedFeatures[art + '_y'] = 0;
        if (featureHistory[art + '_x']) {
          featureHistory[art + '_x'][featureHistory[art + '_x'].length - 1] = 0;
        }
        if (featureHistory[art + '_y']) {
          featureHistory[art + '_y'][featureHistory[art + '_y'].length - 1] = 0;
        }
      }
      updateCharts();
      console.log("✓ All markers should now be at (0,0)");
    });
  }
  
  const soundSelector = document.getElementById('sound-selector');
  if (soundSelector) {
    soundSelector.addEventListener('change', function() {
      if (isRecording) {
        alert("Stop recording first to test sounds");
        soundSelector.value = '';
        return;
      }
      
      const vowel = soundSelector.value;
      if (vowel && VOWEL_POSITIONS[vowel]) {
        const pos = VOWEL_POSITIONS[vowel];
        console.log(`Testing vowel /${vowel}/`, pos);
        updateStatus(`Demo: /${vowel}/`);
        
        // SET POSITIONS DIRECTLY - no smoothing for dropdown selection
        const articulators = ['ul', 'll', 'li', 'tt', 'tb', 'td'];
        
        for (const art of articulators) {
          if (pos[art]) {
            // Scale values to display range (vowel presets use model-range values)
            const scaledX = scaleToDisplay(pos[art].x);
            const scaledY = scaleToDisplay(pos[art].y);
            
            // Set directly to smoothedFeatures (no smoothing!)
            smoothedFeatures[art + '_x'] = scaledX;
            smoothedFeatures[art + '_y'] = scaledY;
            // Also update feature history directly
            if (featureHistory[art + '_x']) {
              featureHistory[art + '_x'][featureHistory[art + '_x'].length - 1] = scaledX;
            }
            if (featureHistory[art + '_y']) {
              featureHistory[art + '_y'][featureHistory[art + '_y'].length - 1] = scaledY;
            }
          }
        }
        
        // Set jaw opening directly
        smoothedFeatures.jaw_opening = pos.jaw_opening;
        
        updateCharts();
      } else if (!vowel) {
        // Reset selected - go back to neutral
        initializeDefaultPositions();
        updateCharts();
        updateStatus("Ready to start.");
      }
    });
  }
}

// VOWEL POSITIONS - Values in MODEL RANGE (-4 to +4), scaled by 0.7 to fit display
// MNGU0: X+ = forward (lips), X- = back (throat)
//        Y+ = down (jaw), Y- = up (palate)
const VOWEL_POSITIONS = {
  'i': { // /i/ "ee" - High front vowel: tongue HIGH & FRONT, jaw CLOSED
    ul: {x: 2.5, y: -1.5},    // Upper lip - forward, up
    ll: {x: 2.5, y: -0.8},    // Lower lip - close to UL (closed)
    li: {x: 1.8, y: -0.3},    // Lower incisor
    tt: {x: 3.2, y: -3.2},    // Tongue tip - HIGH and FRONT
    tb: {x: 1.2, y: -3.5},    // Tongue body - VERY HIGH
    td: {x: -0.8, y: -2.2},   // Tongue dorsum
    jaw_opening: 0.2
  },
  'e': { // /e/ "eh" - Mid front vowel: tongue mid-front, jaw medium
    ul: {x: 2.2, y: -1.2},    // Upper lip
    ll: {x: 2.2, y: 0.5},     // Lower lip - moderate gap
    li: {x: 1.3, y: 1.0},     // Lower incisor
    tt: {x: 2.8, y: -1.2},    // Tongue tip - mid height, front
    tb: {x: 0.5, y: -1.8},    // Tongue body
    td: {x: -1.2, y: -0.8},   // Tongue dorsum
    jaw_opening: 0.6
  },
  'a': { // /a/ "ah" - Low vowel: tongue LOW, jaw WIDE OPEN
    ul: {x: 1.8, y: -0.8},    // Upper lip - stays up
    ll: {x: 1.8, y: 3.2},     // Lower lip - FAR DOWN (wide open!)
    li: {x: 0.8, y: 3.5},     // Lower incisor - way down
    tt: {x: 1.2, y: 2.2},     // Tongue tip - LOW
    tb: {x: -0.5, y: 1.8},    // Tongue body - LOW
    td: {x: -2.2, y: 0.8},    // Tongue dorsum - low and back
    jaw_opening: 1.5
  },
  'o': { // /o/ "oh" - Mid back vowel: tongue BACK, lips rounded
    ul: {x: 0.8, y: -1.0},    // Upper lip - protruded/rounded
    ll: {x: 0.8, y: 1.2},     // Lower lip - moderate opening
    li: {x: 0.0, y: 1.8},     // Lower incisor
    tt: {x: -0.5, y: 0.5},    // Tongue tip - RETRACTED
    tb: {x: -2.2, y: -0.5},   // Tongue body - BACK
    td: {x: -3.2, y: -1.2},   // Tongue dorsum - FAR BACK & raised
    jaw_opening: 0.8
  },
  'u': { // /u/ "oo" - High back vowel: tongue HIGH & FAR BACK, jaw closed
    ul: {x: 0.0, y: -1.2},    // Upper lip - protruded back
    ll: {x: 0.0, y: -0.5},    // Lower lip - close (closed jaw)
    li: {x: -0.5, y: 0.0},    // Lower incisor
    tt: {x: -1.2, y: -0.8},   // Tongue tip - RETRACTED
    tb: {x: -2.8, y: -2.8},   // Tongue body - HIGH & FAR BACK
    td: {x: -3.5, y: -3.5},   // Tongue dorsum - HIGHEST & FARTHEST BACK
    jaw_opening: 0.2
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
  }, 5000); // 5 seconds per vowel - slow enough to observe
}

function testArticulatorAnimation() {
  const speechPositions = [
    {
      name: '/i/ (see)',
      ul: { x: 0.9, y: -1.05 }, ll: { x: 0.9, y: -0.9 }, li: { x: 0.9, y: -0.975 },
      tt: { x: 0.6, y: -1.0 }, tb: { x: 0.2, y: -1.0 }, td: { x: -0.2, y: -0.8 },
      jaw_opening: 0.1
    },
    {
      name: '/a/ (father)',
      ul: { x: 0.9, y: -0.9 }, ll: { x: 0.9, y: -0.4 }, li: { x: 0.9, y: -0.65 },
      tt: { x: 0.2, y: -0.3 }, tb: { x: -0.2, y: -0.35 }, td: { x: -0.6, y: -0.3 },
      jaw_opening: 0.8
    },
    {
      name: '/u/ (boot)',
      ul: { x: 0.6, y: -1.0 }, ll: { x: 0.6, y: -0.85 }, li: { x: 0.6, y: -0.925 },
      tt: { x: -0.2, y: -0.7 }, tb: { x: -0.6, y: -0.9 }, td: { x: -1.0, y: -0.8 },
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
    
    // SIMPLIFIED: Just update the 6 marker positions
    const articulators = ['ul', 'll', 'li', 'tt', 'tb', 'td'];
    const latestFeatures = {};

    // Get latest positions from feature history
    for (const art of articulators) {
      const xKey = art + '_x';
      const yKey = art + '_y';
      
      if (featureHistory[xKey] && featureHistory[yKey]) {
        latestFeatures[art] = {
          x: featureHistory[xKey][featureHistory[xKey].length - 1],
          y: featureHistory[yKey][featureHistory[yKey].length - 1]
        };
      } else {
        latestFeatures[art] = { x: 0, y: 0 };
      }
    }

    // Update each marker's position (just set cx, cy directly)
    for (const art of articulators) {
      const marker = document.getElementById(`${art}-marker`);
      const label = document.getElementById(`${art}-label`);
      
      if (marker && latestFeatures[art]) {
        // AGGRESSIVE clamping - absolutely force within -2.0 to +2.0
        let rawX = latestFeatures[art].x || 0;
        let rawY = latestFeatures[art].y || 0;
        
        const svgX = Math.max(-2.0, Math.min(2.0, rawX));
        const svgY = Math.max(-2.0, Math.min(2.0, rawY));
        
        // Debug log if values were clamped
        if (Math.abs(rawX) > 2.0 || Math.abs(rawY) > 2.0) {
          console.warn(`${art} clamped: (${rawX.toFixed(2)}, ${rawY.toFixed(2)}) → (${svgX.toFixed(2)}, ${svgY.toFixed(2)})`);
        }
        
        marker.setAttribute('cx', svgX);
        marker.setAttribute('cy', svgY);
        
        // Position label slightly offset
        if (label) {
          label.setAttribute('x', svgX + 0.12);
          label.setAttribute('y', svgY - 0.12);
        }
      }
    }
    
    // Log positions every 50 updates
    if (debugCounters.chartsUpdated % 50 === 0) {
      console.log("📍 Current marker positions:", 
        Object.fromEntries(articulators.map(art => 
          [art, latestFeatures[art] ? 
            `(${latestFeatures[art].x.toFixed(2)}, ${latestFeatures[art].y.toFixed(2)})` : 
            'undefined']
        ))
      );
    }

    // Update pitch/loudness bars
    if (featureHistory.pitch && featureHistory.loudness) {
      const latestPitch = featureHistory.pitch[featureHistory.pitch.length - 1];
      const latestLoudness = featureHistory.loudness[featureHistory.loudness.length - 1];
      updateSourceFeatures(latestPitch, latestLoudness);
    }
    
    // Log current positions for debugging
    if (debugCounters.featuresUpdated % 50 === 0) {
      console.log("Current positions:", {
        ul: `(${latestFeatures.ul.x.toFixed(2)}, ${latestFeatures.ul.y.toFixed(2)})`,
        ll: `(${latestFeatures.ll.x.toFixed(2)}, ${latestFeatures.ll.y.toFixed(2)})`,
        tt: `(${latestFeatures.tt.x.toFixed(2)}, ${latestFeatures.tt.y.toFixed(2)})`,
        tb: `(${latestFeatures.tb.x.toFixed(2)}, ${latestFeatures.tb.y.toFixed(2)})`,
        td: `(${latestFeatures.td.x.toFixed(2)}, ${latestFeatures.td.y.toFixed(2)})`
      });
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