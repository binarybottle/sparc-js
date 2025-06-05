/******************************************************************************
 * SPARC Feature Extraction - Web Client (Main Application)
 * 
 * This file contains the main application logic for the SPARC system, handling:
 * - User interface and visualization
 * - Audio capture and buffering
 * - Animation and rendering
 * - Communication with the worker thread for ML processing
 * 
 * Part of the Speech Articulatory Coding (SPARC) system that provides 
 * real-time visualization of speech articulatory features from microphone input.
******************************************************************************/

/******************************************************************************
* CONFIGURATION & GLOBAL VARIABLES *
******************************************************************************/
const config = {
  sampleRate: 16000,
  frameSize: 512,
  // For approximately 20ms frames at 16kHz
  // 16000 samples/sec * 0.02 sec ≈ 320 samples
  // Nearest power of 2 is 512
  bufferSize: 16000,  // 1 second of audio at 16kHz
  updateInterval: 50, // Update features every 50ms (20 Hz)
  extractPitchFn: 2   // 1 for original, 2 for smoothed
};

// Global variables
let audioContext;
let audioStream;
let workletNode;
let waveformHistory = Array(500).fill(0); // More points for smoother waveform
let animationRunning = false;
let animationFrame = null;
let isRecording = false;
let audioBuffer = new Float32Array(config.bufferSize);
let audioBufferIndex = 0;
let smoothedFeatures = {
  ul_x: 0, ul_y: 0,
  ll_x: 0, ll_y: 0,
  li_x: 0, li_y: 0,
  tt_x: 0, tt_y: 0,
  tb_x: 0, tb_y: 0,
  td_x: 0, td_y: 0
};

// Feature history
let featureHistory = {
  ul_x: Array(100).fill(0),
  ul_y: Array(100).fill(0),
  ll_x: Array(100).fill(0),
  ll_y: Array(100).fill(0),
  li_x: Array(100).fill(0),
  li_y: Array(100).fill(0),
  tt_x: Array(100).fill(0),
  tt_y: Array(100).fill(0),
  tb_x: Array(100).fill(0),
  tb_y: Array(100).fill(0),
  td_x: Array(100).fill(0),
  td_y: Array(100).fill(0),
  pitch: Array(100).fill(0),
  loudness: Array(100).fill(0)
};

// Web Worker Management
let SparcWorker = null;
let workerInitialized = false;
let pendingWorkerResponses = 0;

/******************************************************************************
* CORE UTILITY FUNCTIONS *
******************************************************************************/
function updateStatus(message) {
  document.getElementById('status').textContent = "Status: " + message;
}

function initializeDefaultPositions() {
  // Default positions for a neutral expression
  const defaultPositions = {
    ul: { x: 0.9, y: -1.05 },  // Upper lip
    ll: { x: 0.9, y: -0.8 },   // Lower lip
    li: { x: 0.85, y: -0.92 }, // Lip interface
    tt: { x: 0.5, y: -0.7 },   // Tongue tip
    tb: { x: 0.0, y: -0.6 },   // Tongue body
    td: { x: -0.5, y: -0.5 }   // Tongue dorsum
  };
  
  // Set initial values in feature history
  for (let i = 0; i < featureHistory.ul_x.length; i++) {
    featureHistory.ul_x[i] = defaultPositions.ul.x;
    featureHistory.ul_y[i] = defaultPositions.ul.y;
    featureHistory.ll_x[i] = defaultPositions.ll.x;
    featureHistory.ll_y[i] = defaultPositions.ll.y;
    featureHistory.li_x[i] = defaultPositions.li.x;
    featureHistory.li_y[i] = defaultPositions.li.y;
    featureHistory.tt_x[i] = defaultPositions.tt.x;
    featureHistory.tt_y[i] = defaultPositions.tt.y;
    featureHistory.tb_x[i] = defaultPositions.tb.x;
    featureHistory.tb_y[i] = defaultPositions.tb.y;
    featureHistory.td_x[i] = defaultPositions.td.x;
    featureHistory.td_y[i] = defaultPositions.td.y;
  }
  
  // Update visualization with these positions
  updateCharts();
}

function getRecentAudioBuffer() {
  // Create a new buffer with the most recent audio data
  const recentAudio = new Float32Array(config.bufferSize);
  
  // Copy from circular buffer in the correct order
  for (let i = 0; i < config.bufferSize; i++) {
    const index = (audioBufferIndex + i) % config.bufferSize;
    recentAudio[i] = audioBuffer[index];
  }
  
  return recentAudio;
}

// Helper function to ensure points have valid coordinates
function sanitizePoint(point, prevPoint) {
  if (!point || typeof point.x !== 'number' || typeof point.y !== 'number' || 
      isNaN(point.x) || isNaN(point.y)) {
      return { x: 0, y: 0 };
  }
  
  // Basic range limiting
  let result = {
      x: Math.min(Math.max(point.x, -1.8), 1.8),
      y: Math.min(Math.max(point.y, -1.8), 1.8)
  };
  
  // If we have a previous point, limit the maximum change per frame
  if (prevPoint && isRecording) {
    const maxDelta = 0.05; // Maximum change per frame when recording
    
    // Limit the change in each coordinate
    if (Math.abs(result.x - prevPoint.x) > maxDelta) {
      result.x = prevPoint.x + Math.sign(result.x - prevPoint.x) * maxDelta;
    }
    
    if (Math.abs(result.y - prevPoint.y) > maxDelta) {
      result.y = prevPoint.y + Math.sign(result.y - prevPoint.y) * maxDelta;
    }
  }
  
  return result;
}

// Apply anatomical constraints to tongue points
function applyAnatomicalConstraints(tt, tb, td) {
  // Define palate curve as a series of points
  const palate = [
    { x: 1.0, y: -1.25 },   // Front of palate (near teeth)
    { x: 0.7, y: -1.35 },   // Front palate
    { x: 0.4, y: -1.3 },    // Mid-front palate
    { x: 0.0, y: -1.15 },   // Middle of palate
    { x: -0.4, y: -0.95 },  // Mid-back palate
    { x: -0.8, y: -0.7 },   // Back palate
    { x: -1.2, y: -0.4 }    // Back of throat
  ];
  
  // Helper function to find the y-coordinate on the palate for a given x
  function getPalateY(x) {
    // Find the palate segments that x falls between
    for (let i = 0; i < palate.length - 1; i++) {
      if (x <= palate[i].x && x >= palate[i + 1].x) {
        // Interpolate to find precise y value
        const ratio = (x - palate[i].x) / (palate[i + 1].x - palate[i].x);
        return palate[i].y + ratio * (palate[i + 1].y - palate[i].y);
      }
    }
    
    // Handle x values outside the defined palate range
    if (x > palate[0].x) return palate[0].y;
    if (x < palate[palate.length - 1].x) return palate[palate.length - 1].y;
    
    return -1.0; // Default fallback
  }

  // Apply constraints to each tongue point
  const minDistance = 0.08; // Minimum distance from palate
  
  // Check each tongue point against the palate
  const palateY_tt = getPalateY(tt.x);
  const palateY_tb = getPalateY(tb.x);
  const palateY_td = getPalateY(td.x);
  
  // Prevent tongue from going through the palate
  if (tt.y < palateY_tt + minDistance) {
    tt.y = palateY_tt + minDistance;
  }
  if (tb.y < palateY_tb + minDistance) {
    tb.y = palateY_tb + minDistance;
  }
  if (td.y < palateY_td + minDistance) {
    td.y = palateY_td + minDistance;
  }
  
  // Add constraints to maintain tongue shape integrity
  // Ensure tongue dorsum stays behind tongue body
  if (td.x > tb.x - 0.15) {
    td.x = tb.x - 0.15;
  }
  
  // Ensure tongue body stays behind tongue tip
  if (tb.x > tt.x - 0.15) {
    tb.x = tt.x - 0.15;
  }
  
  // Limit extreme positions
  tt.x = Math.max(-1.5, Math.min(tt.x, 1.2));
  tt.y = Math.max(-1.5, Math.min(tt.y, 0.5));
  tb.x = Math.max(-1.5, Math.min(tb.x, 1.0));
  tb.y = Math.max(-1.5, Math.min(tb.y, 0.5));
  td.x = Math.max(-1.5, Math.min(td.x, 0.5));
  td.y = Math.max(-1.5, Math.min(td.y, 0.5));
}

/******************************************************************************
* INITIALIZATION & SETUP *
******************************************************************************/
// Initialize application
async function init() {
  updateStatus("Loading models...");
  try {
    // Initialize worker
    await initSparcWorker();
    
    // Setup visualization
    setupCharts();
    
    // Enable UI
    document.getElementById('startButton').disabled = false;
    updateStatus("Models loaded. Ready to start.");
    
    // Add event listeners
    document.getElementById('startButton').addEventListener('click', startRecording);
    document.getElementById('stopButton').addEventListener('click', stopRecording);
    
    // Initialize debug mode
    document.getElementById('debug-mode').checked = true;
    const debugMarkers = document.querySelectorAll('.debug-marker');
    debugMarkers.forEach(marker => {
      marker.style.display = 'block';
    });

    // Start animation when not recording
    if (!isRecording) {
      testArticulatorAnimation();
    }

  } catch (error) {
    updateStatus("Error loading models: " + error.message);
    console.error("Model loading error:", error);
  }
}

// Initialize the ML worker
function initSparcWorker() {
  if (SparcWorker) return Promise.resolve();
  
  return new Promise((resolve, reject) => {
    try {
      console.log("Initializing ML worker...");
      SparcWorker = new Worker('sparc-worker.js');
      
      SparcWorker.onmessage = function(e) {
        const message = e.data;
        
        switch(message.type) {
          case 'initialized':
            console.log("ML worker initialized successfully");
            workerInitialized = true;
            resolve();
            break;
            
          case 'features':
            pendingWorkerResponses--;
            
            // Got features from the worker
            const { articulationFeatures, pitch, loudness } = message;
            
            // Update feature history
            updateFeatureHistory(articulationFeatures, pitch, loudness);
            
            // Update UI
            requestAnimationFrame(() => {
              updateCharts();
            });
            break;
            
          case 'status':
            console.log("Worker status:", message.message);
            updateStatus(message.message);
            break;
            
          case 'error':
            console.error("Worker error:", message.error);
            pendingWorkerResponses--;
            if (!workerInitialized) {
              reject(new Error(message.error));
            }
            break;
        }
      };
      
      // Initialize the worker
      SparcWorker.postMessage({
        type: 'init',
        onnxPath: 'models/wavlm_base_layer9_quantized.onnx',
        linearModelPath: 'models/wavlm_linear_model.json'
      });
      
    } catch (error) {
      console.error("Error creating worker:", error);
      reject(error);
    }
  });
}

// Setup the SVG vocal tract visualization
function setupVocalTractVisualization() {
  const svg = document.getElementById('vocal-tract-svg');
  
  // Clear any existing elements
  while (svg.firstChild) {
      svg.removeChild(svg.firstChild);
  }
  
  // Create static elements
  createStaticElements(svg);
  
  // Create dynamic elements with initial positions
  createDynamicElements(svg);
}

// Create static vocal tract elements
function createStaticElements(svg) {
  // Reduced size pharynx wall with better anatomical shape
  const pharynxWall = document.createElementNS('http://www.w3.org/2000/svg', 'path');
  pharynxWall.setAttribute('class', 'pharynx');
  pharynxWall.setAttribute('d', 'M-1.2,-0.2 C-1.2,-0.3 -1.15,-0.45 -1.05,-0.6 C-0.95,-0.75 -0.85,-0.9 -0.75,-1.0 C-0.6,-1.1 -0.45,-1.2 -0.3,-1.25 C-0.25,-1.3 -0.2,-1.25 -0.2,-1.2 L-0.25,-1.0 L-0.35,-0.8 L-0.5,-0.6 L-0.65,-0.4 L-0.85,-0.25 Z');
  svg.appendChild(pharynxWall);
  
  // Hard palate with more natural curve - reduced size
  const palate = document.createElementNS('http://www.w3.org/2000/svg', 'path');
  palate.setAttribute('class', 'palate');
  palate.setAttribute('id', 'palate');
  palate.setAttribute('d', 'M0.9,-0.9 C0.7,-1.0 0.4,-0.95 0.1,-0.85 C-0.25,-0.75 -0.5,-0.6 -0.75,-0.4');
  svg.appendChild(palate);
  
  // Jaw outline with more natural shape - reduced size
  const jaw = document.createElementNS('http://www.w3.org/2000/svg', 'path');
  jaw.setAttribute('class', 'jaw');
  jaw.setAttribute('id', 'jaw');
  jaw.setAttribute('d', 'M0.9,-0.1 C0.7,0.0 0.5,0.05 0.3,0.07 C0.1,0.08 -0.1,0.09 -0.3,0.07 C-0.5,0.05 -0.7,0.0 -0.85,-0.1 C-0.95,-0.15 -1.05,-0.2 -1.1,-0.25');
  svg.appendChild(jaw);
  
  // Upper teeth with subtle shape
  const upperTeeth = document.createElementNS('http://www.w3.org/2000/svg', 'path');
  upperTeeth.setAttribute('class', 'teeth');
  upperTeeth.setAttribute('d', 'M0.85,-0.8 L0.85,-0.7 L0.75,-0.7 L0.75,-0.8 Z');
  svg.appendChild(upperTeeth);
  
  // Lower teeth with subtle shape
  const lowerTeeth = document.createElementNS('http://www.w3.org/2000/svg', 'path');
  lowerTeeth.setAttribute('class', 'teeth');
  lowerTeeth.setAttribute('d', 'M0.85,-0.2 L0.85,-0.1 L0.75,-0.1 L0.75,-0.2 Z');
  svg.appendChild(lowerTeeth);
  
  // Labels for orientation - moved closer to border
  addLabel(svg, "FRONT", 0.75, 0.35);
  addLabel(svg, "BACK", -0.75, 0.35);
}

// Create dynamic elements
function createDynamicElements(svg) {
  // Upper lip - will be dynamically shaped
  const upperLip = document.createElementNS('http://www.w3.org/2000/svg', 'path');
  upperLip.setAttribute('class', 'lips');
  upperLip.setAttribute('id', 'upper-lip');
  svg.appendChild(upperLip);
  
  // Lower lip - will be dynamically shaped
  const lowerLip = document.createElementNS('http://www.w3.org/2000/svg', 'path');
  lowerLip.setAttribute('class', 'lips');
  lowerLip.setAttribute('id', 'lower-lip');
  svg.appendChild(lowerLip);
  
  // Tongue - will be shaped by the feature points
  const tongue = document.createElementNS('http://www.w3.org/2000/svg', 'path');
  tongue.setAttribute('class', 'tongue');
  tongue.setAttribute('id', 'tongue');
  svg.appendChild(tongue);
  
  // Debug markers for articulator positions (hidden by default)
  const articulators = ['ul', 'll', 'li', 'tt', 'tb', 'td'];
  const colors = ['#e74c3c', '#3498db', '#f1c40f', '#2ecc71', '#9b59b6', '#e67e22'];
  
  articulators.forEach((art, i) => {
      const marker = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      marker.setAttribute('id', `${art}-marker`);
      marker.setAttribute('r', '0.03'); // Reduced from 0.05
      marker.setAttribute('fill', colors[i]);
      marker.setAttribute('stroke', '#fff');
      marker.setAttribute('stroke-width', '0.005'); // Reduced from 0.01
      marker.setAttribute('class', 'debug-marker');
      marker.style.display = 'none'; // Hidden by default
      svg.appendChild(marker);
  });
}

// Helper function to add a label to the SVG
function addLabel(svg, text, x, y) {
  const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
  label.setAttribute('x', x);
  label.setAttribute('y', y);
  label.setAttribute('font-size', '0.12'); // Smaller font
  label.setAttribute('text-anchor', 'middle');
  label.setAttribute('fill', '#888');
  label.textContent = text;
  svg.appendChild(label);
}

// Helper function to create an SVG path from points
function createPathFromPoints(points) {
  if (!points || points.length === 0) return '';
  
  let path = `M ${points[0].x},${points[0].y}`;
  
  // If only one point, return just that point
  if (points.length === 1) return path;
  
  // Create a smooth curve through points
  for (let i = 1; i < points.length; i++) {
      path += ` L ${points[i].x},${points[i].y}`;
  }
  
  return path;
}

// Tongue path generation based directly on feature points
function createTonguePath(tt, tb, td) {
  // Start with the tongue root (relatively fixed point at the back)
  const tongueRoot = { x: -1.0, y: -0.2 };
  
  // Ensure all coordinates are valid
  tt = sanitizePoint(tt);
  tb = sanitizePoint(tb);
  td = sanitizePoint(td);
  
  // Apply anatomical constraints
  applyAnatomicalConstraints(tt, tb, td);
  
  // Reduce the size of the tongue by scaling control points closer to the main points
  // Calculate control points for smooth Bezier curves - use smaller multipliers
  const c1 = {
    x: tongueRoot.x + (td.x - tongueRoot.x) * 0.2,
    y: tongueRoot.y + (td.y - tongueRoot.y) * 0.2
  };
  
  const c2 = {
    x: td.x - (td.x - tongueRoot.x) * 0.2,
    y: td.y - (td.y - tongueRoot.y) * 0.2
  };
  
  const c3 = {
    x: td.x + (tb.x - td.x) * 0.3, 
    y: td.y + (tb.y - td.y) * 0.3
  };
  
  const c4 = {
    x: tb.x - (tb.x - td.x) * 0.3,
    y: tb.y - (tb.y - td.y) * 0.3
  };
  
  const c5 = {
    x: tb.x + (tt.x - tb.x) * 0.3,
    y: tb.y + (tt.y - tb.y) * 0.3
  };
  
  const c6 = {
    x: tt.x - (tt.x - tb.x) * 0.3,
    y: tt.y - (tt.y - tb.y) * 0.3
  };
  
  // Create the tongue tip end point - make it smaller
  const tipEnd = {
    x: tt.x + 0.08, // Reduced from 0.15
    y: tt.y + 0.03  // Reduced from 0.05
  };
  
  // Create underside control points for a natural shape - make smaller
  const underC1 = {
    x: tipEnd.x - 0.03, // Reduced from 0.05
    y: tipEnd.y + 0.08  // Reduced from 0.15
  };
  
  const underC2 = {
    x: tt.x - 0.06,    // Reduced from 0.1
    y: tt.y + 0.12     // Reduced from 0.18
  };
  
  const underC3 = {
    x: tb.x - 0.06,    // Reduced from 0.1
    y: tb.y + 0.15     // Reduced from 0.25
  };
  
  const underC4 = {
    x: td.x - 0.06,    // Reduced from 0.1
    y: td.y + 0.12     // Reduced from 0.2
  };
  
  const underC5 = {
    x: tongueRoot.x + 0.12, // Reduced from 0.2
    y: tongueRoot.y + 0.08  // Reduced from 0.15
  };
  
  // Generate the tongue path using Bezier curves
  return `
    M ${tongueRoot.x},${tongueRoot.y}
    C ${c1.x},${c1.y} ${c2.x},${c2.y} ${td.x},${td.y}
    C ${c3.x},${c3.y} ${c4.x},${c4.y} ${tb.x},${tb.y}
    C ${c5.x},${c5.y} ${c6.x},${c6.y} ${tt.x},${tt.y}
    Q ${tipEnd.x},${tt.y} ${tipEnd.x},${tipEnd.y}
    C ${underC1.x},${underC1.y} ${underC2.x},${underC2.y} ${tt.x - 0.03},${tt.y + 0.08}
    C ${underC3.x},${underC3.y} ${underC4.x},${underC4.y} ${td.x - 0.03},${td.y + 0.08}
    C ${underC5.x},${underC5.y} ${tongueRoot.x + 0.06},${tongueRoot.y + 0.06} ${tongueRoot.x},${tongueRoot.y}
    Z
  `;
}

// Lip shape creation from feature points
function createLipPaths(ul, ll, li) {
  // Ensure coordinates are valid
  ul = sanitizePoint(ul);
  ll = sanitizePoint(ll);
  li = sanitizePoint(li);
  
  // Enhanced lip profile - extend back to show more realistic shape
  // Corner points with greater horizontal extension
  const lipExtendBack = 0.25; // How far back the lips extend from li.x
  const backX = li.x - lipExtendBack;
  
  // Upper lip profile path
  const upperLipPath = `
    M ${backX},${ul.y - 0.05}
    Q ${li.x - lipExtendBack*0.7},${ul.y - 0.07} ${li.x - lipExtendBack*0.4},${ul.y - 0.05}
    Q ${li.x - lipExtendBack*0.2},${ul.y - 0.02} ${ul.x},${ul.y}
    Q ${ul.x + 0.05},${ul.y} ${ul.x + 0.1},${ul.y + 0.02}
    L ${ul.x + 0.1},${(ul.y + ll.y)/2 - 0.02}
    Q ${li.x - 0.05},${(ul.y + ll.y)/2 - 0.01} ${backX},${(ul.y + ll.y)/2}
    Z
  `;
  
  // Lower lip profile path
  const lowerLipPath = `
    M ${backX},${(ul.y + ll.y)/2}
    Q ${li.x - 0.05},${(ul.y + ll.y)/2 + 0.01} ${ll.x - 0.05},${(ul.y + ll.y)/2 + 0.02}
    L ${ll.x + 0.1},${ll.y - 0.02}
    Q ${ll.x + 0.05},${ll.y} ${ll.x},${ll.y}
    Q ${li.x - lipExtendBack*0.2},${ll.y + 0.02} ${li.x - lipExtendBack*0.4},${ll.y + 0.05}
    Q ${li.x - lipExtendBack*0.7},${ll.y + 0.07} ${backX},${ll.y + 0.05}
    Z
  `;
  
  return {
    upperLip: upperLipPath,
    lowerLip: lowerLipPath
  };
}

// Compact display for pitch and loudness
function updateSourceFeatures(pitch, loudness) {
  // Normalize pitch (typical speech range 75-300 Hz)
  const normalizedPitch = Math.min(100, Math.max(0, ((pitch - 75) / 225) * 100));
  
  // Normalize loudness (typical range -60 to 0 dB)
  const normalizedLoudness = Math.min(100, Math.max(0, ((loudness + 60) / 60) * 100));
  
  // Update bars
  document.getElementById('pitch-bar').style.height = normalizedPitch + '%';
  document.getElementById('loudness-bar').style.height = normalizedLoudness + '%';
}

// Set up charts
function setupCharts() {
  // Set up SVG visualization
  setupVocalTractVisualization();
  
  console.log("Tongue element exists:", !!document.getElementById('tongue'));
  console.log("Upper lip element exists:", !!document.getElementById('upper-lip'));
  console.log("Lower lip element exists:", !!document.getElementById('lower-lip'));
  console.log("Debug markers:", document.querySelectorAll('.debug-marker').length);

  // Initialize with default positions after SVG is set up
  initializeDefaultPositions();

  if (!isRecording) {
    testArticulatorAnimation();
  }
}

/******************************************************************************
* DEBUGGING *
******************************************************************************/
function addDebugPanel() {
  const debugPanel = document.createElement('div');
  debugPanel.style.position = 'fixed';
  debugPanel.style.right = '10px';
  debugPanel.style.top = '10px';
  debugPanel.style.backgroundColor = 'rgba(0,0,0,0.7)';
  debugPanel.style.color = 'white';
  debugPanel.style.padding = '10px';
  debugPanel.style.borderRadius = '5px';
  debugPanel.style.fontSize = '12px';
  debugPanel.style.fontFamily = 'monospace';
  debugPanel.style.zIndex = '1000';
  debugPanel.id = 'debug-panel';
  document.body.appendChild(debugPanel);
  
  setInterval(() => {
    if (!document.getElementById('debug-panel')) return;
    
    const articulators = ['ul', 'll', 'li', 'tt', 'tb', 'td'];
    let debugHTML = '<b>Articulator Positions:</b><br>';
    
    articulators.forEach(art => {
      const x = featureHistory[art + '_x'][featureHistory[art + '_x'].length - 1].toFixed(2);
      const y = featureHistory[art + '_y'][featureHistory[art + '_y'].length - 1].toFixed(2);
      debugHTML += `${art}: (${x}, ${y})<br>`;
    });
    
    document.getElementById('debug-panel').innerHTML = debugHTML;
  }, 100);
}

// Call after initialization
addDebugPanel();
 
function testArticulatorAnimation() {
  // Create sample animation path with more natural positions
  const vowelPositions = [
    // [ul_x, ul_y, ll_x, ll_y, li_x, li_y, tt_x, tt_y, tb_x, tb_y, td_x, td_y]
    // /a/ as in "father" - slightly smaller movements
    [0.9, -1.0, 0.9, -0.7, 0.9, -0.85, 0.4, -0.5, -0.1, -0.6, -0.5, -0.5],
    // /i/ as in "see" - higher tongue position
    [0.9, -1.05, 0.9, -0.9, 0.9, -0.95, 0.4, -0.9, 0.0, -0.9, -0.4, -0.6],
    // /u/ as in "boot" - more rounded lips, back tongue raised
    [0.7, -1.0, 0.7, -0.85, 0.7, -0.9, 0.3, -0.7, -0.1, -0.6, -0.4, -0.5],
    // /o/ as in "go" - mid-open rounded position
    [0.8, -1.0, 0.8, -0.8, 0.8, -0.9, 0.3, -0.6, -0.1, -0.6, -0.5, -0.5]
  ];
  
  let frame = 0;
  const frameDuration = 500; // ms per position
  const frameTransitions = 20; // Smooth steps between positions
  
  // Set the animation flag
  animationRunning = true;

  function animateFrame() {
    // Check if we should continue animating
    if (!document.getElementById('tongue') || isRecording || !animationRunning) {
      animationRunning = false;
      return; // Stop the animation if recording starts
    }

    const currentVowelIdx = Math.floor(frame / frameTransitions) % vowelPositions.length;
    const nextVowelIdx = (currentVowelIdx + 1) % vowelPositions.length;
    const transitionProgress = (frame % frameTransitions) / frameTransitions;
    
    const currentVowel = vowelPositions[currentVowelIdx];
    const nextVowel = vowelPositions[nextVowelIdx];
    
    // Interpolate between current and next position
    const interpolatedPosition = currentVowel.map((val, idx) => {
      return val + (nextVowel[idx] - val) * transitionProgress;
    });
    
    // Apply to feature history
    const features = {
      ul: {x: interpolatedPosition[0], y: interpolatedPosition[1]},
      ll: {x: interpolatedPosition[2], y: interpolatedPosition[3]},
      li: {x: interpolatedPosition[4], y: interpolatedPosition[5]},
      tt: {x: interpolatedPosition[6], y: interpolatedPosition[7]},
      tb: {x: interpolatedPosition[8], y: interpolatedPosition[9]},
      td: {x: interpolatedPosition[10], y: interpolatedPosition[11]}
    };
    
    // Update history and visualization
    updateFeatureHistory(features, 120 + Math.sin(frame/10)*60, -30 + Math.sin(frame/5)*20);
    updateChartsWithDebug(); // Use the new function that adds debug lines
    
    frame++;
    // Store the animation frame ID so we can cancel it
    animationFrame = setTimeout(animateFrame, frameDuration / frameTransitions);
  }
  
  // Start animation
  animateFrame();
}

function updateDebugLines(showLines) {
  const debugLines = document.querySelectorAll('.debug-line');
  
  // Remove existing lines
  debugLines.forEach(line => line.remove());
  
  if (showLines) {
    const svg = document.getElementById('vocal-tract-svg');
    const articulators = ['ul', 'll', 'li', 'tt', 'tb', 'td'];
    const markers = {};
    
    // Get all marker positions
    articulators.forEach(art => {
      const marker = document.getElementById(`${art}-marker`);
      if (marker) {
        markers[art] = {
          x: parseFloat(marker.getAttribute('cx')),
          y: parseFloat(marker.getAttribute('cy'))
        };
      }
    });
    
    // Create lines between tongue points
    if (markers.tt && markers.tb) {
      const ttTbLine = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      ttTbLine.setAttribute('class', 'debug-line');
      ttTbLine.setAttribute('x1', markers.tt.x);
      ttTbLine.setAttribute('y1', markers.tt.y);
      ttTbLine.setAttribute('x2', markers.tb.x);
      ttTbLine.setAttribute('y2', markers.tb.y);
      svg.appendChild(ttTbLine);
    }
    
    if (markers.tb && markers.td) {
      const tbTdLine = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      tbTdLine.setAttribute('class', 'debug-line');
      tbTdLine.setAttribute('x1', markers.tb.x);
      tbTdLine.setAttribute('y1', markers.tb.y);
      tbTdLine.setAttribute('x2', markers.td.x);
      tbTdLine.setAttribute('y2', markers.td.y);
      svg.appendChild(tbTdLine);
    }
    
    // Create lines between lip points
    if (markers.ul && markers.li) {
      const ulLiLine = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      ulLiLine.setAttribute('class', 'debug-line');
      ulLiLine.setAttribute('x1', markers.ul.x);
      ulLiLine.setAttribute('y1', markers.ul.y);
      ulLiLine.setAttribute('x2', markers.li.x);
      ulLiLine.setAttribute('y2', markers.li.y);
      svg.appendChild(ulLiLine);
    }
    
    if (markers.li && markers.ll) {
      const liLlLine = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      liLlLine.setAttribute('class', 'debug-line');
      liLlLine.setAttribute('x1', markers.li.x);
      liLlLine.setAttribute('y1', markers.li.y);
      liLlLine.setAttribute('x2', markers.ll.x);
      liLlLine.setAttribute('y2', markers.ll.y);
      svg.appendChild(liLlLine);
    }
  }
}

// Call this after debug markers are updated
function updateChartsWithDebug() {
  // Regular update
  updateCharts();
  
  // Add debug lines if debug mode is checked
  const debugMode = document.getElementById('debug-mode').checked;
  updateDebugLines(debugMode);
}

/******************************************************************************
* AUDIO RECORDING & PROCESSING *
******************************************************************************/
// AudioWorklet processor code as a string
const audioProcessorCode = `
class AudioProcessor extends AudioWorkletProcessor {
constructor() {
  super();
  this.bufferSize = 512;
  this.buffer = new Float32Array(this.bufferSize);
  this.bufferIndex = 0;
}
process(inputs, outputs, parameters) {
  // Get input data
  const input = inputs[0][0];
  
  if (input && input.length > 0) {
    // Send audio data to main thread
    this.port.postMessage({
      audio: input.slice()
    });
  }
  
  // Keep processor alive
  return true;
}
}
registerProcessor('audio-processor', AudioProcessor);
`;

// Start recording
async function startRecording() {
  try {
    // Stop any running test animation
    animationRunning = false;
    if (animationFrame) {
      clearTimeout(animationFrame);
      animationFrame = null;
    }
      
    // Request microphone access
    audioStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        sampleRate: config.sampleRate,
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true
      }
    });
      
    // Create audio context
    audioContext = new (window.AudioContext || window.webkitAudioContext)({
      sampleRate: config.sampleRate
    });
      
    // Check if AudioWorklet is supported
    if (audioContext.audioWorklet) {
      // Create a Blob URL for the processor code
      const blob = new Blob([audioProcessorCode], { type: 'application/javascript' });
      const processorUrl = URL.createObjectURL(blob);
          
      // Add the module to the audio worklet
      await audioContext.audioWorklet.addModule(processorUrl);
          
      // Create audio worklet node
      workletNode = new AudioWorkletNode(audioContext, 'audio-processor');
          
      // Handle messages from the audio processor
      workletNode.port.onmessage = (event) => {
        if (event.data.audio) {
          processAudioData(event.data.audio);
        }
      };
          
      // Connect nodes
      const source = audioContext.createMediaStreamSource(audioStream);
      source.connect(workletNode);
      workletNode.connect(audioContext.destination);
          
    } else {
      // Fallback to ScriptProcessorNode for older browsers
      console.warn("AudioWorklet not supported, falling back to ScriptProcessorNode");
      const source = audioContext.createMediaStreamSource(audioStream);
      const processor = audioContext.createScriptProcessor(config.frameSize, 1, 1);
      processor.onaudioprocess = (event) => {
        const input = event.inputBuffer.getChannelData(0);
        processAudioData(input);
      };
      source.connect(processor);
      processor.connect(audioContext.destination);
      workletNode = processor; // Store for cleanup
    }
      
    // Update UI
    isRecording = true;
    document.getElementById('startButton').disabled = true;
    document.getElementById('stopButton').disabled = false;
    updateStatus("Recording...");
      
    // Start feature extraction loop
    extractFeaturesLoop();
  } catch (error) {
    updateStatus("Error starting recording: " + error.message);
    console.error("Recording error:", error);
  }
}

function stopRecording() {
  if (audioStream) {
    // Stop all tracks in the stream
    audioStream.getTracks().forEach(track => track.stop());
      
    // Disconnect audio processor
    if (workletNode) {
      workletNode.disconnect();
      workletNode = null;
    }
      
    // Close audio context
    if (audioContext) {
      audioContext.close();
      audioContext = null;
    }
      
    // Update UI
    isRecording = false;
    document.getElementById('startButton').disabled = false;
    document.getElementById('stopButton').disabled = true;
    updateStatus("Recording stopped.");

    // After stopping recording, restart the animation if not already running
    if (!animationRunning) {
      testArticulatorAnimation();
    }
  }
}

function processAudioData(audioData) {
  // Add to circular buffer
  for (let i = 0; i < audioData.length; i++) {
    audioBuffer[audioBufferIndex] = audioData[i];
    audioBufferIndex = (audioBufferIndex + 1) % config.bufferSize;
  }
}

// Feature extraction loop using worker
async function extractFeaturesLoop() {
  if (!isRecording) return;
  
  try {
    // Schedule next iteration first
    setTimeout(extractFeaturesLoop, config.updateInterval);
    
    // If we have too many pending responses, skip this frame
    if (pendingWorkerResponses > 2) {
      console.warn("Skipping frame, too many pending responses:", pendingWorkerResponses);
      return;
    }
    
    // Get the latest audio
    const recentAudio = getRecentAudioBuffer();
    
    // Send to worker
    SparcWorker.postMessage({
      type: 'process',
      audio: recentAudio.buffer,
      config: config
    }, [recentAudio.buffer.slice().buffer]); // Clone and transfer buffer ownership
    
    pendingWorkerResponses++;
    
  } catch (error) {
    console.error("Feature extraction error:", error);
  }
}

/******************************************************************************
* UI UPDATES & VISUALIZATION *
******************************************************************************/
// Main chart update function
function updateCharts() {
  const articulators = ['ul', 'll', 'li', 'tt', 'tb', 'td'];
  const latestFeatures = {};

  // Get latest positions for all articulators
  for (let i = 0; i < articulators.length; i++) {
    const art = articulators[i];
    const latestX = featureHistory[art + '_x'][featureHistory[art + '_x'].length - 1];
    const latestY = featureHistory[art + '_y'][featureHistory[art + '_y'].length - 1];
    latestFeatures[art] = { x: latestX, y: latestY };
    
    // Update the marker position (for debugging)
    const marker = document.getElementById(`${art}-marker`);
    if (marker) {
      marker.setAttribute('cx', latestX);
      marker.setAttribute('cy', latestY);
    }
  }

  // Update tongue shape FIRST based on tongue articulator positions
  const tongue = document.getElementById('tongue');
  if (tongue) {
    tongue.setAttribute('d', createTonguePath(
      latestFeatures.tt,
      latestFeatures.tb,
      latestFeatures.td
    ));
  }

  // Update lip shapes using lip articulator positions
  const lipPaths = createLipPaths(
    latestFeatures.ul, 
    latestFeatures.ll,
    latestFeatures.li
  );

  const upperLip = document.getElementById('upper-lip');
  if (upperLip) {
    upperLip.setAttribute('d', lipPaths.upperLip);
  }

  const lowerLip = document.getElementById('lower-lip');
  if (lowerLip) {
    lowerLip.setAttribute('d', lipPaths.lowerLip);
  }

  // Update source features visualization (pitch and loudness)
  const latestPitch = featureHistory.pitch[featureHistory.pitch.length - 1];
  const latestLoudness = featureHistory.loudness[featureHistory.loudness.length - 1];
  updateSourceFeatures(latestPitch, latestLoudness);
}

function updateFeatureHistory(articulationFeatures, pitch, loudness) {
  // Smoothing factor (0-1, lower = smoother)
  // Adaptive smoothing - more smoothing when recording
  const alpha = isRecording ? 0.1 : 0.3; // Lower alpha = more smoothing

  // Initialize smoothed features if they're all zeros
  if (smoothedFeatures.ul_x === 0 && smoothedFeatures.ul_y === 0) {
    smoothedFeatures.ul_x = articulationFeatures.ul.x;
    smoothedFeatures.ul_y = articulationFeatures.ul.y;
    smoothedFeatures.ll_x = articulationFeatures.ll.x;
    smoothedFeatures.ll_y = articulationFeatures.ll.y;
    smoothedFeatures.li_x = articulationFeatures.li.x;
    smoothedFeatures.li_y = articulationFeatures.li.y;
    smoothedFeatures.tt_x = articulationFeatures.tt.x;
    smoothedFeatures.tt_y = articulationFeatures.tt.y;
    smoothedFeatures.tb_x = articulationFeatures.tb.x;
    smoothedFeatures.tb_y = articulationFeatures.tb.y;
    smoothedFeatures.td_x = articulationFeatures.td.x;
    smoothedFeatures.td_y = articulationFeatures.td.y;
  }

  // Smooth articulation features
  smoothedFeatures.ul_x = alpha * articulationFeatures.ul.x + (1 - alpha) * smoothedFeatures.ul_x;
  smoothedFeatures.ul_y = alpha * articulationFeatures.ul.y + (1 - alpha) * smoothedFeatures.ul_y;
  smoothedFeatures.ll_x = alpha * articulationFeatures.ll.x + (1 - alpha) * smoothedFeatures.ll_x;
  smoothedFeatures.ll_y = alpha * articulationFeatures.ll.y + (1 - alpha) * smoothedFeatures.ll_y;
  smoothedFeatures.li_x = alpha * articulationFeatures.li.x + (1 - alpha) * smoothedFeatures.li_x;
  smoothedFeatures.li_y = alpha * articulationFeatures.li.y + (1 - alpha) * smoothedFeatures.li_y;
  smoothedFeatures.tt_x = alpha * articulationFeatures.tt.x + (1 - alpha) * smoothedFeatures.tt_x;
  smoothedFeatures.tt_y = alpha * articulationFeatures.tt.y + (1 - alpha) * smoothedFeatures.tt_y;
  smoothedFeatures.tb_x = alpha * articulationFeatures.tb.x + (1 - alpha) * smoothedFeatures.tb_x;
  smoothedFeatures.tb_y = alpha * articulationFeatures.tb.y + (1 - alpha) * smoothedFeatures.tb_y;
  smoothedFeatures.td_x = alpha * articulationFeatures.td.x + (1 - alpha) * smoothedFeatures.td_x;
  smoothedFeatures.td_y = alpha * articulationFeatures.td.y + (1 - alpha) * smoothedFeatures.td_y;

  // Shift all arrays to make room for new values
  for (const key in featureHistory) {
    featureHistory[key].shift();
  }

  // Add new values with the correct arrays
  featureHistory.ul_x.push(smoothedFeatures.ul_x);
  featureHistory.ul_y.push(smoothedFeatures.ul_y);
  featureHistory.ll_x.push(smoothedFeatures.ll_x);
  featureHistory.ll_y.push(smoothedFeatures.ll_y);
  featureHistory.li_x.push(smoothedFeatures.li_x);
  featureHistory.li_y.push(smoothedFeatures.li_y);
  featureHistory.tt_x.push(smoothedFeatures.tt_x);
  featureHistory.tt_y.push(smoothedFeatures.tt_y);
  featureHistory.tb_x.push(smoothedFeatures.tb_x);
  featureHistory.tb_y.push(smoothedFeatures.tb_y);
  featureHistory.td_x.push(smoothedFeatures.td_x);
  featureHistory.td_y.push(smoothedFeatures.td_y);

  // Add source features
  featureHistory.pitch.push(pitch);
  featureHistory.loudness.push(loudness);
}

/******************************************************************************
* EVENT LISTENERS *
******************************************************************************/
document.addEventListener('DOMContentLoaded', function() {
  init().catch(error => {
    console.error("Error during initialization:", error);
    updateStatus("Initialization error: " + error.message);
  });
});