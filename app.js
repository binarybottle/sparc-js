/******************************************************************************
 * SPARC Feature Extraction - Web Client
 * 
 * This application provides real-time visualization of
 * speech articulatory coding features from microphone input.
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
let wavlmSession = null;
let linearModel = null;
let linearModelWorkingMemory = null;
let waveformHistory = Array(500).fill(0); // More points for smoother waveform
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

// Pitch history for smoothing
let pitchHistory = Array(5).fill(0);

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

function calculateLoudness(audioData) {
  // Calculate RMS loudness
  let sum = 0;
  for (let i = 0; i < audioData.length; i++) {
      sum += audioData[i] * audioData[i];
  }
  const rms = Math.sqrt(sum / audioData.length);
  const dbFS = 20 * Math.log10(rms);
  
  return dbFS;
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
function sanitizePoint(point) {
  if (!point || typeof point.x !== 'number' || typeof point.y !== 'number' || 
      isNaN(point.x) || isNaN(point.y)) {
      return { x: 0, y: 0 };
  }
  return {
      x: Math.min(Math.max(point.x, -1.8), 1.8),
      y: Math.min(Math.max(point.y, -1.8), 1.8)
  };
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
      // Load WavLM model using the ONNX Runtime
      updateStatus("Loading WavLM model...");
      wavlmSession = await initOnnxRuntime();
      
      // Load linear model for articulation feature extraction
      updateStatus("Loading linear projection model...");
      await initLinearModel();
      
      // Setup visualization
      setupCharts();
      
      // Enable UI
      document.getElementById('startButton').disabled = false;
      updateStatus("Models loaded. Ready to start.");
      
      // Add event listeners
      document.getElementById('startButton').addEventListener('click', startRecording);
      document.getElementById('stopButton').addEventListener('click', stopRecording);
      
      /*/ Add debug mode toggle handler
      if (document.getElementById('debug-mode')) {
          document.getElementById('debug-mode').addEventListener('change', function() {
              const debugMarkers = document.querySelectorAll('.debug-marker');
              debugMarkers.forEach(marker => {
                  marker.style.display = this.checked ? 'block' : 'none';
              });
          });
      } */
      document.getElementById('debug-mode').checked = true;
      const debugMarkers = document.querySelectorAll('.debug-marker');
      debugMarkers.forEach(marker => {
          marker.style.display = 'block';
      });

  } catch (error) {
      updateStatus("Error loading models: " + error.message);
      console.error("Model loading error:", error);
  }
}

// Initialize ONNX Runtime for SPARC
async function initOnnxRuntime() {
  try {
      // Use locally hosted WASM files
      if (window.ort && window.ort.env && window.ort.env.wasm) {
          ort.env.wasm.wasmPaths = {
              'ort-wasm.wasm': 'wasm/ort-wasm.wasm',
              'ort-wasm-simd.wasm': 'wasm/ort-wasm-simd.wasm',
              'ort-wasm-threaded.wasm': 'wasm/ort-wasm-threaded.wasm',
              'ort-wasm-simd-threaded.wasm': 'wasm/ort-wasm-simd-threaded.wasm'
          };
          
          console.log("Local WASM paths set for ONNX Runtime");
      }
      
      const options = {
          executionProviders: ['wasm'],
          graphOptimizationLevel: 'all'
      };
      
      const modelPath = 'models/wavlm_base_layer9_quantized.onnx';
      const wavlmSession = await ort.InferenceSession.create(modelPath, options);
      
      console.log("WavLM model loaded successfully");
      return wavlmSession;
  } catch (error) {
      console.error("Failed to initialize ONNX Runtime:", error);
      throw error;
  }
}

// Initialize the linear model
async function initLinearModel() {
  try {
      // Load the linear model weights and biases from a JSON file
      const response = await fetch('models/wavlm_linear_model.json');
      const modelData = await response.json();
      
      // Convert model data to typed arrays for better performance
      linearModel = {
          weights: modelData.weights.map(w => new Float32Array(w)),
          biases: new Float32Array(modelData.biases),
          inputDim: modelData.input_dim,
          outputDim: modelData.output_dim,
          metadata: modelData.metadata
      };
      
      // Pre-allocate working memory for calculations
      linearModelWorkingMemory = new Float32Array(linearModel.outputDim);
      
      console.log(`Linear model loaded successfully. Input dim: ${linearModel.inputDim}, Output dim: ${linearModel.outputDim}`);
      return true;
  } catch (error) {
      console.error("Failed to load linear model:", error);
      return false;
  }
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
  // Pharynx wall with better anatomical shape
  const pharynxWall = document.createElementNS('http://www.w3.org/2000/svg', 'path');
  pharynxWall.setAttribute('class', 'pharynx');
  pharynxWall.setAttribute('d', 'M-1.6,-0.1 C-1.6,-0.3 -1.5,-0.5 -1.4,-0.7 C-1.3,-0.9 -1.2,-1.1 -1.0,-1.3 C-0.8,-1.5 -0.6,-1.7 -0.4,-1.8 C-0.3,-1.9 -0.2,-1.8 -0.2,-1.7 L-0.3,-1.4 L-0.5,-1.1 L-0.7,-0.8 L-1.0,-0.5 L-1.3,-0.3 Z');
  svg.appendChild(pharynxWall);
  
  // Hard palate with more natural curve
  const palate = document.createElementNS('http://www.w3.org/2000/svg', 'path');
  palate.setAttribute('class', 'palate');
  palate.setAttribute('id', 'palate');
  palate.setAttribute('d', 'M1.0,-1.25 C0.7,-1.35 0.4,-1.3 0.0,-1.15 C-0.4,-0.95 -0.8,-0.7 -1.2,-0.4');
  svg.appendChild(palate);
  
  // Jaw outline with more natural shape
  const jaw = document.createElementNS('http://www.w3.org/2000/svg', 'path');
  jaw.setAttribute('class', 'jaw');
  jaw.setAttribute('id', 'jaw');
  jaw.setAttribute('d', 'M1.0,-0.1 C0.8,0.0 0.6,0.05 0.4,0.08 C0.2,0.1 0.0,0.11 -0.2,0.11 C-0.4,0.11 -0.6,0.09 -0.8,0.05 C-1.0,0.0 -1.2,-0.1 -1.4,-0.25');
  svg.appendChild(jaw);
  
  // Upper teeth with better shape
  const upperTeeth = document.createElementNS('http://www.w3.org/2000/svg', 'path');
  upperTeeth.setAttribute('class', 'teeth');
  upperTeeth.setAttribute('d', 'M0.95,-1.2 C0.97,-1.15 0.99,-1.1 0.98,-1.05 L0.98,-0.95 C0.98,-0.93 0.96,-0.9 0.93,-0.9 L0.85,-0.9 C0.83,-0.9 0.8,-0.93 0.8,-0.95 L0.8,-1.2 C0.8,-1.22 0.83,-1.25 0.85,-1.25 L0.9,-1.25 C0.92,-1.25 0.95,-1.22 0.95,-1.2 Z');
  svg.appendChild(upperTeeth);
  
  // Lower teeth with better shape
  const lowerTeeth = document.createElementNS('http://www.w3.org/2000/svg', 'path');
  lowerTeeth.setAttribute('class', 'teeth');
  lowerTeeth.setAttribute('d', 'M0.95,-0.15 C0.97,-0.1 0.99,-0.05 0.98,0.0 L0.98,0.1 C0.98,0.12 0.96,0.15 0.93,0.15 L0.85,0.15 C0.83,0.15 0.8,0.12 0.8,0.1 L0.8,-0.15 C0.8,-0.17 0.83,-0.2 0.85,-0.2 L0.9,-0.2 C0.92,-0.2 0.95,-0.17 0.95,-0.15 Z');
  svg.appendChild(lowerTeeth);
  
  // Labels for orientation
  addLabel(svg, "FRONT", 1.5, 0);
  addLabel(svg, "BACK", -1.5, 0);
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
  label.setAttribute('font-size', '0.15');
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

// Improved lip shape creation from feature points
function createLipPaths(ul, ll, li) {
  // Ensure coordinates are valid
  ul = sanitizePoint(ul);
  ll = sanitizePoint(ll);
  li = sanitizePoint(li);
  
  // Calculate lip aperture and derived values - reduce width
  const lipAperture = Math.max(0.03, Math.abs(ul.y - ll.y)); // Reduced from 0.05
  const lipWidth = 0.25 + (0.1 * Math.min(1, lipAperture * 2)); // Reduced from 0.35 and 0.15
  
  // Corner points for mouth
  const mouthCornerLeft = { 
    x: Math.min(ul.x - lipWidth, li.x - 0.03), // Reduced from 0.05
    y: (ul.y + ll.y) / 2 
  };
  
  const mouthCornerRight = { 
    x: Math.max(ul.x + 0.1, li.x + lipWidth), // Reduced from 0.15
    y: (ul.y + ll.y) / 2 
  };
  
  // Control points for lip curves - reduce bulge
  const ulc1 = {
    x: mouthCornerLeft.x + (ul.x - mouthCornerLeft.x) * 0.3,
    y: mouthCornerLeft.y - (mouthCornerLeft.y - ul.y) * 0.15 // Reduced from 0.2
  };
  
  const ulc2 = {
    x: ul.x - (ul.x - mouthCornerLeft.x) * 0.5,
    y: ul.y - 0.01 // Reduced from 0.02
  };
  
  const ulc3 = {
    x: ul.x + (mouthCornerRight.x - ul.x) * 0.5,
    y: ul.y - 0.01 // Reduced from 0.02
  };
  
  const ulc4 = {
    x: mouthCornerRight.x - (mouthCornerRight.x - ul.x) * 0.3,
    y: mouthCornerRight.y - (mouthCornerRight.y - ul.y) * 0.15 // Reduced from 0.2
  };
  
  // Control points for lower lip
  const llc1 = {
    x: mouthCornerLeft.x + (ll.x - mouthCornerLeft.x) * 0.3,
    y: mouthCornerLeft.y + (ll.y - mouthCornerLeft.y) * 0.15 // Reduced from 0.2
  };
  
  const llc2 = {
    x: ll.x - (ll.x - mouthCornerLeft.x) * 0.5,
    y: ll.y + 0.01 // Reduced from 0.02
  };
  
  const llc3 = {
    x: ll.x + (mouthCornerRight.x - ll.x) * 0.5,
    y: ll.y + 0.01 // Reduced from 0.02
  };
  
  const llc4 = {
    x: mouthCornerRight.x - (mouthCornerRight.x - ll.x) * 0.3,
    y: mouthCornerRight.y + (ll.y - mouthCornerRight.y) * 0.15 // Reduced from 0.2
  };
  
  // Generate the paths
  const upperLipPath = `
    M ${mouthCornerLeft.x},${mouthCornerLeft.y}
    C ${ulc1.x},${ulc1.y} ${ulc2.x},${ulc2.y} ${ul.x},${ul.y}
    C ${ulc3.x},${ulc3.y} ${ulc4.x},${ulc4.y} ${mouthCornerRight.x},${mouthCornerRight.y}
  `;
  
  const lowerLipPath = `
    M ${mouthCornerLeft.x},${mouthCornerLeft.y}
    C ${llc1.x},${llc1.y} ${llc2.x},${llc2.y} ${ll.x},${ll.y}
    C ${llc3.x},${llc3.y} ${llc4.x},${llc4.y} ${mouthCornerRight.x},${mouthCornerRight.y}
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
  
  function animateFrame() {
    if (!document.getElementById('tongue')) return; // Stop if not on page
    
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
    setTimeout(animateFrame, frameDuration / frameTransitions);
  }
  
  // Start animation
  animateFrame();
}

// Call to test without microphone
testArticulatorAnimation();

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
  }
}

function processAudioData(audioData) {
  // Add to circular buffer
  for (let i = 0; i < audioData.length; i++) {
      audioBuffer[audioBufferIndex] = audioData[i];
      audioBufferIndex = (audioBufferIndex + 1) % config.bufferSize;
  }
  
  // We don't need waveform visualization anymore
}

/******************************************************************************
* FEATURE EXTRACTION *
******************************************************************************/

// ----- WavLM & Linear Projection ----- 

// Process audio through WavLM model
async function extractWavLMFeatures(audioData, session) {
  try {
      // Ensure audioData is a Float32Array of the right length
      const inputLength = 16000; // 1 second at 16kHz
      
      // Create a properly sized array
      const inputData = new Float32Array(inputLength);
      
      // Copy available data (with zero-padding if needed)
      const copyLength = Math.min(audioData.length, inputLength);
      for (let i = 0; i < copyLength; i++) {
          inputData[i] = audioData[i];
      }
      
      // Create tensor with the shape the model expects
      const inputTensor = new ort.Tensor('float32', inputData, [1, inputLength]);
      
      // Run inference
      const feeds = {};
      feeds[session.inputNames[0]] = inputTensor;
      
      const outputData = await session.run(feeds);
      
      // Extract the output tensor
      let output = outputData[session.outputNames[0]];
      
      // Apply Butterworth filtering (10Hz cutoff as in the Python version)
      output = filterWavLMFeatures(output);
      
      return output;
  } catch (error) {
      console.error("Error in WavLM feature extraction:", error);
      throw error;
  }
}

// Extract articulation features from WavLM output
function extractArticulationFeatures(wavlmFeatures) {
  try {
      if (!linearModel) {
          console.error("Linear model not loaded");
          return null;
      }
      // Extract the output tensor data from ONNX model output
      const features = wavlmFeatures.data;
      const dims = wavlmFeatures.dims;

      // WavLM outputs [batch_size, sequence_length, hidden_size]
      // Take the middle frame as the current representation
      const batchSize = dims[0];  // Should be 1 for real-time
      const seqLength = dims[1];  // Number of frames
      const hiddenSize = dims[2]; // 768 for WavLM base

      // Take the middle frame for real-time feedback
      const middleFrameIdx = Math.floor(seqLength / 2);
      const startIdx = middleFrameIdx * hiddenSize;
      
      // Apply the linear model with optimized matrix multiplication
      // Compute y = Wx + b where W is weights, x is features, b is bias
      const output = linearModelWorkingMemory;
      
      // First set output to bias values
      output.set(linearModel.biases);
      
      // Then add the weighted input features
      for (let i = 0; i < linearModel.outputDim; i++) {
          const weights = linearModel.weights[i];
          for (let j = 0; j < hiddenSize; j++) {
              output[i] += weights[j] * features[startIdx + j];
          }
      }
      
      // Map the output to articulators
      const articulationFeatures = {
        ul: {x: output[0], y: output[1]},
        ll: {x: output[2], y: output[3]},
        li: {x: output[4], y: output[5]},
        tt: {x: output[6], y: output[7]},
        tb: {x: output[8], y: output[9]},
        td: {x: output[10], y: output[11]}
      };

      // Apply scaling and offsets to map to SVG coordinate space
      const scaleFactorX = 1.0;
      const scaleFactorY = 1.0;
      const offsetX = 0.0;      
      const offsetY = 0.0;      

      for (const key in articulationFeatures) {
        articulationFeatures[key].x = articulationFeatures[key].x * scaleFactorX + offsetX;
        articulationFeatures[key].y = articulationFeatures[key].y * scaleFactorY + offsetY;
      }

      return articulationFeatures;
  } catch (error) {
      console.error("Error in articulation feature extraction:", error);
      return null;
  }
}

// ----- Pitch Detection ----- 

// YIN pitch detection implementation (based on algorithm by de Cheveigné and Kawahara)
class YINPitchDetector {
  constructor(options = {}) {
      this.sampleRate = options.sampleRate || 16000;
      this.threshold = options.threshold || 0.15;
      this.minFrequency = options.minFrequency || 70;
      this.maxFrequency = options.maxFrequency || 400;
      
      // Pre-calculate some values based on our constraints
      this.minPeriod = Math.floor(this.sampleRate / this.maxFrequency);
      this.maxPeriod = Math.floor(this.sampleRate / this.minFrequency);
  }
  
  detect(audioBuffer) {
    // Ensure we have a Float32Array
    const buffer = audioBuffer instanceof Float32Array ? audioBuffer : new Float32Array(audioBuffer);
    
    // Use a reasonable window size for speech
    const bufferSize = Math.min(buffer.length, 2048);
    
    // Calculate difference function (step 1 & 2 of YIN algorithm)
    const yinBuffer = new Float32Array(bufferSize / 2);
    
    // Step 1: Autocorrelation method
    for (let tau = 0; tau < yinBuffer.length; tau++) {
        yinBuffer[tau] = 0;
        
        for (let j = 0; j < yinBuffer.length; j++) {
            const delta = buffer[j] - buffer[j + tau];
            yinBuffer[tau] += delta * delta;
        }
    }
    
    // Step 2: Cumulative mean normalized difference
    yinBuffer[0] = 1;
    let runningSum = 0;
    for (let tau = 1; tau < yinBuffer.length; tau++) {
        runningSum += yinBuffer[tau];
        if (runningSum === 0) {
            yinBuffer[tau] = 1;
        } else {
            yinBuffer[tau] *= tau / runningSum;
        }
    }
    
    // Step 3: Find the first minimum below the threshold
    let minTau = 0;
    let minVal = 1000; // Initialize with a value higher than possible values
    
    // Only look for minima between our min and max period
    for (let tau = this.minPeriod; tau <= this.maxPeriod && tau < yinBuffer.length; tau++) {
        if (yinBuffer[tau] < minVal) {
            minVal = yinBuffer[tau];
            minTau = tau;
        }
        
        if (minVal < this.threshold) {
            // Found a minimum below threshold
            
            // Step 4: Interpolate for better accuracy
            const exactTau = this.parabolicInterpolation(yinBuffer, minTau);
            
            // Convert period to frequency
            const frequency = this.sampleRate / exactTau;
            
            return frequency;
        }
    }
    
    // If no minimum found below threshold, check the lowest value
    if (minTau > 0) {
        // Convert to frequency
        const frequency = this.sampleRate / minTau;
        
        // Only return if confidence is reasonable
        if (minVal < 0.3) {
            return frequency;
        }
    }
    
    // No pitch detected
    return 0;
}

parabolicInterpolation(array, position) {
    // Handle edge cases
    if (position === 0 || position === array.length - 1) {
        return position;
    }
    
    // Quadratic interpolation using the point and its neighbors
    const x1 = position - 1;
    const x2 = position;
    const x3 = position + 1;
    
    const y1 = array[x1];
    const y2 = array[x2];
    const y3 = array[x3];
    
    // Fit a parabola: y = a*x^2 + b*x + c
    const a = (y3 + y1 - 2 * y2) / 2;
    const b = (y3 - y1) / 2;
    
    if (a === 0) {
        // If a is zero, the parabola is actually a line
        return position;
    }
    
    // The minimum of the parabola is at: x = -b / (2*a)
    const exactPosition = x2 - b / (2 * a);
    
    return exactPosition;
}
}

// Function to extract pitch using our YIN implementation
function extractPitch(audioData) {
try {
    // Create detector if it doesn't exist yet
    if (!window.yinDetector) {
        window.yinDetector = new YINPitchDetector({
            sampleRate: config.sampleRate,
            threshold: 0.15,  // YIN threshold (lower = stricter)
            minFrequency: 70, // Min frequency for speech
            maxFrequency: 400 // Max frequency for speech
        });
    }
    
    // Extract a smaller window from the audio buffer for efficiency
    const bufferSize = Math.min(audioData.length, 2048);
    const startIdx = Math.floor((audioData.length - bufferSize) / 2);
    const audioSlice = audioData.slice(startIdx, startIdx + bufferSize);
    
    // Detect pitch
    const pitch = window.yinDetector.detect(audioSlice);
    
    return pitch || 0;
} catch (error) {
    console.error("Pitch detection error:", error);
    return 0;
}
}

// Smoothed version with history for stable output
function extractPitchSmoothed(audioData) {
const rawPitch = extractPitch(audioData);

// Update history
pitchHistory.push(rawPitch);
pitchHistory.shift();

// Filter out zeros for median calculation
const nonZeroPitches = pitchHistory.filter(p => p > 0);

if (nonZeroPitches.length === 0) {
    return 0;
}

// Use median for smoothing (more robust than average)
const sortedPitches = [...nonZeroPitches].sort((a, b) => a - b);
const medianPitch = sortedPitches[Math.floor(sortedPitches.length / 2)];

return medianPitch;
}

// ----- Filtering ----- 

// Butterworth filter implementation with pre-computed coefficients for a 10Hz cutoff at 50Hz sample rate
class LowpassFilter {
constructor() {
    // Pre-computed coefficients for a 5th order Butterworth low-pass filter
    // with 10Hz cutoff at 50Hz sample rate
    this.b = [0.0008, 0.0039, 0.0078, 0.0078, 0.0039, 0.0008];
    this.a = [1.0000, -3.0756, 3.8289, -2.3954, 0.7475, -0.0930];
    
    // Initialize delay lines
    this.x_history = new Float32Array(this.b.length).fill(0);
    this.y_history = new Float32Array(this.a.length-1).fill(0);
}

// Process a single sample
processSample(x) {
    // Shift x history
    for (let i = this.x_history.length - 1; i > 0; i--) {
        this.x_history[i] = this.x_history[i-1];
    }
    this.x_history[0] = x;
    
    // Apply filter
    let y = 0;
    for (let i = 0; i < this.b.length; i++) {
        y += this.b[i] * this.x_history[i];
    }
    
    for (let i = 0; i < this.y_history.length; i++) {
        y -= this.a[i+1] * this.y_history[i];
    }
    
    // Shift y history
    for (let i = this.y_history.length - 1; i > 0; i--) {
        this.y_history[i] = this.y_history[i-1];
    }
    this.y_history[0] = y;
    
    return y;
}

// Process an array of samples
process(inputArray) {
    const outputArray = new Float32Array(inputArray.length);
    for (let i = 0; i < inputArray.length; i++) {
        outputArray[i] = this.processSample(inputArray[i]);
    }
    return outputArray;
}

// Reset the filter state
reset() {
    this.x_history.fill(0);
    this.y_history.fill(0);
}
}

// For filtering WavLM feature dimensions
function createFilterBank(numFilters) {
const filters = [];
for (let i = 0; i < numFilters; i++) {
    filters.push(new LowpassFilter());
}
return filters;
}

// Filter WavLM features in the time dimension
function filterWavLMFeatures(wavlmFeatures) {
const dims = wavlmFeatures.dims;
const data = wavlmFeatures.data;
const batchSize = dims[0];
const seqLength = dims[1];
const hiddenSize = dims[2];

// Only create filters when needed
if (!window.featuresFilterBank || window.featuresFilterBank.length !== hiddenSize) {
    window.featuresFilterBank = createFilterBank(hiddenSize);
}

// Apply filtering to each feature dimension across time
const filteredData = new Float32Array(data.length);

for (let h = 0; h < hiddenSize; h++) {
    // Extract this feature dimension across all time steps for the first batch item
    const featureTimeSeries = new Float32Array(seqLength);
    for (let t = 0; t < seqLength; t++) {
        const idx = t * hiddenSize + h; // Assuming batch size 1
        featureTimeSeries[t] = data[idx];
    }
    
    // Apply filter
    const filteredTimeSeries = window.featuresFilterBank[h].process(featureTimeSeries);
    
    // Put filtered data back
    for (let t = 0; t < seqLength; t++) {
        const idx = t * hiddenSize + h; // Assuming batch size 1
        filteredData[idx] = filteredTimeSeries[t];
    }
}

return new ort.Tensor('float32', filteredData, dims);
}

// ----- Main Extraction Loop ----- 

async function extractFeaturesLoop() {
if (!isRecording) return;

try {
    // Get the latest 1 second of audio
    const recentAudio = getRecentAudioBuffer();
    
    // Extract WavLM features using the new function
    const wavlmOutput = await extractWavLMFeatures(recentAudio, wavlmSession);
    
    // The wavlmOutput is already the Layer 9 features since your model is truncated
    // You can now apply your linear projection to get articulatory features
    const articulationFeatures = extractArticulationFeatures(wavlmOutput);
    
    if (!articulationFeatures) {
        console.error("Failed to extract articulation features");
        setTimeout(extractFeaturesLoop, config.updateInterval);
        return;
    }
    
    // Extract pitch based on configuration
    let pitch = 0;
    if (config.extractPitchFn === 1) {
        pitch = extractPitch(recentAudio);
    } else if (config.extractPitchFn === 2) {
        pitch = extractPitchSmoothed(recentAudio);
    }
    
    // Calculate loudness
    const loudness = calculateLoudness(recentAudio);
    
    // Update feature history
    updateFeatureHistory(articulationFeatures, pitch, loudness);
    
    // Update UI
    updateCharts();
    
} catch (error) {
    console.error("Feature extraction error:", error);
}

// Schedule next extraction
setTimeout(extractFeaturesLoop, config.updateInterval);
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
  const alpha = 0.3;

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
* 7. EVENT LISTENERS *
******************************************************************************/
document.addEventListener('DOMContentLoaded', function() {
init().catch(error => {
    console.error("Error during initialization:", error);
    updateStatus("Initialization error: " + error.message);
});
});