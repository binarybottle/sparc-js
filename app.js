/******************************************************************************
 * SPARC Feature Extraction - Web Client
 * 
 * This application provides real-time visualization of
 * speech articulatory coding features from microphone input.
******************************************************************************/

/******************************************************************************
* 1. CONFIGURATION & GLOBAL VARIABLES *
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
let isRecording = false;
let audioBuffer = new Float32Array(config.bufferSize);
let audioBufferIndex = 0;

// Chart variables
let xValuesChart;
let yValuesChart;
let xyPositionsChart;
  
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
  
// Articulator traces for the 2D plot
let articulatorTraces = {
    ul: Array(10).fill({x: 0, y: 0}),
    ll: Array(10).fill({x: 0, y: 0}),
    li: Array(10).fill({x: 0, y: 0}),
    tt: Array(10).fill({x: 0, y: 0}),
    tb: Array(10).fill({x: 0, y: 0}),
    td: Array(10).fill({x: 0, y: 0})
};
  
// Pitch history for smoothing
let pitchHistory = Array(5).fill(0);

/******************************************************************************
* 2. CORE UTILITY FUNCTIONS *
******************************************************************************/
function updateStatus(message) {
  document.getElementById('status').textContent = "Status: " + message;
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

// Fallback function that simulates articulation features
function simulateArticulationFeatures() {
  return {
    ul: {x: Math.sin(Date.now() * 0.001) * 0.5, y: Math.cos(Date.now() * 0.001) * 0.5},
    ll: {x: Math.sin(Date.now() * 0.001 + 1) * 0.5, y: Math.cos(Date.now() * 0.001 + 1) * 0.5},
    li: {x: Math.sin(Date.now() * 0.001 + 2) * 0.5, y: Math.cos(Date.now() * 0.001 + 2) * 0.5},
    tt: {x: Math.sin(Date.now() * 0.001 + 3) * 0.5, y: Math.cos(Date.now() * 0.001 + 3) * 0.5},
    tb: {x: Math.sin(Date.now() * 0.001 + 4) * 0.5, y: Math.cos(Date.now() * 0.001 + 4) * 0.5},
    td: {x: Math.sin(Date.now() * 0.001 + 5) * 0.5, y: Math.cos(Date.now() * 0.001 + 5) * 0.5}
  };
}

/******************************************************************************
* 3. INITIALIZATION & SETUP *
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
      
      // Setup audio chart
      setupCharts();
      
      // Enable UI
      document.getElementById('startButton').disabled = false;
      updateStatus("Models loaded. Ready to start.");
      
      // Add event listeners
      document.getElementById('startButton').addEventListener('click', startRecording);
      document.getElementById('stopButton').addEventListener('click', stopRecording);
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
    
    // Rest of your code remains the same
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

// Setup all charts
function setupCharts() {
  // Clean up any existing charts
  if (xValuesChart) xValuesChart.destroy();
  if (yValuesChart) yValuesChart.destroy();
  if (xyPositionsChart) xyPositionsChart.destroy();
  
  // 1. X Values Chart
  const xCtx = document.getElementById('xValuesChart').getContext('2d');
  xValuesChart = new Chart(xCtx, {
    type: 'line',
    data: {
      labels: Array(100).fill(''),
      datasets: [
        {
          label: 'Upper Lip X',
          data: featureHistory.ul_x,
          borderColor: 'rgb(255, 99, 132)',
          tension: 0.2
        },
        {
          label: 'Lower Lip X',
          data: featureHistory.ll_x,
          borderColor: 'rgb(54, 162, 235)',
          tension: 0.2
        },
        {
          label: 'Lower Incisor X',
          data: featureHistory.li_x,
          borderColor: 'rgb(255, 206, 86)',
          tension: 0.2
        },
        {
          label: 'Tongue Tip X',
          data: featureHistory.tt_x,
          borderColor: 'rgb(75, 192, 192)',
          tension: 0.2
        },
        {
          label: 'Tongue Blade X',
          data: featureHistory.tb_x,
          borderColor: 'rgb(153, 102, 255)',
          tension: 0.2
        },
        {
          label: 'Tongue Dorsum X',
          data: featureHistory.td_x,
          borderColor: 'rgb(255, 159, 64)',
          tension: 0.2
        }
      ]
    },
    options: {
      animation: { duration: 0 },
      scales: { y: { min: -2, max: 2 } },
      plugins: {
        title: { display: true, text: 'X Coordinates - Front/Back Movement' },
        legend: { position: 'right' }
      }
    }
  });
  
  // 2. Y Values Chart
  const yCtx = document.getElementById('yValuesChart').getContext('2d');
  yValuesChart = new Chart(yCtx, {
    type: 'line',
    data: {
      labels: Array(100).fill(''),
      datasets: [
        {
          label: 'Upper Lip Y',
          data: featureHistory.ul_y,
          borderColor: 'rgb(255, 99, 132)',
          tension: 0.2
        },
        {
          label: 'Lower Lip Y',
          data: featureHistory.ll_y,
          borderColor: 'rgb(54, 162, 235)',
          tension: 0.2
        },
        {
          label: 'Lower Incisor Y',
          data: featureHistory.li_y,
          borderColor: 'rgb(255, 206, 86)',
          tension: 0.2
        },
        {
          label: 'Tongue Tip Y',
          data: featureHistory.tt_y,
          borderColor: 'rgb(75, 192, 192)',
          tension: 0.2
        },
        {
          label: 'Tongue Blade Y',
          data: featureHistory.tb_y,
          borderColor: 'rgb(153, 102, 255)',
          tension: 0.2
        },
        {
          label: 'Tongue Dorsum Y',
          data: featureHistory.td_y,
          borderColor: 'rgb(255, 159, 64)',
          tension: 0.2
        }
      ]
    },
    options: {
      animation: { duration: 0 },
      scales: { y: { min: -2, max: 2 } },
      plugins: {
        title: { display: true, text: 'Y Coordinates - Up/Down Movement' },
        legend: { position: 'right' }
      }
    }
  });
  
  // Initialize articulator traces
  const articulatorTraces = {
    ul: Array(10).fill({x: 0, y: 0}),
    ll: Array(10).fill({x: 0, y: 0}),
    li: Array(10).fill({x: 0, y: 0}),
    tt: Array(10).fill({x: 0, y: 0}),
    tb: Array(10).fill({x: 0, y: 0}),
    td: Array(10).fill({x: 0, y: 0})
  };
  
  // 3. 2D Positions Chart (Scatter plot)
  const xyCtx = document.getElementById('xyPositionsChart').getContext('2d');
  xyPositionsChart = new Chart(xyCtx, {
    type: 'scatter',
    data: {
      datasets: [
        // Current positions (large points)
        {
          label: 'Upper Lip',
          data: [{ x: 0, y: 0 }],
          backgroundColor: 'rgb(255, 99, 132)',
          pointRadius: 8
        },
        {
          label: 'Lower Lip',
          data: [{ x: 0, y: 0 }],
          backgroundColor: 'rgb(54, 162, 235)',
          pointRadius: 8
        },
        {
          label: 'Lower Incisor',
          data: [{ x: 0, y: 0 }],
          backgroundColor: 'rgb(255, 206, 86)',
          pointRadius: 8
        },
        {
          label: 'Tongue Tip',
          data: [{ x: 0, y: 0 }],
          backgroundColor: 'rgb(75, 192, 192)',
          pointRadius: 8
        },
        {
          label: 'Tongue Blade',
          data: [{ x: 0, y: 0 }],
          backgroundColor: 'rgb(153, 102, 255)',
          pointRadius: 8
        },
        {
          label: 'Tongue Dorsum',
          data: [{ x: 0, y: 0 }],
          backgroundColor: 'rgb(255, 159, 64)',
          pointRadius: 8
        },
        // Traces for each articulator
        {
          label: 'UL Trace',
          data: articulatorTraces.ul,
          backgroundColor: 'rgba(255, 99, 132, 0.3)',
          pointRadius: 2,
          showLine: true,
          fill: false,
          borderWidth: 1,
          borderColor: 'rgba(255, 99, 132, 0.5)',
          pointHoverRadius: 0
        },
        {
          label: 'LL Trace',
          data: articulatorTraces.ll,
          backgroundColor: 'rgba(54, 162, 235, 0.3)',
          pointRadius: 2,
          showLine: true,
          fill: false,
          borderWidth: 1,
          borderColor: 'rgba(54, 162, 235, 0.5)',
          pointHoverRadius: 0
        },
        {
          label: 'LI Trace',
          data: articulatorTraces.li,
          backgroundColor: 'rgba(255, 206, 86, 0.3)',
          pointRadius: 2,
          showLine: true,
          fill: false,
          borderWidth: 1,
          borderColor: 'rgba(255, 206, 86, 0.5)',
          pointHoverRadius: 0
        },
        {
          label: 'TT Trace',
          data: articulatorTraces.tt,
          backgroundColor: 'rgba(75, 192, 192, 0.3)',
          pointRadius: 2,
          showLine: true,
          fill: false,
          borderWidth: 1,
          borderColor: 'rgba(75, 192, 192, 0.5)',
          pointHoverRadius: 0
        },
        {
          label: 'TB Trace',
          data: articulatorTraces.tb,
          backgroundColor: 'rgba(153, 102, 255, 0.3)',
          pointRadius: 2,
          showLine: true,
          fill: false,
          borderWidth: 1,
          borderColor: 'rgba(153, 102, 255, 0.5)',
          pointHoverRadius: 0
        },
        {
          label: 'TD Trace',
          data: articulatorTraces.td,
          backgroundColor: 'rgba(255, 159, 64, 0.3)',
          pointRadius: 2,
          showLine: true,
          fill: false,
          borderWidth: 1,
          borderColor: 'rgba(255, 159, 64, 0.5)',
          pointHoverRadius: 0
        },
        // Landmarks
        {
          label: 'Landmarks',
          data: [
            { x: 0, y: 0, label: 'Origin' },
            { x: 0.5, y: -1.2, label: 'Alveolar Ridge' },
            { x: -0.8, y: -0.5, label: 'Velum' }
          ],
          backgroundColor: 'rgba(200, 200, 200, 0.5)',
          pointStyle: 'triangle',
          pointRadius: 5,
          pointHoverRadius: 7
        }
      ]
    },
    options: {
      animation: { duration: 0 },
      scales: {
        x: { 
          min: -2, max: 2, 
          title: { display: true, text: 'Front/Back Position' } 
        },
        y: { 
          min: -2, max: 2, 
          title: { display: true, text: 'Up/Down Position' } 
        }
      },
      plugins: {
        title: { display: true, text: 'Articulator Positions' },
        legend: { 
          position: 'right',
          labels: {
            filter: function(legendItem, chartData) {
              // Only show main articulators and landmarks in legend
              return !legendItem.text.includes('Trace');
            }
          }
        },
        tooltip: {
          callbacks: {
            label: function(context) {
              // Check if this is a landmark point with a label
              const point = context.raw;
              if (point && point.label) {
                return point.label;
              }
              // Default tooltip behavior
              return context.dataset.label + ': (' + 
                context.parsed.x.toFixed(2) + ', ' + 
                context.parsed.y.toFixed(2) + ')';
            }
          }
        }
      }
    }
  });
  
  // Store the articulator traces in a global variable for updates
  window.articulatorTraces = articulatorTraces;
}

/******************************************************************************
* 4. AUDIO RECORDING & PROCESSING *
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
}

/******************************************************************************
* 5. FEATURE EXTRACTION *
******************************************************************************/

// 5.1 WavLM & Linear Projection

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
      return simulateArticulationFeatures(); // Fallback to simulated features
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
    
    return articulationFeatures;
  } catch (error) {
    console.error("Error in articulation feature extraction:", error);
    return simulateArticulationFeatures(); // Fallback to simulated features
  }
}

// 5.2 Pitch Detection

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

// 5.3 Filtering

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

// 5.4 Main Extraction Loop

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
    updateFeatureUI(articulationFeatures, pitch, loudness);
    updateCharts();
    
  } catch (error) {
    console.error("Feature extraction error:", error);
  }
  
  // Schedule next extraction
  setTimeout(extractFeaturesLoop, config.updateInterval);
}

/******************************************************************************
* 6. UI UPDATES & VISUALIZATION *
******************************************************************************/
function updateCharts() {
  // Update X chart
  xValuesChart.data.datasets[0].data = featureHistory.ul_x;
  xValuesChart.data.datasets[1].data = featureHistory.ll_x;
  xValuesChart.data.datasets[2].data = featureHistory.li_x;
  xValuesChart.data.datasets[3].data = featureHistory.tt_x;
  xValuesChart.data.datasets[4].data = featureHistory.tb_x;
  xValuesChart.data.datasets[5].data = featureHistory.td_x;
  xValuesChart.update();
  
  // Update Y chart
  yValuesChart.data.datasets[0].data = featureHistory.ul_y;
  yValuesChart.data.datasets[1].data = featureHistory.ll_y;
  yValuesChart.data.datasets[2].data = featureHistory.li_y;
  yValuesChart.data.datasets[3].data = featureHistory.tt_y;
  yValuesChart.data.datasets[4].data = featureHistory.tb_y;
  yValuesChart.data.datasets[5].data = featureHistory.td_y;
  yValuesChart.update();
  
  // Update XY positions and traces
  const articulators = ['ul', 'll', 'li', 'tt', 'tb', 'td'];
  
  for (let i = 0; i < articulators.length; i++) {
    const art = articulators[i];
    const latestX = featureHistory[art + '_x'][featureHistory[art + '_x'].length - 1];
    const latestY = featureHistory[art + '_y'][featureHistory[art + '_y'].length - 1];
    const latestPos = { x: latestX, y: latestY };
    
    // Update current position (big dot)
    xyPositionsChart.data.datasets[i].data = [latestPos];
    
    // Update trace
    window.articulatorTraces[art].push({...latestPos});
    window.articulatorTraces[art].shift();
    xyPositionsChart.data.datasets[i + 6].data = [...window.articulatorTraces[art]];
  }
  
  xyPositionsChart.update();
}

function updateFeatureHistory(articulationFeatures, pitch, loudness) {
  // Shift all arrays to make room for new values
  for (const key in featureHistory) {
    featureHistory[key].shift();
  }
  
  // Add new values - now with both X and Y
  featureHistory.ul_x.push(articulationFeatures.ul.x);
  featureHistory.ul_y.push(articulationFeatures.ul.y);
  
  featureHistory.ll_x.push(articulationFeatures.ll.x);
  featureHistory.ll_y.push(articulationFeatures.ll.y);
  
  featureHistory.li_x.push(articulationFeatures.li.x);
  featureHistory.li_y.push(articulationFeatures.li.y);
  
  featureHistory.tt_x.push(articulationFeatures.tt.x);
  featureHistory.tt_y.push(articulationFeatures.tt.y);
  
  featureHistory.tb_x.push(articulationFeatures.tb.x);
  featureHistory.tb_y.push(articulationFeatures.tb.y);
  
  featureHistory.td_x.push(articulationFeatures.td.x);
  featureHistory.td_y.push(articulationFeatures.td.y);
  
  featureHistory.pitch.push(pitch);
  featureHistory.loudness.push(loudness);
}

function updateFeatureUI(articulationFeatures, pitch, loudness) {
  // Update vocal tract feature displays - now with both X and Y
  document.getElementById('ul-x-value').textContent = articulationFeatures.ul.x.toFixed(3);
  document.getElementById('ul-y-value').textContent = articulationFeatures.ul.y.toFixed(3);
  
  document.getElementById('ll-x-value').textContent = articulationFeatures.ll.x.toFixed(3);
  document.getElementById('ll-y-value').textContent = articulationFeatures.ll.y.toFixed(3);
  
  document.getElementById('li-x-value').textContent = articulationFeatures.li.x.toFixed(3);
  document.getElementById('li-y-value').textContent = articulationFeatures.li.y.toFixed(3);
  
  document.getElementById('tt-x-value').textContent = articulationFeatures.tt.x.toFixed(3);
  document.getElementById('tt-y-value').textContent = articulationFeatures.tt.y.toFixed(3);
  
  document.getElementById('tb-x-value').textContent = articulationFeatures.tb.x.toFixed(3);
  document.getElementById('tb-y-value').textContent = articulationFeatures.tb.y.toFixed(3);
  
  document.getElementById('td-x-value').textContent = articulationFeatures.td.x.toFixed(3);
  document.getElementById('td-y-value').textContent = articulationFeatures.td.y.toFixed(3);
  
  // Update source feature displays
  document.getElementById('pitch-value').textContent = pitch.toFixed(1);
  document.getElementById('loudness-value').textContent = loudness.toFixed(1);
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