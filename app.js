// Configuration
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
let isRecording = false;
let audioBuffer = new Float32Array(config.bufferSize);
let audioBufferIndex = 0;
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

// Initialize application
async function init() {
    updateStatus("Loading models...");
    try {
        // Load WavLM model using the ONNX Runtime
        updateStatus("Loading WavLM model...");
        wavlmSession = await initOnnxRuntime();
        
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

// Start application when the page loads
window.addEventListener('DOMContentLoaded', init);

// Chart for visualization
let articulatorsChart;

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

function updateStatus(message) {
    document.getElementById('status').textContent = "Status: " + message;
}

function setupCharts() {
    const ctx = document.getElementById('articulatorsChart').getContext('2d');
    
    articulatorsChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: Array(100).fill(''),
            datasets: [
                // Upper Lip
                {
                    label: 'Upper Lip X',
                    data: featureHistory.ul_x,
                    borderColor: 'rgb(255, 99, 132)',
                    tension: 0.2,
                    borderDash: [5, 5] // Dashed line for X values
                },
                {
                    label: 'Upper Lip Y',
                    data: featureHistory.ul_y,
                    borderColor: 'rgb(255, 99, 132)',
                    tension: 0.2
                },
                
                // Lower Lip
                {
                    label: 'Lower Lip X',
                    data: featureHistory.ll_x,
                    borderColor: 'rgb(54, 162, 235)',
                    tension: 0.2,
                    borderDash: [5, 5]
                },
                {
                    label: 'Lower Lip Y',
                    data: featureHistory.ll_y,
                    borderColor: 'rgb(54, 162, 235)',
                    tension: 0.2
                },
                
                // Lower Incisor
                {
                    label: 'Lower Incisor X',
                    data: featureHistory.li_x,
                    borderColor: 'rgb(255, 206, 86)',
                    tension: 0.2,
                    borderDash: [5, 5]
                },
                {
                    label: 'Lower Incisor Y',
                    data: featureHistory.li_y,
                    borderColor: 'rgb(255, 206, 86)',
                    tension: 0.2
                },
                
                // Tongue Tip
                {
                    label: 'Tongue Tip X',
                    data: featureHistory.tt_x,
                    borderColor: 'rgb(75, 192, 192)',
                    tension: 0.2,
                    borderDash: [5, 5]
                },
                {
                    label: 'Tongue Tip Y',
                    data: featureHistory.tt_y,
                    borderColor: 'rgb(75, 192, 192)',
                    tension: 0.2
                },
                
                // Tongue Blade
                {
                    label: 'Tongue Blade X',
                    data: featureHistory.tb_x,
                    borderColor: 'rgb(153, 102, 255)',
                    tension: 0.2,
                    borderDash: [5, 5]
                },
                {
                    label: 'Tongue Blade Y',
                    data: featureHistory.tb_y,
                    borderColor: 'rgb(153, 102, 255)',
                    tension: 0.2
                },
                
                // Tongue Dorsum
                {
                    label: 'Tongue Dorsum X',
                    data: featureHistory.td_x,
                    borderColor: 'rgb(255, 159, 64)',
                    tension: 0.2,
                    borderDash: [5, 5]
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
            animation: {
                duration: 0
            },
            scales: {
                y: {
                    min: -2,
                    max: 2
                }
            },
            plugins: {
                legend: {
                    position: 'right',
                    labels: {
                        boxWidth: 15,
                        font: {
                            size: 10
                        }
                    }
                }
            }
        }
    });
}

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

// YIN pitch detection implementation
// Based on the algorithm by Alain de Cheveigné and Hideki Kawahara
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
let pitchHistory = Array(5).fill(0);
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

// Main feature extraction loop
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
      const output = outputData[session.outputNames[0]];
      
      // Return output features - shape [1, 49, 768]
      return output;
    } catch (error) {
      console.error("Error in WavLM feature extraction:", error);
      throw error;
    }
}

function extractArticulationFeatures(wavlmFeatures) {
    // In a real implementation, this would be a linear projection from WavLM features
    // to the 12 EMA channels (6 articulators, each with x and y coordinates)
    
    // Here we're simulating the output
    const articulationFeatures = {
        ul: {x: Math.sin(Date.now() * 0.001) * 0.5, y: Math.cos(Date.now() * 0.001) * 0.5},
        ll: {x: Math.sin(Date.now() * 0.001 + 1) * 0.5, y: Math.cos(Date.now() * 0.001 + 1) * 0.5},
        li: {x: Math.sin(Date.now() * 0.001 + 2) * 0.5, y: Math.cos(Date.now() * 0.001 + 2) * 0.5},
        tt: {x: Math.sin(Date.now() * 0.001 + 3) * 0.5, y: Math.cos(Date.now() * 0.001 + 3) * 0.5},
        tb: {x: Math.sin(Date.now() * 0.001 + 4) * 0.5, y: Math.cos(Date.now() * 0.001 + 4) * 0.5},
        td: {x: Math.sin(Date.now() * 0.001 + 5) * 0.5, y: Math.cos(Date.now() * 0.001 + 5) * 0.5}
    };
    
    /*
    // Real implementation would use proper matrices instead of simulated values
    const features = wavlmFeatures.data;
    const dims = wavlmFeatures.dims;
    const featureSize = dims[2]; // 768
    
    // Apply a linear projection for each articulator
    // Using pre-trained weights from your model
    const articulationFeatures = {
        ul: {x: 0, y: 0},
        ll: {x: 0, y: 0},
        li: {x: 0, y: 0},
        tt: {x: 0, y: 0},
        tb: {x: 0, y: 0},
        td: {x: 0, y: 0}
    };
    
    // This is just an example - you'd need the actual weight matrices
    // from your trained model
    for (const articulator in articulationFeatures) {
        // For each x and y coordinate
        for (const coord of ['x', 'y']) {
            // Apply the projection (dot product)
            let sum = 0;
            for (let i = 0; i < featureSize; i++) {
                // weights[articulator][coord][i] would be your trained weights
                // For now using a placeholder value 
                const weight = (i % featureSize) / featureSize * 0.01;
                sum += features[i] * weight;
            }
            articulationFeatures[articulator][coord] = sum;
        }
    }
    */

    return articulationFeatures;
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

function updateCharts() {
    // Update articulators chart with both X and Y values
    articulatorsChart.data.datasets[0].data = featureHistory.ul_x;
    articulatorsChart.data.datasets[1].data = featureHistory.ul_y;
    
    articulatorsChart.data.datasets[2].data = featureHistory.ll_x;
    articulatorsChart.data.datasets[3].data = featureHistory.ll_y;
    
    articulatorsChart.data.datasets[4].data = featureHistory.li_x;
    articulatorsChart.data.datasets[5].data = featureHistory.li_y;
    
    articulatorsChart.data.datasets[6].data = featureHistory.tt_x;
    articulatorsChart.data.datasets[7].data = featureHistory.tt_y;
    
    articulatorsChart.data.datasets[8].data = featureHistory.tb_x;
    articulatorsChart.data.datasets[9].data = featureHistory.tb_y;
    
    articulatorsChart.data.datasets[10].data = featureHistory.td_x;
    articulatorsChart.data.datasets[11].data = featureHistory.td_y;
    
    articulatorsChart.update();
}

// Initialize the application when the page loads
window.addEventListener('DOMContentLoaded', init);