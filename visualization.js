/******************************************************************************
 * SPARC Visualization - Vocal Tract Display
 *
 * Renders articulatory feature positions as colored markers on an SVG grid.
 * Also handles demo animation, smoothing control, and
 * pitch/loudness/jaw-opening bar displays.
 *
 * Depends on global state from app.js:
 *   smoothedFeatures, featureHistory, debugCounters,
 *   smoothingFactor, isRecording, animationRunning, animationFrame,
 *   DISPLAY_MIN, DISPLAY_MAX,
 *   scaleToDisplay, clampToDisplay, updateFeatureHistory,
 *   calculateJawOpening, updateStatus, debugLog
 ******************************************************************************/

/******************************************************************************
 * SVG SETUP
 ******************************************************************************/

function setupVocalTractVisualization() {
  const svg = document.getElementById('vocal-tract-svg');
  if (!svg) {
    console.error("SVG element 'vocal-tract-svg' not found");
    return;
  }

  svg.setAttribute('viewBox', '-5 -5 9 9');
  svg.setAttribute('width', '600');
  svg.setAttribute('height', '600');

  while (svg.firstChild) svg.removeChild(svg.firstChild);

  createReferenceGrid(svg);
  createLegend(svg);
  createArticulatorMarkers(svg);
}

function createReferenceGrid(svg) {
  const grid = document.createElementNS('http://www.w3.org/2000/svg', 'g');
  grid.setAttribute('id', 'reference-grid');
  grid.setAttribute('opacity', '0.15');

  for (let y = -5; y <= 4; y++) {
    const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    line.setAttribute('x1', '-5'); line.setAttribute('y1', y);
    line.setAttribute('x2', '4');  line.setAttribute('y2', y);
    line.setAttribute('stroke', y === 0 ? '#666' : '#999');
    line.setAttribute('stroke-width', '0.03');
    grid.appendChild(line);
  }

  for (let x = -5; x <= 4; x++) {
    const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    line.setAttribute('x1', x);  line.setAttribute('y1', '-5');
    line.setAttribute('x2', x);  line.setAttribute('y2', '4');
    line.setAttribute('stroke', x === 0 ? '#666' : '#999');
    line.setAttribute('stroke-width', '0.03');
    grid.appendChild(line);
  }

  svg.appendChild(grid);

  addSvgLabel(svg, 'FRONT (+X)', 3.0, 0.3);
  addSvgLabel(svg, 'BACK (-X)', -4.2, 0.3);
  addSvgLabel(svg, 'UP (+Y)', 0.2, -4.5);
  addSvgLabel(svg, 'DOWN (-Y)', 0.2, 3.8);
}

function createLegend(svg) {
  const legend = document.createElementNS('http://www.w3.org/2000/svg', 'g');
  legend.setAttribute('id', 'legend');

  const items = [
    { color: '#e74c3c', label: 'UL (upper lip)' },
    { color: '#3498db', label: 'LL (lower lip)' },
    { color: '#f1c40f', label: 'LI (lower incisor)' },
    { color: '#2ecc71', label: 'TT (tongue tip)' },
    { color: '#9b59b6', label: 'TB (tongue body)' },
    { color: '#e67e22', label: 'TD (tongue dorsum)' }
  ];

  items.forEach((item, i) => {
    const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    circle.setAttribute('cx', -4.8);
    circle.setAttribute('cy', -4.6 + i * 0.5);
    circle.setAttribute('r', '0.1');
    circle.setAttribute('fill', item.color);
    legend.appendChild(circle);

    const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    text.setAttribute('x', -4.55);
    text.setAttribute('y', -4.5 + i * 0.5);
    text.setAttribute('font-size', '0.25');
    text.setAttribute('fill', '#333');
    text.textContent = item.label;
    legend.appendChild(text);
  });

  svg.appendChild(legend);
}

function createArticulatorMarkers(svg) {
  const articulators = [
    { id: 'ul', color: '#e74c3c', label: 'UL' },
    { id: 'll', color: '#3498db', label: 'LL' },
    { id: 'li', color: '#f1c40f', label: 'LI' },
    { id: 'tt', color: '#2ecc71', label: 'TT' },
    { id: 'tb', color: '#9b59b6', label: 'TB' },
    { id: 'td', color: '#e67e22', label: 'TD' }
  ];

  articulators.forEach(art => {
    const marker = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    marker.setAttribute('id', `${art.id}-marker`);
    marker.setAttribute('r', '0.15');
    marker.setAttribute('fill', art.color);
    marker.setAttribute('stroke', '#fff');
    marker.setAttribute('stroke-width', '0.03');
    marker.setAttribute('class', 'articulator-marker');
    svg.appendChild(marker);

    const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    label.setAttribute('id', `${art.id}-label`);
    label.setAttribute('class', 'articulator-label');
    label.setAttribute('fill', '#333');
    label.setAttribute('font-size', '0.3');
    label.setAttribute('font-weight', 'bold');
    label.setAttribute('text-anchor', 'middle');
    label.textContent = art.label;
    svg.appendChild(label);
  });
}

function addSvgLabel(svg, text, x, y) {
  const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
  label.setAttribute('x', x);
  label.setAttribute('y', y);
  label.setAttribute('font-size', '0.25');
  label.setAttribute('text-anchor', 'middle');
  label.setAttribute('fill', '#888');
  label.textContent = text;
  svg.appendChild(label);
}

/******************************************************************************
 * CHART / MARKER UPDATE
 ******************************************************************************/

function updateCharts() {
  try {
    if (!featureHistory || Object.keys(featureHistory).length === 0) return;

    const articulators = ['ul', 'll', 'li', 'tt', 'tb', 'td'];
    const latestFeatures = {};

    for (const art of articulators) {
      const xHist = featureHistory[art + '_x'];
      const yHist = featureHistory[art + '_y'];
      latestFeatures[art] = (xHist && yHist)
        ? { x: xHist[xHist.length - 1], y: yHist[yHist.length - 1] }
        : { x: 0, y: 0 };
    }

    for (const art of articulators) {
      const marker = document.getElementById(`${art}-marker`);
      const label = document.getElementById(`${art}-label`);
      if (!marker || !latestFeatures[art]) continue;

      const svgX = clampToDisplay(latestFeatures[art].x || 0);
      const svgY = clampToDisplay(latestFeatures[art].y || 0);

      marker.setAttribute('cx', svgX);
      marker.setAttribute('cy', svgY);

      if (label) {
        label.setAttribute('x', svgX + 0.25);
        label.setAttribute('y', svgY - 0.2);
      }
    }

    if (featureHistory.pitch && featureHistory.loudness) {
      const p = featureHistory.pitch[featureHistory.pitch.length - 1];
      const l = featureHistory.loudness[featureHistory.loudness.length - 1];
      updateSourceFeatures(p, l);
    }

    if (typeof window !== 'undefined' && typeof window.updateJawOpeningDisplay === 'function') {
      window.updateJawOpeningDisplay(smoothedFeatures.jaw_opening);
    }
  } catch (error) {
    debugLog('Error in updateCharts', error);
    debugCounters.errors++;
  }
}

function updateSourceFeatures(pitch, loudness) {
  const normalizedPitch = Math.min(100, Math.max(0, ((pitch - 75) / 225) * 100));
  const normalizedLoudness = Math.min(100, Math.max(0, ((loudness + 60) / 60) * 100));

  const pitchBar = document.getElementById('pitch-bar');
  const loudnessBar = document.getElementById('loudness-bar');
  if (pitchBar) pitchBar.style.height = normalizedPitch + '%';
  if (loudnessBar) loudnessBar.style.height = normalizedLoudness + '%';
}

/******************************************************************************
 * DEFAULT POSITIONS & DEMO ANIMATION
 ******************************************************************************/

function stopAnimation() {
  animationRunning = false;
  if (animationFrame) {
    clearTimeout(animationFrame);
    animationFrame = null;
  }
}

function initializeDefaultPositions() {
  const defaultPositions = {
    ul: { x: 1.5, y: -0.5 },
    ll: { x: 1.5, y: 0.5 },
    li: { x: 0.8, y: 1.0 },
    tt: { x: 0.5, y: 0.0 },
    tb: { x: -1.0, y: -0.5 },
    td: { x: -2.0, y: -0.3 }
  };

  Object.keys(defaultPositions).forEach(art => {
    smoothedFeatures[art + '_x'] = scaleToDisplay(defaultPositions[art].x);
    smoothedFeatures[art + '_y'] = scaleToDisplay(defaultPositions[art].y);
  });

  smoothedFeatures.jaw_opening = calculateJawOpening(defaultPositions.ul.y, defaultPositions.ll.y);

  if (featureHistory && Object.keys(featureHistory).length > 0) {
    for (let i = 0; i < 100; i++) {
      Object.keys(defaultPositions).forEach(art => {
        if (featureHistory[art + '_x']) featureHistory[art + '_x'][i] = smoothedFeatures[art + '_x'];
        if (featureHistory[art + '_y']) featureHistory[art + '_y'][i] = smoothedFeatures[art + '_y'];
      });
      if (featureHistory.jaw_opening) featureHistory.jaw_opening[i] = smoothedFeatures.jaw_opening;
    }
    updateCharts();
  }
}

const VOWEL_POSITIONS = {
  'i': {
    ul: {x: 2.5, y: -1.5}, ll: {x: 2.5, y: -0.8}, li: {x: 1.8, y: -0.3},
    tt: {x: 3.2, y: -3.2}, tb: {x: 1.2, y: -3.5}, td: {x: -0.8, y: -2.2},
    jaw_opening: 0.2
  },
  'e': {
    ul: {x: 2.2, y: -1.2}, ll: {x: 2.2, y: 0.5}, li: {x: 1.3, y: 1.0},
    tt: {x: 2.8, y: -1.2}, tb: {x: 0.5, y: -1.8}, td: {x: -1.2, y: -0.8},
    jaw_opening: 0.6
  },
  'a': {
    ul: {x: 1.8, y: -0.8}, ll: {x: 1.8, y: 3.2}, li: {x: 0.8, y: 3.5},
    tt: {x: 1.2, y: 2.2}, tb: {x: -0.5, y: 1.8}, td: {x: -2.2, y: 0.8},
    jaw_opening: 1.5
  },
  'o': {
    ul: {x: 0.8, y: -1.0}, ll: {x: 0.8, y: 1.2}, li: {x: 0.0, y: 1.8},
    tt: {x: -0.5, y: 0.5}, tb: {x: -2.2, y: -0.5}, td: {x: -3.2, y: -1.2},
    jaw_opening: 0.8
  },
  'u': {
    ul: {x: 0.0, y: -1.2}, ll: {x: 0.0, y: -0.5}, li: {x: -0.5, y: 0.0},
    tt: {x: -1.2, y: -0.8}, tb: {x: -2.8, y: -2.8}, td: {x: -3.5, y: -3.5},
    jaw_opening: 0.2
  }
};

function testArticulatorAnimation() {
  const speechPositions = [
    {
      name: '/i/ (see)',
      ul: {x: 0.9, y: -1.05}, ll: {x: 0.9, y: -0.9}, li: {x: 0.9, y: -0.975},
      tt: {x: 0.6, y: -1.0}, tb: {x: 0.2, y: -1.0}, td: {x: -0.2, y: -0.8},
      jaw_opening: 0.1
    },
    {
      name: '/a/ (father)',
      ul: {x: 0.9, y: -0.9}, ll: {x: 0.9, y: -0.4}, li: {x: 0.9, y: -0.65},
      tt: {x: 0.2, y: -0.3}, tb: {x: -0.2, y: -0.35}, td: {x: -0.6, y: -0.3},
      jaw_opening: 0.8
    },
    {
      name: '/u/ (boot)',
      ul: {x: 0.6, y: -1.0}, ll: {x: 0.6, y: -0.85}, li: {x: 0.6, y: -0.925},
      tt: {x: -0.2, y: -0.7}, tb: {x: -0.6, y: -0.9}, td: {x: -1.0, y: -0.8},
      jaw_opening: 0.15
    }
  ];

  let frame = 0;
  const frameDuration = 800;
  const frameTransitions = 30;
  animationRunning = true;

  function animateFrame() {
    if (!document.getElementById('vocal-tract-svg') || isRecording || !animationRunning) {
      animationRunning = false;
      return;
    }

    const currentIdx = Math.floor(frame / frameTransitions) % speechPositions.length;
    const nextIdx = (currentIdx + 1) % speechPositions.length;
    const t = (frame % frameTransitions) / frameTransitions;

    const curr = speechPositions[currentIdx];
    const next = speechPositions[nextIdx];

    const features = {};
    for (const art of ['ul', 'll', 'li', 'tt', 'tb', 'td']) {
      features[art] = {
        x: curr[art].x + (next[art].x - curr[art].x) * t,
        y: curr[art].y + (next[art].y - curr[art].y) * t
      };
    }

    smoothedFeatures.jaw_opening = curr.jaw_opening + (next.jaw_opening - curr.jaw_opening) * t;

    updateFeatureHistory(features, 120 + Math.sin(frame / 15) * 80, -25 + Math.sin(frame / 10) * 25);
    updateCharts();

    if (frame % frameTransitions === 0) {
      updateStatus(`Demo: ${curr.name}`);
    }

    frame++;
    animationFrame = setTimeout(animateFrame, frameDuration / frameTransitions);
  }

  updateStatus('Demo: Showing articulator movement...');
  animateFrame();
}

/******************************************************************************
 * CONTROLS SETUP
 ******************************************************************************/

function setupCharts() {
  setupVocalTractVisualization();
  initializeDefaultPositions();
  if (!isRecording) {
    testArticulatorAnimation();
  }
}

function setupSensitivityControls() {
  const smoothingSlider = document.getElementById('smoothing-slider');
  const smoothingValue = document.getElementById('smoothing-value');
  if (smoothingSlider) {
    smoothingSlider.addEventListener('input', function() {
      smoothingFactor = parseFloat(this.value);
      if (smoothingValue) smoothingValue.textContent = smoothingFactor.toFixed(1);
    });
  }

  const soundSelector = document.getElementById('sound-selector');
  if (soundSelector) {
    soundSelector.addEventListener('change', function() {
      if (isRecording) {
        alert('Stop recording first to test sounds');
        soundSelector.value = '';
        return;
      }

      // Stop the cycling demo animation when a sound is selected
      stopAnimation();

      const vowel = soundSelector.value;
      if (vowel && VOWEL_POSITIONS[vowel]) {
        const pos = VOWEL_POSITIONS[vowel];
        updateStatus(`/${vowel}/`);

        for (const art of ['ul', 'll', 'li', 'tt', 'tb', 'td']) {
          if (!pos[art]) continue;
          const sx = scaleToDisplay(pos[art].x);
          const sy = scaleToDisplay(pos[art].y);
          smoothedFeatures[art + '_x'] = sx;
          smoothedFeatures[art + '_y'] = sy;
          if (featureHistory[art + '_x']) featureHistory[art + '_x'][featureHistory[art + '_x'].length - 1] = sx;
          if (featureHistory[art + '_y']) featureHistory[art + '_y'][featureHistory[art + '_y'].length - 1] = sy;
        }
        smoothedFeatures.jaw_opening = pos.jaw_opening;
        updateCharts();
      } else if (!vowel) {
        // "-- Select --" chosen: restart animation
        testArticulatorAnimation();
      }
    });
  }
}
