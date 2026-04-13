/******************************************************************************
 * SPARC Visualization - Vocal Tract Display
 *
 * Renders articulatory feature positions as colored markers on an SVG grid.
 * Also handles demo animation and smoothing control.
 *
 * Depends on global state from app.js:
 *   smoothedFeatures, featureHistory, debugCounters,
 *   smoothingFactor, isRecording, animationRunning, animationFrame,
 *   DISPLAY_MIN, DISPLAY_MAX,
 *   scaleToDisplay, clampToDisplay, updateFeatureHistory,
 *   calculateJawOpening, updateStatus, debugLog
 ******************************************************************************/

/******************************************************************************
 * ARTICULATOR COLOR MAP
 ******************************************************************************/

const ARTICULATOR_COLORS = {
  ul: { fill: '#b71c1c', stroke: '#fff', label: 'UL (upper lip)' },
  ll: { fill: '#ef9a9a', stroke: '#fff', label: 'LL (lower lip)' },
  li: { fill: '#ffffff', stroke: '#333', label: 'LI (lower incisor)' },
  tt: { fill: '#0d47a1', stroke: '#fff', label: 'TT (tongue tip)' },
  tb: { fill: '#1976d2', stroke: '#fff', label: 'TB (tongue body)' },
  td: { fill: '#64b5f6', stroke: '#fff', label: 'TD (tongue dorsum)' }
};

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
  svg.removeAttribute('width');
  svg.removeAttribute('height');

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

  // MNGU0: +x = anterior, +y = inferior; SVG: +x = right, +y = down
  addSvgLabel(svg, 'FRONT', 3.0, 0.3);
  addSvgLabel(svg, 'BACK', -4.2, 0.3);
  addSvgLabel(svg, 'UP', 0.2, -4.5);
  addSvgLabel(svg, 'DOWN', 0.2, 3.8);
}

function createLegend(svg) {
  const legend = document.createElementNS('http://www.w3.org/2000/svg', 'g');
  legend.setAttribute('id', 'legend');

  const order = ['ul', 'll', 'li', 'tt', 'tb', 'td'];

  order.forEach((id, i) => {
    const art = ARTICULATOR_COLORS[id];

    const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    circle.setAttribute('cx', -4.8);
    circle.setAttribute('cy', -4.6 + i * 0.5);
    circle.setAttribute('r', '0.1');
    circle.setAttribute('fill', art.fill);
    circle.setAttribute('stroke', art.stroke);
    circle.setAttribute('stroke-width', '0.03');
    legend.appendChild(circle);

    const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    text.setAttribute('x', -4.55);
    text.setAttribute('y', -4.6 + i * 0.5);
    text.style.fontSize = '0.22px';
    text.style.fill = '#333';
    text.style.textAnchor = 'start';
    text.style.dominantBaseline = 'central';
    text.textContent = art.label;
    legend.appendChild(text);
  });

  svg.appendChild(legend);
}

function createArticulatorMarkers(svg) {
  const order = ['ul', 'll', 'li', 'tt', 'tb', 'td'];

  order.forEach(id => {
    const art = ARTICULATOR_COLORS[id];

    const marker = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    marker.setAttribute('id', `${id}-marker`);
    marker.setAttribute('r', '0.15');
    marker.setAttribute('fill', art.fill);
    marker.setAttribute('stroke', art.stroke);
    marker.setAttribute('stroke-width', '0.03');
    marker.setAttribute('class', 'articulator-marker');
    svg.appendChild(marker);
  });
}

function addSvgLabel(svg, text, x, y) {
  const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
  label.setAttribute('class', 'grid-label');
  label.setAttribute('x', x);
  label.setAttribute('y', y);
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
      if (!marker || !latestFeatures[art]) continue;

      marker.setAttribute('cx', clampToDisplay(latestFeatures[art].x || 0));
      marker.setAttribute('cy', clampToDisplay(latestFeatures[art].y || 0));
    }
  } catch (error) {
    debugLog('Error in updateCharts', error);
    debugCounters.errors++;
  }
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
    ul: { x: 1.5, y: -0.3 },
    ll: { x: 1.3, y: 0.3 },
    li: { x: 0.8, y: 0.5 },
    tt: { x: 0.5, y: 0.6 },
    tb: { x: -0.5, y: -0.3 },
    td: { x: -1.5, y: -0.2 }
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

// Approximate MNGU0 EMA z-scored coordinates for vowel demo positions.
// MNGU0 axes: +x = anterior (toward lips), +y = inferior (downward).
// Constraints from real EMA data (see COMPARISON.md):
//   - UL is the most anterior (highest x); tongue articulators are behind lips
//   - TT y >= LI y for vowels (tongue tip rests at or below lower incisor)
//   - TB and TD vertical position varies with vowel height
//   - Typical range: roughly -3 to +3 (z-scored)
// These positions are for the UI demo only and do not affect model output.
const VOWEL_POSITIONS = {
  'i': {
    ul: {x: 2.0, y: -0.4}, ll: {x: 1.8, y: 0.0}, li: {x: 1.2, y: 0.2},
    tt: {x: 0.8, y: 0.3},  tb: {x: -0.5, y: -1.5}, td: {x: -1.5, y: -1.0},
    jaw_opening: 0.15
  },
  'e': {
    ul: {x: 1.8, y: -0.3}, ll: {x: 1.6, y: 0.5}, li: {x: 1.0, y: 0.6},
    tt: {x: 0.6, y: 0.6},  tb: {x: -0.5, y: -0.8}, td: {x: -1.5, y: -0.3},
    jaw_opening: 0.4
  },
  'a': {
    ul: {x: 1.5, y: -0.3}, ll: {x: 1.3, y: 1.8}, li: {x: 0.8, y: 2.0},
    tt: {x: 0.3, y: 2.2},  tb: {x: -0.8, y: 1.0},  td: {x: -2.0, y: 0.3},
    jaw_opening: 0.9
  },
  'o': {
    ul: {x: 0.8, y: -0.6}, ll: {x: 0.6, y: 0.5}, li: {x: 0.2, y: 0.8},
    tt: {x: -0.3, y: 0.9}, tb: {x: -1.8, y: -0.8}, td: {x: -2.5, y: -0.5},
    jaw_opening: 0.5
  },
  'u': {
    ul: {x: 0.5, y: -0.6}, ll: {x: 0.3, y: -0.1}, li: {x: -0.1, y: 0.1},
    tt: {x: -0.5, y: 0.2}, tb: {x: -2.0, y: -1.8}, td: {x: -2.8, y: -1.5},
    jaw_opening: 0.15
  }
};

function testArticulatorAnimation() {
  const vowelSequence = ['i', 'a', 'u'];

  let frame = 0;
  const frameDuration = 800;
  const frameTransitions = 30;
  animationRunning = true;

  function animateFrame() {
    if (!document.getElementById('vocal-tract-svg') || isRecording || !animationRunning) {
      animationRunning = false;
      return;
    }

    const currentIdx = Math.floor(frame / frameTransitions) % vowelSequence.length;
    const nextIdx = (currentIdx + 1) % vowelSequence.length;
    const t = (frame % frameTransitions) / frameTransitions;

    const curr = VOWEL_POSITIONS[vowelSequence[currentIdx]];
    const next = VOWEL_POSITIONS[vowelSequence[nextIdx]];

    const features = {};
    for (const art of ['ul', 'll', 'li', 'tt', 'tb', 'td']) {
      features[art] = {
        x: curr[art].x + (next[art].x - curr[art].x) * t,
        y: curr[art].y + (next[art].y - curr[art].y) * t
      };
    }

    smoothedFeatures.jaw_opening = curr.jaw_opening + (next.jaw_opening - curr.jaw_opening) * t;

    updateFeatureHistory(features, 0, -60);
    updateCharts();

    if (frame % frameTransitions === 0) {
      const vowel = vowelSequence[currentIdx];
      updateStatus(`Demo: /${vowel}/`);
    }

    frame++;
    animationFrame = setTimeout(animateFrame, frameDuration / frameTransitions);
  }

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
        testArticulatorAnimation();
      }
    });
  }
}
