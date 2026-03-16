const canvas = document.getElementById('stageCanvas');
const ctx = canvas.getContext('2d');
const video = document.getElementById('webcam');
const startBtn = document.getElementById('startBtn');
const statusBadge = document.getElementById('statusBadge');

const modelStatusEl = document.getElementById('modelStatus');
const cameraStatusEl = document.getElementById('cameraStatus');
const detectorStatusEl = document.getElementById('detectorStatus');
const handStatusEl = document.getElementById('handStatus');
const hitCountEl = document.getElementById('hitCount');
const lastHitEl = document.getElementById('lastHit');
const fpsEl = document.getElementById('fps');
const objectListEl = document.getElementById('objectList');
const mappingListEl = document.getElementById('mappingList');
const errorDetailsEl = document.getElementById('errorDetails');
const initLogEl = document.getElementById('initLog');

const confidenceInput = document.getElementById('confidence');
const cooldownInput = document.getElementById('cooldown');
const smoothingInput = document.getElementById('smoothing');
const stabilityInput = document.getElementById('stability');
const confValue = document.getElementById('confValue');
const cooldownValue = document.getElementById('cooldownValue');
const smoothValue = document.getElementById('smoothValue');
const stableValue = document.getElementById('stableValue');

const URLS = {
  mpTasks: 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/+esm',
  mpWasmRoot: 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm',
  mpHandTask: 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
};

const SOUND_TYPES = ['kick', 'snare', 'hihat', 'tom', 'clap', 'cowbell', 'shaker', 'conga', 'rim'];
const DEFAULT_MAP = { bottle: 'cowbell', bowl: 'tom', cup: 'hihat', scissors: 'clap', book: 'snare', 'cell phone': 'rim', keyboard: 'kick', spoon: 'shaker', laptop: 'tom', mouse: 'hihat', remote: 'clap', backpack: 'conga', 'potted plant': 'cowbell' };

const BLOCKED_CLASS_ALIASES = new Set(['person', 'people', 'human', 'man', 'woman', 'boy', 'girl']);
const SUPPRESSED_CLASSES = new Set(['bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter']);
const INDOOR_PRIORITY_CLASSES = new Set([
  'book', 'cell phone', 'keyboard', 'mouse', 'laptop', 'remote', 'backpack', 'bottle', 'cup', 'bowl', 'scissors', 'spoon',
  'potted plant', 'vase', 'chair', 'couch', 'bed', 'dining table', 'tv', 'clock', 'toothbrush', 'teddy bear', 'wine glass'
]);

const TRACKING = {
  minArea: 1400,
  detectEveryMs: 90,
  maxDetections: 35,
  trackCreateBoostNonPriority: 0.08,
  trackUpdateSlack: 0.16,
  iouMatchThreshold: 0.2,
  distanceMatchNorm: 0.6,
  confirmHits: 2,
  maxMisses: 14,
  staleMs: 1800,
  edgeExpireGraceMs: 500,
  maxRenderedTracks: 12
};

const state = {
  detector: null,
  handLandmarker: null,
  modelsLoadPromise: null,
  lastDetectionAt: 0,
  tracks: new Map(),
  nextTrackId: 1,
  fingertips: [],
  hitCount: 0,
  lastHit: '–',
  insideState: new Map(),
  cooldownByTrack: new Map(),
  soundMap: JSON.parse(localStorage.getItem('object-drum-map') || '{}'),
  settings: {
    confidence: Number(confidenceInput.value),
    cooldownMs: Number(cooldownInput.value),
    smoothing: Number(smoothingInput.value),
    stabilityFrames: Number(stabilityInput.value)
  },
  audio: null,
  running: false,
  fpsSamples: [],
  lastMappingKey: ''
};

function normalizeLabel(label) {
  return String(label || '').toLowerCase().trim().replace(/[_-]+/g, ' ');
}

function isBlockedClass(label) {
  return BLOCKED_CLASS_ALIASES.has(normalizeLabel(label));
}

function isSuppressedClass(label) {
  return SUPPRESSED_CLASSES.has(normalizeLabel(label));
}

function isIndoorPriorityClass(label) {
  return INDOOR_PRIORITY_CLASSES.has(normalizeLabel(label));
}

function labelThreshold(label) {
  const base = state.settings.confidence;
  return isIndoorPriorityClass(label) ? base : Math.min(0.92, base + TRACKING.trackCreateBoostNonPriority);
}

function logInit(message) {
  const line = `[${new Date().toLocaleTimeString()}] ${message}`;
  console.info('[init]', message);
  initLogEl.textContent = `${line}\n${initLogEl.textContent}`.split('\n').slice(0, 18).join('\n');
}

function setError(message) {
  errorDetailsEl.textContent = message || 'No errors.';
}

function statusText(err) {
  if (!err) return 'Unknown error';
  return `${err.name || 'Error'}: ${err.message || String(err)}`;
}

function setComponentStatus(component, text) {
  if (component === 'model') modelStatusEl.textContent = text;
  if (component === 'camera') cameraStatusEl.textContent = text;
  if (component === 'detector') detectorStatusEl.textContent = text;
  if (component === 'hand') handStatusEl.textContent = text;
}

function ensureSecureContext() {
  const isLocalhost = ['localhost', '127.0.0.1', '::1'].includes(location.hostname);
  if (window.isSecureContext || isLocalhost) return;
  throw new Error(`Webcam requires secure context. Open via https:// or localhost (current origin: ${location.origin})`);
}

async function importModule(url, label) {
  logInit(`Importing ${label}: ${url}`);
  try {
    return await import(url);
  } catch (err) {
    throw new Error(`${label} import failed from ${url}. ${statusText(err)}`);
  }
}

async function verifyUrl(url, label) {
  try {
    const res = await fetch(url, { method: 'GET', mode: 'cors' });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
  } catch (err) {
    throw new Error(`${label} resource not reachable at ${url}. ${statusText(err)}`);
  }
}

function assignDefaultSound(label) {
  if (!state.soundMap[label]) {
    state.soundMap[label] = DEFAULT_MAP[label] || SOUND_TYPES[label.length % SOUND_TYPES.length];
    localStorage.setItem('object-drum-map', JSON.stringify(state.soundMap));
  }
  return state.soundMap[label];
}

function playSound(type) {
  const Ctx = window.AudioContext || window.webkitAudioContext;
  if (!state.audio) state.audio = new Ctx();
  if (state.audio.state === 'suspended') state.audio.resume();
  const now = state.audio.currentTime;
  const output = state.audio.createGain();
  output.gain.value = 0.9;
  output.connect(state.audio.destination);

  const noiseBuffer = () => {
    const buffer = state.audio.createBuffer(1, state.audio.sampleRate * 0.25, state.audio.sampleRate);
    const channel = buffer.getChannelData(0);
    for (let i = 0; i < channel.length; i++) channel[i] = Math.random() * 2 - 1;
    return buffer;
  };

  const hitTone = (freq, decay, wave = 'sine', gain = 0.8) => {
    const osc = state.audio.createOscillator();
    const g = state.audio.createGain();
    osc.type = wave;
    osc.frequency.setValueAtTime(freq, now);
    osc.frequency.exponentialRampToValueAtTime(Math.max(50, freq * 0.25), now + decay);
    g.gain.setValueAtTime(gain, now);
    g.gain.exponentialRampToValueAtTime(0.001, now + decay);
    osc.connect(g).connect(output);
    osc.start(now);
    osc.stop(now + decay);
  };

  const hitNoise = (decay, filterFreq, gain = 0.35) => {
    const src = state.audio.createBufferSource();
    src.buffer = noiseBuffer();
    const filter = state.audio.createBiquadFilter();
    filter.type = 'highpass';
    filter.frequency.value = filterFreq;
    const g = state.audio.createGain();
    g.gain.setValueAtTime(gain, now);
    g.gain.exponentialRampToValueAtTime(0.001, now + decay);
    src.connect(filter).connect(g).connect(output);
    src.start(now);
    src.stop(now + decay);
  };

  ({
    kick: () => hitTone(160, 0.25, 'sine', 1),
    snare: () => { hitNoise(0.2, 1200, 0.45); hitTone(220, 0.12, 'triangle', 0.25); },
    hihat: () => hitNoise(0.08, 4500, 0.22),
    tom: () => hitTone(210, 0.22, 'triangle', 0.65),
    clap: () => { hitNoise(0.11, 1500, 0.4); setTimeout(() => hitNoise(0.08, 1700, 0.28), 22); },
    cowbell: () => { hitTone(620, 0.16, 'square', 0.32); hitTone(840, 0.15, 'square', 0.22); },
    shaker: () => hitNoise(0.05, 3800, 0.18),
    conga: () => hitTone(280, 0.2, 'sine', 0.68),
    rim: () => { hitTone(1200, 0.05, 'triangle', 0.2); hitNoise(0.03, 5500, 0.08); }
  }[type] || (() => hitTone(220, 0.2)))();
}

async function loadModels() {
  if (state.modelsLoadPromise) return state.modelsLoadPromise;

  state.modelsLoadPromise = (async () => {
    setComponentStatus('model', 'Loading');
    setComponentStatus('detector', 'Loading');
    setComponentStatus('hand', 'Loading');
    statusBadge.textContent = 'Loading models…';

    try {
      ensureSecureContext();
      const tf = globalThis.tf;
      const cocoSsd = globalThis.cocoSsd;
      if (!tf) throw new Error('TensorFlow.js global failed to load from CDN script tag.');
      if (!cocoSsd?.load) throw new Error('COCO-SSD global failed to load from CDN script tag.');
      logInit(`TFJS backend: ${tf.getBackend?.() || 'unknown'}`);

      const mediapipe = await importModule(URLS.mpTasks, 'MediaPipe Tasks Vision');
      state.detector = await cocoSsd.load({ base: 'lite_mobilenet_v2' });
      setComponentStatus('detector', 'Ready');
      logInit('Object detector ready');

      await verifyUrl(URLS.mpHandTask, 'Hand landmark model');
      const vision = await mediapipe.FilesetResolver.forVisionTasks(URLS.mpWasmRoot);
      state.handLandmarker = await mediapipe.HandLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: URLS.mpHandTask },
        runningMode: 'VIDEO',
        numHands: 2,
        minHandDetectionConfidence: 0.45,
        minHandPresenceConfidence: 0.45,
        minTrackingConfidence: 0.4
      });
      setComponentStatus('hand', 'Ready');
      logInit('Hand tracker ready');

      setComponentStatus('model', 'Ready');
      if (!state.running) statusBadge.textContent = 'Models ready';
      setError('No errors.');
    } catch (err) {
      setComponentStatus('model', 'Failed');
      if (!state.detector) setComponentStatus('detector', 'Failed');
      if (!state.handLandmarker) setComponentStatus('hand', 'Failed');
      statusBadge.textContent = 'Model load failed';
      setError(`Model initialization failed. ${statusText(err)}`);
      logInit(`Model load failed: ${statusText(err)}`);
      throw err;
    }
  })();

  return state.modelsLoadPromise;
}

function iou(a, b) {
  const x1 = Math.max(a.x, b.x);
  const y1 = Math.max(a.y, b.y);
  const x2 = Math.min(a.x + a.w, b.x + b.w);
  const y2 = Math.min(a.y + a.h, b.y + b.h);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  return inter ? inter / (a.w * a.h + b.w * b.h - inter) : 0;
}

function centerDistanceNorm(a, b) {
  const ax = a.x + a.w / 2;
  const ay = a.y + a.h / 2;
  const bx = b.x + b.w / 2;
  const by = b.y + b.h / 2;
  const dx = ax - bx;
  const dy = ay - by;
  const diagonal = Math.hypot(canvas.width, canvas.height) || 1;
  return Math.hypot(dx, dy) / diagonal;
}

function isNearFrameEdge(box, margin = 16) {
  return box.x <= margin || box.y <= margin || box.x + box.w >= canvas.width - margin || box.y + box.h >= canvas.height - margin;
}

function detectionPassesCreateGate(pred) {
  if (isBlockedClass(pred.class) || isSuppressedClass(pred.class)) return false;
  const [x, y, w, h] = pred.bbox;
  if (w * h < TRACKING.minArea) return false;
  return pred.score >= labelThreshold(pred.class);
}

function detectionPassesUpdateGate(pred) {
  if (isBlockedClass(pred.class) || isSuppressedClass(pred.class)) return false;
  const [x, y, w, h] = pred.bbox;
  if (w * h < TRACKING.minArea * 0.7) return false;
  const relaxed = Math.max(0.15, labelThreshold(pred.class) - TRACKING.trackUpdateSlack);
  return pred.score >= relaxed;
}

function createTrack(pred, now) {
  const [x, y, w, h] = pred.bbox;
  const id = String(state.nextTrackId++);
  state.tracks.set(id, {
    id,
    label: pred.class,
    score: pred.score,
    box: { x, y, w, h },
    hitStreak: 1,
    missCount: 0,
    confirmed: false,
    createdAt: now,
    lastSeen: now,
    lastSeenNearEdge: isNearFrameEdge({ x, y, w, h }),
    flashUntil: 0
  });
  assignDefaultSound(pred.class);
}

function bestTrackForDetection(pred, usedTrackIds) {
  const [x, y, w, h] = pred.bbox;
  const detBox = { x, y, w, h };
  let best = null;
  let bestScore = -Infinity;

  state.tracks.forEach((track) => {
    if (usedTrackIds.has(track.id)) return;
    if (track.label !== pred.class) return;

    const overlap = iou(detBox, track.box);
    const dist = centerDistanceNorm(detBox, track.box);
    if (overlap < TRACKING.iouMatchThreshold && dist > TRACKING.distanceMatchNorm) return;

    const score = overlap * 1.2 - dist;
    if (score > bestScore) {
      bestScore = score;
      best = track;
    }
  });

  return best;
}

function updateTracks(predictions) {
  const now = performance.now();
  const candidatePreds = predictions.filter((p) => detectionPassesUpdateGate(p));
  const usedTrackIds = new Set();

  candidatePreds.forEach((pred) => {
    const track = bestTrackForDetection(pred, usedTrackIds);

    if (track) {
      const [x, y, w, h] = pred.bbox;
      const a = state.settings.smoothing;
      track.box = {
        x: track.box.x * (1 - a) + x * a,
        y: track.box.y * (1 - a) + y * a,
        w: track.box.w * (1 - a) + w * a,
        h: track.box.h * (1 - a) + h * a
      };
      track.score = pred.score;
      track.lastSeen = now;
      track.lastSeenNearEdge = isNearFrameEdge(track.box);
      track.missCount = 0;
      track.hitStreak += 1;
      if (track.hitStreak >= TRACKING.confirmHits) track.confirmed = true;
      usedTrackIds.add(track.id);
      assignDefaultSound(track.label);
      return;
    }

    if (detectionPassesCreateGate(pred)) {
      createTrack(pred, now);
    }
  });

  state.tracks.forEach((track) => {
    if (usedTrackIds.has(track.id)) return;
    track.missCount += 1;

    const tooManyMisses = track.missCount > TRACKING.maxMisses;
    const tooOld = now - track.lastSeen > TRACKING.staleMs;
    const edgeLikelyLeft = track.lastSeenNearEdge && now - track.lastSeen > TRACKING.edgeExpireGraceMs;

    if (tooManyMisses || tooOld || edgeLikelyLeft) {
      state.tracks.delete(track.id);
      state.cooldownByTrack.delete(track.id);
      for (const k of state.insideState.keys()) {
        if (k.endsWith(`|${track.id}`)) state.insideState.delete(k);
      }
    }
  });
}

function rankedVisibleTracks() {
  return Array.from(state.tracks.values())
    .filter((t) => t.confirmed || t.hitStreak >= state.settings.stabilityFrames)
    .filter((t) => t.missCount <= TRACKING.maxMisses)
    .sort((a, b) => {
      const indoorDiff = Number(isIndoorPriorityClass(b.label)) - Number(isIndoorPriorityClass(a.label));
      if (indoorDiff !== 0) return indoorDiff;
      const freshDiff = b.lastSeen - a.lastSeen;
      if (freshDiff !== 0) return freshDiff;
      return b.score - a.score;
    })
    .slice(0, TRACKING.maxRenderedTracks);
}

function updateMappingsUi() {
  const labels = Array.from(new Set(rankedVisibleTracks().map((t) => t.label))).sort();
  const nextKey = labels.join('|');

  const active = document.activeElement;
  const interacting = active && mappingListEl.contains(active) && active.tagName === 'SELECT';
  if (interacting) return;
  if (nextKey === state.lastMappingKey) return;

  state.lastMappingKey = nextKey;
  mappingListEl.innerHTML = '';

  labels.forEach((label) => {
    const row = document.createElement('div');
    row.className = 'mapping-row';
    const span = document.createElement('span');
    span.textContent = label;

    const sel = document.createElement('select');
    SOUND_TYPES.forEach((soundName) => {
      const option = document.createElement('option');
      option.value = soundName;
      option.textContent = soundName;
      if ((state.soundMap[label] || assignDefaultSound(label)) === soundName) option.selected = true;
      sel.appendChild(option);
    });

    sel.addEventListener('change', () => {
      state.soundMap[label] = sel.value;
      localStorage.setItem('object-drum-map', JSON.stringify(state.soundMap));
    });

    row.append(span, sel);
    mappingListEl.append(row);
  });
}

function updateObjectList() {
  const active = rankedVisibleTracks();
  objectListEl.innerHTML = active.length
    ? active.map((t) => `<li><span>${t.label}</span><strong>${Math.round(t.score * 100)}%</strong></li>`).join('')
    : '<li><span>No stable detections yet</span><strong>–</strong></li>';
}

function pointInBox(pt, box) {
  return pt.x >= box.x && pt.x <= box.x + box.w && pt.y >= box.y && pt.y <= box.y + box.h;
}

function handleHits() {
  const now = performance.now();
  const visibleTracks = rankedVisibleTracks();

  state.fingertips.forEach((tip, idx) => {
    const containing = visibleTracks.filter((t) => pointInBox(tip, t.box));
    containing.sort((a, b) => a.box.w * a.box.h - b.box.w * b.box.h);
    const winner = containing[0];

    visibleTracks.forEach((track) => {
      const key = `${idx}|${track.id}`;
      const inside = winner?.id === track.id;
      const wasInside = state.insideState.get(key) || false;

      if (inside && !wasInside) {
        const canHit = now - (state.cooldownByTrack.get(track.id) || 0) > state.settings.cooldownMs;
        if (canHit) {
          const sound = state.soundMap[track.label] || assignDefaultSound(track.label);
          playSound(sound);
          track.flashUntil = now + 130;
          state.cooldownByTrack.set(track.id, now);
          state.hitCount += 1;
          state.lastHit = `${track.label} → ${sound}`;
        }
      }

      state.insideState.set(key, inside);
    });
  });

  hitCountEl.textContent = String(state.hitCount);
  lastHitEl.textContent = state.lastHit;
}

function drawFrame() {
  const w = canvas.width;
  const h = canvas.height;
  ctx.save();
  ctx.translate(w, 0);
  ctx.scale(-1, 1);
  ctx.drawImage(video, 0, 0, w, h);
  ctx.restore();

  rankedVisibleTracks().forEach((t) => {
    const x = w - (t.box.x + t.box.w);
    const flash = performance.now() < t.flashUntil;

    ctx.lineWidth = flash ? 4 : 2;
    ctx.strokeStyle = flash ? '#2ee8a6' : (isIndoorPriorityClass(t.label) ? '#80a4ff' : '#64759c');
    ctx.fillStyle = flash ? 'rgba(46, 232, 166, .18)' : 'rgba(28, 44, 88, .22)';
    ctx.fillRect(x, t.box.y, t.box.w, t.box.h);
    ctx.strokeRect(x, t.box.y, t.box.w, t.box.h);

    const staleTag = t.missCount > 0 ? ` • hold ${t.missCount}` : '';
    const label = `${t.label} ${Math.round(t.score * 100)}%${staleTag}`;
    ctx.font = '13px Inter, sans-serif';
    const tw = ctx.measureText(label).width;
    ctx.fillStyle = '#0c1220';
    ctx.fillRect(x, Math.max(0, t.box.y - 20), tw + 12, 18);
    ctx.fillStyle = '#dfe8ff';
    ctx.fillText(label, x + 6, Math.max(14, t.box.y - 6));
  });

  state.fingertips.forEach((tip) => {
    const x = w - tip.x;
    ctx.beginPath();
    ctx.arc(x, tip.y, 7, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(255, 104, 165, .95)';
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.strokeStyle = 'rgba(255, 255, 255, .8)';
    ctx.stroke();
  });
}

async function step() {
  if (!state.running) return;
  const now = performance.now();

  try {
    if (now - state.lastDetectionAt > TRACKING.detectEveryMs) {
      const preds = await state.detector.detect(video, TRACKING.maxDetections);
      updateTracks(preds);
      updateMappingsUi();
      updateObjectList();
      state.lastDetectionAt = now;
    }

    const hands = state.handLandmarker.detectForVideo(video, now);
    state.fingertips = (hands.landmarks || []).map((landmarks) => ({
      x: landmarks[8].x * canvas.width,
      y: landmarks[8].y * canvas.height
    }));

    handleHits();
    drawFrame();

    state.fpsSamples.push(now);
    while (state.fpsSamples.length && now - state.fpsSamples[0] > 1000) state.fpsSamples.shift();
    fpsEl.textContent = String(state.fpsSamples.length);

    requestAnimationFrame(step);
  } catch (err) {
    const msg = `Runtime loop failed. ${statusText(err)}`;
    setError(msg);
    statusBadge.textContent = 'Runtime error';
    logInit(msg);
    state.running = false;
    startBtn.disabled = false;
  }
}

function cameraErrorMessage(err) {
  if (!err) return 'Unknown camera error';
  if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') return 'Camera permission was denied. Allow webcam access in browser site settings and retry.';
  if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') return 'No camera device found. Connect/enable a webcam and retry.';
  if (err.name === 'NotReadableError' || err.name === 'TrackStartError') return 'Camera is busy or blocked by another app. Close competing apps and retry.';
  if (err.name === 'OverconstrainedError') return `Requested camera constraints are unsupported (${err.constraint || 'unknown constraint'}).`;
  return statusText(err);
}

async function start() {
  setError('No errors.');
  startBtn.disabled = true;

  try {
    ensureSecureContext();
    setComponentStatus('camera', 'Requesting');
    statusBadge.textContent = 'Requesting camera…';
    logInit(`Requesting webcam from origin ${location.origin}`);

    if (!navigator.mediaDevices?.getUserMedia) throw new Error('getUserMedia is not available in this browser. Use latest Chrome/Edge/Safari.');

    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 1600 },
        height: { ideal: 900 },
        facingMode: 'environment'
      },
      audio: false
    });

    video.srcObject = stream;
    await video.play();
    canvas.width = video.videoWidth || 960;
    canvas.height = video.videoHeight || 540;
    setComponentStatus('camera', 'Ready');
    logInit(`Camera ready (${canvas.width}x${canvas.height})`);
  } catch (err) {
    const msg = cameraErrorMessage(err);
    setComponentStatus('camera', 'Failed');
    statusBadge.textContent = 'Camera error';
    setError(`Camera startup failed. ${msg}`);
    logInit(`Camera startup failed: ${msg}`);
    startBtn.disabled = false;
    return;
  }

  try {
    await loadModels();
  } catch (err) {
    statusBadge.textContent = 'Model error';
    setError(`Model loading failed. ${statusText(err)}`);
    startBtn.disabled = false;
    return;
  }

  state.running = true;
  statusBadge.textContent = 'Live';
  logInit('Live processing started. Indoor classes prioritized; person class filtered.');
  step();
}

[confidenceInput, cooldownInput, smoothingInput, stabilityInput].forEach((input) => {
  input.addEventListener('input', () => {
    state.settings.confidence = Number(confidenceInput.value);
    state.settings.cooldownMs = Number(cooldownInput.value);
    state.settings.smoothing = Number(smoothingInput.value);
    state.settings.stabilityFrames = Number(stabilityInput.value);
    confValue.textContent = confidenceInput.value;
    cooldownValue.textContent = cooldownInput.value;
    smoothValue.textContent = smoothingInput.value;
    stableValue.textContent = stabilityInput.value;
  });
});

window.addEventListener('unhandledrejection', (event) => {
  const msg = `Unhandled promise rejection: ${statusText(event.reason)}`;
  setError(msg);
  logInit(msg);
});

window.addEventListener('error', (event) => {
  const msg = `Window error: ${event.message}`;
  setError(msg);
  logInit(msg);
});

startBtn.addEventListener('click', start);
logInit('Booted. Click "Start Camera" to begin. Persistent track hold is enabled.');
loadModels().catch((err) => {
  logInit(`Background model preload failed: ${statusText(err)}`);
});
