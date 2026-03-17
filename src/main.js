const canvas = document.getElementById('stageCanvas');
const ctx = canvas.getContext('2d');
const video = document.getElementById('webcam');
const startBtn = document.getElementById('startBtn');
const statusBadge = document.getElementById('statusBadge');

const modelStatusEl = document.getElementById('modelStatus');
const cameraStatusEl = document.getElementById('cameraStatus');
const detectorStatusEl = document.getElementById('detectorStatus');
const handStatusEl = document.getElementById('handStatus');
const detectorModelEl = document.getElementById('detectorModel');
const rawDetectionsEl = document.getElementById('rawDetections');
const trackedCountEl = document.getElementById('trackedCount');
const handModeEl = document.getElementById('handMode');

const hitCountEl = document.getElementById('hitCount');
const lastHitEl = document.getElementById('lastHit');
const fpsEl = document.getElementById('fps');
const objectListEl = document.getElementById('objectList');
const rawListEl = document.getElementById('rawList');
const mappingListEl = document.getElementById('mappingList');
const errorDetailsEl = document.getElementById('errorDetails');
const initLogEl = document.getElementById('initLog');

const confidenceInput = document.getElementById('confidence');
const cooldownInput = document.getElementById('cooldown');
const smoothingInput = document.getElementById('smoothing');
const stabilityInput = document.getElementById('stability');
const debugRawInput = document.getElementById('debugRaw');
const confValue = document.getElementById('confValue');
const cooldownValue = document.getElementById('cooldownValue');
const smoothValue = document.getElementById('smoothValue');
const stableValue = document.getElementById('stableValue');

const URLS = {
  mpTasks: 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/+esm',
  mpWasmRoot: 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm',
  mpHandTask: 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
  transformers: 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2'
};

const STAGE = { width: 720, height: 1280 };
const SOUND_TYPES = ['kick', 'snare', 'hihat', 'tom', 'clap', 'cowbell', 'shaker', 'conga', 'rim'];
const DEFAULT_MAP = {
  bottle: 'cowbell', bowl: 'tom', cup: 'hihat', mug: 'hihat', scissors: 'clap', book: 'snare', notebook: 'snare',
  'cell phone': 'rim', keyboard: 'kick', spoon: 'shaker', laptop: 'tom', mouse: 'hihat', remote: 'clap',
  backpack: 'conga', 'potted plant': 'shaker', plant: 'shaker', pen: 'rim', pencil: 'rim', marker: 'clap',
  headphones: 'conga', camera: 'cowbell', candle: 'tom', fan: 'kick', tissue: 'shaker', 'tissue box': 'snare',
  'charging cable': 'shaker'
};
const BLOCKED_CLASS_ALIASES = new Set(['person', 'people', 'human', 'man', 'woman', 'boy', 'girl']);
const INDOOR_PRIORITY = new Set([
  'book', 'notebook', 'cell phone', 'keyboard', 'mouse', 'laptop', 'remote', 'backpack', 'bottle', 'cup', 'mug',
  'bowl', 'scissors', 'spoon', 'potted plant', 'plant', 'vase', 'pen', 'pencil', 'marker', 'headphones',
  'camera', 'fan', 'candle', 'tissue', 'tissue box', 'charging cable'
]);

const OPEN_VOCAB_PROMPTS = [
  'pen', 'pencil', 'marker', 'notebook', 'mug', 'charger', 'charging cable', 'headphones',
  'camera', 'tissue box', 'tissue', 'candle', 'flower', 'desk fan', 'plant pot'
];

const CANONICAL_LABELS = {
  charger: 'charging cable',
  cable: 'charging cable',
  'usb cable': 'charging cable',
  'charging cord': 'charging cable',
  'desk fan': 'fan',
  flower: 'plant',
  'plant pot': 'potted plant',
  mug: 'cup',
  cellphone: 'cell phone',
  phone: 'cell phone'
};

const DETECTION_CONFIG = {
  modelBase: 'mobilenet_v1',
  detectEveryMs: 80,
  maxNumBoxes: 45,
  minArea: 900,
  minRawScore: 0.12,
  trackKeepAliveMs: 1400,
  maxMisses: 18,
  openVocabEveryMs: 520,
  openVocabThreshold: 0.15,
  detectorMergeIou: 0.56
};

const state = {
  detector: null,
  openVocabDetector: null,
  handLandmarker: null,
  modelsLoadPromise: null,
  lastDetectionAt: 0,
  lastOpenVocabAt: 0,
  tracked: new Map(),
  rawDetections: [],
  nextTrackId: 1,
  fingertips: [],
  hands: [],
  hitCount: 0,
  lastHit: '–',
  insideState: new Map(),
  cooldownByTrack: new Map(),
  holdByPair: new Map(),
  soundMap: JSON.parse(localStorage.getItem('object-drum-map') || '{}'),
  stageTransform: null,
  settings: {
    confidence: Number(confidenceInput.value),
    cooldownMs: Number(cooldownInput.value),
    smoothing: Number(smoothingInput.value),
    stabilityFrames: Number(stabilityInput.value),
    minArea: DETECTION_CONFIG.minArea,
    debugRaw: false
  },
  audio: null,
  running: false,
  fpsSamples: [],
  lastMappingKey: '',
  detectorStatusBySource: { coco: 'idle', openvocab: 'idle' }
};

function normalizeLabel(label) {
  const normalized = String(label || '').toLowerCase().trim().replace(/[_-]+/g, ' ');
  return CANONICAL_LABELS[normalized] || normalized;
}
function isBlockedClass(label) { return BLOCKED_CLASS_ALIASES.has(normalizeLabel(label)); }
function isIndoorPriority(label) { return INDOOR_PRIORITY.has(normalizeLabel(label)); }
function labelThreshold(label) {
  const base = state.settings.confidence;
  return isIndoorPriority(label) ? base : Math.min(0.92, base + 0.06);
}


function updateDetectorSummary() {
  const coco = state.detectorStatusBySource.coco;
  const open = state.detectorStatusBySource.openvocab;
  detectorStatusEl.textContent = `COCO:${coco} / OV:${open}`;
}

function setDetectorSourceStatus(source, status) {
  state.detectorStatusBySource[source] = status;
  updateDetectorSummary();
}

function toUnifiedPrediction({ label, score, bbox, source }) {
  return { class: normalizeLabel(label), score, bbox, source };
}

function dedupePredictions(predictions) {
  const sorted = [...predictions].sort((a, b) => b.score - a.score);
  const kept = [];
  sorted.forEach((p) => {
    const clash = kept.some((k) => {
      if (k.class !== p.class) return false;
      const a = { x: p.bbox[0], y: p.bbox[1], w: p.bbox[2], h: p.bbox[3] };
      const b = { x: k.bbox[0], y: k.bbox[1], w: k.bbox[2], h: k.bbox[3] };
      return iou(a, b) >= DETECTION_CONFIG.detectorMergeIou;
    });
    if (!clash) kept.push(p);
  });
  return kept;
}

async function detectOpenVocab() {
  if (!state.openVocabDetector) return [];
  try {
    const outputs = await state.openVocabDetector(video, OPEN_VOCAB_PROMPTS, {
      threshold: DETECTION_CONFIG.openVocabThreshold,
      percentage: false
    });
    return (outputs || []).map((o) => {
      const box = o.box || {};
      const x = box.xmin ?? box.x ?? 0;
      const y = box.ymin ?? box.y ?? 0;
      const xmax = box.xmax ?? (x + (box.width || 0));
      const ymax = box.ymax ?? (y + (box.height || 0));
      return toUnifiedPrediction({
        label: o.label,
        score: Number(o.score || 0),
        bbox: [x, y, Math.max(0, xmax - x), Math.max(0, ymax - y)],
        source: 'openvocab'
      });
    });
  } catch (err) {
    setDetectorSourceStatus('openvocab', 'error');
    logInit(`Open-vocab detection failed: ${statusText(err)}`);
    return [];
  }
}

function logInit(message) {
  const line = `[${new Date().toLocaleTimeString()}] ${message}`;
  console.info('[init]', message);
  initLogEl.textContent = `${line}\n${initLogEl.textContent}`.split('\n').slice(0, 18).join('\n');
}
function setError(message) { errorDetailsEl.textContent = message || 'No errors.'; }
function statusText(err) { return !err ? 'Unknown error' : `${err.name || 'Error'}: ${err.message || String(err)}`; }
function setComponentStatus(component, text) {
  if (component === 'model') modelStatusEl.textContent = text;
  if (component === 'camera') cameraStatusEl.textContent = text;
  if (component === 'detector') detectorStatusEl.textContent = text;
  if (component === 'hand') handStatusEl.textContent = text;
}

function ensureSecureContext() {
  const isLocalhost = ['localhost', '127.0.0.1', '::1'].includes(location.hostname);
  if (!window.isSecureContext && !isLocalhost) throw new Error(`Webcam requires secure context. Use localhost/https (origin: ${location.origin})`);
}

async function importModule(url, label) {
  try {
    logInit(`Importing ${label}: ${url}`);
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
  const out = state.audio.createGain();
  out.gain.value = 0.9;
  out.connect(state.audio.destination);

  const noiseBuffer = () => {
    const buffer = state.audio.createBuffer(1, state.audio.sampleRate * 0.25, state.audio.sampleRate);
    const ch = buffer.getChannelData(0);
    for (let i = 0; i < ch.length; i++) ch[i] = Math.random() * 2 - 1;
    return buffer;
  };
  const tone = (freq, decay, wave = 'sine', gain = 0.8) => {
    const osc = state.audio.createOscillator();
    const g = state.audio.createGain();
    osc.type = wave;
    osc.frequency.setValueAtTime(freq, now);
    osc.frequency.exponentialRampToValueAtTime(Math.max(50, freq * 0.25), now + decay);
    g.gain.setValueAtTime(gain, now);
    g.gain.exponentialRampToValueAtTime(0.001, now + decay);
    osc.connect(g).connect(out);
    osc.start(now);
    osc.stop(now + decay);
  };
  const noise = (decay, freq, gain = 0.35) => {
    const src = state.audio.createBufferSource();
    src.buffer = noiseBuffer();
    const f = state.audio.createBiquadFilter();
    f.type = 'highpass';
    f.frequency.value = freq;
    const g = state.audio.createGain();
    g.gain.setValueAtTime(gain, now);
    g.gain.exponentialRampToValueAtTime(0.001, now + decay);
    src.connect(f).connect(g).connect(out);
    src.start(now);
    src.stop(now + decay);
  };

  ({
    kick: () => tone(160, 0.25, 'sine', 1),
    snare: () => { noise(0.2, 1200, 0.45); tone(220, 0.12, 'triangle', 0.25); },
    hihat: () => noise(0.08, 4500, 0.22),
    tom: () => tone(210, 0.22, 'triangle', 0.65),
    clap: () => { noise(0.11, 1500, 0.4); setTimeout(() => noise(0.08, 1700, 0.28), 22); },
    cowbell: () => { tone(620, 0.16, 'square', 0.32); tone(840, 0.15, 'square', 0.22); },
    shaker: () => noise(0.05, 3800, 0.18),
    conga: () => tone(280, 0.2, 'sine', 0.68),
    rim: () => { tone(1200, 0.05, 'triangle', 0.2); noise(0.03, 5500, 0.08); }
  }[type] || (() => tone(220, 0.2)))();
}

async function loadModels() {
  if (state.modelsLoadPromise) return state.modelsLoadPromise;
  state.modelsLoadPromise = (async () => {
    setComponentStatus('model', 'Loading');
    setDetectorSourceStatus('coco', 'loading');
    setDetectorSourceStatus('openvocab', 'loading');
    setComponentStatus('hand', 'Loading');
    statusBadge.textContent = 'Loading models…';
    try {
      ensureSecureContext();
      const cocoSsd = globalThis.cocoSsd;
      if (!globalThis.tf) throw new Error('TensorFlow.js global failed to load from CDN script tag.');
      if (!cocoSsd?.load) throw new Error('COCO-SSD global failed to load from CDN script tag.');

      detectorModelEl.textContent = `Hybrid: COCO-SSD (${DETECTION_CONFIG.modelBase}) + OWLViT open-vocab`;
      logInit(`Loading detector model: COCO-SSD (${DETECTION_CONFIG.modelBase})`);
      state.detector = await cocoSsd.load({ base: DETECTION_CONFIG.modelBase });
      setDetectorSourceStatus('coco', 'ready');

      try {
        logInit('Loading open-vocabulary detector (OWLViT)...');
        const transformers = await importModule(URLS.transformers, 'Transformers.js');
        state.openVocabDetector = await transformers.pipeline('zero-shot-object-detection', 'Xenova/owlvit-base-patch32');
        setDetectorSourceStatus('openvocab', 'ready');
      } catch (err) {
        setDetectorSourceStatus('openvocab', 'degraded');
        logInit(`Open-vocab model unavailable; continuing with COCO only. ${statusText(err)}`);
      }

      const mediapipe = await importModule(URLS.mpTasks, 'MediaPipe Tasks Vision');
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

      setComponentStatus('model', 'Ready');
      if (!state.running) statusBadge.textContent = 'Models ready';
      setError('No errors.');
      logInit(`Detector config: conf=${state.settings.confidence}, maxBoxes=${DETECTION_CONFIG.maxNumBoxes}, detectEvery=${DETECTION_CONFIG.detectEveryMs}ms, openVocabEvery=${DETECTION_CONFIG.openVocabEveryMs}ms, minArea=${state.settings.minArea}`);
    } catch (err) {
      setComponentStatus('model', 'Failed');
      if (!state.detector) setDetectorSourceStatus('coco', 'failed');
      if (!state.openVocabDetector) setDetectorSourceStatus('openvocab', 'failed');
      if (!state.handLandmarker) setComponentStatus('hand', 'Failed');
      statusBadge.textContent = 'Model load failed';
      setError(`Model initialization failed. ${statusText(err)}`);
      logInit(`Model load failed: ${statusText(err)}`);
      throw err;
    }
  })();
  return state.modelsLoadPromise;
}

function updateStageTransform() {
  const vw = video.videoWidth || STAGE.width;
  const vh = video.videoHeight || STAGE.height;
  const cw = canvas.width;
  const ch = canvas.height;
  const scale = Math.max(cw / vw, ch / vh);
  const srcW = cw / scale;
  const srcH = ch / scale;
  const srcX = (vw - srcW) / 2;
  const srcY = (vh - srcH) / 2;
  state.stageTransform = { vw, vh, cw, ch, srcX, srcY, srcW, srcH, mirror: true };
}

function videoToStagePoint(vx, vy) {
  const t = state.stageTransform;
  if (!t) return null;
  const nx = (vx - t.srcX) / t.srcW;
  const ny = (vy - t.srcY) / t.srcH;
  const sx = nx * t.cw;
  const sy = ny * t.ch;
  const mx = t.mirror ? t.cw - sx : sx;
  return { x: mx, y: sy };
}

function videoToStageBox(vx, vy, vw, vh) {
  const p1 = videoToStagePoint(vx, vy);
  const p2 = videoToStagePoint(vx + vw, vy + vh);
  if (!p1 || !p2) return null;
  let x = Math.min(p1.x, p2.x);
  let y = Math.min(p1.y, p2.y);
  let w = Math.abs(p2.x - p1.x);
  let h = Math.abs(p2.y - p1.y);

  const x2 = Math.min(canvas.width, x + w);
  const y2 = Math.min(canvas.height, y + h);
  x = Math.max(0, x);
  y = Math.max(0, y);
  w = x2 - x;
  h = y2 - y;
  if (w <= 0 || h <= 0) return null;
  return { x, y, w, h };
}

function iou(a, b) {
  const x1 = Math.max(a.x, b.x);
  const y1 = Math.max(a.y, b.y);
  const x2 = Math.min(a.x + a.w, b.x + b.w);
  const y2 = Math.min(a.y + a.h, b.y + b.h);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  return inter ? inter / (a.w * a.h + b.w * b.h - inter) : 0;
}

function updateTracks(predictions) {
  const now = performance.now();
  const unmatched = new Set(state.tracked.keys());
  const cleaned = [];

  predictions.forEach((p) => {
    const label = normalizeLabel(p.class);
    const [vx, vy, vw, vh] = p.bbox;
    const box = videoToStageBox(vx, vy, vw, vh);
    if (!box) return;
    if (isBlockedClass(label) || p.score < DETECTION_CONFIG.minRawScore || box.w * box.h < state.settings.minArea) return;
    cleaned.push({ ...p, class: label, box, source: p.source || 'coco' });
  });

  state.rawDetections = cleaned;
  rawDetectionsEl.textContent = String(cleaned.length);

  cleaned.forEach((p) => {
    let bestId = null;
    let bestIou = 0;
    for (const [id, t] of state.tracked) {
      if (t.label !== p.class) continue;
      const overlap = iou(p.box, t.box);
      if (overlap > 0.16 && overlap > bestIou) {
        bestIou = overlap;
        bestId = id;
      }
    }

    if (bestId) {
      unmatched.delete(bestId);
      const t = state.tracked.get(bestId);
      const a = state.settings.smoothing;
      t.box = {
        x: t.box.x * (1 - a) + p.box.x * a,
        y: t.box.y * (1 - a) + p.box.y * a,
        w: t.box.w * (1 - a) + p.box.w * a,
        h: t.box.h * (1 - a) + p.box.h * a
      };
      t.score = p.score;
      t.lastSeen = now;
      t.seenFrames += 1;
      t.missFrames = 0;
    } else if (p.score >= labelThreshold(p.class)) {
      const id = String(state.nextTrackId++);
      state.tracked.set(id, {
        id,
        label: p.class,
        score: p.score,
        box: p.box,
        seenFrames: 1,
        missFrames: 0,
        lastSeen: now,
        flashUntil: 0
      });
    }
    assignDefaultSound(p.class);
  });

  unmatched.forEach((id) => {
    const t = state.tracked.get(id);
    t.missFrames += 1;
    const tooLongMissing = now - t.lastSeen > DETECTION_CONFIG.trackKeepAliveMs || t.missFrames > DETECTION_CONFIG.maxMisses;
    if (tooLongMissing) {
      state.tracked.delete(id);
      state.cooldownByTrack.delete(id);
      for (const k of state.insideState.keys()) if (k.endsWith(`|${id}`)) state.insideState.delete(k);
    }
  });

  trackedCountEl.textContent = String(state.tracked.size);
}

function stableTracks() {
  return Array.from(state.tracked.values())
    .filter((t) => t.seenFrames >= state.settings.stabilityFrames)
    .sort((a, b) => Number(isIndoorPriority(b.label)) - Number(isIndoorPriority(a.label)) || b.score - a.score)
    .slice(0, 16);
}

function updateMappingsUi() {
  const labels = Array.from(new Set(stableTracks().map((t) => t.label))).sort();
  const nextKey = labels.join('|');
  const active = document.activeElement;
  const interacting = active && mappingListEl.contains(active) && active.tagName === 'SELECT';
  if (interacting || nextKey === state.lastMappingKey) return;

  state.lastMappingKey = nextKey;
  mappingListEl.innerHTML = '';
  labels.forEach((label) => {
    const row = document.createElement('div');
    row.className = 'mapping-row';
    const span = document.createElement('span');
    span.textContent = label;
    const sel = document.createElement('select');
    SOUND_TYPES.forEach((soundName) => {
      const o = document.createElement('option');
      o.value = soundName;
      o.textContent = soundName;
      if ((state.soundMap[label] || assignDefaultSound(label)) === soundName) o.selected = true;
      sel.appendChild(o);
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
  const active = stableTracks();
  objectListEl.innerHTML = active.length
    ? active.map((t) => `<li><span>${t.label}</span><strong>${Math.round(t.score * 100)}%</strong></li>`).join('')
    : '<li><span>No stable detections yet</span><strong>–</strong></li>';

  rawListEl.innerHTML = state.settings.debugRaw
    ? (state.rawDetections.length
      ? state.rawDetections.slice(0, 20).map((d) => `<li><span>${d.class} <em>(${d.source})</em></span><strong>${Math.round(d.score * 100)}%</strong></li>`).join('')
      : '<li><span>No raw detections</span><strong>–</strong></li>')
    : '<li><span>Debug mode off</span><strong>–</strong></li>';
}

function pointInBox(pt, box) {
  return pt.x >= box.x && pt.x <= box.x + box.w && pt.y >= box.y && pt.y <= box.y + box.h;
}


function handTrackKey(handIndex, trackId) {
  return `${handIndex}|${trackId}`;
}

function handInsideCount(hand, box) {
  let count = 0;
  hand.points.forEach((pt) => {
    if (pointInBox(pt, box)) count += 1;
  });
  return count;
}

function pinchClosed(hand) {
  if (!hand.points[4] || !hand.points[8]) return false;
  const dx = hand.points[4].x - hand.points[8].x;
  const dy = hand.points[4].y - hand.points[8].y;
  const pinch = Math.hypot(dx, dy);
  const handScale = Math.max(24, hand.scale || 24);
  return pinch < handScale * 0.45;
}

function updateHoldState(tracks) {
  const activeKeys = new Set();

  state.hands.forEach((hand, handIndex) => {
    tracks.forEach((track) => {
      const key = handTrackKey(handIndex, track.id);
      activeKeys.add(key);

      const insideCount = handInsideCount(hand, track.box);
      const tipInside = pointInBox(hand.tip, track.box);
      const wristInside = pointInBox(hand.points[0] || hand.tip, track.box);
      const graspLike = insideCount >= 7 && (pinchClosed(hand) || wristInside);
      const pair = state.holdByPair.get(key) || { insideFrames: 0, releaseFrames: 0, suppressed: false, reason: '' };

      if (graspLike || (tipInside && insideCount >= 6)) {
        pair.insideFrames += 1;
        pair.releaseFrames = 0;
      } else {
        pair.releaseFrames += 1;
      }

      if (!pair.suppressed && pair.insideFrames >= 8 && graspLike) {
        pair.suppressed = true;
        pair.reason = 'holding';
      }

      if (pair.suppressed) {
        const released = insideCount <= 2 && !tipInside && pair.releaseFrames >= 5;
        if (released) {
          pair.suppressed = false;
          pair.insideFrames = 0;
          pair.reason = '';
        }
      }

      state.holdByPair.set(key, pair);
    });
  });

  for (const [key, pair] of state.holdByPair.entries()) {
    if (!activeKeys.has(key) && pair.releaseFrames >= 2) state.holdByPair.delete(key);
    else if (!activeKeys.has(key)) pair.releaseFrames += 1;
  }

  const handLabels = state.hands.map((_, idx) => {
    const holdingAny = tracks.some((t) => state.holdByPair.get(handTrackKey(idx, t.id))?.suppressed);
    return holdingAny ? `H${idx + 1}:holding` : `H${idx + 1}:free`;
  });
  handModeEl.textContent = handLabels.length ? handLabels.join(' / ') : 'free/free';
}

function handleHits() {
  const now = performance.now();
  const tracks = stableTracks();
  updateHoldState(tracks);

  state.hands.forEach((hand, idx) => {
    const tip = hand.tip;
    const containing = tracks.filter((t) => pointInBox(tip, t.box));
    containing.sort((a, b) => a.box.w * a.box.h - b.box.w * b.box.h);
    const winner = containing[0];

    tracks.forEach((track) => {
      const key = handTrackKey(idx, track.id);
      const inside = winner?.id === track.id;
      const wasInside = state.insideState.get(key) || false;
      const suppressed = state.holdByPair.get(key)?.suppressed;

      if (inside && !wasInside && !suppressed) {
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

      // Keep inside state while suppressed so release/re-entry is required before re-triggering.
      state.insideState.set(key, inside || Boolean(suppressed && pointInBox(tip, track.box)));
    });
  });

  hitCountEl.textContent = String(state.hitCount);
  lastHitEl.textContent = state.lastHit;
}

function drawFrame() {
  const t = state.stageTransform;
  if (!t) return;
  const w = canvas.width;
  const h = canvas.height;

  if (t.mirror) {
    ctx.save();
    ctx.translate(w, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(video, t.srcX, t.srcY, t.srcW, t.srcH, 0, 0, w, h);
    ctx.restore();
  } else {
    ctx.drawImage(video, t.srcX, t.srcY, t.srcW, t.srcH, 0, 0, w, h);
  }

  stableTracks().forEach((track) => {
    const flash = performance.now() < track.flashUntil;
    ctx.lineWidth = flash ? 4 : 2;
    ctx.strokeStyle = flash ? '#2ee8a6' : (isIndoorPriority(track.label) ? '#84a8ff' : '#6f83aa');
    ctx.fillStyle = flash ? 'rgba(46, 232, 166, .18)' : 'rgba(28, 44, 88, .22)';
    ctx.fillRect(track.box.x, track.box.y, track.box.w, track.box.h);
    ctx.strokeRect(track.box.x, track.box.y, track.box.w, track.box.h);

    const hold = track.missFrames > 0 ? ` • hold ${track.missFrames}` : '';
    const label = `${track.label} ${Math.round(track.score * 100)}%${hold}`;
    ctx.font = '13px Inter, sans-serif';
    const tw = ctx.measureText(label).width;
    ctx.fillStyle = '#0c1220';
    ctx.fillRect(track.box.x, Math.max(0, track.box.y - 20), tw + 12, 18);
    ctx.fillStyle = '#dfe8ff';
    ctx.fillText(label, track.box.x + 6, Math.max(14, track.box.y - 6));
  });

  state.hands.forEach((hand, idx) => {
    const tip = hand.tip;
    ctx.beginPath();
    ctx.arc(tip.x, tip.y, 7, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(255, 104, 165, .95)';
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.strokeStyle = 'rgba(255, 255, 255, .8)';
    ctx.stroke();

    const holding = stableTracks().some((track) => state.holdByPair.get(handTrackKey(idx, track.id))?.suppressed);
    if (holding) {
      ctx.font = '11px Inter, sans-serif';
      ctx.fillStyle = 'rgba(255, 184, 58, .95)';
      ctx.fillText('HOLD', tip.x + 10, tip.y - 10);
    }
  });
}

async function step() {
  if (!state.running) return;
  const now = performance.now();
  try {
    if (now - state.lastDetectionAt > DETECTION_CONFIG.detectEveryMs) {
      const cocoPreds = (await state.detector.detect(video, DETECTION_CONFIG.maxNumBoxes))
        .map((p) => toUnifiedPrediction({ label: p.class, score: p.score, bbox: p.bbox, source: 'coco' }));

      let openVocabPreds = [];
      if (now - state.lastOpenVocabAt > DETECTION_CONFIG.openVocabEveryMs) {
        openVocabPreds = await detectOpenVocab();
        state.lastOpenVocabAt = now;
      }

      const merged = dedupePredictions([...cocoPreds, ...openVocabPreds]);
      updateTracks(merged);
      updateMappingsUi();
      updateObjectList();
      state.lastDetectionAt = now;
    }

    const hands = state.handLandmarker.detectForVideo(video, now);
    state.hands = (hands.landmarks || []).map((landmarks) => {
      const points = landmarks.map((lm) => videoToStagePoint(lm.x * video.videoWidth, lm.y * video.videoHeight) || { x: -9999, y: -9999 });
      const tip = points[8] || { x: -9999, y: -9999 };
      const wrist = points[0] || tip;
      const middleMcp = points[9] || tip;
      const scale = Math.hypot(middleMcp.x - wrist.x, middleMcp.y - wrist.y);
      return { tip, points, scale };
    });
    state.fingertips = state.hands.map((h) => h.tip);

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
  if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') return 'Camera permission denied. Allow webcam access and retry.';
  if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') return 'No camera device found.';
  if (err.name === 'NotReadableError' || err.name === 'TrackStartError') return 'Camera is busy in another app/tab.';
  if (err.name === 'OverconstrainedError') return `Requested camera constraints unsupported (${err.constraint || 'unknown'})`;
  return statusText(err);
}

async function start() {
  setError('No errors.');
  startBtn.disabled = true;
  try {
    ensureSecureContext();
    setComponentStatus('camera', 'Requesting');
    statusBadge.textContent = 'Requesting camera…';
    if (!navigator.mediaDevices?.getUserMedia) throw new Error('getUserMedia is not available in this browser.');

    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1080 }, height: { ideal: 1920 }, facingMode: 'user' },
      audio: false
    });

    video.srcObject = stream;
    await video.play();
    canvas.width = STAGE.width;
    canvas.height = STAGE.height;
    updateStageTransform();
    setComponentStatus('camera', 'Ready');
    logInit(`Camera ready ${video.videoWidth}x${video.videoHeight}; stage ${canvas.width}x${canvas.height} (9:16)`);
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
  logInit('Live processing started with portrait 9:16 stage and quality-tuned detector.');
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
    logInit(`Control updated: conf=${state.settings.confidence}, cooldown=${state.settings.cooldownMs}, smoothing=${state.settings.smoothing}, stability=${state.settings.stabilityFrames}`);
  });
});

debugRawInput.addEventListener('change', () => {
  state.settings.debugRaw = debugRawInput.checked;
  updateObjectList();
  logInit(`Raw detection debug ${state.settings.debugRaw ? 'enabled' : 'disabled'}`);
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
logInit('Booted. Hybrid detector ready (COCO + optional open-vocab). Person/human classes are filtered out.');
loadModels().catch((err) => logInit(`Background model preload failed: ${statusText(err)}`));
