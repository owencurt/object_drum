
const canvas = document.getElementById('stageCanvas');
const ctx = canvas.getContext('2d');
const video = document.getElementById('webcam');
const startBtn = document.getElementById('startBtn');
const statusBadge = document.getElementById('statusBadge');

const modelStatusEl = document.getElementById('modelStatus');
const hitCountEl = document.getElementById('hitCount');
const lastHitEl = document.getElementById('lastHit');
const fpsEl = document.getElementById('fps');
const objectListEl = document.getElementById('objectList');
const mappingListEl = document.getElementById('mappingList');

const confidenceInput = document.getElementById('confidence');
const cooldownInput = document.getElementById('cooldown');
const smoothingInput = document.getElementById('smoothing');
const stabilityInput = document.getElementById('stability');
const confValue = document.getElementById('confValue');
const cooldownValue = document.getElementById('cooldownValue');
const smoothValue = document.getElementById('smoothValue');
const stableValue = document.getElementById('stableValue');

const SOUND_TYPES = ['kick', 'snare', 'hihat', 'tom', 'clap', 'cowbell', 'shaker', 'conga', 'rim'];
const DEFAULT_MAP = {
  bottle: 'cowbell',
  bowl: 'tom',
  cup: 'hihat',
  scissors: 'clap',
  book: 'snare',
  cellphone: 'rim',
  keyboard: 'kick',
  spoon: 'shaker'
};

const state = {
  detector: null,
  handLandmarker: null,
  lastDetectionAt: 0,
  tracked: new Map(),
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
    stabilityFrames: Number(stabilityInput.value),
    minArea: 2200
  },
  audio: null,
  running: false,
  fpsSamples: []
};

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
    osc.connect(g);
    g.connect(output);
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
  statusBadge.textContent = 'Loading models…';
  await import('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/+esm');
  const cocoSsd = await import('https://cdn.jsdelivr.net/npm/@tensorflow-models/coco-ssd@2.2.3/+esm');
  const mediapipe = await import('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/+esm');
  state.detector = await cocoSsd.load({ base: 'lite_mobilenet_v2' });
  const vision = await mediapipe.FilesetResolver.forVisionTasks('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm');
  state.handLandmarker = await mediapipe.HandLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task' },
    runningMode: 'VIDEO',
    numHands: 2,
    minHandDetectionConfidence: 0.45,
    minHandPresenceConfidence: 0.45,
    minTrackingConfidence: 0.4
  });
  modelStatusEl.textContent = 'Ready';
}

function iou(a, b) {
  const x1 = Math.max(a.x, b.x);
  const y1 = Math.max(a.y, b.y);
  const x2 = Math.min(a.x + a.w, b.x + b.w);
  const y2 = Math.min(a.y + a.h, b.y + b.h);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  if (!inter) return 0;
  return inter / (a.w * a.h + b.w * b.h - inter);
}

function updateTracks(predictions) {
  const now = performance.now();
  const unmatched = new Set(state.tracked.keys());

  predictions.forEach((p) => {
    const [x, y, w, h] = p.bbox;
    if (p.score < state.settings.confidence || w * h < state.settings.minArea) return;

    let bestId = null;
    let bestIou = 0;
    for (const [id, t] of state.tracked) {
      if (t.label !== p.class) continue;
      const overlap = iou({ x, y, w, h }, t.box);
      if (overlap > 0.25 && overlap > bestIou) {
        bestIou = overlap;
        bestId = id;
      }
    }

    if (bestId) {
      unmatched.delete(bestId);
      const t = state.tracked.get(bestId);
      const a = state.settings.smoothing;
      t.box = {
        x: t.box.x * (1 - a) + x * a,
        y: t.box.y * (1 - a) + y * a,
        w: t.box.w * (1 - a) + w * a,
        h: t.box.h * (1 - a) + h * a
      };
      t.score = p.score;
      t.lastSeen = now;
      t.seenFrames += 1;
    } else {
      const id = String(state.nextTrackId++);
      state.tracked.set(id, {
        id,
        label: p.class,
        score: p.score,
        box: { x, y, w, h },
        seenFrames: 1,
        lastSeen: now,
        flashUntil: 0
      });
    }
    assignDefaultSound(p.class);
  });

  unmatched.forEach((id) => {
    const t = state.tracked.get(id);
    if (now - t.lastSeen > 450) {
      state.tracked.delete(id);
      state.cooldownByTrack.delete(id);
      for (const k of state.insideState.keys()) if (k.endsWith(`|${id}`)) state.insideState.delete(k);
    }
  });
}

function updateMappingsUi() {
  const labels = Array.from(new Set(Array.from(state.tracked.values()).map((t) => t.label))).sort();
  mappingListEl.innerHTML = '';
  labels.forEach((label) => {
    const row = document.createElement('div');
    row.className = 'mapping-row';
    const span = document.createElement('span');
    span.textContent = label;
    const sel = document.createElement('select');
    SOUND_TYPES.forEach((s) => {
      const o = document.createElement('option');
      o.value = s;
      o.textContent = s;
      if ((state.soundMap[label] || assignDefaultSound(label)) === s) o.selected = true;
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
  const active = Array.from(state.tracked.values())
    .filter((t) => t.seenFrames >= state.settings.stabilityFrames)
    .sort((a, b) => b.score - a.score);
  objectListEl.innerHTML = active.length
    ? active.map((t) => `<li><span>${t.label}</span><strong>${Math.round(t.score * 100)}%</strong></li>`).join('')
    : '<li><span>No stable detections yet</span><strong>–</strong></li>';
}

function pointInBox(pt, box) {
  return pt.x >= box.x && pt.x <= box.x + box.w && pt.y >= box.y && pt.y <= box.y + box.h;
}

function handleHits() {
  const now = performance.now();
  const stableTracks = Array.from(state.tracked.values()).filter((t) => t.seenFrames >= state.settings.stabilityFrames);

  state.fingertips.forEach((tip, idx) => {
    const containing = stableTracks.filter((t) => pointInBox(tip, t.box));
    // Overlap strategy: pick the smallest containing box to prefer deliberate small targets.
    containing.sort((a, b) => a.box.w * a.box.h - b.box.w * b.box.h);
    const winner = containing[0];

    stableTracks.forEach((track) => {
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

  const stableTracks = Array.from(state.tracked.values()).filter((t) => t.seenFrames >= state.settings.stabilityFrames);
  stableTracks.forEach((t) => {
    const x = w - (t.box.x + t.box.w);
    const flash = performance.now() < t.flashUntil;
    ctx.lineWidth = flash ? 4 : 2;
    ctx.strokeStyle = flash ? '#2ee8a6' : '#80a4ff';
    ctx.fillStyle = flash ? 'rgba(46, 232, 166, .18)' : 'rgba(28, 44, 88, .22)';
    ctx.fillRect(x, t.box.y, t.box.w, t.box.h);
    ctx.strokeRect(x, t.box.y, t.box.w, t.box.h);

    const label = `${t.label} ${Math.round(t.score * 100)}%`;
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

  if (now - state.lastDetectionAt > 120) {
    const preds = await state.detector.detect(video, 20);
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
}

async function start() {
  try {
    startBtn.disabled = true;
    statusBadge.textContent = 'Requesting camera…';
    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720, facingMode: 'user' }, audio: false });
    video.srcObject = stream;
    await video.play();

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    if (!state.detector || !state.handLandmarker) await loadModels();

    state.running = true;
    statusBadge.textContent = 'Live';
    step();
  } catch (err) {
    console.error(err);
    statusBadge.textContent = 'Camera/Model error';
    startBtn.disabled = false;
  }
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

startBtn.addEventListener('click', start);
loadModels().catch((e) => {
  console.error(e);
  modelStatusEl.textContent = 'Failed';
  statusBadge.textContent = 'Model load failed';
});
