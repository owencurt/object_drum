# Object Drum (Local MVP)

A polished local webcam instrument that detects household objects + tracks both hands and triggers percussion when your index fingertip *enters* an object's box.

## Why this stack

- **Static HTML/CSS/vanilla JS**: zero-build local run with very low setup friction.
- **TensorFlow.js COCO-SSD (`lite_mobilenet_v2`)**: practical, pretrained common-object detector that runs in-browser locally with no server.
- **MediaPipe Hand Landmarker (Tasks Vision)**: reliable 2-hand fingertip tracking with good real-time performance.
- **Web Audio API synthesis**: no audio assets needed, polyphonic percussion playback, low latency.

This is optimized for a local demo on a MacBook, with minimal setup friction.

## Features

- Live webcam canvas with dark polished UI.
- Real-time object boxes with labels/confidence.
- Both hands tracked; index fingertip markers rendered.
- Hit detection on **entry** only (`outside -> inside`).
- Per-object cooldown to avoid rapid retriggering.
- Track smoothing + stability gating to reduce flicker/jitter hits.
- Overlap policy: fingertip chooses the **smallest containing box**.
- Box flash feedback on hit.
- Side panel with:
  - stats (hits, last hit, FPS, model status)
  - sliders (confidence, cooldown, smoothing, stability)
  - active detection list
  - editable sound mapping dropdown per detected class
- Auto-default sound assignment for unseen classes.
- Mapping persistence in `localStorage`.

## Run (no install needed)

Option A (npm script wrapper):

```bash
npm run dev
```

Option B (direct):

```bash
python3 -m http.server 5173
```

Then open `http://localhost:5173`.

## Interaction behavior details

- A hit triggers when an index fingertip enters a stable object track's box.
- One hit per fingertip entry.
- Cooldown applies per object track.
- Tiny detections are ignored (`minArea` guard in app state).
- Sound selection depends only on object class mapping.
- Multiple simultaneous hits can overlap (polyphony).

## Practical limitations

- COCO-SSD class coverage is limited to pretrained COCO categories.
- Fast movement/lighting issues can still cause detection jitter.
- Browser performance varies by hardware/browser.
- First load downloads model assets from CDN.

## Next-step improvements

- Add optional local backend for model caching/offline startup.
- Add velocity-sensitive hit dynamics.
- Add calibration zones / manual object lock.
- Add custom sample packs and MIDI output.
- Add adaptive detection throttling by measured FPS.
