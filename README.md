# Object Drum (Local MVP)

A polished local webcam instrument that detects household objects + tracks both hands and triggers percussion when your index fingertip *enters* an object's box.

## Why this stack

- **Static HTML/CSS/vanilla JS**: zero-build local run with very low setup friction.
- **TensorFlow.js COCO-SSD (`lite_mobilenet_v2`)**: practical, pretrained common-object detector that runs in-browser locally with no server.
- **MediaPipe Hand Landmarker (Tasks Vision)**: reliable 2-hand fingertip tracking with good real-time performance.
- **Web Audio API synthesis**: no audio assets needed, polyphonic percussion playback, low latency.

## Run

```bash
npm run dev
```

(or `python3 -m http.server 5173`)

Then open:

```text
http://localhost:5173
```

> Use `localhost` (or `127.0.0.1`) — not `file://` and not arbitrary insecure origins — because webcam access requires a secure context.

## Startup diagnostics now included

The UI now shows separate startup statuses for:

- Camera
- Object Detector
- Hand Tracker
- Overall Model Bundle

And a diagnostics area that reports exact errors, including:

- permission denied / camera missing / camera busy messages
- exact CDN import URL failures
- exact model resource URL failures
- runtime-loop failures

Initialization steps are also logged in-browser console with `[init]` markers.

## Interaction behavior details

- A hit triggers when an index fingertip enters a stable object track's box.
- One hit per fingertip entry.
- Cooldown applies per object track.
- Tiny detections are ignored (`minArea` guard in app state).
- Sound selection depends only on object class mapping.
- Multiple simultaneous hits can overlap (polyphony).

## Notes

- `favicon.ico` 404s were harmless and unrelated to model/camera startup; a `favicon.svg` is now included to remove this server-log noise.
- First model load downloads model assets from CDN.

## Practical limitations

- COCO-SSD class coverage is limited to pretrained COCO categories.
- Fast movement/lighting issues can still cause detection jitter.
- Browser performance varies by hardware/browser.
- Network/firewall rules that block CDN URLs will prevent model startup (now explicitly reported in diagnostics).
