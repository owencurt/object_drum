# Object Drum (Local MVP)

A polished local webcam instrument that detects household objects, tracks fingertips, and triggers percussion when an index fingertip enters a tracked object's box.

## Why this stack

- **Static HTML/CSS/vanilla JS**: zero-build local run with very low setup friction.
- **TensorFlow.js COCO-SSD (`lite_mobilenet_v2`)**: practical pretrained detector for common objects in browser.
- **MediaPipe Hand Landmarker (Tasks Vision)**: reliable two-hand fingertip tracking.
- **Web Audio API synthesis**: low-latency polyphonic percussion without bundled sample files.

## Run

```bash
npm run dev
```

(or `python3 -m http.server 5173`)

Open:

```text
http://localhost:5173
```

> Use `localhost`/`127.0.0.1` (or HTTPS). Webcam APIs require secure context.

## What changed for persistence + recall

- Added a **tracking-by-detection layer** on top of raw model outputs.
- Tracks now survive temporary misses (miss counters + stale timeout) instead of disappearing immediately.
- Track association uses **class + IoU + center-distance matching**.
- Track boxes are smoothed every update to reduce jitter.
- Tracks are only expired after meaningful absence (`maxMisses`, stale age, edge-leave logic).
- Confirmed/visible tracks stay consistent for labels, hit logic, and sound mappings.

## Detection tuning and class prioritization

- Detection cadence tightened (`~90ms`) and max detections increased (`35`) for better recall of smaller/background objects.
- Default camera request favors higher resolution (`1600x900` ideal) to help background object detection.
- Indoor/desk classes are **prioritized** for creation/visibility.
- Outdoor/road clutter classes are **suppressed** (e.g., bike/car/bus/train/etc.) to reduce UI noise.
- Person/human labels remain fully filtered out.

## Diagnostics

The UI shows separate startup status for Camera / Detector / Hand Tracker / Model Bundle and includes an on-screen diagnostics log + explicit startup/runtime errors.

## Class support details

See `DETECTABLE_OBJECTS.md` for:
- model-supported class scope,
- direct vs approximate support for requested dorm/desk items,
- intentionally filtered classes,
- unsupported items and limitations.

## Remaining practical limitations

- COCO-SSD class coverage is fixed to COCO-style categories; unsupported custom labels cannot be detected directly.
- Small/occluded objects remain sensitive to lighting, blur, and framing.
- CDN/network restrictions can block model startup.
