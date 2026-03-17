# Object Drum (Local MVP)

A local webcam instrument that detects indoor objects, tracks both index fingertips, and triggers percussion when a fingertip enters an object's tracked box.

## What changed in this upgrade

The detection stack is now **hybrid**:

- **Primary detector:** TensorFlow.js **COCO-SSD** (`mobilenet_v1`) for fast, stable baseline categories.
- **Secondary detector:** Transformers.js **OWL-ViT zero-shot detector** (open-vocabulary prompts) for desk/dorm items COCO commonly misses.
- **Merge strategy:** both detector outputs are normalized, deduplicated (IoU merge), then passed through the existing track persistence + hit pipeline.

This preserves the existing behavior (hand tracking, hold suppression, hit logic, sound mapping, UI), while improving practical detection of objects like **pen/pencil/marker/notebook/headphones/candle/tissue box/charger cable**.

If OWL-ViT fails to load (CDN/model issue), the app automatically degrades to COCO-only mode and keeps running.

## Portrait 9:16 presentation layout

- Main stage remains portrait **9:16** for recording-friendly demos.
- Small windows stack controls below stage.
- Large windows keep stage left and controls right.

## Run

```bash
npm run dev
```

Then open:

```text
http://localhost:5173
```

> Use localhost/127.0.0.1 (or HTTPS). Webcam APIs require secure context.

## Hybrid detector notes

- First run may take longer because OWL-ViT weights download in-browser.
- Session panel now shows detector status as `COCO:<state> / OV:<state>`.
- Raw debug list shows source tag per detection (`coco` or `openvocab`).

## Controls

All existing controls are unchanged:

- Confidence
- Hit cooldown
- Smoothing
- Stability frames
- Raw detection debug toggle
- Per-label sound mapping

## Object coverage and limits

See `DETECTABLE_OBJECTS.md` for:

- why pens/pencils were previously missed,
- exact hybrid model behavior,
- classes now improved,
- still unreliable/unsupported edge cases.
