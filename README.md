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


## Open-vocabulary model loading behavior

The OV model (`Xenova/owlvit-base-patch32`) now loads with this strategy:

1. Probe local path: `/models/Xenova/owlvit-base-patch32/config.json`
2. If present, try loading locally first.
3. If local assets are missing/broken, automatically retry from remote Hugging Face.
4. If both fail, app degrades to COCO-only mode.

This makes the default `npm run dev` static-server workflow work even when no local `/models/...` folder exists.

If you want local OV assets, place the full exported model under:

- `/models/Xenova/owlvit-base-patch32/`

Required files include config/tokenizer/preprocessor and ONNX weights.

## Hybrid detector notes

- First run may take longer because OWL-ViT weights download in-browser.
- Session panel now shows detector status as `COCO:<state> / OV:<state(detail)>` for clearer OV failure reasons.
- Diagnostics log includes exact OV model source/probe path and fallback attempts.
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
