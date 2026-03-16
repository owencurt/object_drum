# Object Drum (Local MVP)

A polished local webcam instrument that detects household objects, tracks both index fingertips, and triggers percussion when a fingertip enters an object's tracked box.

## Detection-quality upgrade summary

This app now treats object detection quality as a full pipeline problem, not just a threshold tweak.

### What was changed

- **Detector quality mode:** switched COCO-SSD base from `lite_mobilenet_v2` to **`mobilenet_v1`** for stronger object quality (at some performance cost).
- **Higher camera input quality:** requests up to `1920x1080` ideal webcam resolution for better small/background object recall.
- **Higher detection throughput:** tighter detection cadence and more returned boxes.
- **Two-stage scoring:** low raw acceptance for debug/track updates + higher class-aware creation threshold for stable tracks.
- **Miss-tolerant track persistence:** tracks now hold through temporary misses and age out later, reducing flicker/disappear behavior.
- **Indoor class bias:** indoor/common desk classes are prioritized for track creation and ranking.
- **Debug visibility:** optional raw detection panel plus detector model/config status in UI.

## Why this detector choice

We kept COCO-SSD but upgraded configuration/model base because:

- It is still the most practical no-build browser option in this project.
- `mobilenet_v1` generally gives better detection quality than `lite_mobilenet_v2` for indoor scenes.
- It avoids introducing heavy new build/runtime dependencies while improving real-world recall.

## Run

```bash
npm run dev
```

(or `python3 -m http.server 5173`)

Open:

```text
http://localhost:5173
```

> Use `localhost` / `127.0.0.1` (or HTTPS). Webcam APIs require secure context.

## Key controls

- **Confidence:** base score threshold for creating stable tracks (indoor classes are allowed a bit earlier than non-priority classes).
- **Smoothing:** dampens box jitter.
- **Stability Frames:** number of consecutive updates before a track is treated as stable for UI/hits.
- **Show raw detections:** debug panel to inspect what the detector sees before stable-track filtering.

## Diagnostics in UI

Session panel now shows:

- active detector model
- raw detection count
- tracked object count
- existing camera / detector / hand / model startup statuses

## Class policy

- Person/human classes are fully filtered out from playable object flow.
- See `DETECTABLE_OBJECTS.md` for supported vs approximate vs unsupported classes for indoor/dorm items.

## Practical limitations

- COCO label space still does **not** include some requested objects (e.g., vinyl, tissue box, pen/pencil, candle, headphones).
- Low light, motion blur, and severe occlusion still degrade detection quality.
- `mobilenet_v1` improves quality but may reduce FPS on slower machines.
