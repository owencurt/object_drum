# Object Drum (Local MVP)

A local webcam instrument that detects household objects, tracks both index fingertips, and triggers percussion when a fingertip enters an object's tracked box.

## Portrait 9:16 presentation layout

The app is now intentionally framed for **vertical recording**:

- The main stage is a portrait **9:16 camera viewport**.
- On smaller windows, controls stack below the stage.
- On larger desktop windows, the portrait stage stays left and controls stay on the right.
- The visual style, overlays, hit flashes, and sound interaction are preserved.

This keeps demos easy to screen-record and repost to phone-first/social formats.

## Detector quality approach

The app keeps COCO-SSD but uses a stronger quality-oriented setup:

- `mobilenet_v1` base (better recall than the lighter base)
- tighter inference cadence
- higher max detections
- miss-tolerant object track persistence
- indoor-priority class bias
- optional raw detection debug list

Person/human detections remain filtered out from playable interaction.

## Run

```bash
npm run dev
```

(or `python3 -m http.server 5173`)

Then open:

```text
http://localhost:5173
```

> Use localhost/127.0.0.1 (or HTTPS). Webcam APIs require secure context.

---

## Controls explained (what each one does)

### 1) Confidence

- **What it controls:** base score required before a raw detection is promoted to a stable playable track.
- **Higher value:** fewer detections, cleaner results, more missed small/far objects.
- **Lower value:** more detections and recall, but more false positives/noise.
- **Effect on experience:** directly affects how often objects appear in tracking, mapping list, and hit targets.

If you want **more detections**, lower confidence gradually.
If you want **fewer false positives**, raise confidence.

### 2) Hit Cooldown (ms)

- **What it controls:** minimum time before the same tracked object can retrigger another hit.
- **Higher value:** fewer accidental rapid-fire retriggers while finger lingers.
- **Lower value:** more responsive repeated hits, but easier to over-trigger.
- **Effect on experience:** changes drum “playability” feel, not detector quality.

If you get **too many accidental hits**, increase cooldown.
If hits feel **too slow**, decrease cooldown.

### 3) Smoothing

- **What it controls:** how quickly track boxes adapt to new detection positions.
- **Higher smoothing factor:** boxes follow motion faster but may jitter more.
- **Lower smoothing factor:** boxes look steadier but lag behind movement.
- **Effect on experience:** visual stability and hit alignment feel.

If boxes are **jittery**, reduce smoothing a bit.
If boxes feel **too sluggish**, increase smoothing.

### 4) Stability Frames

- **What it controls:** number of consecutive updates before a track is treated as stable/interactive.
- **Higher value:** fewer flickers/false starts, but slower object activation.
- **Lower value:** quicker activation, but noisier object presence.
- **Effect on experience:** balance between responsiveness and stability.

If objects appear/disappear too quickly, increase stability frames.
If objects take too long to become playable, decrease it.

### 5) Show raw detections (debug toggle)

- **What it controls:** whether to show the raw detection list before stable-track filtering.
- **Why useful:** helps diagnose whether misses come from model output or from post-processing/tracking thresholds.
- **Effect on experience:** no sound/hit logic change; diagnostic visibility only.

Use this when tuning confidence/stability to understand pipeline behavior.

### 6) Sound mapping dropdowns

- **What it controls:** per-object-class sound assignment.
- **Effect on experience:** does not change detection/tracking; only changes sound output when hit triggers occur.

---

## Reading the status/debug panel

Session stats include:

- **Detector Model**: confirms active detector variant.
- **Raw Detections**: number of model detections that passed minimal filtering.
- **Tracked**: number of currently persisted tracks.
- **Camera/Detector/Hand/Model statuses**: startup health per component.

If raw detections are high but tracked is low, tuning is likely too strict.
If raw detections are low, lighting/framing/model coverage is likely the main issue.


## Hold suppression (anti self-trigger)

When a hand appears to be **grasping** an overlapped object (many landmarks inside the box + pinch/hand-shape cues sustained over frames), that hand/object pair is temporarily suppressed for hit-triggering.

- This prevents a held object from repeatedly self-triggering.
- The suppression is **per hand + per object**, so the other hand can still hit normally.
- The pair rearms after release (hand landmarks leave the object area for multiple frames).

In-session debug:
- **Hand Mode** shows whether each hand is currently free or holding.
- `HOLD` appears near fingertip markers when suppression is active for that hand.

## Recording recommendations

- Keep browser window narrow enough that the portrait stage is dominant.
- Ensure the full portrait frame is visible when screen recording.
- Use consistent indoor lighting and avoid heavy motion blur.
- Frame objects so they fill more of the portrait viewport for better recall.

## Object coverage / limitations

See `DETECTABLE_OBJECTS.md` for:

- directly supported indoor classes,
- approximate mappings,
- unsupported requested items (e.g., candle, vinyl, tissue box, pen/pencil, headphones),
- where to tune policy in code.
