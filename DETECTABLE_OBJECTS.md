# Detectable Objects in Object Drum

This app uses **TensorFlow.js COCO-SSD** with base **`mobilenet_v1`**.

## 1) Detector fit assessment (for this project)

### Why prior quality was weak

In the previous setup, detection quality was limited by a combination of:

- lighter detector base (`lite_mobilenet_v2`) with weaker recall on cluttered indoor scenes
- high default confidence for stable-track creation
- modest camera input constraints
- short track persistence window after misses

This caused missed small/background items and unstable object presence.

### What changed now

- switched detector base to `mobilenet_v1`
- increased ideal camera resolution request
- increased detection frequency and max returned boxes
- added miss-tolerant track persistence
- added indoor-priority thresholding and ranking
- added optional raw-detection debug list

## 2) Model class coverage vs app behavior

COCO-SSD can detect COCO-family categories (80 classes), including many indoor-relevant classes such as:

- book
- cell phone
- keyboard
- mouse
- laptop
- remote
- backpack
- bottle
- cup
- bowl
- spoon
- scissors
- potted plant
- vase

The app still filters person/human classes from playable object flow.

## 3) Requested objects: direct vs approximate vs unsupported

### Directly represented by COCO class names

- books
- phone (`cell phone`)
- cup / mug (`cup`)
- bowl
- bottle
- scissors
- spoon
- keyboard
- mouse
- laptop
- backpack
- remote
- plant pot / pot (`potted plant`)

### Approximate only

- notebook (often appears as `book`)
- flower (often via `potted plant` or `vase` context)
- polaroid/camera-like items (may be inconsistent)
- desk lamp (no dedicated lamp class)
- fan (no dedicated fan class)

### Not reliably supported in COCO label space

- candle
- vinyl records
- tissue / tissue box
- pen
- pencil
- headphones

## 4) Real-world quality factors

Detection reliability still depends on:

- lighting and contrast
- distance/size in frame
- camera angle
- occlusion
- motion blur
- chosen confidence setting

## 5) Tuning locations in code

Primary detection-quality settings are in `src/main.js`:

- `DETECTION_CONFIG`
- `INDOOR_PRIORITY`
- `labelThreshold()`
- track-expiry handling in `updateTracks()`
