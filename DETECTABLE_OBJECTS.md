# Detectable Objects (Model vs App Policy)

This app uses **TensorFlow.js COCO-SSD** (`@tensorflow-models/coco-ssd`, base `lite_mobilenet_v2`).
The detector can output COCO object categories, while the app applies additional filtering/prioritization rules for interaction quality.

## 1) Model can detect (COCO-SSD class family)

COCO-SSD commonly includes classes such as:

- person
- vehicle/road classes (bicycle, car, bus, train, truck, motorcycle, airplane, boat, traffic light, stop sign, etc.)
- indoor/common classes (book, cell phone, keyboard, mouse, laptop, remote, backpack, bottle, cup, bowl, spoon, scissors, potted plant, vase, chair, couch, dining table, bed, tv, clock, etc.)
- animals and assorted household categories.

## 2) App currently allows as interactive objects

The app does **not** expose every model class equally.

### Fully filtered (never shown / never playable)
- `person`, `people`, `human`, `man`, `woman`, `boy`, `girl`

### Suppressed to reduce clutter
- `bicycle`, `car`, `motorcycle`, `airplane`, `bus`, `train`, `truck`, `boat`, `traffic light`, `fire hydrant`, `stop sign`, `parking meter`

### Prioritized indoor/common classes
Examples include:
- `book`, `cell phone`, `keyboard`, `mouse`, `laptop`, `remote`, `backpack`
- `bottle`, `cup`, `bowl`, `scissors`, `spoon`
- `potted plant`, `vase`, `chair`, `couch`, `bed`, `dining table`, `tv`, `clock`

## Requested dorm/desk items: direct vs approximate vs unsupported

### Directly supported by model class name
- book
- phone (as `cell phone`)
- cup / mug (as `cup`)
- bowl
- bottle
- scissors
- spoon
- keyboard
- mouse
- laptop
- backpack
- remote
- plant / pot (as `potted plant`)

### Approximate only (closest model category)
- notebook (often detected as `book`)
- flower (may appear as `potted plant` or `vase` context)
- desk lamp (no dedicated lamp class; sometimes weakly mapped to nearby household classes)
- camera-style items / polaroid (can be inconsistently detected, often as `cell phone` or not at all)
- fan (no dedicated fan class)

### Not directly supported in COCO label space
- vinyl record
- tissue / tissue box
- pen
- pencil
- headphones
- candle

## Real-world detection quality factors

Results depend on:
- lighting and contrast
- object size in frame
- camera angle and distance
- occlusion and motion blur
- confidence threshold setting

## Tracking/persistence behavior

The app uses a tracked-object layer over raw detections:
- tracks are matched by class + IoU + center distance,
- temporary misses are tolerated,
- boxes are smoothed,
- tracks expire only after sustained loss/edge-leave conditions.

## How to adjust class policy

In `src/main.js`:
- edit `BLOCKED_CLASS_ALIASES` to exclude additional labels,
- edit `SUPPRESSED_CLASSES` for clutter reduction,
- edit `INDOOR_PRIORITY_CLASSES` to bias class ranking/retention,
- edit `DEFAULT_MAP` to tune default sound assignment.
