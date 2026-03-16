# Detectable Objects in Object Drum

This app uses **TensorFlow.js COCO-SSD** (`@tensorflow-models/coco-ssd`, base `lite_mobilenet_v2`).
That means raw detections come from the model's pretrained **COCO object categories** (80 classes).

## Model can detect (COCO-SSD classes)

The COCO-SSD model is expected to detect classes from the COCO label set, including common categories such as:

- person
- bicycle, car, motorcycle, bus, train, truck, boat, airplane
- traffic light, fire hydrant, stop sign, parking meter
- bench, chair, couch, bed, dining table, toilet
- backpack, handbag, suitcase, tie, umbrella
- bottle, wine glass, cup, bowl, fork, knife, spoon
- banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake
- tv, laptop, mouse, remote, keyboard, cell phone
- microwave, oven, toaster, sink, refrigerator
- book, clock, vase, scissors, teddy bear, hair drier, toothbrush
- sports ball, baseball bat, baseball glove, skateboard, surfboard, tennis racket, frisbee, kite
- potted plant, dog, cat, bird, horse, sheep, cow, elephant, bear, zebra, giraffe

> Source basis: the project loads COCO-SSD (`@tensorflow-models/coco-ssd`) and therefore follows its COCO class vocabulary.

## App currently allows as interactive objects

- The app allows all model-detected classes **except person/human-related labels**.
- Specifically, these labels are intentionally filtered out before tracking/render/hit/mapping:
  - `person`, `people`, `human`, `man`, `woman`, `boy`, `girl`

So although the model can detect `person`, the app deliberately does **not** show or use person detections.

## Why results vary in real usage

Detection quality depends on:

- lighting
- camera angle/distance
- occlusion
- object size in frame
- motion blur
- confidence threshold setting in the UI

## How to add or filter classes in the future

In `src/main.js`:

1. Update `BLOCKED_CLASS_ALIASES` to filter additional classes.
2. Update `DEFAULT_MAP` to tune default drum assignments for class names.
3. (Optional) add a whitelist strategy if you only want selected classes to be playable.

Filtering is centralized in `isBlockedClass()` and applied in detection processing and UI list/mapping flows.
