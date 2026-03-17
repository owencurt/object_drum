# Detectable Objects in Object Drum

## Previous stack (root cause)

The app originally used only **TensorFlow.js COCO-SSD** (`mobilenet_v1`).

COCO is limited to fixed class labels (80 classes). It includes useful indoor labels like `book`, `cup`, `bottle`, `laptop`, `keyboard`, etc., but **does not include dedicated classes for pen/pencil/marker/headphones/tissue box/candle/charger cable**.

So even with threshold tuning, those items often produced no result because the class space itself was missing.

## New stack (implemented)

A **layered detector architecture** is now used:

1. **COCO-SSD** (fast baseline)
2. **OWL-ViT zero-shot detector** via Transformers.js (targeted open-vocabulary prompts)
3. **Prediction merge + dedupe** before existing tracking/hit pipeline

Open-vocab prompts used:

- `pen`, `pencil`, `marker`, `notebook`, `mug`, `charger`, `charging cable`, `headphones`
- `camera`, `tissue box`, `tissue`, `candle`, `flower`, `desk fan`, `plant pot`

Canonical label normalization maps variants into stable app labels (e.g. `charger` → `charging cable`, `desk fan` → `fan`, `flower` → `plant`, `mug` → `cup`).

## What should now work better

### Strong baseline (COCO + tracking)

- book / notebook (notebook may also come from open-vocab)
- cell phone
- keyboard
- mouse
- laptop
- remote
- bottle
- cup / mug
- bowl
- spoon
- scissors
- backpack
- potted plant / vase

### Improved by open-vocabulary layer

- pen
- pencil
- marker
- notebook (as explicit class, not only book approximation)
- headphones
- charging cable / charger (best-effort)
- tissue / tissue box (best-effort)
- candle (best-effort)
- camera (best-effort)
- fan (best-effort)
- flower/plant variants (normalized to plant/potted plant)

## Still not guaranteed

Even with open-vocab, these remain variable depending on scale/lighting/occlusion/model confidence:

- thin cables at long distance
- very small stationery objects far from camera
- uncommon camera form factors (e.g., polaroid-style)
- highly stylized decor objects

## Practical tuning guidance

If detection is noisy:

- increase confidence slider,
- keep objects larger in frame,
- use raw debug list to inspect whether detections are coming from `coco` or `openvocab`.

If OWL-ViT fails to load, app runs in COCO-only fallback mode.
