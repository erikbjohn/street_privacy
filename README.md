# Placemetry Privacy Blur Pipeline

> Automated face & license-plate blurring for street-level imagery

---

## Overview

This pipeline removes personally identifiable information (PII) from imagery collected by Placemetry's vehicle-mounted camera rigs before images are stored, delivered to municipal clients, or used for model training. It operates as a standalone Python script designed for local batch processing on any machine with a CUDA-capable GPU or CPU.

| Component | Model | Notes |
|---|---|---|
| Face detection | `deepgaze/face-detection` (RetinaFace-based) | Handles partial/occluded faces; GPU-accelerated |
| Plate detection | `nickmuchi/yolos-small-rego-plates-detection` (YOLOS fine-tuned) | North American + international plates; robust to angle & distance |
| Blurring | OpenCV GaussianBlur | Configurable kernel size; runs in-place for memory efficiency |

---

## Installation

Python 3.10+ is required.

```bash
# Install dependencies
pip install torch torchvision transformers pillow opencv-python-headless

# GPU (CUDA 12) — recommended for batch workloads
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Models are downloaded automatically on first run and cached at `~/.cache/huggingface/hub/`. No manual download required.

---

## Quick Start

**Single image**
```bash
python3 placemetry_blur.py \
    --input  /data/raw/property_123.jpg \
    --output /data/clean/property_123.jpg
```

**Batch directory** (preserves sub-directory structure)
```bash
python3 placemetry_blur.py \
    --input  /data/raw/ \
    --output /data/clean/ \
    --json-report /logs/run_$(date +%Y%m%d).json
```

**QA / debug mode** — overlay showing all detected boxes
```bash
python3 placemetry_blur.py \
    --input  /data/raw/sample/ \
    --output /data/qa/ \
    --debug
# Produces both a blurred image and a DEBUG_<name>.jpg overlay
# for visual verification of detection quality.
```

**Plates only with heavy blur** (insurance deliverables)
```bash
python3 placemetry_blur.py \
    --input  /data/raw/ \
    --no-faces \
    --blur-strength 91 \
    --plate-threshold 0.35
```

---

## CLI Reference

| Flag | Default | Description |
|---|---|---|
| `--input / -i` | — | Input image file or directory. Directories are searched recursively. |
| `--output / -o` | `<input>_blurred` | Output file or directory. |
| `--debug` | off | Write a `DEBUG_<name>` overlay image showing raw detection boxes: green = faces, orange = plates. |
| `--face-threshold` | `0.50` | Confidence threshold for face detections (0–1). Lower catches more faces at the cost of more false positives. |
| `--plate-threshold` | `0.45` | Confidence threshold for plate detections (0–1). |
| `--blur-strength` | `51` | Gaussian kernel size (odd integer). Higher = stronger blur. Use 91+ for insurance deliverables. |
| `--no-faces` | off | Disable face detection and blurring entirely. |
| `--no-plates` | off | Disable plate detection and blurring entirely. |
| `--device` | auto | Force `cpu` or `cuda`. Auto-detected by default. |
| `--json-report <path>` | — | Write a JSON stats file on completion. |

---

## Architecture Notes

### Processing flow

Each image follows this path:

1. Load BGR via OpenCV
2. Convert to RGB PIL for HuggingFace inference pipeline
3. Collect bounding boxes from each detector
4. Apply padded Gaussian blur in-place on the BGR array
5. Write result — the original file is never modified

### Model details

**`deepgaze/face-detection`** — RetinaFace architecture fine-tuned for single-shot face detection. The full-range model handles faces up to ~5 m from camera, covering most street-level pedestrian distances. Lower threshold to 0.35 in dense pedestrian scenes.

**`nickmuchi/yolos-small-rego-plates-detection`** — YOLOS transformer model fine-tuned on a diverse plate dataset including US, EU, and AU formats. Works well on plates up to ~40 m; struggles with extreme perspective. Supplement with a Haar cascade fallback for edge cases if needed.

### GPU throughput

| Hardware | Resolution | Throughput (approx.) |
|---|---|---|
| CPU (8-core) | 1920 × 1080 | ~3 img/s |
| NVIDIA RTX 3080 | 1920 × 1080 | ~25 img/s |
| NVIDIA RTX 4090 | 1920 × 1080 | ~55 img/s |

---

## Batch Processing Tips

For large local runs, a few patterns work well.

**Parallel processing across CPU cores**
```bash
# Split a directory into N chunks and process in parallel
ls /data/raw/*.jpg | split -n l/4 - /tmp/chunk_
for f in /tmp/chunk_*; do
    python3 placemetry_blur.py -i "$(cat $f | tr '\n' ' ')" -o /data/clean/ &
done
wait
```

**Watch a folder for new images as they arrive from the rig**
```bash
# Requires: brew install fswatch  (macOS) or apt install inotify-tools (Linux)

# macOS
fswatch -0 /data/incoming/ | xargs -0 -I {} python3 placemetry_blur.py -i {} -o /data/clean/

# Linux
inotifywait -m /data/incoming/ -e close_write |
    while read dir _ file; do
        python3 placemetry_blur.py -i "${dir}${file}" -o /data/clean/
    done
```

**Log runs to a date-stamped report**
```bash
python3 placemetry_blur.py \
    --input  /data/raw/ \
    --output /data/clean/ \
    --json-report "/logs/run_$(date +%Y%m%d_%H%M%S).json"
```

---

## Tuning Tips

| Symptom | Fix |
|---|---|
| Missing distant faces | Lower `--face-threshold` to `0.35`. Run `--debug` on a sample set to check false positive rate. |
| Missing angled plates | Lower `--plate-threshold` to `0.35`. Angled/partially occluded plates are the main failure mode of the YOLOS model. |
| Over-blurring large regions | Raise `--face-threshold` to `0.65+` to reduce false positives on windows and signs. |
| Blur too weak | Raise `--blur-strength`. Values of `91` or `121` are effectively unreadable under any conditions. |
| False positives on signs | High-contrast rectangular shapes can trigger the plate detector. Raise `--plate-threshold` to `0.55`. |
| Slow on CPU | Use `--no-faces` or `--no-plates` if only one redaction type is needed. Face detection is the slower of the two models. |

---

## JSON Report Schema

When `--json-report` is specified, the script writes:

```json
{
  "images":            42,
  "faces":             117,
  "plates":            89,
  "elapsed_seconds":   14.2,
  "images_per_second": 2.96,
  "output_dir":        "/data/clean/",
  "device":            "cuda",
  "models": {
    "face":  "deepgaze/face-detection",
    "plate": "nickmuchi/yolos-small-rego-plates-detection"
  }
}
```

---

*Placemetry — Internal documentation. Not for external distribution.*
