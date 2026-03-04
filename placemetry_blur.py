#!/usr/bin/env python3
"""
Placemetry Privacy Blur Pipeline
=====================================
Removes PII from street-level and aerial imagery before storage/delivery.

Detection stack:
  - Faces  : HuggingFace deepgaze/face-detection (RetinaFace-based, handles
              partial/occluded faces well in street-view imagery)
  - Plates : HuggingFace nickmuchi/yolos-small-rego-plates-detection
             (YOLOS fine-tuned on North American & international plates)

Both models auto-download from HuggingFace Hub on first run and are
cached in ~/.cache/huggingface/.

Usage:
    # Single image
    python3 placemetry_blur.py -i photo.jpg -o photo_clean.jpg

    # Batch a whole directory (preserves sub-directory structure)
    python3 placemetry_blur.py -i /data/raw/ -o /data/blurred/

    # QA mode: save DEBUG_ overlay showing bounding boxes
    python3 placemetry_blur.py -i /data/raw/ -o /data/blurred/ --debug

    # Only blur plates, extra strong blur
    python3 placemetry_blur.py -i input.jpg --no-faces --blur-strength 91

    # Write a JSON run report
    python3 placemetry_blur.py -i /data/raw/ --json-report run_stats.json

Requirements:
    pip install torch torchvision transformers pillow opencv-python-headless

    GPU is used automatically if available; falls back to CPU.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
from transformers import (
    AutoFeatureExtractor,
    AutoModelForObjectDetection,
    pipeline,
)

# ─── Constants ────────────────────────────────────────────────────────────────

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

# HuggingFace model IDs
FACE_MODEL_ID   = "deepgaze/face-detection"          # RetinaFace-based
PLATE_MODEL_ID  = "nickmuchi/yolos-small-rego-plates-detection"

# Runtime config dict — modified by CLI args before detectors are created
CFG = {
    "face_threshold":  0.50,   # confidence threshold for face detections
    "plate_threshold": 0.45,   # confidence threshold for plate detections
    "blur_kernel":     51,     # Gaussian blur kernel (must be odd; higher = stronger)
    "padding_face":    0.12,   # fractional padding added around face boxes
    "padding_plate":   0.06,   # fractional padding around plate boxes
    "device":          "cuda" if torch.cuda.is_available() else "cpu",
}


# ─── Geometry helpers ─────────────────────────────────────────────────────────

def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def pad_box(x, y, w, h, pad_frac, img_w, img_h):
    """Expand a (x,y,w,h) box by pad_frac, clamped to image bounds."""
    pw, ph = int(w * pad_frac), int(h * pad_frac)
    return (
        clamp(x - pw, 0, img_w),
        clamp(y - ph, 0, img_h),
        clamp(x + w + pw, 0, img_w),
        clamp(y + h + ph, 0, img_h),
    )


def blur_region(img: np.ndarray, x1, y1, x2, y2) -> None:
    """Apply Gaussian blur to a rectangular region in-place."""
    if x2 <= x1 or y2 <= y1:
        return
    k = CFG["blur_kernel"]
    k = k if k % 2 == 1 else k + 1
    img[y1:y2, x1:x2] = cv2.GaussianBlur(img[y1:y2, x1:x2], (k, k), 0)


def draw_box(img, x1, y1, x2, y2, label, color):
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(img, label, (x1, max(y1 - 6, 14)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


# ─── Detectors ────────────────────────────────────────────────────────────────

class HFObjectDetector:
    """
    Generic HuggingFace object-detection pipeline wrapper.
    Returns a list of (x, y, w, h) pixel boxes for the target label(s).
    """

    def __init__(self, model_id: str, target_labels: list[str], threshold: float):
        print(f"  Loading {model_id} on {CFG['device']}...")
        self._pipe = pipeline(
            "object-detection",
            model=model_id,
            device=0 if CFG["device"] == "cuda" else -1,
            threshold=threshold,
        )
        self._labels = {lbl.lower() for lbl in target_labels}

    def detect(self, img_bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Run detection; return (x, y, w, h) boxes."""
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        results = self._pipe(pil_img)

        boxes = []
        for det in results:
            label = det["label"].lower()
            # Accept any result if target_labels is empty (catch-all)
            if self._labels and not any(lbl in label for lbl in self._labels):
                continue
            b = det["box"]
            x = int(b["xmin"])
            y = int(b["ymin"])
            w = int(b["xmax"]) - x
            h = int(b["ymax"]) - y
            boxes.append((x, y, w, h))
        return boxes


class FaceDetector(HFObjectDetector):
    def __init__(self):
        super().__init__(
            model_id=FACE_MODEL_ID,
            target_labels=["face"],
            threshold=CFG["face_threshold"],
        )

    def detect(self, img_bgr):
        return super().detect(img_bgr)


class PlateDetector(HFObjectDetector):
    """
    YOLOS-based license plate detector.
    The model labels plates as 'licence-plate' (British spelling).
    We cast a wide net on label matching so it works even if label names change.
    """

    def __init__(self):
        super().__init__(
            model_id=PLATE_MODEL_ID,
            target_labels=["licence", "license", "plate", "rego"],
            threshold=CFG["plate_threshold"],
        )

    def detect(self, img_bgr):
        return super().detect(img_bgr)


# ─── Per-image processing ─────────────────────────────────────────────────────

def process_image(
    img_path: Path,
    out_path: Path,
    face_det,
    plate_det,
    debug: bool = False,
    stats: dict | None = None,
) -> bool:
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  [SKIP] Cannot read: {img_path}")
        return False

    h, w = img.shape[:2]
    result = img.copy()
    dbg = img.copy() if debug else None

    face_boxes  = face_det.detect(img)  if face_det  else []
    plate_boxes = plate_det.detect(img) if plate_det else []

    for (x, y, bw, bh) in face_boxes:
        x1, y1, x2, y2 = pad_box(x, y, bw, bh, CFG["padding_face"], w, h)
        blur_region(result, x1, y1, x2, y2)
        if debug and dbg is not None:
            draw_box(dbg, x1, y1, x2, y2, "face", (0, 220, 0))

    for (x, y, bw, bh) in plate_boxes:
        x1, y1, x2, y2 = pad_box(x, y, bw, bh, CFG["padding_plate"], w, h)
        blur_region(result, x1, y1, x2, y2)
        if debug and dbg is not None:
            draw_box(dbg, x1, y1, x2, y2, "plate", (0, 100, 255))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), result)

    if debug and dbg is not None:
        cv2.imwrite(str(out_path.parent / ("DEBUG_" + out_path.name)), dbg)

    nf, np_ = len(face_boxes), len(plate_boxes)
    print(f"  + {str(img_path.name):<48} faces={nf}  plates={np_}")

    if stats is not None:
        stats["images"] += 1
        stats["faces"]  += nf
        stats["plates"] += np_
    return True


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Placemetry Privacy Blur — face & license plate removal",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input",  "-i", required=True,
                        help="Input image file or directory")
    parser.add_argument("--output", "-o", default=None,
                        help="Output file or directory")
    parser.add_argument("--debug", action="store_true",
                        help="Write DEBUG_ overlay images showing detection boxes")
    parser.add_argument("--face-threshold",  type=float, default=0.50,
                        help="Face detection confidence threshold (0–1)")
    parser.add_argument("--plate-threshold", type=float, default=0.45,
                        help="Plate detection confidence threshold (0–1)")
    parser.add_argument("--blur-strength", type=int, default=51,
                        help="Gaussian blur kernel size (odd int; 91+ = heavy blur)")
    parser.add_argument("--no-faces",  action="store_true", help="Skip face blurring")
    parser.add_argument("--no-plates", action="store_true", help="Skip plate blurring")
    parser.add_argument("--device", default=None,
                        help="Force device: 'cpu' or 'cuda' (auto-detected by default)")
    parser.add_argument("--json-report", default=None,
                        help="Write JSON stats report to this path")
    args = parser.parse_args()

    # Apply CLI settings
    CFG["face_threshold"]  = args.face_threshold
    CFG["plate_threshold"] = args.plate_threshold
    k = args.blur_strength
    CFG["blur_kernel"] = k if k % 2 == 1 else k + 1
    if args.device:
        CFG["device"] = args.device

    input_path = Path(args.input)

    if input_path.is_file():
        images = [input_path]
        default_out = input_path.parent / (input_path.stem + "_blurred" + input_path.suffix)
    elif input_path.is_dir():
        images = sorted(
            p for p in input_path.rglob("*") if p.suffix.lower() in SUPPORTED_EXTS
        )
        default_out = input_path / "blurred"
    else:
        print(f"[ERROR] Input not found: {input_path}")
        sys.exit(1)

    if not images:
        print("[ERROR] No supported images found.")
        sys.exit(1)

    out_arg = Path(args.output) if args.output else default_out

    print("\nPlacemetry Privacy Blur")
    print(f"  Input    : {input_path}")
    print(f"  Output   : {out_arg}")
    print(f"  Images   : {len(images)}")
    print(f"  Device   : {CFG['device']}")
    face_str = 'OFF' if args.no_faces else f'ON  (threshold={CFG["face_threshold"]})'
    print(f'  Faces    : {face_str}')
    plate_str = 'OFF' if args.no_plates else f'ON  (threshold={CFG["plate_threshold"]})'
    print(f'  Plates   : {plate_str}')
    print(f"  Blur k   : {CFG['blur_kernel']}")
    print(f"  Debug    : {'ON' if args.debug else 'OFF'}")
    print()

    print("Loading models...")
    face_det  = None if args.no_faces  else FaceDetector()
    plate_det = None if args.no_plates else PlateDetector()
    print()

    stats = {"images": 0, "faces": 0, "plates": 0}
    t0 = time.time()

    for img_path in images:
        out_path = out_arg if input_path.is_file() else out_arg / img_path.relative_to(input_path)
        process_image(img_path, out_path, face_det, plate_det,
                      debug=args.debug, stats=stats)

    elapsed = time.time() - t0
    rate = stats["images"] / elapsed if elapsed > 0 else 0

    print(f"\nDone in {elapsed:.1f}s  ({rate:.1f} img/s)")
    print(f"  Images processed : {stats['images']}")
    print(f"  Faces blurred    : {stats['faces']}")
    print(f"  Plates blurred   : {stats['plates']}")

    if args.json_report:
        stats.update({
            "elapsed_seconds": round(elapsed, 2),
            "images_per_second": round(rate, 2),
            "output_dir": str(out_arg),
            "device": CFG["device"],
            "models": {
                "face":  FACE_MODEL_ID  if not args.no_faces  else None,
                "plate": PLATE_MODEL_ID if not args.no_plates else None,
            },
        })
        Path(args.json_report).write_text(json.dumps(stats, indent=2))
        print(f"  JSON report      : {args.json_report}")


if __name__ == "__main__":
    main()
