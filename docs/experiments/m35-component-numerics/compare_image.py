# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Compare a completed native RGBA8 export with a pinned diffusers receipt.

python3 compare_image.py NATIVE.rgba8 REFERENCE_DIRECTORY
Uses the registered RGB8 PSNR and image-gguf Gaussian luminance SSIM.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("reference", type=Path)
    args = parser.parse_args()
    receipt = args.reference / "reference.json"
    record = json.loads(receipt.read_text())
    entry = record["files"]["image"]
    # A reference receipt cannot redirect the comparator outside its bundle.
    if Path(entry["file"]).name != entry["file"]:
        raise ValueError("reference image path is not a filename")
    path = args.reference / entry["file"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
        raise ValueError("reference image hash differs")
    image = Image.open(path)
    if list(image.size) != entry["size"]:
        raise ValueError("reference image dimensions differ")
    spec = record["report"]["spec"]
    if image.size != (spec["width"], spec["height"]):
        raise ValueError("reference image and request dimensions differ")
    raw = args.candidate.read_bytes()
    if len(raw) != image.width * image.height * 4:
        raise ValueError("native RGBA8 extent differs")
    a = np.frombuffer(raw, dtype=np.uint8).reshape(image.height, image.width, 4)[..., :3]
    a = a.astype(np.float64)
    b = np.asarray(image.convert("RGB"), dtype=np.float64)
    source = Path(__file__).resolve().parent.parent / "image-gguf/compare.py"
    module_spec = importlib.util.spec_from_file_location("image_compare", source)
    compare = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(compare)
    mse = float(np.mean((a - b) ** 2))
    psnr = float("inf") if mse == 0 else float(10 * np.log10(255 ** 2 / mse))
    luma = np.array([0.299, 0.587, 0.114])
    ssim = float(compare.ssim(a @ luma, b @ luma))
    passed = psnr >= 32.0 and ssim >= 0.98
    print(json.dumps({"reference_receipt_sha256": hashlib.sha256(receipt.read_bytes()).hexdigest(),
                      "candidate_rgba_sha256": hashlib.sha256(raw).hexdigest(),
                      "psnr_db": None if mse == 0 else psnr, "ssim_luma": ssim,
                      "pixels_equal": mse == 0, "pass": bool(passed)}, indent=2))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
