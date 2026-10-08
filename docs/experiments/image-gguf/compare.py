#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Compare generated images: exact identity, PSNR and luminance SSIM.

Metrics describe pixel agreement with a named control; they are not a
perceptual quality score and no acceptance threshold is implied.
Usage: compare.py CANDIDATE.png REFERENCE.png [...pairs]
"""

import hashlib
import json
import sys

import numpy as np
from PIL import Image


def load(path):
    image = Image.open(path)
    return image.convert("RGB"), hashlib.sha256(image.tobytes()).hexdigest(), image.mode


def gaussian_kernel(size=11, sigma=1.5):
    x = np.arange(size) - size // 2
    k = np.exp(-(x ** 2) / (2 * sigma ** 2))
    return k / k.sum()


def blur(a, k):
    pad = len(k) // 2
    a = np.pad(a, pad, mode="reflect")
    a = np.apply_along_axis(lambda r: np.convolve(r, k, mode="valid"), 1, a)
    return np.apply_along_axis(lambda c: np.convolve(c, k, mode="valid"), 0, a)


def ssim(x, y):
    """Wang et al. SSIM on BT.601 luminance, 11x11 Gaussian window, sigma 1.5."""
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    k = gaussian_kernel()
    mx, my = blur(x, k), blur(y, k)
    sxx = blur(x * x, k) - mx * mx
    syy = blur(y * y, k) - my * my
    sxy = blur(x * y, k) - mx * my
    value = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx * mx + my * my + c1) * (sxx + syy + c2))
    return float(value.mean())


def compare(candidate, reference):
    a, ha, ma = load(candidate)
    b, hb, mb = load(reference)
    if a.size != b.size:
        raise ValueError(f"size mismatch {a.size} vs {b.size}")
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    mse = float(np.mean((x - y) ** 2))
    luma = np.array([0.299, 0.587, 0.114])
    return {
        "candidate": candidate, "reference": reference, "size": list(a.size),
        "modes": [ma, mb],
        "identical_pixels": ha == hb and ma == mb,
        "mean_abs_diff": round(float(np.mean(np.abs(x - y))), 4),
        "max_abs_diff": int(np.max(np.abs(x - y))),
        "psnr_db": None if mse == 0 else round(10 * np.log10(255 ** 2 / mse), 3),
        "ssim_luma": round(ssim(x @ luma, y @ luma), 5),
    }


if __name__ == "__main__":
    paths = sys.argv[1:]
    if not paths or len(paths) % 2:
        sys.exit(__doc__)
    print(json.dumps([compare(paths[i], paths[i + 1]) for i in range(0, len(paths), 2)], indent=2))
