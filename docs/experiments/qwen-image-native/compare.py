# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Judge llmpalooza's Qwen-Image-2.1 outputs against diffusers' (README's bounds).

Runs in the image-reference container (NumPy and Pillow), over the output
directories of llmp_qwen_image_exec and the reference.py directory:

  python3 compare.py REF_DIR [--tokens RUN] [--text RUN] [--dit-first RUN]
                     [--dit-forced RUN] [--vae RUN] [--image RUN] [--size N]

Each RUN is the harness's --out for that check (see README "Reproduce");
the checks given are judged against the pre-registered bounds and the
verdicts printed as JSON. An image run's RGBA pixels are also written as
image.png beside them.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "image-gguf"))
import compare as image_compare  # noqa: E402 (the image-gguf study's SSIM)

BOUNDS = {
    "text": {"rel_rms_max": 0.20, "cosine_min": 0.99},
    "dit_first": {"rel_rms_max": 0.0125, "cosine_min": 0.99996},
    "dit_forced": {"rel_rms_max": 0.046},
    "vae": {"rel_rms_max": 0.0055, "psnr_min": 47.5},
    "image": {"psnr_min": 32.0, "ssim_min": 0.98},
}


def bf16(path, count=None):
    raw = np.fromfile(path, dtype=np.uint16)
    if count is not None and raw.size != count:
        raise SystemExit(f"{path}: {raw.size} values, expected {count}")
    return (raw.astype(np.uint32) << 16).view(np.float32).astype(np.float64)


def tensor_compare(a, b):
    d = a - b
    return {"rel_rms": float(np.sqrt(np.sum(d * d) / np.sum(b * b))),
            "cosine": float(a @ b / np.sqrt((a @ a) * (b @ b))),
            "max_abs": float(np.max(np.abs(d)))}


def image_metrics(candidate, reference):
    a = np.asarray(candidate.convert("RGB"), dtype=np.float64)
    b = np.asarray(reference.convert("RGB"), dtype=np.float64)
    mse = float(np.mean((a - b) ** 2))
    luma = np.array([0.299, 0.587, 0.114])
    return {"psnr_db": None if mse == 0 else round(10 * np.log10(255 ** 2 / mse), 3),
            "ssim_luma": round(image_compare.ssim(a @ luma, b @ luma), 5),
            "mean_abs_diff": round(float(np.mean(np.abs(a - b))), 4),
            "identical_pixels": bool(mse == 0)}


def rgba(run, size):
    raw = np.fromfile(Path(run) / "image.rgba8", dtype=np.uint8)
    if raw.size != size * size * 4:
        raise SystemExit(f"{run}: image.rgba8 is not {size}x{size} RGBA")
    image = Image.fromarray(raw.reshape(size, size, 4), mode="RGBA")
    image.save(Path(run) / "image.png")
    return image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    for name in ("tokens", "text", "dit-first", "dit-forced", "vae", "image"):
        ap.add_argument(f"--{name}")
    ap.add_argument("--size", type=int, default=1024)
    args = ap.parse_args()
    ref = Path(args.ref)
    index = json.loads((ref / "reference.json").read_text())["files"]
    reference_image = Image.open(ref / index["image"]["file"])
    out = {}
    if args.tokens:
        report = json.loads((Path(args.tokens) / "report.json").read_text())
        out["tokens"] = {"match": report.get("tokens_match_reference") is True,
                         "tokens": report["tokens"], "drop": report["drop_tokens"]}
        out["tokens"]["pass"] = out["tokens"]["match"] and report["drop_tokens"] == 14
    if args.text:
        n = int(np.prod(index["prompt_embeds"]["shape"]))
        c = tensor_compare(bf16(Path(args.text) / "prompt_embeds.bf16", n),
                           bf16(ref / index["prompt_embeds"]["file"], n))
        c["pass"] = c["rel_rms"] <= BOUNDS["text"]["rel_rms_max"] and c["cosine"] >= BOUNDS["text"]["cosine_min"]
        out["text"] = c
    if args.dit_first:
        n = 4096 * 64
        c = tensor_compare(bf16(Path(args.dit_first) / "step00_noise_pred.bf16", n),
                           bf16(ref / index["step00_noise_pred"]["file"], n))
        c["pass"] = (c["rel_rms"] <= BOUNDS["dit_first"]["rel_rms_max"]
                     and c["cosine"] >= BOUNDS["dit_first"]["cosine_min"])
        out["dit_first"] = c
    if args.dit_forced:
        rows = []
        for i in range(40):
            n = 4096 * 64
            rows.append(tensor_compare(bf16(Path(args.dit_forced) / f"step{i:02d}_noise_pred.bf16", n),
                                       bf16(ref / index[f"step{i:02d}_noise_pred"]["file"], n)))
        worst = max(r["rel_rms"] for r in rows)
        out["dit_forced"] = {"rel_rms_per_step": [round(r["rel_rms"], 6) for r in rows],
                             "worst_rel_rms": worst, "min_cosine": min(r["cosine"] for r in rows),
                             "pass": worst <= BOUNDS["dit_forced"]["rel_rms_max"]}
    if args.vae:
        n = int(np.prod(index["vae_decoded"]["shape"]))
        c = tensor_compare(bf16(Path(args.vae) / "vae_decoded.bf16", n), bf16(ref / index["vae_decoded"]["file"], n))
        c.update(image_metrics(rgba(args.vae, args.size), reference_image))
        c["pass"] = c["rel_rms"] <= BOUNDS["vae"]["rel_rms_max"] and (
            c["psnr_db"] is None or c["psnr_db"] >= BOUNDS["vae"]["psnr_min"])
        out["vae"] = c
    if args.image:
        c = image_metrics(rgba(args.image, args.size), reference_image)
        c["pass"] = c["identical_pixels"] or (c["psnr_db"] >= BOUNDS["image"]["psnr_min"]
                                              and c["ssim_luma"] >= BOUNDS["image"]["ssim_min"])
        out["image"] = c
    out["bounds"] = BOUNDS
    # NumPy scalars (the verdicts' booleans among them) as Python values.
    print(json.dumps(out, indent=2, default=lambda o: o.item()))


if __name__ == "__main__":
    main()
