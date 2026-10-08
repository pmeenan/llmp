# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Qwen-Image-2.1 BF16 baseline in diffusers: load, per-step and full generation.

Runs inside the image-reference container (../image-reference/Dockerfile) on a
Spark. PROCESS_START (epoch seconds, set just before `docker run`) marks the
process start. The fixed prompt, size, steps and seed are prompts.json's
"image" entry. The first request is instrumented (per-step times, time to
the first denoising step's output); the rest are plain full generations.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time


def available():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable missing")


def main():
    model, output, prompts = map(Path, sys.argv[1:4])
    repeats = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    process_start = float(os.environ.get("PROCESS_START", time.time()))
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    spec = json.loads(prompts.read_text())["image"]
    samples = []
    running = True

    def sample():
        while running:
            samples.append(available())
            time.sleep(0.2)
    baseline = available()
    threading.Thread(target=sample, daemon=True).start()
    import torch
    from diffusers import QwenImage21Pipeline
    report = {"spec": spec, "torch": torch.__version__, "cuda": torch.version.cuda,
              "device": torch.cuda.get_device_name(), "baseline_available_bytes": baseline}
    t0 = time.time()
    pipe = QwenImage21Pipeline.from_pretrained(str(model), torch_dtype=torch.bfloat16,
                                               local_files_only=True, use_safetensors=True)
    t1 = time.time()
    pipe.to("cuda")
    torch.cuda.synchronize()
    t2 = time.time()
    report["load"] = {"imports_s": t0 - process_start, "from_pretrained_s": t1 - t0,
                      "to_cuda_s": t2 - t1, "process_start_to_loaded_s": t2 - process_start}

    def run(instrumented):
        steps = []

        def callback(pipeline, index, timestep, tensors):
            torch.cuda.synchronize()
            steps.append(time.time())
            return tensors
        torch.cuda.synchronize()
        start = time.time()
        result = pipe(prompt=spec["prompt"], width=spec["width"], height=spec["height"],
                      output_resolution=spec["width"], num_inference_steps=spec["steps"],
                      true_cfg_scale=spec["true_cfg_scale"], use_kv_cache=spec["prefix_kv_cache"],
                      generator=torch.Generator("cuda").manual_seed(spec["seed"]),
                      callback_on_step_end=callback if instrumented else None)
        torch.cuda.synchronize()
        end = time.time()
        image = result.images[0]
        row = {"instrumented": instrumented, "generation_s": end - start,
               "pixels_sha256": hashlib.sha256(image.tobytes()).hexdigest()}
        if instrumented:
            row["first_step_output_s"] = steps[0] - start
            row["process_start_to_first_step_output_s"] = steps[0] - process_start
            gaps = [b - a for a, b in zip(steps, steps[1:])]
            row["step_s_median"] = sorted(gaps)[len(gaps) // 2]
            row["step_s_min_max"] = [min(gaps), max(gaps)]
            row["after_last_step_s"] = end - steps[-1]
        return row, image

    runs = []
    first, image = run(True)
    runs.append(first)
    image.save(output / "reference.png")
    for _ in range(repeats):
        row, _ = run(False)
        runs.append(row)
    running = False
    report["runs"] = runs
    report["pixels_identical"] = len({r["pixels_sha256"] for r in runs}) == 1
    report["cuda_max_allocated_bytes"] = torch.cuda.max_memory_allocated()
    report["peak_drop_bytes"] = baseline - min(samples)
    (output / "image.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    os.umask(0o077)
    main()
