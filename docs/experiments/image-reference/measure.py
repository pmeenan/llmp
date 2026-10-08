# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Instrument a pinned, local Qwen Image pipeline. Raw output stays external."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time


def main(argv=None, retained=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--cancel-after", type=int)
    parser.add_argument("--plain", action="store_true")
    parser.add_argument("--edit", action="store_true")
    parser.add_argument("--release-phases", action="store_true")
    args = parser.parse_args(argv)
    if args.steps < 1 or args.size < 64 or args.size % 32:
        parser.error("Positive steps and a size divisible by 32 are required")
    if args.cancel_after is not None and not 1 <= args.cancel_after < args.steps:
        parser.error("Cancellation must precede the last step")
    if args.release_phases and (args.plain or args.edit or args.cancel_after):
        parser.error("Phase release is a separate instrumented text-to-image case")
    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    import torch
    from diffusers import QwenImage21Pipeline
    from fetch import verify_directory
    pins = json.loads(Path(__file__).with_name("pins.json").read_text())
    if not retained or "pipe" not in retained:
        verify_directory(args.model, pins)
    torch.set_num_threads(8)
    records = []
    def snapshot():
        mem = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
        io = dict(line.split(":", 1) for line in Path("/proc/self/io").read_text().splitlines())
        return {"available_bytes": int(mem["MemAvailable"].split()[0]) * 1024,
                "cuda_allocated": torch.cuda.memory_allocated(),
                "cuda_reserved": torch.cuda.memory_reserved(),
                "cuda_peak_allocated": torch.cuda.max_memory_allocated(),
                "cuda_peak_reserved": torch.cuda.max_memory_reserved(),
                "process_read_bytes": int(io["read_bytes"]),
                "process_write_bytes": int(io["write_bytes"])}
    before = snapshot()
    start = time.monotonic()
    reused = retained is not None and "pipe" in retained
    if reused:
        pipe = retained["pipe"]
        cpu_load_seconds = transfer_seconds = load_seconds = 0.0
    else:
        pipe = QwenImage21Pipeline.from_pretrained(
            str(args.model), torch_dtype=torch.bfloat16, local_files_only=True,
            use_safetensors=True)
        cpu_load_seconds = time.monotonic() - start
        print(json.dumps({"cpu_load_seconds": cpu_load_seconds}), flush=True)
        transfer_start = time.monotonic()
        pipe.to("cuda")
        torch.cuda.synchronize()
        transfer_seconds = time.monotonic() - transfer_start
        load_seconds = time.monotonic() - start
        if retained is not None:
            retained["pipe"] = pipe
    print(json.dumps({"load_seconds": load_seconds, "transfer_seconds": transfer_seconds}), flush=True)
    components = {}
    for name in ("text_encoder", "transformer", "vae"):
        module = getattr(pipe, name)
        components[name] = {
            "parameters": sum(p.numel() for p in module.parameters()),
            "parameter_bytes": sum(p.numel() * p.element_size() for p in module.parameters()),
            "buffer_bytes": sum(p.numel() * p.element_size() for p in module.buffers())}
    after_load = snapshot()
    handles = []
    original_vae = {"encode": pipe.vae.encode, "decode": pipe.vae.decode}
    starts = {}
    block_calls = [0] * len(pipe.transformer.transformer_blocks)
    def count_block(index):
        def hook(module, inputs, output):
            block_calls[index] += 1
        return hook
    def pre(name):
        def hook(module, inputs, kwargs):
            torch.cuda.synchronize()
            starts[name] = time.monotonic()
        return hook
    def post(name):
        def hook(module, inputs, kwargs, output):
            torch.cuda.synchronize()
            row = {"component": name, "seconds": time.monotonic() - starts[name], **snapshot()}
            cache = kwargs.get("kv_cache")
            if cache is not None:
                row["prefix_kv_bytes"] = sum(
                    t.numel() * t.element_size() for layer in cache.layer_caches
                    for t in (layer.k, layer.v) if t is not None)
            row["cache_mode"] = kwargs.get("kv_cache_mode")
            records.append(row)
        return hook
    if not args.plain:
        for name in ("text_encoder", "transformer"):
            module = getattr(pipe, name)
            handles.append(module.register_forward_pre_hook(pre(name), with_kwargs=True))
            handles.append(module.register_forward_hook(post(name), with_kwargs=True))
        for index, block in enumerate(pipe.transformer.transformer_blocks):
            handles.append(block.register_forward_hook(count_block(index)))
        def wrap_vae(method, label):
            def wrapped(*values, **kwargs):
                torch.cuda.synchronize()
                began = time.monotonic()
                output = method(*values, **kwargs)
                torch.cuda.synchronize()
                records.append({"component": label, "seconds": time.monotonic() - began, **snapshot()})
                return output
            return wrapped
        pipe.vae.encode = wrap_vae(pipe.vae.encode, "vae_encode")
        pipe.vae.decode = wrap_vae(pipe.vae.decode, "vae_decode")
    steps = []
    phase_releases = []
    original_encode_prompt = pipe.encode_prompt
    if args.release_phases:
        def encode_and_release(*values, **kwargs):
            output = original_encode_prompt(*values, **kwargs)
            torch.cuda.synchronize()
            prior = snapshot()
            began = time.monotonic()
            # Discard backing, retaining only shapes/configuration. This is
            # not CPU offload: a later request must reload these weights.
            pipe.text_encoder.to("meta")
            torch.cuda.empty_cache()
            phase_releases.append({"component": "text_encoder", "before": prior,
                                   "after": snapshot(), "seconds": time.monotonic() - began})
            return output
        pipe.encode_prompt = encode_and_release
    cancellation_at = None
    def callback(pipeline, index, timestep, tensors):
        nonlocal cancellation_at
        torch.cuda.synchronize()
        latent = tensors["latents"]
        steps.append({"index": index, "latent_bytes": latent.numel() * latent.element_size(),
                      "finite": bool(torch.isfinite(latent).all()), **snapshot()})
        if args.cancel_after == index + 1:
            cancellation_at = time.monotonic()
            pipeline._interrupt = True
        if args.release_phases and index + 1 == args.steps:
            prior = snapshot()
            began = time.monotonic()
            pipeline.transformer.to("meta")
            torch.cuda.empty_cache()
            phase_releases.append({"component": "transformer", "before": prior,
                                   "after": snapshot(), "seconds": time.monotonic() - began})
        return tensors
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    prompt = "A red ceramic teapot on a plain wooden table, soft daylight, no text."
    image_args = {}
    input_sha256 = None
    if args.edit:
        from PIL import Image
        pixels = bytes(channel for y in range(256) for x in range(256)
                       for channel in ((220, 30, 30) if 64 <= x < 192 and 64 <= y < 192 else (240, 240, 240)))
        input_sha256 = hashlib.sha256(pixels).hexdigest()
        image_args["image"] = Image.frombytes("RGB", (256, 256), pixels)
        prompt = "Change the red square to blue, preserve the pale background."
    result = pipe(prompt=prompt, width=args.size, height=args.size,
                  output_resolution=args.size,
                  num_inference_steps=args.steps, true_cfg_scale=1.0,
                  generator=torch.Generator("cuda").manual_seed(42),
                  use_kv_cache=not args.no_cache,
                  output_type="latent" if args.cancel_after else "pil",
                  callback_on_step_end=callback if not args.plain or args.cancel_after else None,
                  **image_args)
    torch.cuda.synchronize()
    seconds = time.monotonic() - start
    after_run = snapshot()
    cancellation_return_seconds = None if cancellation_at is None else time.monotonic() - cancellation_at
    output = result.images
    if args.cancel_after:
        if len(steps) != args.cancel_after:
            raise RuntimeError("Cancellation did not stop at the requested boundary")
        output_info = {"cancelled": True, "completed_steps": len(steps)}
    else:
        image = output[0]
        image.save(args.output / "image.png")
        output_info = {"mode": image.mode, "size": list(image.size),
                       "pixels_sha256": hashlib.sha256(image.tobytes()).hexdigest()}
    if any(not step["finite"] for step in steps):
        raise RuntimeError("Nonfinite latent")
    for handle in handles:
        handle.remove()
    pipe.vae.encode = original_vae["encode"]
    pipe.vae.decode = original_vae["decode"]
    pipe.encode_prompt = original_encode_prompt
    if not args.plain:
        del block
    del output, result, pipe, module, original_vae, original_encode_prompt
    if args.release_phases:
        del encode_and_release
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    after_release = snapshot()
    cancellation_full_release_seconds = None if cancellation_at is None else time.monotonic() - cancellation_at
    report = {"arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "model_revision": pins["revision"], "torch": torch.__version__,
              "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(),
              "seed": 42, "prompt": prompt, "dtype": "bfloat16", "true_cfg_scale": 1.0,
              "components": components, "load_seconds": load_seconds, "run_seconds": seconds,
              "cpu_load_seconds": cpu_load_seconds, "transfer_seconds": transfer_seconds,
              "before": before, "after_load": after_load, "after_run": after_run,
              "after_cleanup": after_release, "pipeline_released": retained is None,
              "pipeline_reused": reused, "calls": records, "steps": steps, "output": output_info}
    report["cancellation_return_seconds"] = cancellation_return_seconds
    report["cancellation_cleanup_seconds"] = cancellation_full_release_seconds
    report["transformer_block_calls"] = block_calls
    report["input_rgb_sha256"] = input_sha256
    report["phase_releases"] = phase_releases
    (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"load_seconds": load_seconds, "run_seconds": seconds, "output": output_info}), flush=True)
    return report


if __name__ == "__main__":
    os.umask(0o077)
    main()
