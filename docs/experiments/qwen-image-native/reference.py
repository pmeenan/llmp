# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Qwen-Image-2.1 BF16 reference tensors and bound calibration, from diffusers.

Runs inside the image-reference container (../image-reference/Dockerfile) on a
Spark, with the checkpoint at MODEL, the fast-swap prompt set's "image" entry
(the teapot, 1024x1024, 40 steps, seed 42, no guidance, prefix KV cache on)
and a new private output directory:

  python3 reference.py MODEL OUT PROMPTS [--skip-fp32]

The pipeline's denoising loop is re-run by hand (the same calls, in the same
order and dtypes as QwenImage21Pipeline.__call__ at diffusers 8b3c707e) so
every tensor llmpalooza is compared with can be kept; its image must be pixel for
pixel the baseline's (fast-swap/baselines.md), or the run is refused. It then
writes, as raw little-endian files with a JSON index (shape, dtype, SHA-256):

- the prompt's token IDs and the text encoder's kept hidden states
  (prompt_embeds, the system turn dropped);
- the initial latents (the seeded noise, packed), the schedule, and every
  step's latents in and noise prediction out (teacher-forcing references);
- the final latents, the VAE's decoded tensor and the image;
- debugging taps: text-encoder layer 0, and DiT block 0 and 31 of steps 0
  and 1.

Calibration (the scale of "small" for the pre-registered bounds): the same
components in FP32 on the same inputs (text encoder; DiT at steps 0 and 1;
VAE on the final latents), a 40-step trajectory with an FP32 DiT, the
pipeline's own cache-off sample, and an unrelated sample (seed 43). A
torch.profiler table of one cached DiT step is written for the kernel mix.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

SPEC_KEYS = ("prompt", "width", "height", "steps", "seed", "true_cfg_scale", "prefix_kv_cache")
BASELINE_PIXELS = "7d00b052878a03cc01baccbede3c3cafeca58b8160687ff93e915614387aac8f"


class Store:
    def __init__(self, out):
        self.out = out
        self.index = {}

    def tensor(self, name, t):
        t = t.detach().contiguous().cpu()
        if t.dtype == torch.bfloat16:
            raw, dtype = t.view(torch.int16).numpy().tobytes(), "bf16"
        elif t.dtype == torch.float32:
            raw, dtype = t.numpy().tobytes(), "f32"
        elif t.dtype == torch.int64:
            raw, dtype = t.numpy().astype("<i8").tobytes(), "i64"
        else:
            raise ValueError(f"{name}: unexpected dtype {t.dtype}")
        path = self.out / f"{name}.{dtype}"
        path.write_bytes(raw)
        self.index[name] = {"file": path.name, "dtype": dtype, "shape": list(t.shape),
                            "sha256": hashlib.sha256(raw).hexdigest()}

    def image(self, name, image):
        path = self.out / f"{name}.png"
        image.save(path)
        self.index[name] = {"file": path.name, "mode": image.mode, "size": list(image.size),
                            "pixels_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def sync():
    torch.cuda.synchronize()
    return time.time()


def rel_rms(a, b):
    """RMS of (a - b) over RMS of b, in float64."""
    a, b = a.double(), b.double()
    return float(((a - b).pow(2).mean().sqrt() / b.pow(2).mean().sqrt()).item())


def cosine(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    return float((a @ b / (a.norm() * b.norm())).item())


def compare(a, b):
    return {"rel_rms": rel_rms(a, b), "cosine": cosine(a, b),
            "max_abs": float((a.double() - b.double()).abs().max().item())}


def psnr_ssim(candidate, reference):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "image-gguf"))
    import compare as image_compare  # noqa: E402 (the image-gguf study's PSNR and SSIM)
    a = np.asarray(candidate.convert("RGB"), dtype=np.float64)
    b = np.asarray(reference.convert("RGB"), dtype=np.float64)
    mse = float(np.mean((a - b) ** 2))
    luma = np.array([0.299, 0.587, 0.114])
    return {"psnr_db": None if mse == 0 else round(10 * np.log10(255 ** 2 / mse), 3),
            "ssim_luma": round(image_compare.ssim(a @ luma, b @ luma), 5),
            "mean_abs_diff": round(float(np.mean(np.abs(a - b))), 4)}


class Pipeline:
    """QwenImage21Pipeline.__call__'s text-to-image path, step by step."""

    def __init__(self, pipe, spec):
        from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_shift
        self.pipe, self.spec = pipe, spec
        self.size = spec["width"]
        self.calculate_shift = calculate_shift

    def encode(self):
        pipe = self.pipe
        embeds, mask, image_pad_mask = pipe.encode_prompt(prompt=self.spec["prompt"], image=None,
                                                          device="cuda")
        return embeds, mask, image_pad_mask

    def schedule(self, latents):
        from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import retrieve_timesteps
        pipe, steps = self.pipe, self.spec["steps"]
        sigmas = np.linspace(1.0, 1 / steps, steps)
        cfg = pipe.scheduler.config
        mu = self.calculate_shift(latents.shape[1], cfg.get("base_image_seq_len", 256),
                                  cfg.get("max_image_seq_len", 4096), cfg.get("base_shift", 0.5),
                                  cfg.get("max_shift", 1.15))
        timesteps, _ = retrieve_timesteps(pipe.scheduler, steps, "cuda", sigmas=sigmas, mu=mu)
        return timesteps, mu

    def noise(self, seed, dtype):
        latents, _ = self.pipe.prepare_latents(None, 1, self.pipe.transformer.config.in_channels,
                                               self.size, self.size, dtype, "cuda",
                                               torch.Generator("cuda").manual_seed(seed))
        return latents

    def denoise(self, embeds, mask, image_pad_mask, latents, use_cache=True, taps=None,
                on_step=None, force=None):
        """The denoising loop. `force(i)` may return the latents to feed at step i
        (teacher forcing); `on_step(i, latents_in, noise_pred, latents_out)`."""
        from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21KVCache
        pipe = self.pipe
        timesteps, _ = self.schedule(latents)
        img_mask = torch.cat([image_pad_mask, image_pad_mask.new_ones(1, latents.shape[1] // 4)], dim=1)
        grid = self.size // pipe.vae_scale_factor
        img_shapes = [[(1, grid, grid)]]
        cache = QwenImage21KVCache(len(pipe.transformer.transformer_blocks)) if use_cache else None
        pipe.scheduler.set_begin_index(0)
        for i, t in enumerate(timesteps):
            if force is not None:
                latents = force(i, latents)
            mode = ("extract" if i == 0 else "cached") if use_cache else None
            timestep = t.expand(latents.shape[0]).to(latents.dtype)
            if taps is not None:
                taps.step = i
            with pipe.transformer.cache_context("cond"):
                noise_pred = pipe.transformer(hidden_states=latents, timestep=timestep / 1000,
                                              encoder_hidden_states=embeds,
                                              encoder_hidden_states_mask=mask, img_shapes=img_shapes,
                                              img_mask=img_mask, attention_kwargs={}, kv_cache=cache,
                                              kv_cache_mode=mode, return_dict=False)[0]
            noise_pred = noise_pred[:, -latents.size(1):]
            out = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            if on_step is not None:
                on_step(i, latents, noise_pred, out)
            latents = out
        return latents, timesteps

    def decode(self, latents, vae=None):
        pipe = self.pipe
        vae = vae or pipe.vae
        z = pipe._unpack_latents(latents, self.size, self.size, pipe.vae_scale_factor).to(vae.dtype)
        mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(z.device, z.dtype)
        std = torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(z.device, z.dtype)
        z = z * std + mean
        decoded = vae.decode(z, return_dict=False)[0][:, :, 0]
        return decoded, pipe.image_processor.postprocess(decoded, output_type="pil")[0]


class Taps:
    """Forward hooks keeping the joint hidden states after chosen DiT blocks."""

    def __init__(self, store, pipe, blocks, steps):
        self.step, self.store = -1, store
        for b in blocks:
            pipe.transformer.transformer_blocks[b].register_forward_hook(self.hook(b, steps))

    def hook(self, block, steps):
        def keep(module, args, output):
            if self.step in steps:
                self.store.tensor(f"dit_step{self.step}_block{block}", output[0])
        return keep


@torch.no_grad()  # as QwenImage21Pipeline.__call__: no autograd state across steps
def main():
    model, out, prompts = map(Path, sys.argv[1:4])
    skip_fp32 = "--skip-fp32" in sys.argv[4:]
    out.mkdir(mode=0o700, parents=True, exist_ok=False)
    spec = json.loads(prompts.read_text())["image"]
    spec = {k: spec[k] for k in SPEC_KEYS}
    if spec["true_cfg_scale"] > 1 or not spec["prefix_kv_cache"] or spec["width"] != spec["height"]:
        raise SystemExit("this harness covers the unguided, cached, square case only")
    from diffusers import AutoencoderKLQwenImage21, QwenImage21Pipeline
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    store = Store(out)
    report = {"spec": spec, "torch": torch.__version__, "cuda": torch.version.cuda,
              "device": torch.cuda.get_device_name(),
              "allow_bf16_reduced_precision_reduction":
                  torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction}
    t0 = time.time()
    pipe = QwenImage21Pipeline.from_pretrained(str(model), torch_dtype=torch.bfloat16,
                                               local_files_only=True, use_safetensors=True)
    pipe.to("cuda")
    report["load_s"] = sync() - t0
    run = Pipeline(pipe, spec)

    # ---- The reference sample, kept tensor by tensor.
    text_model = pipe.text_encoder.model.language_model
    layer0 = {}
    handle = text_model.layers[0].register_forward_hook(
        lambda m, a, o: layer0.__setitem__("out", o[0] if isinstance(o, tuple) else o))
    ids = pipe.processor(text=[pipe.prompt_template_t2i.format(spec["prompt"])], padding=True,
                         padding_side="left", return_tensors="pt").input_ids[0]
    t_start = sync()
    embeds, mask, image_pad_mask = run.encode()
    t_encoded = sync()
    handle.remove()
    report["text_encode_s"] = t_encoded - t_start
    report["drop_tokens"] = pipe._drop_idx
    report["prompt_mask_is_none"] = mask is None
    store.tensor("input_ids", ids.to(torch.int64))
    store.tensor("text_layer0_out", layer0["out"][0])
    store.tensor("prompt_embeds", embeds[0])
    noise = run.noise(spec["seed"], embeds.dtype)
    store.tensor("latents_init", noise[0])
    taps = Taps(store, pipe, (0, 31), (0, 1))
    step_times = []

    def keep_step(i, latents_in, noise_pred, latents_out):
        step_times.append(sync())
        store.tensor(f"step{i:02d}_latents_in", latents_in[0])
        store.tensor(f"step{i:02d}_noise_pred", noise_pred[0])

    t_denoise = sync()
    final, timesteps = run.denoise(embeds, mask, image_pad_mask, noise, taps=taps, on_step=keep_step)
    taps.step = -1
    store.tensor("latents_final", final[0])
    store.tensor("sigmas", pipe.scheduler.sigmas.float().cpu())
    store.tensor("timesteps", timesteps.float().cpu())
    report["mu"] = run.schedule(noise)[1]
    t_decode = sync()
    decoded, image = run.decode(final)
    t_done = sync()
    report["vae_decode_s"] = t_done - t_decode
    report["denoise_s"] = t_decode - t_denoise
    gaps = [b - a for a, b in zip(step_times, step_times[1:])]
    report["step_s_median_synced"] = sorted(gaps)[len(gaps) // 2]
    report["first_step_s_synced"] = step_times[0] - t_denoise
    store.tensor("vae_decoded", decoded[0])
    store.image("image", image)
    pixels = store.index["image"]["pixels_sha256"]
    report["pixels_match_baseline"] = pixels == BASELINE_PIXELS
    if pixels != BASELINE_PIXELS:
        (out / "reference.json").write_text(json.dumps({"report": report, "files": store.index}, indent=2))
        raise SystemExit(f"the hand-run loop's image {pixels} is not the baseline's {BASELINE_PIXELS}")
    # The documented postprocess, recomputed: (x * 0.5 + 0.5).clamp(0, 1) in
    # BF16, then float32 * 255, round half to even, uint8.
    manual = ((decoded[0] * 0.5 + 0.5).clamp(0, 1)).permute(1, 2, 0).float().cpu().numpy()
    manual = (manual * 255).round().astype("uint8")
    report["postprocess_formula_matches"] = bool((manual == np.asarray(image)).all())

    # ---- Calibration: pipeline-level controls.
    calib = {}
    latents_nc, _ = run.denoise(embeds, mask, image_pad_mask, noise, use_cache=False)
    _, image_nc = run.decode(latents_nc)
    store.image("control_cache_off", image_nc)
    calib["cache_off_vs_reference"] = psnr_ssim(image_nc, image)
    other = run.noise(spec["seed"] + 1, embeds.dtype)
    latents_43, _ = run.denoise(embeds, mask, image_pad_mask, other)
    _, image_43 = run.decode(latents_43)
    store.image("control_seed43", image_43)
    calib["seed43_vs_reference"] = psnr_ssim(image_43, image)

    # ---- The kernel mix of one cached DiT step (step 1's inputs).
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        def stop_after(i, latents):
            if i >= 2:
                raise StopIteration
            return latents
        try:
            run.denoise(embeds, mask, image_pad_mask, noise, force=stop_after)
        except StopIteration:
            pass
        sync()
    table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=60)
    (out / "profile_steps01.txt").write_text(table)
    groups = {"gemm": 0.0, "attention": 0.0, "other": 0.0}
    for e in prof.key_averages():
        name, us = e.key.lower(), e.device_time_total
        if e.device_type is not None and str(e.device_type) != "DeviceType.CUDA":
            continue
        if any(k in name for k in ("gemm", "cutlass", "nvjet", "sm90_", "sm100_", "sm120_", "cublas")):
            groups["gemm"] += us
        elif any(k in name for k in ("flash", "fmha", "attention", "attn")):
            groups["attention"] += us
        else:
            groups["other"] += us
    report["profile_steps01_us"] = groups

    # ---- Calibration: components in FP32 on the same inputs.
    if not skip_fp32:
        te = pipe.text_encoder
        te.to(torch.float32)
        embeds32, _, _ = pipe.encode_prompt(prompt=spec["prompt"], image=None, device="cuda")
        store.tensor("prompt_embeds_fp32", embeds32[0])
        calib["text_bf16_vs_fp32"] = compare(embeds[0].float(), embeds32[0])
        te.to("meta")
        torch.cuda.empty_cache()
        dit = pipe.transformer
        dit.to(torch.float32)
        e32 = embeds.float()
        n32 = {}

        def keep32(i, latents_in, noise_pred, latents_out):
            if i < 2:
                n32[i] = noise_pred[0]

        def forced(i, latents):
            if i >= 2:
                raise StopIteration
            return torch.from_numpy(np.frombuffer(
                (out / store.index[f"step{i:02d}_latents_in"]["file"]).read_bytes(), dtype=np.int16).copy()
            ).view(torch.bfloat16).view(1, -1, 64).to("cuda").float()
        try:
            run.denoise(e32, mask, image_pad_mask, noise.float(), force=forced, on_step=keep32)
        except StopIteration:
            pass
        for i in (0, 1):
            ref16 = torch.from_numpy(np.frombuffer(
                (out / store.index[f"step{i:02d}_noise_pred"]["file"]).read_bytes(), dtype=np.int16).copy()
            ).view(torch.bfloat16).view(-1, 64).float()
            store.tensor(f"step{i:02d}_noise_pred_fp32", n32[i])
            calib[f"dit_step{i}_bf16_vs_fp32"] = compare(ref16, n32[i].cpu())
        t32 = sync()
        final32, _ = run.denoise(e32, mask, image_pad_mask, noise.float())
        report["fp32_denoise_s"] = sync() - t32
        store.tensor("latents_final_fp32dit", final32[0])
        _, image32 = run.decode(final32.to(torch.bfloat16))
        store.image("control_fp32_dit", image32)
        calib["fp32_dit_vs_reference"] = psnr_ssim(image32, image)
        calib["fp32_dit_final_latents_vs_reference"] = compare(final[0].float(), final32[0])
        dit.to("meta")
        torch.cuda.empty_cache()
        vae32 = AutoencoderKLQwenImage21.from_pretrained(str(model), subfolder="vae", torch_dtype=torch.float32,
                                                        local_files_only=True, use_safetensors=True).to("cuda")
        decoded32, image_v32 = run.decode(final, vae=vae32)
        store.tensor("vae_decoded_fp32", decoded32[0])
        store.image("control_fp32_vae", image_v32)
        calib["vae_bf16_vs_fp32_tensor"] = compare(decoded[0].float(), decoded32[0])
        calib["fp32_vae_vs_reference"] = psnr_ssim(image_v32, image)
    report["calibration"] = calib
    report["cuda_max_allocated_bytes"] = torch.cuda.max_memory_allocated()
    (out / "reference.json").write_text(json.dumps({"report": report, "files": store.index}, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
