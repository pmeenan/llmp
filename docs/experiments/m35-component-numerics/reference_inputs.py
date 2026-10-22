# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Freeze U4's 512-square inputs, or generate its independent BF16 reference.

Inside the pinned image-reference container:
  python3 reference_inputs.py MODEL NEW_OUTPUT --inputs-only
  python3 reference_inputs.py MODEL OTHER_NEW_OUTPUT
Input preparation runs no model inference. Compare the input entries between
both receipts exactly before consuming the reference. Output stays outside Git.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import torch

SPEC = {"prompt": "A blue glass vase beside a yellow book, viewed from above.",
        "width": 512, "height": 512, "steps": 2, "seed": 43,
        "true_cfg_scale": 1.0, "prefix_kv_cache": True}


def helper():
    path = Path(__file__).resolve().parent.parent / "qwen-image-native/reference.py"
    spec = importlib.util.spec_from_file_location("native_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--inputs-only", action="store_true")
    args = parser.parse_args()
    from diffusers import FlowMatchEulerDiscreteScheduler, QwenImage21Pipeline
    from transformers import AutoProcessor
    ref = helper()
    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    store = ref.Store(args.output)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    # Construct the pipeline with no weights for input preparation. Its own
    # processor, template, latent packing and scheduler supply every input.
    if args.inputs_only:
        pipe = QwenImage21Pipeline(
            scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(
                str(args.model / "scheduler"), local_files_only=True),
            vae=None, text_encoder=None,
            processor=AutoProcessor.from_pretrained(str(args.model / "processor"),
                                                   local_files_only=True),
            transformer=None)
    else:
        pipe = QwenImage21Pipeline.from_pretrained(
            str(args.model), torch_dtype=torch.bfloat16,
            local_files_only=True, use_safetensors=True).to("cuda")
    ids = pipe.processor(text=[pipe.prompt_template_t2i.format(SPEC["prompt"])],
                         padding=True, padding_side="left", return_tensors="pt").input_ids[0]
    store.tensor("input_ids", ids.to(torch.int64))
    # Explicit checkpoint dimensions let input preparation avoid loading weights.
    config = json.loads((args.model / "transformer/config.json").read_text())
    noise, _ = pipe.prepare_latents(None, 1, config["in_channels"],
                                    SPEC["height"], SPEC["width"], torch.bfloat16,
                                    "cuda", torch.Generator("cuda").manual_seed(SPEC["seed"]))
    store.tensor("latents_init", noise[0])
    run = ref.Pipeline(pipe, SPEC)
    timesteps, mu = run.schedule(noise)
    store.tensor("timesteps", timesteps.float())
    store.tensor("sigmas", pipe.scheduler.sigmas.float())
    report = {"spec": SPEC, "torch": torch.__version__, "cuda": torch.version.cuda,
              "device": torch.cuda.get_device_name(), "drop_tokens": pipe._drop_idx,
              "input_ids": ids.tolist(), "kept_rows": len(ids) - pipe._drop_idx,
              "mu": mu, "inputs_only": args.inputs_only}
    if not args.inputs_only:
        embeds, mask, image_pad_mask = run.encode()
        store.tensor("prompt_embeds", embeds[0])
        def keep(i, before, prediction, after):
            store.tensor(f"step{i:02d}_latents_in", before[0])
            store.tensor(f"step{i:02d}_noise_pred", prediction[0])
        final, _ = run.denoise(embeds, mask, image_pad_mask, noise.clone(), on_step=keep)
        store.tensor("latents_final", final[0])
        decoded, image = run.decode(final)
        store.tensor("vae_decoded", decoded[0])
        store.image("image", image)
        official = pipe(prompt=SPEC["prompt"], width=512, height=512,
                        output_resolution=512, num_inference_steps=2,
                        true_cfg_scale=1.0,
                        generator=torch.Generator("cuda").manual_seed(43),
                        use_kv_cache=True, output_type="pil").images[0]
        store.image("official_image", official)
        if image.tobytes() != official.tobytes() or image.size != official.size:
            raise RuntimeError("manual reference differs from the official pipeline")
        report["official_pipeline_pixels_equal"] = True
    report["generator_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (args.output / "reference.json").write_text(
        json.dumps({"report": report, "files": store.index}, indent=2) + "\n")


if __name__ == "__main__":
    main()
