# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Pinned TensorFold BF16 Teacher32 or fixed-prefix frontier diagnostic.

Use the pinned source/package and isolated supervisor described in README.
No timing endpoint or natural generation is measured. Teacher-forced heads
score the same retained native history as the FP8 reference.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
import tensorfold
from tensorfold.cuda import prompt_precision
from tensorfold.cuda.nvfp4.linear import Mx8Linear
from tensorfold.families.qwen4_exp.cuda import decode
from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine


def sha(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--histories", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fixed-prefix-rows", type=int, choices=[0, 29], default=0,
                        help="append exactly29 retained history IDs and observe only its prefill frontier")
    args = parser.parse_args()
    args.out.mkdir(exist_ok=False)
    manifest = json.loads(Path("/reference/tf-source.json").read_text())
    if manifest["commit"] != "ed78d6fc204d89d90b045bf033d6551e7714f3a1":
        raise RuntimeError("reference commit differs")
    package = Path("/reference/package")
    if not Path(tensorfold.__file__).is_relative_to(package):
        raise RuntimeError("old dependency package imported")
    omitted_docs = {"src/tensorfold/families/README.md",
                    "src/tensorfold/families/qwen3_5/cuda/README.md",
                    "src/tensorfold/kernels/README.md"}
    for installed in (package / "tensorfold").rglob("*"):
        if not installed.is_file() or "__pycache__" in installed.parts or installed.suffix == ".pyc":
            continue
        name = "src/" + str(installed.relative_to(package))
        if name not in manifest["files"] or sha(installed) != manifest["files"][name]:
            raise RuntimeError("installed reference payload differs: " + name)
    for name, expected in manifest["files"].items():
        if name.startswith("src/tensorfold/") and name not in omitted_docs:
            if sha(package / name.removeprefix("src/")) != expected:
                raise RuntimeError("reference package differs: " + name)
    prompt = json.loads(args.inputs.read_text())["decode"][0]["prompt_token_ids"]
    histories = json.loads(args.histories.read_text())
    if len(prompt) != 1536 or set(histories) != {"plain", "spec-lean"}:
        raise RuntimeError("reference work differs")
    if any(not 0 <= t < 248320 for t in prompt) or any(
            len(ids) != 32 or any(not 0 <= t < 248320 for t in ids)
            for ids in histories.values()):
        raise RuntimeError("invalid literal IDs")
    if os.environ.get("TENSORFOLD_PREFILL_ROWS") != "512":
        raise RuntimeError("reference prefill envelope differs")
    prompt_precision.set_fp8(False)  # upstream default BF16 policy, before allocation/load
    engine = FlashNextEngine(args.model, depth=3, confidence=0.0,
                             draft_vocab="default", max_len=2048,
                             context_explicit=True, graphs=True, kv_dtype="bf16",
                             streams=1)
    original_prefill = Mx8Linear.prefill
    original_prefill8 = Mx8Linear.prefill8
    original_chunk = decode.prefill_chunk
    calls = {"bf16_folded": 0, "bf16_lane_fallback": 0, "fp8": 0}
    cuts = []

    def witnessed_prefill(self, *call_args, **kwargs):
        result = original_prefill(self, *call_args, **kwargs)
        calls["bf16_folded" if self.fold else "bf16_lane_fallback"] += 1
        return result

    def witnessed_fp8(self, *call_args, **kwargs):
        calls["fp8"] += 1
        return original_prefill8(self, *call_args, **kwargs)

    def witnessed_chunk(e, ids, start, **kwargs):
        end = min(start + e.prefill_rows, kwargs.get("end") or len(ids))
        cuts.append([start, end - start])
        return original_chunk(e, ids, start, **kwargs)

    try:
        if histories["plain"] != histories["spec-lean"]:
            raise RuntimeError("one diagnostic history would not cover both recorded modes")
        if (engine.context_window != 2048 or engine.max_len != 2052
                or engine.prefill_rows != 512 or prompt_precision.fp8()):
            raise RuntimeError("BF16 diagnostic admitted policy differs")
        # Match FlashNextEngine._serial's lazy initialization exactly;
        # this teacher-only diagnostic never calls natural generate().
        serial_initialized_here = engine.serial is None
        if serial_initialized_here:
            engine.serial = engine.e.twin()
        serial = engine.serial
        if serial.graphs is not None:
            raise RuntimeError("public serial teacher unexpectedly captured")
        allocation_before_teacher = torch.cuda.memory_allocated()
        history = histories["plain"]
        actual_prompt = prompt + history[:args.fixed_prefix_rows]
        expected_cuts = [[0, 512], [512, 512], [1024, 512]]
        if args.fixed_prefix_rows:
            expected_cuts.append([1536, 29])
        suffix_forward_calls = 0
        heads = []
        original_sample = serial.sample

        def capture_frontier(logits, positions, sampling, **kwargs):
            heads.append(logits[:1].float().cpu().numpy().copy())
            return original_sample(logits, positions, sampling, **kwargs)

        # There is no paid performance endpoint in this diagnostic. Witness
        # actual prompt consumers while preserving upstream arithmetic.
        Mx8Linear.prefill = witnessed_prefill
        Mx8Linear.prefill8 = witnessed_fp8
        decode.prefill_chunk = witnessed_chunk
        serial.sample = capture_frontier
        try:
            decode.prefill(serial, actual_prompt, None, mtp=False)
        finally:
            serial.sample = original_sample
            Mx8Linear.prefill = original_prefill
            Mx8Linear.prefill8 = original_prefill8
            decode.prefill_chunk = original_chunk
        if calls["fp8"] or calls["bf16_folded"] + calls["bf16_lane_fallback"] == 0:
            raise RuntimeError("actual BF16 prompt consumers absent")
        if cuts != expected_cuts:
            raise RuntimeError("actual BF16 teacher chunk cuts differ")
        for token in ([] if args.fixed_prefix_rows else history[:-1]):
            suffix_forward_calls += 1
            logits = serial.forward([token])
            heads.append(logits[:1].float().cpu().numpy().copy())
            decode.commit(serial.w, serial.st, serial.buf, 1, 1)
        data = np.concatenate(heads, axis=0)
        expected_rows = 1 if args.fixed_prefix_rows else 32
        if data.shape != (expected_rows, 248320) or not np.isfinite(data).all():
            raise RuntimeError("incomplete/nonfinite diagnostic teacher heads")
        path = args.out / "bf16-heads.f32"
        data.astype("<f4").tofile(path)
        torch.cuda.synchronize()
        binding = dict(module=tensorfold.__file__, version=tensorfold.__version__,
                       torch=torch.__version__, cuda=torch.version.cuda,
                       prompt_fp8=prompt_precision.fp8(), prompt_calls=calls,
                       cuts=cuts, context=engine.context_window,
                       cache_slots=engine.max_len, prefill_rows=engine.prefill_rows,
                       capacity_plan=engine.capacity_plan,
                       allocated_before_teacher=allocation_before_teacher,
                       allocated_after_teacher=torch.cuda.memory_allocated(),
                       peak_allocated=torch.cuda.max_memory_allocated(),
                       kv_dtype=engine.kv_dtype, depth=engine.depth,
                       confidence=engine.confidence, copy_drafts=engine.copy_drafts,
                       plain_graphs=serial.graphs is not None,
                       serial_initialized_here=serial_initialized_here,
                       actual_draft_vocab=len(engine.w.draft_ids),
                       draft_ids_sha256=hashlib.sha256(
                           engine.w.draft_ids.cpu().numpy().astype("<i4").tobytes()).hexdigest())
        (args.out / "diagnostic.json").write_text(json.dumps(dict(
            scope=("fixed-prefix29 BF16 prompt frontier; no suffix forward or performance endpoint"
                   if args.fixed_prefix_rows else "same-conditioned BF16-prompt Teacher32; no performance endpoint"),
            binding=binding, teacher_history=history, fixed_prefix_rows=args.fixed_prefix_rows,
            prompt_rows=len(actual_prompt),
            prompt_ids_sha256=hashlib.sha256(np.asarray(actual_prompt, dtype="<i4").tobytes()).hexdigest(),
            suffix_forward_calls=suffix_forward_calls, heads_sha256=sha(path)), indent=2) + "\n")
    finally:
        Mx8Linear.prefill = original_prefill
        Mx8Linear.prefill8 = original_prefill8
        decode.prefill_chunk = original_chunk
        engine.shutdown()
        engine.close()
        torch.cuda.synchronize()
    del engine
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print("TENSORFOLD_BF16_DIAGNOSTIC_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
