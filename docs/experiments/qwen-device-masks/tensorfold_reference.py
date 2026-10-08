# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0
"""Bounded external TensorFold reference; inference remains upstream code.

Use the pinned source/package and isolated supervisor described in README.
Timing uses the public generate path without logits observers. Separate
teacher-forced heads score supplied native histories, outside paid timers.
"""
import argparse
import gc
import hashlib
import json
import os
import time
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
    prompt_precision.set_fp8(True)  # upstream --prefill-fp8, before loading weights
    engine = FlashNextEngine(args.model, depth=3, confidence=0.0,
                             draft_vocab="default", max_len=2048,
                             context_explicit=True, graphs=True, kv_dtype="bf16",
                             streams=1)
    records = {}
    warm_cuts = {}
    fp8_calls = 0
    original_prefill8 = Mx8Linear.prefill8
    original_chunk = decode.prefill_chunk
    current_cuts = []

    def witnessed_fp8(self, *call_args, **kwargs):
        nonlocal fp8_calls
        fp8_calls += 1
        return original_prefill8(self, *call_args, **kwargs)

    def witnessed_chunk(e, ids, start, **kwargs):
        end = min(start + e.prefill_rows, kwargs.get("end") or len(ids))
        current_cuts.append([start, end - start])
        return original_chunk(e, ids, start, **kwargs)

    def generate(draft):
        tokens = []
        engine.cache = []  # fresh prompt state, both warm and paid requests
        torch.cuda.synchronize()
        started = time.perf_counter()
        stats = engine.generate(prompt, 32, None,
                                lambda new: (tokens.extend(new), False)[1],
                                draft=draft, stop_eos=False)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        if len(tokens) != 32 or any(not 0 <= t < 248320 for t in tokens):
            raise RuntimeError("reference did not finish 32 valid choices")
        if stats.get("cached") != 0 or stats.get("drafts") != draft:
            raise RuntimeError("reference reused prompt state or wrong decode policy")
        return dict(stats=stats, seconds=seconds, tokens=tokens)

    try:
        # Witness actual chunk and FP8 consumers during warmup only. Restore
        # unmodified upstream methods before any paid endpoint.
        Mx8Linear.prefill8 = witnessed_fp8
        decode.prefill_chunk = witnessed_chunk
        for name, draft in [("plain", False), ("spec-lean", True)]:
            current_cuts.clear()
            generate(draft)
            warm_cuts[name] = list(current_cuts)
        Mx8Linear.prefill8 = original_prefill8
        decode.prefill_chunk = original_chunk
        if fp8_calls <= 0 or any(cuts != [[0, 512], [512, 512], [1024, 512]]
                                 for cuts in warm_cuts.values()):
            raise RuntimeError("actual FP8 prompt/chunk path absent")
        if (engine.prefill_rows != 512 or engine.context_window != 2048
                or engine.max_len != 2052):
            raise RuntimeError("reference actual admitted envelope differs")
        if engine.serial.graphs is not None:
            raise RuntimeError("public plain serial policy unexpectedly changed")
        for name, draft in [("plain", False), ("spec-lean", True)]:
            records[name] = generate(draft)
        # Full vocabulary reads run only after both paid endpoints finish.
        # Each mode scores its supplied native history, even if TF's natural
        # trajectory diverged; no cross-engine bit-exactness assumption.
        serial = engine.serial
        for name, history in histories.items():
            heads = []
            original_sample = serial.sample

            def capture_frontier(logits, positions, sampling, **kwargs):
                heads.append(logits[:1].float().cpu().numpy().copy())
                return original_sample(logits, positions, sampling, **kwargs)

            serial.sample = capture_frontier
            try:
                decode.prefill(serial, prompt, None, mtp=False)
            finally:
                serial.sample = original_sample
            for token in history[:-1]:
                logits = serial.forward([token])
                heads.append(logits[:1].float().cpu().numpy().copy())
                decode.commit(serial.w, serial.st, serial.buf, 1, 1)
            data = np.concatenate(heads, axis=0)
            if data.shape != (32, 248320) or not np.isfinite(data).all():
                raise RuntimeError("incomplete/nonfinite teacher-forced heads")
            path = args.out / (name + "-heads.f32")
            data.astype("<f4").tofile(path)
            records[name]["heads_sha256"] = sha(path)
            records[name]["teacher_history"] = history
        torch.cuda.synchronize()
        binding = dict(module=tensorfold.__file__, version=tensorfold.__version__,
                       torch=torch.__version__, cuda=torch.version.cuda,
                       prompt_fp8=prompt_precision.fp8(), witnessed_fp8_calls=fp8_calls,
                       warm_cuts=warm_cuts, context=engine.context_window,
                       cache_slots=engine.max_len,
                       prefill_rows=engine.prefill_rows, kv_dtype=engine.kv_dtype,
                       depth=engine.depth, confidence=engine.confidence,
                       copy_drafts=engine.copy_drafts,
                       plain_graphs=engine.serial.graphs is not None,
                       spec_graphs=engine.e.graphs is not None,
                       actual_draft_vocab=len(engine.w.draft_ids),
                       draft_ids_sha256=hashlib.sha256(
                           engine.w.draft_ids.cpu().numpy().astype("<i4").tobytes()).hexdigest())
        (args.out / "reference.json").write_text(
            json.dumps(dict(binding=binding, results=records), indent=2) + "\n")
    finally:
        Mx8Linear.prefill8 = original_prefill8
        decode.prefill_chunk = original_chunk
        engine.shutdown()
        engine.close()
        torch.cuda.synchronize()
    del engine
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print("TENSORFOLD_REFERENCE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
