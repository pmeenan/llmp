#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""External teacher forcing over the pinned engine's own prefill and output head."""
from pathlib import Path

from contract import ActualTargets, RECIPE, SOURCE, row_groups, sha


def authenticate_source():
    import tensorfold
    root = Path(tensorfold.__file__).resolve().parent
    manifest = Path(__file__).with_name("tensorfold-source-hashes.txt")
    rows = manifest.read_text().splitlines()
    if len(rows) != 449:
        raise ValueError("installed-source inventory changed")
    for row in rows:
        expected, name = row.split("  ", 1)
        path = (root / name).resolve()
        if not path.is_relative_to(root) or sha(path.read_bytes()) != expected:
            raise ValueError(f"pinned TensorFold source differs: {name}")
    return {"commit": SOURCE, "inventory_sha256": sha(manifest.read_bytes()), "files": len(rows)}


class SourceAdapter:
    def __init__(self, checkpoint: Path):
        self.identity = authenticate_source()
        import torch
        from tensorfold.cuda import prompt_precision
        from tensorfold.families.qwen3_5.cuda.engine import Qwen27Engine
        prompt_precision.set_fp8(False)
        if prompt_precision.fp8():
            raise ValueError("source prompt precision is not BF16")
        self.torch = torch
        self.engine = Qwen27Engine(checkpoint, None, streams=1, tp=1, context=12288,
                                   context_explicit=True, vision=False, max_rows=16)
        if self.engine.context_window != 12288 or self.engine.room is None or self.engine.draft is not None:
            raise ValueError("source capacity, KVRoom or draft contract changed")
        # Use the source's public prefill chunk knob for both the adapter and normal-prefill control.
        self.admitted_prompt_rows = self.engine.w.prompt_rows
        if self.admitted_prompt_rows < RECIPE["prefill_rows"]:
            raise ValueError("source admission does not cover the registered prefill rows")
        self.engine.w.prompt_rows = RECIPE["prefill_rows"]

    def close(self):
        self.engine.close()

    def state(self):
        from tensorfold.families.qwen3_5.cuda.forward import State
        st = State(self.engine.w)
        st.limit = self.engine.context_window
        st.room = self.engine.room
        return st

    def logits(self, hidden):
        from tensorfold.families.qwen3_5.cuda.forward import _mm
        result = _mm(hidden, self.engine.w.head)
        if (result.ndim != 2 or result.shape[1] != self.engine.w.config.vocab
                or not self.torch.isfinite(result).all().item()):
            raise ValueError("source head returned invalid/nonfinite full-vocabulary logits")
        return result

    def score(self, ids, first, document):
        from tensorfold.cuda.logprobs import capture
        from tensorfold.families.qwen3_5.cuda.prefill import chunks, prefill_chunk
        if not 2 <= len(ids) <= self.engine.context_window:
            raise ValueError("teacher history exceeds admitted context or is empty")
        spans = chunks(0, len(ids) - 1, self.engine.w.prompt_rows)
        groups = row_groups(ids, first, spans, RECIPE["head_rows"])
        collector = ActualTargets(ids, first, document)
        st = self.state()
        with self.torch.inference_mode():
            for a, b, batches in groups:
                if st.pos != a:
                    raise ValueError("source state did not retain the complete prefix")
                tokens = self.torch.tensor(ids[a:b], dtype=self.torch.int32, device=self.engine.w.norm.device)
                hidden, taps = prefill_chunk(self.engine.w, tokens, st, every=True)
                if taps is not None or hidden is None or hidden.shape[0] != b - a or st.pos != b:
                    raise ValueError("source prefill row/state contract changed")
                for lo, hi, positions, targets in batches:
                    logits = self.logits(hidden[lo:hi].contiguous())
                    capture(logits, targets, positions, collector)
                    del logits
                del hidden, tokens
            for pair in st.kv:
                if pair is not None and any(t.dtype != self.torch.bfloat16 for t in pair):
                    raise ValueError("source KV precision differs from BF16")
        return collector.finish(), {"spans": spans, "final_state_position": st.pos,
                                     "kv_dtype": "bfloat16", "capacity": self.engine.capacity_plan,
                                     "admitted_prompt_rows": self.admitted_prompt_rows}

    def validate(self, ids):
        """Fresh states; normal source prefill remains unmodified and independently invoked."""
        from tensorfold.families.qwen3_5.cuda.decode import prefill
        from tensorfold.families.qwen3_5.cuda.prefill import chunks, prefill_chunk
        results = []
        with self.torch.inference_mode():
            for count in (513, 1024):
                prompt = ids[:count]
                normal_state, token = prefill(self.engine.w, prompt, None, limit=12288,
                                              room=self.engine.room)
                if normal_state.pos != count:
                    raise ValueError("normal source prefill did not consume the control prompt")
                del normal_state  # controls do not retain two full KV domains
                st = self.state()
                spans = chunks(0, count, self.engine.w.prompt_rows)
                for a, b in spans:
                    tokens = self.torch.tensor(prompt[a:b], dtype=self.torch.int32,
                                               device=self.engine.w.norm.device)
                    hidden, _ = prefill_chunk(self.engine.w, tokens, st, every=True)
                    if st.pos != b or hidden.shape[0] != b - a:
                        raise ValueError("adapter control row count changed")
                single = self.logits(hidden[-1:].contiguous())
                batch = self.logits(hidden[-16:].contiguous())
                exact = self.torch.equal(single[0], batch[-1])
                picked = int(single.argmax(-1).item())
                if not exact or picked != token:
                    raise ValueError("adapter differs from normal prefill or changes head row arithmetic")
                results.append({"prompt_tokens": count, "spans": spans, "normal_first_token": token,
                                "adapter_first_token": picked, "head_1_vs_16_exact": exact})
                del st, hidden, single, batch
        return results

    def behavior(self, cases, tokenizer):
        records = []
        for case in cases:
            tokens = []
            def accept(values):
                if not values or any(type(i) is not int for i in values):
                    raise ValueError("invalid generation tokens")
                tokens.extend(values)
                if len(tokens) > 512:
                    raise ValueError("source generation exceeded the registered reply bound")
                return False
            with self.torch.inference_mode():
                stats = self.engine.generate(case["ids"], 512, None, accept, draft=False, stop_eos=True)
            if not tokens:
                raise ValueError("empty source behavior generation")
            finish = "stop" if tokens[-1] in self.engine.eos else "length" if len(tokens) == 512 else "unknown"
            if finish == "unknown":
                raise ValueError("source generation terminated without EOS or the registered bound")
            records.append({"id": case["id"], "prompt_tokens": len(case["ids"]), "tokens": tokens,
                            "finish_reason": finish, "text": tokenizer.decode(tokens, skip_special_tokens=True),
                            "raw_text": tokenizer.decode(tokens, skip_special_tokens=False), "stats": stats})
        return records
