# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Observe a pinned Mia first verify; no timing results from this process."""
import hashlib
import json
import os
from pathlib import Path

def enable(RejectionSampler):
    if not __debug__:
        raise RuntimeError("first-verify evidence checks require assertions")
    import torch

    original = RejectionSampler.__call__
    seen = set()

    def observed(self, logits, batch, draft_logits=None):
        eligible = (
            batch.num_reqs == 1 and batch.num_draft_tokens == 3
            and int(batch.prefill_len_np[0]) in range(31743, 31747)
            and batch.req_ids[0] not in seen
        )
        # Always let the unmodified sampler decide before observing its result.
        result = original(self, logits, batch, draft_logits)
        if eligible:
            req = batch.req_ids[0]
            ids = batch.input_ids[batch.logits_indices].detach().cpu().tolist()
            positions = batch.positions[batch.logits_indices].detach().cpu().tolist()
            values = logits.detach().float().cpu()
            assert list(values.shape) == [4, 248320]
            assert bool(torch.isfinite(values).all())
            assert len(ids) == len(positions) == 4
            assert positions == list(range(positions[0], positions[0] + 4))
            assert positions[0] == int(batch.prefill_len_np[0])
            verdicts = values.argmax(dim=-1).tolist()
            accepted = 0
            for proposal, verdict in zip(ids[1:], verdicts[:3]):
                if proposal != verdict:
                    break
                accepted += 1
            sampled = result.sampled_token_ids.detach().cpu().tolist()[0]
            kept = int(result.num_sampled.detach().cpu().tolist()[0])
            rejected = int(result.num_rejected.detach().cpu().tolist()[0])
            assert kept == accepted + 1 and rejected == 3 - accepted
            assert sampled[:kept] == ids[1:accepted + 1] + [verdicts[accepted]]
            raw = values.numpy().astype('<f4', copy=False).tobytes()
            out = Path(os.environ['LLMP_MIA_TRACE_DIR'])
            name = 'first-verify-' + hashlib.sha256(req.encode()).hexdigest()[:24]
            with (out / (name + '.f32')).open('xb') as file:
                file.write(raw)
            row = dict(format='llmp-mia-first-verify-v1', request_id=req,
                       observer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       prompt_tokens=int(batch.prefill_len_np[0]), position=positions[0],
                       anchor_token=ids[0], drafts=ids[1:], verdicts=verdicts,
                       rows=4, accepted=accepted, kept=kept, rejected=rejected,
                       sampled_tokens=sampled[:kept], logits_sha256=hashlib.sha256(raw).hexdigest(),
                       logits_file=name + '.f32', timing_qualified=False)
            with (out / (name + '.json')).open('x') as file:
                json.dump(row, file, indent=2, allow_nan=False)
                file.write('\n')
            seen.add(req)
        return result

    RejectionSampler.__call__ = observed
    print('llmpalooza first-verify observer installed (timing unqualified)', flush=True)
