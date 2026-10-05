#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0
"""Authenticate common prefixes/counts before summarizing native short bookends."""
import hashlib
import json
import pathlib
import re
import sys

raw = pathlib.Path(sys.argv[1]).read_bytes()
s = raw.decode()
seconds = [float(x) for x in re.findall(r'GEMMA_WARM seconds=(\S+) completed_chunks=32', s)]
seeds = [(int(x), int(y)) for x, y in re.findall(r'GEMMA_TIMED_PREFIX appended=(\d+) past=(\d+)', s)]
tokens = [(int(x), int(y)) for x, y in re.findall(r'GEMMA_WARM_TOKEN step=(\d+) id=(\d+)', s)]
policies = [dict(item.split('=') for item in line.split())
            for line in re.findall(r'GEMMA_POLICY (.*)', s)]
assert len(seconds) == 3 and all(x > 0 for x in seconds)
assert seeds == [(45518, 7), (107, 8), (101, 9)] * 3
assert len(tokens) == 96 and tokens[:32] == tokens[32:64] == tokens[64:]
assert [x for x, _ in tokens[:32]] == list(range(32))
assert len(policies) == 3
for index, policy in enumerate(policies):
    assert policy['rows'] == policy['segments'] == '1'
    for key in ('norm_fused', 'rope_store', 'shared_vecq', 'row_products', 'lane_steps'):
        assert policy[key] == '0'
    for key, count in (('norm_rope', 60), ('norm_add', 90), ('gemma_route', 30), ('gemma_reduce', 30)):
        assert int(policy[key]) == (count if index == 1 else 0)
baseline = (seconds[0] + seconds[2]) / 2
print(json.dumps(dict(log_sha256=hashlib.sha256(raw).hexdigest(),
                      policies=['ordinary', 'all', 'ordinary'], seconds=seconds,
                      completed_units=32, timed_start_prefix=[2,818,5279,529,7001,563,45518,107,101],
                      timed_argmax_ids=[y for _, y in tokens[:32]],
                      throughput=[32/x for x in seconds],
                      candidate_time_reduction_percent=100*(1-seconds[1]/baseline),
                      baseline_bookend_movement_percent=100*abs(seconds[2]-seconds[0])/baseline,
                      scope='Native-only short screen; quality fails, no adoption/reference speed gate.'),
                 indent=2, sort_keys=True))
