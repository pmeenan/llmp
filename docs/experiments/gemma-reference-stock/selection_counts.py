#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Summarize host-side try_fuse decisions, not kernel launches or graph replays."""
import collections
import json
import pathlib
import re
import sys


def summarize(paths):
    arms = {}
    for path in paths:
        arm = None
        for line in path.read_text().splitlines():
            if '] step ' in line and ' start: ' in line and ('/run.sh ' in line or '/run_reference.sh ' in line):
                match = re.search(r'/run(?:_reference)?\.sh ([a-z_]+) ([a-z0-9-]+)', line)
                assert match, 'unexpected diagnostic invocation'
                policy, arm = match.groups()
                assert arm not in arms, 'duplicate process record'
                arms[arm] = dict(policy=policy, banner=None, selections=collections.Counter(),
                                 examples={}, shapes=collections.Counter())
            if line.startswith('LLMP_REF_CONTROLLER '):
                assert arm is not None and arms[arm]['banner'] is None
                arms[arm]['banner'] = int(line.split('=')[1])
            if line.startswith('LLMP_REF_FUSION '):
                assert arm is not None
                fields = line.split(' | ')
                count = int(fields[0].split('=')[1])
                assert len(fields)-1 == count
                tensors = [item.rsplit(':', 3) for item in fields[1:]]
                assert all(len(tensor) == 4 for tensor in tensors)
                operations = ' -> '.join(tensor[1] for tensor in tensors)
                arms[arm]['selections'][operations] += 1
                arms[arm]['examples'].setdefault(operations, fields[1:])
                for _, op, dtype, shape in tensors:
                    arms[arm]['shapes'][op + ':' + dtype + ':' + shape] += 1
    expected_policy = dict(all=0, no_routing=1, no_reduction=2, no_norm_mul=3, no_softcap=4)
    for arm in arms.values():
        assert arm['banner'] == expected_policy[arm['policy']], 'controller did not execute'
        arm['total_host_fusion_decisions'] = sum(arm['selections'].values())
    return dict(scope='host try_fuse selections; not CUDA launch or replay counts', processes=arms)


def main():
    assert len(sys.argv) > 1
    print(json.dumps(summarize(list(map(pathlib.Path, sys.argv[1:]))), indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
