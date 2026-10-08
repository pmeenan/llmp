#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Compact validated replay outputs into aggregate curves, never raw routes."""
import argparse
import json
from pathlib import Path
from run import sha


def summarize(path):
    data=json.loads(path.read_text());curves=[]
    for simulation in data['simulations']:
        curve={k:v for k,v in simulation.items() if k!='requests'}
        if simulation['feasible']:
            requests=simulation['requests']
            curve['total_weight_read_bytes']=sum(r['weight_read_bytes'] for r in requests)
            curve['total_state_written_bytes']=sum(r['state_written_bytes'] for r in requests)
            curve['total_state_restored_bytes']=sum(r['state_restored_bytes'] for r in requests)
            curve['requests']=[{k:r[k] for k in ('model','turn','capture_variant','state_reuse_fallback','reused','prefill_tokens',
                'weight_read_bytes','first_token_weight_bytes','state_written_bytes',
                'state_restored_bytes','state_snapshot_bytes','first_token_storage_seconds',
                'first_token_serial_storage_seconds','first_token_slow_storage_seconds',
                'decode_miss_bytes','decode_no_overlap_ms','decode_half_overlap_ms')} for r in requests]
        curves.append(curve)
    return dict(replay_sha256=sha(path),models=data['models'],prediction_checks=data.get('prediction_checks',[]),capture_summaries=data['captures'],curves=curves)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('output',type=Path)
    p.add_argument('--replay',action='append',nargs=2,metavar=('LABEL','PATH'),required=True)
    p.add_argument('--locality',action='append',nargs=2,metavar=('LABEL','PATH'),default=[])
    p.add_argument('--parallel',type=Path)
    a=p.parse_args();out=dict(replays={label:summarize(Path(path)) for label,path in a.replay},
        locality={label:json.loads(Path(path).read_text()) for label,path in a.locality})
    if a.parallel:out['parallel']=json.loads(a.parallel.read_text())
    with a.output.open('x') as f:json.dump(out,f,indent=2);f.write('\n')


if __name__=='__main__':main()
