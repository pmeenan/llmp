#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Combine complete, disjoint model captures without claiming a common host/build."""
import argparse
import json
from pathlib import Path
from run import IMAGE, REVISION, sha
from prediction_checks import validate_prediction_pair


def merge(roots, output):
    combined=dict(image=IMAGE,revision=REVISION,cuda_disable_fusion=True,cuda_disable_graphs=True,
                  components=[],spec=dict(models=[],configurations=None,trace_sha256=None),runs=[])
    seen=set();links=[];names=set()
    if not roots:raise ValueError('No capture roots')
    for root in roots:
        root=root.resolve();r=json.loads((root/'receipt.json').read_text());spec=r['spec']
        if r.get('parallel') or r.get('restore_probe') or r['image']!=IMAGE or r['revision']!=REVISION or not r.get('cuda_disable_fusion') or not r.get('cuda_disable_graphs'):raise ValueError('Incompatible capture')
        for key in ('configurations','trace_sha256'):
            if combined['spec'][key] is None:combined['spec'][key]=spec[key]
            elif combined['spec'][key]!=spec[key]:raise ValueError('Different workload/configurations')
        if len({m['id'] for m in spec['models']})!=len(spec['models']):raise ValueError('Duplicate model specification')
        configurations=[tuple(c) for c in spec['configurations']]
        if not configurations or len(set(configurations))!=len(configurations):raise ValueError('Empty/duplicate configurations')
        ids={run['model'] for run in r['runs']}
        if not ids or seen&ids:raise ValueError('Empty/duplicate model')
        expected={(name,b,reuse,swa,t) for name in ids for b,reuse,swa in spec['configurations'] for t in (False,True)}
        found={(x['model'],x['batch'],x['reuse'],x['swa'],x['trace']) for x in r['runs']}
        if found!=expected or len(r['runs'])!=len(expected):raise ValueError('Incomplete model capture')
        pairs={(x['model'],x['batch'],x['reuse'],x['swa'],x['trace']):x for x in r['runs']}
        for key,run in pairs.items():
            if key[-1]:
                control=pairs[(*key[:-1],False)]
                validate_prediction_pair(control,run,root/(control['name']+'.predictions'),root/(run['name']+'.predictions'),allow_drift=True)
        models=[m for m in spec['models'] if m['id'] in ids]
        if {m['id'] for m in models}!=ids:raise ValueError('Missing model spec')
        combined['spec']['models'].extend(models);seen.update(ids)
        combined['components'].append(dict(receipt_sha256=sha(root/'receipt.json'),receipt=r))
        for run in r['runs']:
            run=dict(run)
            if not run['name'] or Path(run['name']).name!=run['name'] or run['name'] in names:raise ValueError('Invalid/duplicate run name')
            names.add(run['name'])
            if not all(field in run for field in ('events_sha256','predictions_sha256')):raise ValueError('Missing capture identity')
            for suffix,field in [('jsonl','events_sha256'),('predictions','predictions_sha256'),('log','log_sha256')]:
                path=root/(run['name']+'.'+suffix);digest=sha(path)
                if field in run and run[field]!=digest:raise ValueError('Capture hash mismatch')
                run[field]=digest;links.append(path)
            combined['runs'].append(run)
    output.mkdir(mode=0o700)
    for path in links:(output/path.name).symlink_to(path)
    (output/'receipt.json').write_text(json.dumps(combined,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('output',type=Path);p.add_argument('captures',nargs='+',type=Path)
    a=p.parse_args();merge(a.captures,a.output)


if __name__=='__main__':main()
