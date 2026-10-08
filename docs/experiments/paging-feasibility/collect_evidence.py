#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Record external evidence identities and aggregate normal-reference results."""
import argparse
import json
from pathlib import Path
from statistics import median
from run import sha


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('output',type=Path)
    p.add_argument('--capture',action='append',nargs=2,default=[],metavar=('LABEL','DIRECTORY'))
    p.add_argument('--input',action='append',nargs=2,default=[],metavar=('LABEL','FILE'))
    p.add_argument('--normal-reference',type=Path)
    p.add_argument('--prediction-comparison',action='append',nargs=3,default=[],metavar=('LABEL','LEFT','RIGHT'))
    a=p.parse_args();out=dict(captures={},inputs={},harness_sources={},prediction_comparisons={})
    here=Path(__file__).resolve().parent
    for path in sorted(here.iterdir()):
        if path.suffix in ('.py','.cc'):out['harness_sources'][path.name]=sha(path)
    for label,path in a.capture:
        root=Path(path);receipt=json.loads((root/'receipt.json').read_text())
        for run in receipt['runs']:
            for ext,key in [('jsonl','events_sha256'),('predictions','predictions_sha256'),('log','log_sha256')]:
                digest=sha(root/(run['name']+'.'+ext))
                if key in run and digest!=run[key]:raise ValueError('Capture evidence mismatch')
                run[key]=digest
            run['prediction_records']=len((root/(run['name']+'.predictions')).read_text().splitlines())
        out['captures'][label]=dict(external_directory=root.name,receipt_file_sha256=sha(root/'receipt.json'),receipt=receipt)
    for label,path in a.input:
        path=Path(path);out['inputs'][label]=dict(filename=path.name,bytes=path.stat().st_size,sha256=sha(path))
    for label,left,right in a.prediction_comparison:
        left=Path(left);right=Path(right);a_lines=left.read_text().splitlines();b_lines=right.read_text().splitlines()
        out['prediction_comparisons'][label]=dict(left_sha256=sha(left),right_sha256=sha(right),
            left_count=len(a_lines),right_count=len(b_lines),
            different_records=sum(x!=y for x,y in zip(a_lines,b_lines))+abs(len(a_lines)-len(b_lines)))
    if a.normal_reference:
        data=json.loads(a.normal_reference.read_text());rows=[]
        for r in data['requests']:
            result=r['reference_result'];timing=result['final']['timings']
            rows.append(dict(model=r['model'],turn=r['turn'],prompt_tokens=len(r['prompt']),
                output_tokens=len(r['expected_tokens']),reused_tokens=timing['cache_n'],
                evaluated_prompt_tokens=timing['prompt_n'],switch_first_token_seconds=r['switch_first_token_s'],
                generation_tokens_per_second=timing['predicted_per_second'],
                state_saved_bytes=r['state_receipt']['n_written'],
                state_restored_bytes=r['restore_receipt']['n_read'] if r['restore_receipt'] else 0))
        groups=[]
        for model in sorted({r['model'] for r in rows}):
            for phase in ('first_visit','return'):
                selected=[r for r in rows if r['model']==model and (r['turn']==0)==(phase=='first_visit')]
                summary=dict(model=model,phase=phase,count=len(selected))
                for field in rows[0]:
                    if field in ('model','turn'):continue
                    values=[r[field] for r in selected]
                    summary[field]=dict(min=min(values),median=median(values),max=max(values))
                groups.append(summary)
        out['normal_reference']=dict(input_sha256=sha(a.normal_reference),groups=groups,
            caveat='Single run; unconditioned cache; normal optimized/lazy configuration; timing includes harness lifecycle')
    with a.output.open('x') as f:json.dump(out,f,indent=2);f.write('\n')


if __name__=='__main__':main()
