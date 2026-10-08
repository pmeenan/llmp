#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Run with the pinned reference GGUF reader, after full file hash verification."""
import json
import re
import sys
from gguf import GGUFReader


def account(paths):
    metadata={}; tensors={}
    for path in paths:
        reader=GGUFReader(path,mode='r')
        for k,v in reader.fields.items():
            if k.startswith(('general.','gemma4.','qwen35moe.','deepseek4.','qwen4exp.')):
                value=v.contents()
                if k in metadata and metadata[k]!=value: raise ValueError('Conflicting split metadata')
                metadata[k]=value
        for t in reader.tensors:
            if t.name in tensors: raise ValueError('Duplicate split tensor')
            tensors[t.name]={'bytes':int(t.n_bytes),'shape':[int(n) for n in t.shape],'type':t.tensor_type.name}
    arch=metadata['general.architecture']
    count=int(metadata[arch+'.expert_count']); topk=int(metadata[arch+'.expert_used_count'])
    layers=int(metadata[arch+'.block_count'])-int(metadata.get(arch+'.nextn_predict_layers',0))
    closure={i:[] for i in range(layers)}
    for name,t in tensors.items():
        match=re.fullmatch(r'blk\.(\d+)\.ffn_\w+_exps\.(weight|scale)',name)
        if match and int(match[1])<layers:
            if t['shape'][-1]!=count or t['bytes']%count: raise ValueError('Unexpected expert axis: '+name)
            closure[int(match[1])].append(dict(name=name,bytes=t['bytes']//count,type=t['type']))
        elif '_exps' in name and not match: raise ValueError('Unaccounted expert: '+name)
    if any(not v for v in closure.values()): raise ValueError('Missing expert layer')
    sizes=[sum(p['bytes'] for p in closure[i]) for i in range(layers)]
    total=sum(t['bytes'] for t in tensors.values())
    return dict(architecture=arch,layers=layers,experts=count,topk=topk,expert_bytes=sizes,
                nonexpert_bytes=total-sum(sizes)*count,tensor_bytes=total,
                closure=closure,metadata=metadata,
                note='All non-primary-expert tensors charged, including stored unused MTP')


if __name__=='__main__':print(json.dumps(account(sys.argv[1:]),indent=2))
