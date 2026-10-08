#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Record external reference provenance and tokenizer consistency after runs."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

from fetch import digest


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--model-root",type=Path,required=True)
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    from exllamav3 import Config,Tokenizer,Model,Cache
    from transformers import AutoTokenizer
    import torch
    from exllamav3.ext import exllamav3_ext
    import ninja
    torch.set_num_threads(4)
    cpu_flags={name:getattr(exllamav3_ext,name)() for name in
               ('exl3_moe_cpu_has_avx2','exl3_moe_cpu_has_avx512_bw','exl3_moe_cpu_has_avx512_vnni','exl3_moe_cpu_has_avx512_vbmi')}
    if any(cpu_flags.values()):raise ValueError('unexpected x86 capability on ARM')
    try:
        exllamav3_ext.exl3_moe_cpu_pool_stress(1,1,1,1)
    except RuntimeError as error:
        cpu_rejection=str(error)
        if 'unavailable in ARM single-GPU reference' not in cpu_rejection:raise
    else:
        raise ValueError('unsupported CPU MoE path did not fail')
    probes=[]
    for folder in sorted(args.model_root.glob("*bpw")):
        config=Config.from_directory(str(folder))
        tokenizer=Tokenizer.from_config(config)
        hf=AutoTokenizer.from_pretrained(str(folder),local_files_only=True,trust_remote_code=False)
        messages=[{"role":"user","content":"Explain why the sky is blue."}]
        rendered=hf.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        hf_ids=hf.encode(rendered,add_special_tokens=False)
        exl_ids=tokenizer.encode(rendered,encode_special_tokens=True).tolist()[0]
        if hf_ids!=exl_ids: raise ValueError("tokenizer disagreement")
        tc=json.loads((folder/"tokenizer_config.json").read_text())
        probes.append({"fixture":folder.name,"template_sha256":hashlib.sha256(tc['chat_template'].encode()).hexdigest(),
                       "rendered_sha256":hashlib.sha256(rendered.encode()).hexdigest(),"token_ids":exl_ids,
                       "transformers_exllamav3_ids_match":True,"declared_context":config.max_seq_len if hasattr(config,'max_seq_len') else json.loads((folder/'config.json').read_text())['max_position_embeddings']})
    # Exercise both prefill/reconstruction and decode so their loaded DSOs appear.
    model=Model.from_config(config);cache=Cache(model,max_num_tokens=512,max_batch_size=1)
    model.load(device="cuda:0")
    with torch.inference_mode():
        model.prefill(torch.full((1,256),1000,dtype=torch.long),{"attn_mode":"flash_attn","cache":cache,"past_len":0,"batch_shape":(1,512)})
        for i in range(5):
            out=model.forward(torch.tensor([[1001]]),{"attn_mode":"flash_attn","cache":cache,"past_len":256+i,"batch_shape":(1,512)})
        torch.cuda.synchronize()
        if not bool(torch.isfinite(out).all()):raise ValueError("runtime inventory probe is nonfinite")
    mapped={}
    for line in Path('/proc/self/maps').read_text().splitlines():
        part=line.split(maxsplit=5)
        if len(part)==6 and part[5].startswith('/') and '.so' in Path(part[5]).name:
            path=Path(part[5])
            if path.is_file():mapped[str(path)]={"bytes":path.stat().st_size,"sha256":digest(path)}
    packages=[]
    for dist in sorted(importlib.metadata.distributions(),key=lambda d:d.metadata['Name'].lower()):
        notices=[]
        for f in dist.files or []:
            if any(term in f.name.lower() for term in ('license','copying','notice')):
                path=Path(dist.locate_file(f))
                if path.is_file():notices.append({"path":str(f),"sha256":digest(path)})
        declaration=dist.metadata.get('License')
        packages.append({"name":dist.metadata['Name'],"version":dist.version,
                         "license_expression":dist.metadata.get('License-Expression'),
                         "license_field":declaration if not declaration or len(declaration)<300 else declaration.splitlines()[0],
                         "license_field_sha256":hashlib.sha256(declaration.encode()).hexdigest() if declaration else None,
                         "notice_files":notices})
    source=[]
    for path in sorted(args.source.rglob('*')):
        if path.is_file() and path.suffix in ('.cu','.cuh','.cpp','.h','.py') and '__pycache__' not in path.parts:
            source.append({"path":str(path.relative_to(args.source)),"sha256":digest(path)})
    compiler={}
    for path in [Path('/usr/bin/g++'),Path('/usr/local/cuda-13.0/bin/nvcc'),Path('/usr/local/cuda-13.0/bin/ptxas'),
                 Path('/usr/local/cuda-13.0/nvvm/bin/cicc'),Path('/usr/local/cuda-13.0/bin/fatbinary'),
                 Path('/usr/local/cuda-13.0/bin/nvlink'),Path(exllamav3_ext.__file__)]:
        compiler[str(path)]={"sha256":digest(path),"bytes":path.stat().st_size}
    for name,cmd in {'nvcc':['/usr/local/cuda-13.0/bin/nvcc','--version'],'g++':['g++','--version'],
                     'dpkg':['dpkg-query','-W','-f','${Package}\t${Version}\n','g++-13','gcc-13','libc6','libstdc++6','python3'],
                     'nvidia-smi':['nvidia-smi','--query-gpu=name,driver_version,compute_cap',
                                   '--format=csv,noheader']}.items():
        compiler[name]=subprocess.check_output(cmd,text=True)
    result={"python":sys.version,"torch":torch.__version__,"torch_cuda":torch.version.cuda,
            "unsupported_cpu_flags":cpu_flags,"unsupported_cpu_moe_rejection":cpu_rejection,
            "torch_gpu_targets":torch.cuda.get_arch_list(),"tokenizer_probes":probes,
            "packages":packages,"source_file_count":len(source),
            "source_tree_sha256":hashlib.sha256(json.dumps(source,sort_keys=True).encode()).hexdigest(),
            "source_license_sha256":digest(args.source/'LICENSE'),
            "compiled_translation_units":[r for r in source if Path(r['path']).suffix in ('.cu','.cpp','.c')],
            "mapped_libraries":mapped,"build":compiler,
            "scope":"External reference inventory and notices, not native compiled-closure clearance or whole-container redistribution approval"}
    args.output.write_text(json.dumps(result,indent=2,default=str)+'\n')
    print('runtime inventory and tokenizer cross-check complete')


if __name__=='__main__':main()
