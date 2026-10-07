#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0
"""Reproduce the bounded-root ring adoption gates using private authenticated inputs."""
import hashlib,importlib.util,json,shutil,struct,sys
from pathlib import Path
spec=importlib.util.spec_from_file_location('joint',Path(__file__).with_name('analyze_joint_prefill.py'));core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
h=core.digest
mode=sys.argv[1];r=Path(sys.argv[2]);ring=r/'ring'
if mode=='prepare':
 first,previous=map(Path,sys.argv[3:]);expected={'models.json':'5d5721f0556f132b9eab5e596447d24a60ecaaf9b4335f364213d8a1b9f6ee5f','inputs.json':'f9114e679cd2ef007741adba3116bb2edce7dd84b0db66f0f07dfb6a297e2316','quality.json':'4086e51ef34186b0ba0c7b5467c6428d976e11de394dcd5bf1ec8cd6f6eb3ef9','own-control.json':'8ae0a6526562f16421326fcb703066e6381477e262e2bac97da356764c8f5cd6','llama-joint':'49961825ca4b2e1c9ee0f5a2073aee1646cdd961cf9b8f594dc202d883780def'}
 for p,digest in expected.items():assert h(first/p)==digest,(p,'first-screen guard')
 assert json.loads((first/'quality.json').read_text())['passed'];assert json.loads((first/'own-control.json').read_text())['state']['restore_equal']
 for p in expected:
  if p!='llama-joint':shutil.copyfile(first/p,r/p)
 for slot in range(2):
  (r/f'input{slot}').mkdir();shutil.copyfile(first/f'input{slot}/ids.i32',r/f'input{slot}/ids.i32');shutil.copyfile(first/f'prompt{slot}.txt',r/f'prompt{slot}.txt')
 core.inputs(r)
 model=json.loads((r/'models.json').read_text());st=Path(model['source']['path']).stat();assert (st.st_size,st.st_ino,st.st_mtime_ns)==(model['source']['bytes'],model['inode'],model['mtime_ns'])
 for p,digest in model['files'].items():assert h(Path(p))==digest
 ring.mkdir();shutil.copy2(first/'llama-joint',ring/'llama-joint')
 for slot,digest in enumerate(('84a72b3bab88f084898edb6b202d7cae17cad20849c3cf556f4800c3f70159a2','ac2cc3d0953da8ef932203721f91fc9f5f5ba3473208d5598ebaaeeb030f0a89')):
  p=previous/f'prompt{slot}.txt';assert h(p)==digest;shutil.copyfile(p,ring/p.name)
elif mode=='inputs':
 result={}
 for slot,rows in enumerate((4391,4903)):
  p=ring/f'input{slot}/ids.i32';raw=p.read_bytes();ids=struct.unpack(f'<{len(raw)//4}i',raw);assert len(ids)==rows and ids[0]==2 and all(0<=n<256000 for n in ids)
  result[str(slot)]=dict(rows=rows,ids_sha256=h(p),text_sha256=h(ring/f'prompt{slot}.txt'))
 with (ring/'inputs.json').open('x') as f:json.dump(result,f,indent=2)
 paths=[r/p for p in ('models.json','inputs.json','quality.json','own-control.json','input0/ids.i32','input1/ids.i32','prompt0.txt','prompt1.txt')]+[ring/p for p in ('inputs.json','input0/ids.i32','input1/ids.i32','prompt0.txt','prompt1.txt','llama-joint')]
 with (r/'prepared.sha256').open('x') as f:f.write(''.join(f'{h(p)}  {p}\n' for p in paths))
elif mode=='own':
 assert [len(ids)-39 for ids in core.inputs(ring)]==[4352,4864]
 core.own(ring)
 for name in ('prefill.f32','heads.f32','chosen.i32','final.f32','state.json'):
  assert h(ring/'padded-own'/name)==h(ring/'native-own1'/name)==h(ring/'native-own2'/name),(name,'ring policy difference')
 with (r/'ring-own.sha256').open('x') as f:f.write(f'{h(ring/"own-control.json")}  {ring/"own-control.json"}\n')
elif mode=='quality':
 for arm in ('stock-teacher1','stock-teacher2'):
  receipt=json.loads((ring/f'{arm}-container-retired.json').read_text());assert receipt['container_absent_after_checked_docker_query'] is True
 core.quality(ring)
 with (r/'ring-quality.sha256').open('x') as f:f.write(f'{h(ring/"quality.json")}  {ring/"quality.json"}\n')
else:raise ValueError(mode)
