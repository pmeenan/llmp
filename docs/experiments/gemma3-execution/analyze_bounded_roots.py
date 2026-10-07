#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0
"""Reproduce the bounded-root ring adoption gates using private authenticated inputs."""
import hashlib,importlib.util,json,shutil,struct,subprocess,sys
from pathlib import Path
spec=importlib.util.spec_from_file_location('joint',Path(__file__).with_name('analyze_joint_prefill.py'));core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
h=core.digest
mode=sys.argv[1];r=Path(sys.argv[2]);ring=r/'ring'
if mode=='prepare':
 first=Path(sys.argv[3]);expected={'models.json':'41bfc65af7305b53e841a799460649ddc0fe67f77dd4dedcea5aa08677385279','inputs.json':'d1b71004a5b80370dde6549f867accbe501346f681ffedea03fa16faa1e0b475','quality.json':'e0744d84e7d1ced396ae5b9863e44efc1bf43f71725b0540fd215fe88a36ecb1','own-control.json':'053eaf61bacdd9016fa9ebf68bbceadd9841a83207963073eb65c9c5f77d5d04','llama-joint':'f7dbb1163c993d40744c199c8745387ea5e28c8dfcc279bb6191ecfb86b19e02','cycle.json':'92cf9b9cba7432290d4c22f73c409ccff267d16836e4ab82b6482903b493eaa5'}
 for p,digest in expected.items():assert h(first/p)==digest,(p,'first-screen guard')
 assert json.loads((first/'quality.json').read_text())['passed'];assert json.loads((first/'own-control.json').read_text())['state']['restore_equal']
 shutil.copyfile(first/'models.json',r/'models.json')
 unequal=r/'unequal';unequal.mkdir()
 for p in ('inputs.json','quality.json','own-control.json'):shutil.copyfile(first/p,unequal/p)
 for slot in range(2):
  (unequal/f'input{slot}').mkdir();shutil.copyfile(first/f'input{slot}/ids.i32',unequal/f'input{slot}/ids.i32');shutil.copyfile(first/f'prompt{slot}.txt',unequal/f'prompt{slot}.txt')
 core.inputs(unequal)
 model=json.loads((r/'models.json').read_text());st=Path(model['source']['path']).stat();assert (st.st_size,st.st_ino,st.st_mtime_ns)==(model['source']['bytes'],model['inode'],model['mtime_ns'])
 for p,digest in model['files'].items():
  st=Path(p).stat();assert dict(bytes=st.st_size,inode=st.st_ino,mtime_ns=st.st_mtime_ns)==model['file_stats'][p]
  if not p.endswith('.safetensors'):assert h(Path(p))==digest
 ring.mkdir();shutil.copyfile(first/'llama-joint',ring/'llama-joint');(ring/'llama-joint').chmod(0o700)
 for slot in range(2):shutil.copyfile(first/f'prompt{slot}.txt',ring/f'prompt{slot}.txt')
elif mode=='inputs':
 result={}
 for slot,rows in enumerate((1319,1575)):
  p=ring/f'input{slot}/ids.i32';raw=p.read_bytes();ids=struct.unpack(f'<{len(raw)//4}i',raw);assert len(ids)==rows and ids[0]==2 and all(0<=n<262208 for n in ids)
  result[str(slot)]=dict(rows=rows,prefix=rows-39,ids_sha256=h(p),text_sha256=h(ring/f'prompt{slot}.txt'))
 with (ring/'inputs.json').open('x') as f:json.dump(result,f,indent=2)
 paths=[r/'models.json']+[r/'unequal'/p for p in ('inputs.json','quality.json','own-control.json','input0/ids.i32','input1/ids.i32','prompt0.txt','prompt1.txt')]+[ring/p for p in ('inputs.json','input0/ids.i32','input1/ids.i32','prompt0.txt','prompt1.txt','llama-joint')]
 with (r/'prepared.sha256').open('x') as f:f.write(''.join(f'{h(p)}  {p}\n' for p in paths))
elif mode=='own':
 assert [len(ids)-39 for ids in core.inputs(ring)]==[1280,1536]
 core.own(ring)
 for name in ('heads.f32','chosen.i32','final.f32','state.json'):
  assert h(ring/'padded-own'/name)==h(ring/'native-own1'/name)==h(ring/'native-own2'/name),(name,'ring policy difference')
 with (r/'ring-own.sha256').open('x') as f:f.write(f'{h(ring/"own-control.json")}  {ring/"own-control.json"}\n')
elif mode=='quality':
 for arm in ('stock-teacher1','stock-teacher2'):
  receipt=json.loads((ring/f'{arm}-container-retired.json').read_text());assert receipt['container_absent_after_checked_docker_query'] is True
  q=subprocess.run(['sudo','-n','docker','container','inspect',receipt['cid']],capture_output=True,text=True);assert q.returncode!=0 and 'No such container' in q.stderr
 core.quality(ring)
 with (r/'ring-quality.sha256').open('x') as f:f.write(f'{h(ring/"quality.json")}  {ring/"quality.json"}\n')
else:raise ValueError(mode)
