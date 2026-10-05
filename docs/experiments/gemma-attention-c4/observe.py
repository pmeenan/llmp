#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0
"""Extract actual original replay kernel family/geometry, never traced timing."""
import hashlib
import json
import pathlib
import sqlite3
import sys

base, output = map(pathlib.Path, sys.argv[1:])
def identify(path):
    b = path.read_bytes()
    return {'bytes': len(b), 'sha256': hashlib.sha256(b).hexdigest()}
completion = json.loads((base/'original-traced/completion.json').read_text())
assert completion == {'phase': 'completed_after_backend_release', 'owners': 4,
                      'paid_waves': 32, 'completed_units': 128}
for role in ['first', 'repeat', 'fresh']:
    assert (base/'original-traced'/(role+'.f32')).read_bytes() == (base/'original-first'/(role+'.f32')).read_bytes()
connection = sqlite3.connect('file:'+str(base/'original-launches.sqlite')+'?mode=ro', uri=True)
rows = connection.execute('''
    SELECT s.value, k.gridX, k.gridY, k.gridZ, k.blockX, k.blockY, k.blockZ,
           k.staticSharedMemory, k.dynamicSharedMemory, k.registersPerThread, COUNT(*)
    FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName=s.id
    GROUP BY s.value, k.gridX, k.gridY, k.gridZ, k.blockX, k.blockY, k.blockZ,
             k.staticSharedMemory, k.dynamicSharedMemory, k.registersPerThread
''').fetchall()
assert len(rows) == 3
families = {}
for name, gx, gy, gz, bx, by, bz, static, dynamic, registers, count in rows:
    if 'flash_attn_ext_f16<(int)256, (int)256, (int)4, (int)2,' in name:
        family = 'MMA_F16_256_256_tile4_GQA2'
        assert [gx,gy,gz] == [48,1,1] and [bx,by,bz] == [32,4,1]
    elif 'flash_attn_mask_to_KV_max<(int)4>' in name:
        family = 'mask_prepass_tile4'
    elif 'flash_attn_stream_k_fixup_general<(int)256, (int)4, (int)2>' in name:
        family = 'stream_k_fixup_256_tile4_GQA2'
    else: raise AssertionError(name)
    assert family not in families and count == 1
    families[family] = {'demangled_name': name, 'grid': [gx,gy,gz], 'block': [bx,by,bz],
                        'static_shared_bytes': static, 'dynamic_shared_bytes': dynamic,
                        'registers_per_thread': registers, 'observed_kernel_events': count}
launches = connection.execute('''
    SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_RUNTIME k JOIN StringIds s ON k.nameId=s.id
    WHERE s.value LIKE 'cudaGraphLaunch%'
''').fetchone()[0]
assert launches == 38
connection.close()
result = {'schema':1, 'origin':'Actual original public-backend execution on constructed native-origin four-stream descriptors; not original model graph capture.',
          'trace_report':identify(base/'original-launches.nsys-rep'),
          'sqlite_export':identify(base/'original-launches.sqlite'),
          'application_completion':identify(base/'original-traced/completion.json'),
          'complete_traced_first_repeat_fresh_byte_exact_to_untraced':True,
          'kernels':families, 'observed_cuda_graph_launch_calls':launches,
          'limits':'Default graph-level trace reports one eager kernel triad and 38 graph launch calls. Kernel event counts are not total replay kernel launches. No trace duration enters paid performance.'}
with output.open('x') as f:
    json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result,indent=2))
