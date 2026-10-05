#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0
"""Authenticate actual original launch families; no traced performance claims."""
import hashlib
import json
import pathlib
import sqlite3
import sys


def identify(path):
    b = path.read_bytes()
    return {'bytes': len(b), 'sha256': hashlib.sha256(b).hexdigest()}


def main(base, output):
    result = {'schema': 1, 'arms': {}, 'limits': 'Default graph-level trace includes eager kernels and graph launch calls; event counts are not total replay kernel launches. No trace duration enters paid timings.'}
    for arm, untraced in [('A', 'original-first'), ('A0', 'original-unfused')]:
        traced = base/('original-traced-'+arm)
        assert json.loads((traced/'completion.json').read_text()) == {'phase': 'completed_after_backend_release', 'paid_chains': 32}
        for role in ['first', 'repeat', 'fresh']:
            assert (traced/(role+'.f32')).read_bytes() == (base/untraced/(role+'.f32')).read_bytes()
        database = base/('original-launches-'+arm+'.sqlite')
        c = sqlite3.connect('file:'+str(database)+'?mode=ro', uri=True)
        rows = c.execute('''SELECT s.value,k.gridX,k.gridY,k.gridZ,k.blockX,k.blockY,k.blockZ,
                                   k.staticSharedMemory,k.dynamicSharedMemory,k.registersPerThread,COUNT(*)
                           FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName=s.id
                           GROUP BY s.value,k.gridX,k.gridY,k.gridZ,k.blockX,k.blockY,k.blockZ,
                                    k.staticSharedMemory,k.dynamicSharedMemory,k.registersPerThread''').fetchall()
        kernels = []
        fused, separate, glu = 0, 0, 0
        for name, gx, gy, gz, bx, by, bz, static, dynamic, registers, count in rows:
            if name.startswith('quantize_q8_1('):
                family = 'Q8_1_input_preparation'
                assert [gx,gy,gz] in [[22,1,1],[84,1,1]] and [bx,by,bz] == [256,1,1]
            elif 'mul_mat_vec_q<(ggml_type)12, (int)1, (bool)1, (bool)0, (bool)0>' in name:
                family = 'Q4_K_fused_GeGLU'; fused += count
                assert [gx,gy,gz] == [21504,1,1] and [bx,by,bz] == [32,4,1]
            elif 'mul_mat_vec_q<(ggml_type)12, (int)1, (bool)0, (bool)0, (bool)0>' in name:
                family = 'Q4_K_separate_product'; separate += count
                assert [gx,gy,gz] == [21504,1,1] and [bx,by,bz] == [32,4,1]
            elif 'mul_mat_vec_q<(ggml_type)14, (int)1, (bool)0, (bool)0, (bool)1>' in name:
                family = 'Q6_K_down'; assert count == 1
                assert [gx,gy,gz] == [5376,1,1] and [bx,by,bz] == [32,8,1]
            elif 'unary_gated_op_kernel<&op_gelu, float>' in name:
                family = 'separate_GeGLU'; glu += count
                assert [gx,gy,gz] == [84,1,1] and [bx,by,bz] == [256,1,1]
            else:
                raise AssertionError(name)
            kernels.append({'family': family, 'demangled_name': name, 'grid': [gx,gy,gz], 'block': [bx,by,bz],
                            'static_shared_bytes': static, 'dynamic_shared_bytes': dynamic,
                            'registers_per_thread': registers, 'observed_kernel_events': count})
        launches = c.execute("SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_RUNTIME k JOIN StringIds s ON k.nameId=s.id WHERE s.value LIKE 'cudaGraphLaunch%'").fetchone()[0]
        c.close()
        assert (fused,separate,glu,launches,len(rows)) == ((1,0,0,76,4) if arm == 'A' else (0,2,1,152,5))
        result['arms'][arm] = {'trace_report': identify(base/('original-launches-'+arm+'.nsys-rep')),
                              'sqlite_export': identify(database), 'completion': identify(traced/'completion.json'),
                              'complete_traced_first_repeat_fresh_byte_exact_to_untraced': True,
                              'observed_cuda_graph_launch_calls': launches, 'kernels': kernels}
    with output.open('x') as f:
        json.dump(result, f, indent=2, sort_keys=True); f.write('\n')
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    assert len(sys.argv) == 3
    main(*map(pathlib.Path, sys.argv[1:]))
