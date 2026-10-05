#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0
"""Bound analysis to actual NVTX interval; aggregate activity, never headline speed."""
import collections
import importlib.util
import json
import pathlib
import sqlite3
import sys

sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location('profile_validate',pathlib.Path(__file__).with_name('validate.py'))
validate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validate)
LABEL = 'jitllm.gemma26.paid_prefill'


def union_ns(intervals):
    total = 0
    right = None
    for a,b in sorted(intervals):
        assert b >= a
        if right is None or a > right:
            total += b-a
        elif b > right:
            total += b-right
        right = max(b,right) if right is not None else b
    return total


def clipped(a,b,left,right):
    assert b >= a
    return (max(a,left),min(b,right)) if a < right and b > left else None


def selftest():
    assert union_ns([(0,10),(5,15),(20,25)]) == 20
    assert union_ns([]) == 0
    assert clipped(0,20,5,15) == (5,15)
    assert clipped(15,20,5,15) is None
    pid = (1 << 48) + (93 << 24)
    assert owned({'globalPid':pid},pid) is True
    assert owned({'globalTid':pid+94},pid) is True
    assert owned({'globalTid':(1 << 48)+(80 << 24)+80},pid) is False
    assert owned({},pid) is None
    print('overlap/clipping and actual-process/parent/unknown ownership controls passed')


def owned(row, global_pid):
    if row.get('globalPid') is not None:
        return row['globalPid'] == global_pid
    if row.get('globalTid') is not None:
        return row['globalTid'] & ~0xFFFFFF == global_pid
    return None


def extract(path, completion_record, engine, label=LABEL):
    connection = sqlite3.connect('file:'+str(path)+'?mode=ro',uri=True)
    connection.row_factory = sqlite3.Row
    tables = {r[0] for r in connection.execute("select name from sqlite_master where type='table'")}
    strings = dict(connection.execute('select id,value from StringIds'))
    ranges = [dict(r) for r in connection.execute('select * from NVTX_EVENTS')
              if r['text'] == label or strings.get(r['textId']) == label]
    assert len(ranges) == 1 and ranges[0]['end'] > ranges[0]['start']
    left,right = ranges[0]['start'],ranges[0]['end']
    processes = [dict(r) for r in connection.execute('select * from PROCESSES')]
    # The actual PROCESSES rows authenticate the serialized process-ID field.
    # The low 24 bits of globalTid identify a thread, not another process.
    assert processes and all((r['globalPid'] & 0xFFFFFF) == 0 and
                             ((r['globalPid'] >> 24) & 0xFFFFFF) == r['pid'] for r in processes)
    model = [r for r in processes if r['pid'] == completion_record['child_pid']]
    assert len(model) == 1
    model = model[0]
    expected = 'jitllm_gemma26_prefill_profile' if engine == 'native' else 'llama_prefill_profile'
    # Linux comm can truncate to 15 bytes; PID ownership comes from the atomic
    # waitpid completion receipt and the actual process metadata, not the name.
    assert pathlib.Path(model['name']).name in (expected,expected[:15])
    global_pid = model['globalPid']
    assert ranges[0]['globalTid'] & ~0xFFFFFF == global_pid
    assert ranges[0]['globalTid'] & 0xFFFFFF == completion_record['child_pid']
    symbols = collections.defaultdict(lambda:{'count':0,'clipped_duration_ns_sum':0})
    kernels,copies,memsets,apis,osrt = [],[],[],[],[]
    detail = {}
    activities = {
        'CUPTI_ACTIVITY_KIND_KERNEL':kernels,
        'CUPTI_ACTIVITY_KIND_MEMCPY':copies,
        'CUPTI_ACTIVITY_KIND_MEMSET':memsets,
        'CUPTI_ACTIVITY_KIND_RUNTIME':apis,
        'CUPTI_ACTIVITY_KIND_DRIVER':apis,
        'OSRT_API':osrt}
    for table,intervals in activities.items():
        if table not in tables:
            detail[table] = {'present':False}
            continue
        rows = connection.execute('select * from '+table+' where start < ? and end > ?',(right,left))
        count = 0
        excluded = 0
        unmapped = 0
        for row in rows:
            row = dict(row)
            ownership = owned(row,global_pid)
            if ownership is None:
                unmapped += 1
                continue
            if not ownership:
                excluded += 1
                continue
            span = clipped(row['start'],row['end'],left,right)
            assert span is not None
            intervals.append(span); count += 1
            if table == 'CUPTI_ACTIVITY_KIND_KERNEL':
                key = ('kernel',strings.get(row.get('demangledName'),strings.get(row.get('shortName'),'unresolved')),
                       tuple(row.get(k) for k in ['gridX','gridY','gridZ']),
                       tuple(row.get(k) for k in ['blockX','blockY','blockZ']),row.get('streamId'))
            elif table == 'CUPTI_ACTIVITY_KIND_MEMCPY':
                key = ('memcpy',row.get('copyKind'),row.get('bytes'),row.get('streamId'))
            elif table == 'CUPTI_ACTIVITY_KIND_MEMSET':
                key = ('memset',row.get('bytes'),row.get('streamId'))
            else:
                key = (table,strings.get(row.get('nameId'),'unresolved'))
            aggregate = symbols[key]
            aggregate['count'] += 1
            aggregate['clipped_duration_ns_sum'] += span[1]-span[0]
        detail[table] = {'present':True,'model_intersecting_events':count,
                         'excluded_other_process_events':excluded,'unknown_process_events':unmapped}
    assert kernels, 'no actual CUDA kernel activities in paid range'
    gpu = kernels+copies+memsets
    gpu_union = union_ns(gpu)
    assert gpu_union <= right-left
    connection.close()
    return {'model_process':model,'paid_range_model_owned':True,
            'global_id_encoding_validated_against_process_rows':True,
            'paid_range_wall_ns':right-left,'gpu_activity_union_ns':gpu_union,
            'kernel_union_ns':union_ns(kernels),'copy_union_ns':union_ns(copies),
            'memset_union_ns':union_ns(memsets),'cuda_api_union_ns':union_ns(apis),
            'osrt_union_ns':union_ns(osrt),'host_or_unattributed_ns':right-left-gpu_union,
            'tables':detail,'aggregates':[{'key':list(k),**v} for k,v in sorted(symbols.items(),key=lambda i:repr(i[0]))],
            'limitations':['Instrumented spans are diagnostic, not competitive timings.',
                          'Union and summed categories overlap; do not add them.',
                          'Residual wall time is not measured planner/scheduler CPU time.',
                          'Observer parent waits and foreign process events are excluded; unmapped table events are unknown.',
                          'Kernel roles/head attribution require source plus actual geometry; no automatic role labels.',
                          'A launch API count is not a CUDA kernel count.']}


if __name__ == '__main__':
    if sys.argv[1:] == ['--self-test']:
        selftest()
    else:
        assert len(sys.argv) == 10
        root,scratch = pathlib.Path(sys.argv[1]),pathlib.Path(sys.argv[2])
        expected,name,engine,job,own_sha,analysis_sha,validated_sha = sys.argv[3:]
        assert validate.ring.identity(pathlib.Path(__file__))['sha256'] == analysis_sha
        validated_path = scratch/(name+'-validated.json')
        assert validate.ring.identity(validated_path)['sha256'] == validated_sha
        validated = json.loads(validated_path.read_text())
        assert validated['source_freeze']['sha256'] == expected
        control_name = engine+'-control-validated.json'
        assert validate.ring.identity(scratch/control_name)['sha256'] == own_sha
        own = json.loads((scratch/control_name).read_text())
        pre = scratch/'source-identities.json'
        assert validate.ring.identity(pre)['sha256'] == expected and own['source_freeze'] == validate.ring.identity(pre)
        assert json.loads(pre.read_text())['environment'] == validate.environment(root,scratch)
        result = validate.check_run(scratch,name,engine,job)
        assert result['record'] == own['record']
        result.update({'source_freeze':validate.ring.identity(pre),'annotated_control':validate.ring.identity(scratch/control_name),
                       'sqlite':validate.ring.identity(scratch/(name+'.sqlite')),
                       'analysis_source':validate.ring.identity(pathlib.Path(__file__)),
                       'validated_result':validate.ring.identity(validated_path),
                       'profile':extract(scratch/(name+'.sqlite'),json.loads((scratch/(name+'-completion.json')).read_text()),engine)})
        validate.write(scratch/(name+'-analysis-v2.json'),result)
