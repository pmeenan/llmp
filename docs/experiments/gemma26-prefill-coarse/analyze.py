#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0
"""Thread-owned inclusive scopes; elapsed/charged CPU are separate clock domains."""
import collections
import importlib.util
import json
import pathlib
import re
import sqlite3
import sys

sys.dont_write_bytecode = True
HERE = pathlib.Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


validate = load('coarse_validate', HERE/'validate.py')
old = load('prior_trace', HERE.parent/'gemma26-prefill-profile/trace_v2.py')
PHASES = ['wave', 'places.call', 'places.body', 'inputs', 'state.grow', 'state.describe',
          'state.describe.body', 'state.acquire', 'state.register', 'state.register.body',
          'plan.hit', 'plan.miss', 'graph.build_bind', 'plan.first', 'placement', 'plan.second',
          'plan.bind', 'coverage', 'stage', 'job.completed', 'job.submit', 'publish',
          'host.release', 'refresh.call', 'refresh.body', 'cohort.hold', 'probe.busy', 'probe.wait']
PARENTS = {'places.call':'wave', 'places.body':'places.call', 'inputs':'wave',
           'state.grow':'inputs', 'plan.hit':'wave', 'plan.miss':'wave',
           'graph.build_bind':'plan.miss', 'plan.first':'graph.build_bind',
           'placement':'graph.build_bind', 'plan.second':'graph.build_bind',
           'plan.bind':'plan.miss', 'coverage':'plan.miss', 'stage':'wave',
           'job.completed':'wave', 'job.submit':'job.completed', 'publish':'wave',
           'host.release':'wave', 'state.describe':'state.grow',
           'state.describe.body':'state.describe', 'state.acquire':'state.grow',
           'state.register':'state.grow', 'state.register.body':'state.register',
           'refresh.call':'state.grow', 'refresh.body':'refresh.call', 'cohort.hold':'state.grow'}
MANDATORY = {'wave', 'places.call', 'places.body', 'inputs', 'state.grow',
             'stage', 'job.completed', 'job.submit', 'publish', 'host.release'}
MISS = {'graph.build_bind', 'plan.first', 'placement', 'plan.second', 'plan.bind', 'coverage'}
FRESH = {'state.describe', 'state.describe.body', 'state.acquire', 'state.register',
         'state.register.body', 'refresh.call', 'refresh.body', 'cohort.hold'}
SCHEDULER = {'places.body', 'state.describe.body', 'state.register.body', 'refresh.body'}


def roster(events, outer):
    """Source-defined roles and exact conditional rosters, not observed-count inference."""
    roles = {}
    for role, phase in [('caller', 'wave'), ('scheduler', 'places.body'), ('submission', 'job.submit')]:
        rows = [row for row in events if row['phase'] == phase]
        assert len(rows) == 8 and sorted(row['ordinal'] for row in rows) == list(range(8))
        tids = {row['tid'] for row in rows}
        assert len(tids) == 1
        roles[role] = tids.pop()
    assert roles['caller'] == outer['tid'] and len(set(roles.values())) == 3
    for ordinal in range(8):
        rows = [row for row in events if row['ordinal'] == ordinal]
        by_phase = {row['phase']:row for row in rows}
        assert len(by_phase) == len(rows), 'duplicate phase in one chunk'
        phases = set(by_phase)
        assert MANDATORY <= phases
        assert len(phases & {'plan.hit','plan.miss'}) == 1
        expected = MANDATORY | (MISS | {'plan.miss'} if 'plan.miss' in phases else {'plan.hit'})
        if phases & FRESH:
            expected |= FRESH  # All completed fresh work, or genuinely no fresh extents.
        assert phases == expected, 'missing or unexpected source phase'
        for phase,row in by_phase.items():
            role = 'scheduler' if phase in SCHEDULER else 'submission' if phase == 'job.submit' else 'caller'
            assert row['tid'] == roles[role], 'wrong source worker thread'
            if phase == 'wave':
                parent = outer
            else:
                parent = by_phase[PARENTS[phase]]
            assert parent['start'] <= row['start'] and parent['end'] >= row['end'], 'wrong source parent'
        order = ['places.call', 'inputs', 'plan.miss' if 'plan.miss' in phases else 'plan.hit',
                 'stage', 'job.completed', 'publish', 'host.release']
        if 'plan.miss' in phases:
            child_order = ['plan.first','placement','plan.second']
            assert all(by_phase[a]['end'] <= by_phase[b]['start'] for a,b in zip(child_order,child_order[1:]))
            assert by_phase['graph.build_bind']['end'] <= by_phase['plan.bind']['start']
            assert by_phase['plan.bind']['end'] <= by_phase['coverage']['start']
        if phases & FRESH:
            fresh_order = ['state.describe','state.acquire','state.register','refresh.call','cohort.hold']
            assert all(by_phase[a]['end'] <= by_phase[b]['start'] for a,b in zip(fresh_order,fresh_order[1:]))
        assert all(by_phase[a]['end'] <= by_phase[b]['start'] for a,b in zip(order,order[1:]))
    return roles


def counters(text):
    meta = re.findall(r'^PREFILL_COARSE known_static_bytes=(\d+) clock=thread_cpu_ns elapsed=monotonic_ns inclusive=1$', text, re.M)
    assert len(meta) == 1 and 0 < int(meta[0]) <= 4096
    rows = re.findall(r'^PREFILL_SCOPE owner=(\d+) tid=(\d+) phase=([a-z._]+) count=(\d+) wall_ns=(\d+) thread_cpu_ns=(\d+)$', text, re.M)
    assert rows
    parsed = {}
    owners = {}
    for owner, tid, phase, count, wall, cpu in rows:
        owner, tid, count, wall, cpu = map(int, [owner, tid, count, wall, cpu])
        assert 0 <= owner < 4 and tid > 0 and phase in PHASES and count > 0 and wall > 0
        assert owners.setdefault(owner, tid) == tid
        assert (tid, phase) not in parsed
        parsed[tid, phase] = {'count':count, 'wall_ns':wall, 'thread_cpu_ns':cpu}
    assert len(set(owners.values())) == len(owners)
    return parsed, int(meta[0])


def ranges(path, completion, phases):
    connection = sqlite3.connect('file:'+str(path)+'?mode=ro', uri=True)
    connection.row_factory = sqlite3.Row
    processes = [dict(row) for row in connection.execute('select * from PROCESSES')]
    assert processes and all((row['globalPid'] & 0xFFFFFF) == 0 and
                            (row['globalPid'] >> 24) & 0xFFFFFF == row['pid'] for row in processes)
    model = [row for row in processes if row['pid'] == completion['child_pid']]
    assert len(model) == 1
    global_pid = model[0]['globalPid']
    strings = dict(connection.execute('select id,value from StringIds'))
    rows = []
    for row in connection.execute('select * from NVTX_EVENTS'):
        row = dict(row)
        label = row['text'] or strings.get(row['textId'])
        if label in phases or label == old.LABEL:
            assert old.owned(row, global_pid) is True
            assert row['end'] is not None and row['end'] > row['start']
            rows.append({'phase':label, 'tid':row['globalTid'] & 0xFFFFFF,
                         'start':row['start'], 'end':row['end'], 'ordinal':row.get('uint64Value')})
    gpu = []
    tables = {row[0] for row in connection.execute("select name from sqlite_master where type='table'")}
    unknown = {}
    for table in ['CUPTI_ACTIVITY_KIND_KERNEL', 'CUPTI_ACTIVITY_KIND_MEMCPY', 'CUPTI_ACTIVITY_KIND_MEMSET']:
        unknown[table] = 0
        if table not in tables:
            continue
        for row in connection.execute('select * from '+table):
            row = dict(row)
            ownership = old.owned(row, global_pid)
            if ownership is None:
                unknown[table] += 1
            elif ownership:
                assert row['end'] >= row['start']
                gpu.append((row['start'], row['end']))
    connection.close()
    return rows, gpu, unknown


def exclusive(row, peers):
    children = []
    for other in peers:
        if other is row or other['tid'] != row['tid']:
            continue
        if other['start'] >= row['start'] and other['end'] <= row['end']:
            assert other['start'] > row['start'] or other['end'] < row['end'], 'ambiguous equal nested intervals'
            children.append((other['start'], other['end']))
        elif other['start'] < row['end'] and other['end'] > row['start']:
            assert other['start'] <= row['start'] and other['end'] >= row['end'], 'crossing same-thread scopes'
    return row['end']-row['start']-old.union_ns(children)


def summarize(text, events, gpu):
    counts, known_bytes = counters(text)
    outer = [row for row in events if row['phase'] == old.LABEL]
    assert len(outer) == 1
    outer = outer[0]
    events = [row for row in events if row['phase'] != old.LABEL]
    assert events and all(row['ordinal'] in range(8) and row['start'] >= outer['start'] and
                          row['end'] <= outer['end'] for row in events)
    grouped = collections.defaultdict(list)
    for row in events:
        grouped[row['tid'], row['phase']].append(row)
    assert set(grouped) == set(counts)
    assert all(len(rows) == counts[key]['count'] for key, rows in grouped.items())
    roles = roster(events,outer)
    aggregate = []
    for (tid, phase), rows in sorted(grouped.items()):
        spans = [(row['start'], row['end']) for row in rows]
        intersections = [span for a,b in gpu for left,right in spans
                         if (span := old.clipped(a,b,left,right)) is not None]
        wall = old.union_ns(spans)
        aggregate.append({'role':[role for role, actual in roles.items() if tid == actual], 'phase':phase,
                          'count':len(rows), 'nvtx_wall_union_ns':wall,
                          'nvtx_same_thread_exclusive_ns':sum(exclusive(row, events) for row in rows),
                          'recorded_gpu_overlap_union_ns':old.union_ns(intersections),
                          'clock_inclusive_wall_ns':counts[tid,phase]['wall_ns'],
                          'clock_inclusive_thread_cpu_ns':counts[tid,phase]['thread_cpu_ns']})
    step_rows = re.findall(r'^PREFILL_STEP_TIMES steps=(\d+) wall=([^ ]+) dispatch=([^ ]+) job=([^ ]+) after=([^ ]+) device=([^\n]+)$', text, re.M)
    assert len(step_rows) == 1 and int(step_rows[0][0]) >= 8
    return {'phase_aggregates':aggregate, 'known_counter_static_bytes':known_bytes,
            'step_times':dict(zip(['steps','wall','dispatch','job','after','device'], step_rows[0], strict=True)),
            'all_eight_ordinals_and_worker_associations_checked':True,
            'limitations':['NVTX times and monotonic/thread CPU readouts have separate clock origins.',
                           'Inclusive parent/child and concurrent worker scopes overlap; never add them.',
                           'Same-thread exclusive NVTX elapsed excludes recorded nested scopes, not observer cost.',
                           'Charged CPU is scheduled thread work, not a sampled instruction or stack attribution.',
                           'Elapsed minus charged CPU is not proven blocked state.',
                           'GPU intersections and API bodies overlap host work; no subtraction across clock domains.',
                           'Instrumented times are diagnosis only; no competitive speed or adoption result.']}


def probe(scratch, root, job, name):
    retirement = validate.ring.retired(job)
    log = pathlib.Path.home()/'.local/share/jitllm/jobs'/job/'log'
    text = log.read_text()
    parsed, _ = counters(text)
    assert len(parsed) == 2
    assert {phase for _,phase in parsed} == {'probe.busy', 'probe.wait'}
    assert len({tid for tid,_ in parsed}) == 1
    assert all(row['count'] == 1 for row in parsed.values())
    proof = re.findall(r'^CLOCK_PROBE busy_wall_ns=(\d+) busy_cpu_ns=(\d+) wait_wall_ns=(\d+) wait_cpu_ns=(\d+) witness=(\d+)$',text,re.M)
    assert len(proof) == 1
    bw,bc,ww,wc,witness = map(int, proof[0])
    assert bw > 0 and bc > bw//2 and ww >= 150000000 and 0 <= wc < ww//10 and witness > 0
    child = scratch/(name+'-completion.json')
    validate.profile.completion(child)
    events,_,unknown = ranges(scratch/(name+'.sqlite'), json.loads(child.read_text()), ['probe.busy','probe.wait'])
    assert len(events) == 2 and len({row['tid'] for row in events}) == 1
    assert {row['phase'] for row in events} == {'probe.busy','probe.wait'}
    assert all(row['ordinal'] == 0 for row in events)
    assert {row['tid'] for row in events} == {tid for tid,_ in parsed}
    frame = scratch/'source-input-frame.json'
    sources = {path:validate.ring.identity(root/path) for path in validate.SOURCES}
    assert json.loads(frame.read_text())['sources'] == sources
    validate.profile.write(scratch/'probe-validated.json', {
        'known_busy_wait_same_thread_clock_and_nvtx_passed':True,
        'source_frame':validate.ring.identity(frame), 'sources':sources,
        'probe_binary':validate.ring.identity(root/'build/spark-native/benchmarks/jitllm_gemma26_clock_probe'),
        'child_binary':validate.ring.identity(scratch/'profile_child'),
        'completion':validate.ring.identity(child), 'sqlite':validate.ring.identity(scratch/(name+'.sqlite')),
        'official_retirement':retirement, 'clock_deltas_ns':{'busy_wall':bw,'busy_cpu':bc,'wait_wall':ww,'wait_cpu':wc},
        'unknown_gpu_ownership_events':unknown})


def selftest():
    old.selftest()
    parent = {'tid':1,'start':0,'end':100}
    child = {'tid':1,'start':10,'end':40}
    foreign = {'tid':2,'start':0,'end':100}
    assert exclusive(parent,[parent,child,foreign]) == 70
    assert exclusive(child,[parent,child]) == 30
    try:
        exclusive(parent,[parent,{'tid':1,'start':50,'end':110}])
    except AssertionError:
        pass
    else:
        raise AssertionError('crossing scopes accepted')
    # Small synthetic metadata only. Drive the actual summarize() path with
    # counters regenerated from each mutant, so omission cannot hide behind
    # otherwise matching counter/event counts.
    outer = {'phase':old.LABEL,'tid':1,'start':0,'end':8000,'ordinal':None}
    template = [('wave',1,0,900), ('places.call',1,10,80), ('places.body',2,20,60),
                ('inputs',1,90,250), ('state.grow',1,110,240), ('plan.miss',1,260,600),
                ('graph.build_bind',1,270,450), ('plan.first',1,280,320),
                ('placement',1,330,360), ('plan.second',1,370,410),
                ('plan.bind',1,460,510), ('coverage',1,520,570), ('stage',1,610,660),
                ('job.completed',1,670,800), ('job.submit',3,700,740),
                ('publish',1,810,850), ('host.release',1,860,880)]
    fixture = [{'phase':phase,'tid':tid,'start':first+ordinal*1000,'end':last+ordinal*1000,
                'ordinal':ordinal} for ordinal in range(8) for phase,tid,first,last in template]
    def run(rows):
        totals = collections.Counter((row['tid'],row['phase']) for row in rows)
        text = 'PREFILL_COARSE known_static_bytes=2726 clock=thread_cpu_ns elapsed=monotonic_ns inclusive=1\n'
        text += ''.join(f'PREFILL_SCOPE owner={tid-1} tid={tid} phase={phase} count={count} wall_ns=100 thread_cpu_ns=10\n'
                        for (tid,phase),count in totals.items())
        text += 'PREFILL_STEP_TIMES steps=8 wall=1 dispatch=1 job=1 after=1 device=1\n'
        return summarize(text,[outer,*rows],[])
    run(fixture)  # No fresh extents is a valid conditional path.
    fresh = [('state.describe',1,111,130), ('state.describe.body',2,112,120),
             ('state.acquire',1,131,150), ('state.register',1,151,175),
             ('state.register.body',2,152,165), ('refresh.call',1,176,210),
             ('refresh.body',2,180,200), ('cohort.hold',1,211,230)]
    fresh_fixture = fixture + [{'phase':phase,'tid':tid,'start':first,'end':last,'ordinal':0}
                               for phase,tid,first,last in fresh]
    run(fresh_fixture)
    hit_fixture = [{**row,'phase':'plan.hit'} if row['phase']=='plan.miss' else row
                   for row in fixture if row['phase'] not in MISS]
    run(hit_fixture)
    mutants = [[row for row in fixture if row['phase'] != phase]
               for phase in sorted(MANDATORY | MISS)]
    mutants.append([{**row,'tid':1} if row['phase'] == 'places.body' else row for row in fixture])
    mutants.append([{**row,'tid':3} if row['phase'] == 'refresh.body' else row for row in fresh_fixture])
    mutants.append([{**row,'start':605,'end':609} if row['ordinal']==0 and row['phase']=='coverage' else row for row in fixture])
    mutants.append([row for row in fresh_fixture if row['phase'] != 'state.acquire'])
    for rows in mutants:
        try:
            run(rows)
        except AssertionError:
            pass
        else:
            raise AssertionError('missing/wrong-thread/orphan source roster accepted')
    print(f'same-thread nesting and exact source roster: 3 valid, {len(mutants)} omission/thread/parent mutants refused')


if __name__ == '__main__':
    if sys.argv[1:] == ['--self-test']:
        selftest()
    elif sys.argv[1] == 'probe':
        assert len(sys.argv) == 6
        probe(pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3]), sys.argv[4], sys.argv[5])
    else:
        assert len(sys.argv) == 10
        root, scratch = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
        source_sha, name, job, control_sha, validated_sha, analysis_sha, output = sys.argv[3:]
        assert validate.ring.identity(pathlib.Path(__file__))['sha256'] == analysis_sha
        source = scratch/'source-identities.json'
        assert validate.ring.identity(source)['sha256'] == source_sha
        assert json.loads(source.read_text())['environment'] == validate.environment(root,scratch)
        control = scratch/'native-control-validated.json'
        assert validate.ring.identity(control)['sha256'] == control_sha
        assert json.loads(control.read_text())['source_freeze']['sha256'] == source_sha
        record = scratch/(name+'-validated.json')
        assert validate.ring.identity(record)['sha256'] == validated_sha
        assert json.loads(record.read_text())['source_freeze']['sha256'] == source_sha
        actual = validate.profile.check_run(scratch,name,'native',job)
        assert {key:value for key,value in json.loads(record.read_text()).items() if key not in ['source_freeze','created_utc']} == actual
        completion = json.loads((scratch/(name+'-completion.json')).read_text())
        events,gpu,unknown = ranges(scratch/(name+'.sqlite'),completion,PHASES)
        log = pathlib.Path.home()/'.local/share/jitllm/jobs'/job/'log'
        result = summarize(log.read_text(),events,gpu)
        validate.profile.write(pathlib.Path(output), {'analysis_source':validate.ring.identity(pathlib.Path(__file__)),
            'source_freeze':validate.ring.identity(source), 'control':validate.ring.identity(control),
            'trace_validated':validate.ring.identity(record), 'official_retirement':actual['official_retirement'],
            'sqlite':validate.ring.identity(scratch/(name+'.sqlite')), 'unknown_gpu_ownership_events':unknown, **result})
