# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
import sys
sys.dont_write_bytecode = True

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import math
import os
import pathlib
import re
import socket
import statistics
import subprocess
import time

BUNDLE = pathlib.Path(__file__).resolve().parent
DEFAULT_PINS = BUNDLE / 'pins.json'
PINS = {}


def sha(path):
    h = hashlib.sha256()
    with pathlib.Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load_speed_gate(pins):
    path = pins['source']['speed_gate_path']
    if sha(path) != pins['source']['speed_gate_sha256']:
        raise SystemExit('speed_gate.py source hash changed')
    spec = importlib.util.spec_from_file_location('speed_gate_c1', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def profile_override(speed_gate, family, profile):
    if family not in speed_gate.FAMILIES:
        raise SystemExit(f'unknown speed_gate family: {family}')
    speed_gate.FAMILIES[family] = dataclasses.replace(
        speed_gate.FAMILIES[family], prompt_chars=profile['prompt_chars'],
        cells=(1,), repeats=1, max_tokens=128)


def build_command(pins, family, pair, arm):
    assert family in pins['family_profiles'] and pair in (1, 2, 3) and arm in ('A', 'R')
    order = pins['order'][pair - 1]['arms']
    assert order.count(arm) == 1
    position = order.index(arm)
    profile = pins['family_profiles'][family]
    # A is the upstream reference; R is the repaired candidate.
    runtime = pins['runtimes']['reference' if arm == 'A' else 'candidate']['path']
    label = 'old' if arm == 'A' else 'new'
    return [sys.executable, '-B', pins['source']['speed_gate_path'], 'run',
            '--family', family, '--runtime', runtime, '--label', label,
            '--out', profile['output_root'], '--block', str(pair), '--position', str(position),
            '--cells', '1', '--config-template', profile['config_path'],
            '--share', '/home/pmeenan/.local/share/llmp',
            '--corpus', pins['input']['corpus_path']]


def parse_show(pins, family, pair, arm):
    profile = pins['family_profiles'][family]
    argv = build_command(pins, family, pair, arm)
    speed_gate = load_speed_gate(pins)
    profile_override(speed_gate, family, profile)
    captured = []
    original = speed_gate.cmd_run
    def capture(ns):
        captured.append({k: v for k, v in vars(ns).items() if k != 'func'})
        return 0
    speed_gate.cmd_run = capture
    try:
        rc = speed_gate.main(argv[3:])
    finally:
        speed_gate.cmd_run = original
    if rc or len(captured) != 1:
        raise SystemExit('speed_gate run parser dry check failed')
    parsed = captured[0]
    expected_arm = {'A': ('old', 0), 'R': ('new', 1)}[arm]
    position = pins['order'][pair - 1]['arms'].index(arm)
    if parsed['family'] != family or parsed['cells'] != '1' or parsed['label'] != expected_arm[0]:
        raise SystemExit('parsed speed_gate args disagree with C1 registration')
    if parsed['block'] != pair or parsed['position'] != position or parsed['prime']:
        raise SystemExit('parsed speed_gate block/order differs')
    if parsed['runtime'] != PINS['runtimes']['reference' if arm == 'A' else 'candidate']['path']:
        raise SystemExit('parsed runtime assignment differs')
    if pathlib.Path(parsed['config_template']) != pathlib.Path(profile['config_path']):
        raise SystemExit('parsed config template differs')
    return {'family': family, 'pair': pair, 'arm': arm, 'parsed': parsed,
            'argv': argv, 'override': {'prompt_chars': profile['prompt_chars'],
                                       'cells': [1], 'repeats': 1, 'max_tokens': 128}}


def ldd_resolution(runtime, library_dir, expected_libraries):
    env = dict(os.environ)
    env['LD_LIBRARY_PATH'] = str(library_dir)
    proc = subprocess.run(['/usr/bin/ldd', str(runtime)], env=env,
                          capture_output=True, text=True, timeout=30)
    if proc.returncode:
        raise SystemExit(f'ldd failed for {runtime}: {proc.stderr}')
    expected = {x['path']: x['sha256'] for x in expected_libraries}
    resolved = {}
    for raw in proc.stdout.splitlines():
        line = raw.strip()
        if not line or line.startswith('linux-vdso'):
            continue
        if '=>' in line:
            name, rest = line.split('=>', 1)
            pieces = rest.strip().split()
            if not pieces or pieces[0] == 'not':
                raise SystemExit(f'unresolved dependency: {line}')
            path = pathlib.Path(pieces[0])
            soname = name.strip()
        else:
            pieces = line.split()
            if not pieces[0].startswith('/'):
                raise SystemExit(f'unparsed ldd line: {line}')
            path = pathlib.Path(pieces[0])
            soname = path.name
        if soname not in expected:
            raise SystemExit(f'unpinned dependency {soname}: {path}')
        digest = sha(path)
        if digest != expected[soname]:
            raise SystemExit(f'dependency digest mismatch for {soname}: {path}')
        resolved[soname] = {'path':str(path),'sha256':digest}
    if set(resolved) != set(expected):
        raise SystemExit(f'ldd closure differs: resolved={sorted(resolved)} expected={sorted(expected)}')
    return {'stdout':proc.stdout,'stdout_sha256':hashlib.sha256(proc.stdout.encode()).hexdigest(),
            'stderr_sha256':hashlib.sha256(proc.stderr.encode()).hexdigest(),
            'returncode':proc.returncode,'resolved':resolved}


def artifact_receipts(pins, profile):
    receipts = []
    store = pathlib.Path(pins['store']['path'])
    for artifact in profile['artifacts']:
        root = store / artifact['id']
        files = {}
        for filename, field in (('manifest.json','manifest_sha256'),('index.json','index_sha256'),('spec.json','spec_sha256')):
            path = root / filename
            digest = sha(path)
            if digest != artifact[field]:
                raise SystemExit(f'artifact metadata mismatch: {path}')
            files[filename] = digest
        receipts.append({'id':artifact['id'],'path':str(root),'metadata':files})
    return receipts


def immutable_guard(pins, family, arm):
    profile = pins['family_profiles'][family]
    if socket.gethostname().split('.',1)[0] != profile['host']:
        raise SystemExit('run launched on the wrong Spark')
    if sha(pins['source']['speed_gate_path']) != pins['source']['speed_gate_sha256']:
        raise SystemExit('speed_gate source changed')
    if sha(profile['config_path']) != profile['config_sha256']:
        raise SystemExit('C1 config changed')
    if sha(pins['input']['corpus_path']) != pins['input']['corpus_sha256']:
        raise SystemExit('C1 corpus changed')
    runtime_pin = pins['runtimes']['reference' if arm == 'A' else 'candidate']
    runtime = pathlib.Path(runtime_pin['path'])
    if sha(runtime) != runtime_pin['sha256']:
        raise SystemExit('runtime changed')
    if sha(pins['identity_crosswalk']['path']) != pins['identity_crosswalk']['sha256']:
        raise SystemExit('implementation identity crosswalk changed')
    host=socket.gethostname().split('.',1)[0]
    pin=pins['source_manifests'].get(host)
    if not pin or sha(pin['path']) != pin['sha256']:
        raise SystemExit('local source manifest is absent or changed')
    source_hashes = {host:pin['sha256']}
    library_hashes = {}
    for lib in pins['libraries']['entries']:
        path = pathlib.Path(pins['libraries']['directory']) / lib['path']
        if sha(path) != lib['sha256']:
            raise SystemExit(f'library changed: {path}')
        library_hashes[lib['path']] = lib['sha256']
    tokenizer = None
    if family == 'qwen38-gguf':
        tokenizer = {}
        for path_key, hash_key in (('tokenizer_path','tokenizer_sha256'),('chat_template_path','chat_template_sha256')):
            item = pins['qwen_tokenizer']
            if sha(item[path_key]) != item[hash_key]:
                raise SystemExit(f'Qwen tokenizer resource changed: {item[path_key]}')
            tokenizer[item[path_key]] = item[hash_key]
    artifacts = artifact_receipts(pins, profile)
    resolved = ldd_resolution(runtime, pins['libraries']['directory'], pins['libraries']['entries'])
    env = {k:v for k,v in os.environ.items() if k.startswith(('LLMP','LD_','CUDA','NVIDIA'))}
    if env:
        raise SystemExit(f'unregistered runtime environment: {env}')
    return {'host':socket.gethostname().split('.',1)[0], 'arm':arm,
            'speed_gate_sha256':pins['source']['speed_gate_sha256'],
            'config_sha256':profile['config_sha256'],'corpus_sha256':pins['input']['corpus_sha256'],
            'runtime':{'path':str(runtime),'sha256':runtime_pin['sha256']},
            'source_manifests':source_hashes,'identity_crosswalk_sha256':pins['identity_crosswalk']['sha256'],
            'libraries':library_hashes,'ldd':resolved,'tokenizer':tokenizer,'artifacts':artifacts,
            'environment':env}


def same_guard(before, after):
    b=json.loads(json.dumps(before)); a=json.loads(json.dumps(after))
    lb,la=b['ldd']['stdout'],a['ldd']['stdout']
    b['ldd'].pop('stdout');a['ldd'].pop('stdout')
    # The raw output and its hash remain in each pre/post receipt.  The hash
    # includes ASLR addresses, so compare the normalized text instead.
    b['ldd'].pop('stdout_sha256');a['ldd'].pop('stdout_sha256')
    if b != a or re.sub(r'0x[0-9a-fA-F]+','ADDRESS',lb) != re.sub(r'0x[0-9a-fA-F]+','ADDRESS',la):
        raise SystemExit('immutable pre/post guard mismatch')


def validate_registration(path, pins_path, executor_path):
    receipt=json.loads(pathlib.Path(path).read_text())
    if receipt.get('enabled') is not True or receipt.get('reviewed_registration') is not True:
        raise SystemExit('external registration is not enabled and reviewed')
    if receipt.get('preparation_only') is not False:
        raise SystemExit('registration is still preparation-only')
    if not re.fullmatch(r'[0-9a-f]{40}',str(receipt.get('preregistration_commit',''))):
        raise SystemExit('registration must bind full preregistration commit')
    if receipt.get('pins_sha256') != sha(pins_path):
        raise SystemExit('registration does not bind this static disabled pins file')
    if receipt.get('executor_sha256') != sha(executor_path):
        raise SystemExit('registration does not bind this executor')
    return receipt


def write(path, obj):
    path=pathlib.Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f: json.dump(obj,f,indent=2,sort_keys=True);f.write('\n')


def audit_record(pins, family, pair, arm, record_path, registration):
    profile=pins['family_profiles'][family]
    r=json.loads(pathlib.Path(record_path).read_text())
    if r['error'] is not None or r['problems']:
        raise SystemExit('speed_gate recorded an error/problem; no retry')
    if r['facts'].get('gpu_compute_apps'):
        raise SystemExit('unrelated GPU compute process was present')
    label='old' if arm=='A' else 'new'
    pos=pins['order'][pair-1]['arms'].index(arm)
    if (r['family'] != family or r['label'] != label or r['block'] != pair or
            r['position'] != pos or r['session'] is not None or r['prime'] is not False):
        raise SystemExit('run identity does not match the enrolled arm')
    if r['runtime_sha256'] != pins['runtimes']['reference' if arm=='A' else 'candidate']['sha256']:
        raise SystemExit('recorded runtime digest differs')
    if r['config_template'] != pathlib.Path(profile['config_path']).read_text():
        raise SystemExit('recorded configuration differs from pinned template')
    fs=r['family_spec']
    if (fs['prompt_chars'],fs['cells'],fs['repeats'],fs['max_tokens']) != (profile['prompt_chars'],[1],1,profile['max_tokens']):
        raise SystemExit('recorded C1 family workload differs')
    if len(r['passes']) != 2 or r['passes'][0]['warmup'] is not True or r['passes'][1]['warmup'] is not False:
        raise SystemExit('run did not complete exactly one warmup and one measured pass')
    identities=[]
    for i,p in enumerate(r['passes']):
        cells=p['cells']
        if set(cells) != {'C1'}: raise SystemExit('run contains work outside C1')
        c=cells['C1'];expected_prompt=[i]
        if c['width'] != 1 or c['prompts'] != expected_prompt or c['problems'] or len(c['requests']) != 1:
            raise SystemExit('C1 request work is incomplete or invalid')
        req=c['requests'][0];u=req['usage']
        if req['error'] is not None or req['finish'] != 'length' or req['chunks'] != profile['max_tokens']:
            raise SystemExit('C1 did not complete all 128 generated tokens')
        if u['completion_tokens'] != profile['max_tokens'] or u['total_tokens'] != u['prompt_tokens']+profile['max_tokens']:
            raise SystemExit('C1 token accounting incomplete')
        if (not isinstance(u['prompt_tokens'], int) or u['prompt_tokens'] <= 0 or
                u['prompt_tokens'] + profile['max_tokens'] >= profile['context']):
            raise SystemExit('C1 prompt and completion do not fit the registered context')
        if family == 'qwen38-gguf' and u['prompt_tokens'] <= profile['prefill_chunk']:
            raise SystemExit('Qwen C1 request did not pay a complete five-column prefill')
        if (u.get('prompt_tokens_details') or {}).get('cached_tokens') != 0:
            raise SystemExit('C1 prompt unexpectedly reused cached tokens')
        if len(req.get('chunk_times') or []) != 128 or not all(math.isfinite(x) and x>0 for x in req['chunk_times']):
            raise SystemExit('C1 raw chunk timing is incomplete or non-finite')
        if (not math.isfinite(req['end_time']) or not math.isfinite(req['send_offset']) or
                req['end_time'] < req['chunk_times'][-1] or
                any(right <= left for left, right in zip(req['chunk_times'], req['chunk_times'][1:]))):
            raise SystemExit('C1 raw stream timing is not a complete ordered response')
        if not re.fullmatch(r'[0-9a-f]{64}',str(req.get('reply_sha256',''))):
            raise SystemExit('C1 raw reply digest absent')
        identities.append({'warmup':p['warmup'],'prompts':c['prompts'],'usage':u,
                           'reply_sha256':req['reply_sha256'],'chunks':req['chunks']})
    if r['stop'] != {'signal':'TERM','returncode':0}:
        raise SystemExit('runtime did not retire cleanly after measured pass')
    if not (math.isfinite(r['memory']['peak_gib']) and r['memory']['peak_gib'] > 0):
        raise SystemExit('whole-node memory sample is invalid')
    speed_gate=load_speed_gate(pins)
    family=dataclasses.replace(speed_gate.FAMILIES[family],prompt_chars=profile['prompt_chars'],cells=(1,),repeats=1,max_tokens=128)
    metrics=speed_gate.run_metrics(r,family)
    expected={'C1.prefill_s','C1.decode_tok_s','C1.wall_tok_s','peak_gib'}
    if set(metrics)!=expected or not all(math.isfinite(x) and x>0 for x in metrics.values()):
        raise SystemExit('speed_gate metric set is incomplete or invalid')
    logs=list(pathlib.Path(record_path).parent.parent.glob(f'work/{r["family"]}-b{pair:02d}-p{pos}-{label}/runtime.log'))
    if len(logs)!=1: raise SystemExit('expected one complete raw runtime log')
    result={'valid':True,'family':family.name,'pair':pair,'arm':arm,
            'record_path':str(record_path),'record_sha256':sha(record_path),
            'runtime_log_path':str(logs[0]),'runtime_log_sha256':sha(logs[0]),
            'registration':{'preregistration_commit':registration['preregistration_commit'],
                            'pins_sha256':registration['pins_sha256'],
                            'executor_sha256':registration['executor_sha256']},
            'request_identity':identities,'metrics':metrics,'stop':r['stop']}
    write(pathlib.Path(record_path).with_name(pathlib.Path(record_path).stem+'-c1-audit.json'),result)
    return result


def compare_family(pins, family, results_root, pins_path, executor_path):
    profile=pins['family_profiles'][family]
    root=pathlib.Path(results_root)
    if not root.is_dir(): raise SystemExit(f'missing collected result directory: {root}')
    speed_gate=load_speed_gate(pins)
    family_spec=dataclasses.replace(speed_gate.FAMILIES[family],prompt_chars=profile['prompt_chars'],
                                    cells=(1,),repeats=1,max_tokens=profile['max_tokens'])
    arm_data={}
    expected_registration={'pins_sha256':sha(pins_path),'executor_sha256':sha(executor_path)}
    for pair in (1,2,3):
        order=pins['order'][pair-1]['arms']
        if order not in (['A','R'],['R','A']): raise SystemExit('unregistered pair order')
        arm_data[pair]={}
        for arm in ('A','R'):
            position=order.index(arm);label={'A':'old','R':'new'}[arm]
            stem=f'{family}-b{pair:02d}-p{position}-{label}'
            record_path=root/'runs'/f'{stem}.json'
            audit_path=record_path.with_name(record_path.stem+'-c1-audit.json')
            if not record_path.is_file() or not audit_path.is_file():
                raise SystemExit(f'missing enrolled pair result: {record_path}')
            record=json.loads(record_path.read_text());audit=json.loads(audit_path.read_text())
            if not audit.get('valid') or (audit['family'],audit['pair'],audit['arm'])!=(family,pair,arm):
                raise SystemExit(f'audit identity mismatch: {audit_path}')
            if (record['family'],record['label'],record['block'],record['position'],
                    record['session'],record['prime']) != (family,label,pair,position,None,False):
                raise SystemExit(f'raw run identity mismatch: {record_path}')
            if audit['record_sha256']!=sha(record_path):
                raise SystemExit(f'audit does not bind collected record bytes: {record_path}')
            if record['runtime_sha256']!=pins['runtimes']['reference' if arm=='A' else 'candidate']['sha256']:
                raise SystemExit(f'collected record has unexpected runtime: {record_path}')
            reg=audit.get('registration')
            if (not isinstance(reg,dict) or reg.get('pins_sha256')!=expected_registration['pins_sha256'] or
                    reg.get('executor_sha256')!=expected_registration['executor_sha256'] or
                    not re.fullmatch(r'[0-9a-f]{40}',str(reg.get('preregistration_commit','')))):
                raise SystemExit(f'registration identity missing or inconsistent: {audit_path}')
            if set(audit['metrics'])!={'C1.prefill_s','C1.decode_tok_s','C1.wall_tok_s','peak_gib'}:
                raise SystemExit(f'incomplete audited metrics: {audit_path}')
            recomputed=speed_gate.run_metrics(record,family_spec)
            if recomputed!=audit['metrics']:
                raise SystemExit(f'audited metrics do not match hash-bound raw run data: {record_path}')
            if len(record['passes'])!=2:
                raise SystemExit(f'raw C1 record does not contain exactly two passes: {record_path}')
            for i,pass_record in enumerate(record['passes']):
                cells=pass_record['cells']
                if (pass_record['warmup'] is not (i==0) or set(cells)!={'C1'} or
                        cells['C1']['width']!=1 or cells['C1']['prompts']!=[i] or
                        cells['C1']['problems'] or len(cells['C1']['requests'])!=1):
                    raise SystemExit(f'raw C1 work is incomplete or malformed: {record_path}')
            arm_data[pair][arm]={'record':record,'audit':audit,'position':position}
        for pass_index in (0,1):
            a=arm_data[pair]['A']['record']['passes'][pass_index]['cells']['C1']['requests'][0]['usage']['prompt_tokens']
            r=arm_data[pair]['R']['record']['passes'][pass_index]['cells']['C1']['requests'][0]['usage']['prompt_tokens']
            if a!=r: raise SystemExit(f'paired {family} prompt-token count differs in pair {pair}, pass {pass_index}')
    ratios={'C1.prefill_s':[],'C1.decode_tok_s':[],'C1.wall_tok_s':[]}
    pairs=[]
    for pair in (1,2,3):
        a=arm_data[pair]['A']['audit'];r=arm_data[pair]['R']['audit']
        pair_ratios={'C1.prefill_s':r['metrics']['C1.prefill_s']/a['metrics']['C1.prefill_s'],
                     'C1.decode_tok_s':a['metrics']['C1.decode_tok_s']/r['metrics']['C1.decode_tok_s'],
                     'C1.wall_tok_s':a['metrics']['C1.wall_tok_s']/r['metrics']['C1.wall_tok_s']}
        if not all(math.isfinite(v) and v>0 for v in pair_ratios.values()):
            raise SystemExit(f'non-finite or non-positive paired slowdown in pair {pair}')
        for metric,value in pair_ratios.items(): ratios[metric].append(value)
        pairs.append({'pair':pair,'order':pins['order'][pair-1]['arms'],
                      'prompt_tokens':{'warmup':a['request_identity'][0]['usage']['prompt_tokens'],
                                       'measured':a['request_identity'][1]['usage']['prompt_tokens']},
                      'reference_metrics':a['metrics'],'candidate_metrics':r['metrics'],
                      'candidate_slowdown_ratios':pair_ratios,
                      'record_sha256':{'A':a['record_sha256'],'R':r['record_sha256']}})
    medians={metric:statistics.median(values) for metric,values in ratios.items()}
    limit=pins['bounds']['candidate_slowdown_median_max']
    result={'family':family,'valid_pairs':len(pairs),'ratio_definition':pins['bounds']['ratios'],
            'paired_ratios':ratios,'median_candidate_slowdowns':medians,
            'median_limit':limit,'verdict':'PASS' if all(v<=limit for v in medians.values()) else 'FAIL',
            'pairs':pairs}
    out=root/f'{family}-c1-comparison.json';write(out,result)
    return result


def main():
    global PINS
    parser=argparse.ArgumentParser(description='Registered single-arm C1 speed runner; default is preview only.')
    parser.add_argument('--pins',type=pathlib.Path,default=DEFAULT_PINS)
    parser.add_argument('--family',choices=('gemma31','gemma26','deepseek','qwen38-gguf'),required=True)
    parser.add_argument('--pair',type=int,choices=(1,2,3))
    parser.add_argument('--arm',choices=('A','R'))
    parser.add_argument('--results-root',type=pathlib.Path,
                        help='collected immutable run/audit records for --compare')
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--show',action='store_true',help='parse and print the exact speed_gate invocation; no runtime')
    mode.add_argument('--run',action='store_true',help='requires an external reviewed registration receipt')
    mode.add_argument('--compare',action='store_true',help='compare exactly three paired C1 results; no runtime')
    parser.add_argument('--registration',type=pathlib.Path,help='external preregistration receipt; mandatory with --run')
    args=parser.parse_args();PINS=json.loads(args.pins.read_text())
    if PINS.get('collection_enabled') is not False: raise SystemExit('static pins must remain collection-disabled')
    if args.compare:
        if args.pair is not None or args.arm is not None or not args.results_root:
            raise SystemExit('--compare requires --results-root and no --pair/--arm')
        result=compare_family(PINS,args.family,args.results_root,args.pins.resolve(),pathlib.Path(__file__).resolve())
        print(json.dumps(result,indent=2,sort_keys=True));return 0 if result['verdict']=='PASS' else 1
    if args.pair is None or args.arm is None or args.results_root:
        raise SystemExit('--show/--run require --pair and --arm and do not accept --results-root')
    if args.show:
        print(json.dumps(parse_show(PINS,args.family,args.pair,args.arm),indent=2,sort_keys=True));return 0
    if not args.registration: raise SystemExit('--run requires --registration')
    reg=validate_registration(args.registration,args.pins.resolve(),pathlib.Path(__file__).resolve())
    os.umask(0o077)
    before=immutable_guard(PINS,args.family,args.arm)
    profile=PINS['family_profiles'][args.family]
    pairdir=pathlib.Path(profile['output_root'])
    order=PINS['order'][args.pair-1]['arms'];position=order.index(args.arm)
    label='old' if args.arm=='A' else 'new'
    record=pairdir/'runs'/f'{args.family}-b{args.pair:02d}-p{position}-{label}.json'
    work=pairdir/'work'/f'{args.family}-b{args.pair:02d}-p{position}-{label}'
    if record.exists() or work.exists(): raise SystemExit('arm output exists; automatic retries are forbidden')
    d=pairdir/f'pair-{args.pair}-{args.arm}'
    d.mkdir(mode=0o700,parents=True,exist_ok=False)
    write(d/'preflight.json',before)
    argv=build_command(PINS,args.family,args.pair,args.arm)
    write(d/'launch.json',{'argv':argv,'executor_sha256':sha(__file__),'pins_sha256':sha(args.pins),
                           'registration_commit':reg['preregistration_commit'],'start_unix':time.time(),
                           'runtime_library_path':PINS['libraries']['directory']})
    import contextlib
    previous_ld=os.environ.get('LD_LIBRARY_PATH')
    rc=None; run_error=None
    try:
        os.environ['LD_LIBRARY_PATH']=PINS['libraries']['directory']
        with (d/'collector-output.txt').open('x') as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            runner=load_speed_gate(PINS)
            profile_override(runner,args.family,profile)
            rc=runner.main(argv[3:])
    except BaseException as error:
        run_error=f'{type(error).__name__}: {error}'
    finally:
        if previous_ld is None: os.environ.pop('LD_LIBRARY_PATH',None)
        else: os.environ['LD_LIBRARY_PATH']=previous_ld
    write(d/'returncode.json',{'returncode':rc,'error':run_error})
    after=immutable_guard(PINS,args.family,args.arm)
    write(d/'postflight.json',after);same_guard(before,after)
    if run_error: raise SystemExit(f'speed_gate raised {run_error}; preserve arm and do not retry')
    if rc: raise SystemExit(f'speed_gate returned {rc}; preserve arm and do not retry')
    audit=audit_record(PINS,args.family,args.pair,args.arm,record,reg)
    print(json.dumps(audit,sort_keys=True));return 0


if __name__ == '__main__':
    raise SystemExit(main())
