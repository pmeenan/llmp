#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Validate external raw results and retain aggregates, never raw traces."""

import argparse
from collections import Counter
import json
import re
from pathlib import Path

import numpy as np

from checks import compare, summary
from fetch import digest


def noise(a, b):
    middle = (a["median"] + b["median"])/2
    allowance = max(abs(a["median"]/b["median"]-1),
                    (a["median_ci95"][1]-a["median_ci95"][0]
                     +b["median_ci95"][1]-b["median_ci95"][0])/2/middle)
    return {"relative_allowance": allowance, "stable_for_native_gate": allowance<=.10}


def host_ranges(observations):
    values = {}
    for row in observations:
        for key, value in row.items():
            if isinstance(value, (int, float)):
                values.setdefault(key, []).append(value)
        for key, value in zip(("gpu_temp_C", "gpu_sm_MHz", "gpu_mem_MHz", "gpu_watts", "gpu_util_percent"),
                              row.get("gpu_tempC_smMHz_memMHz_watts_util_pct", "").split(",")):
            try:
                values.setdefault(key, []).append(float(value))
            except ValueError:
                pass
    return {key: {"min": min(samples), "max": max(samples)} for key, samples in values.items() if samples}



def require(condition, message):
    if not condition:
        raise ValueError(message)


def exact_keys(rows, key, expected, label):
    actual = [key(row) for row in rows]
    require(Counter(actual) == Counter(expected), f"{label}: missing, duplicate or unexpected cases")


def check_manifest(manifest, fixture, profile, mode, protocol_hash, harness_hash):
    expected = {"fixture": fixture["repository"], "revision": fixture["revision"],
                "weight_sha256": fixture["published_sha256"], "profile": profile,
                "mode": mode, "protocol_sha256": protocol_hash, "harness_sha256": harness_hash}
    for key, value in expected.items():
        require(manifest.get(key) == value, f"manifest {key} mismatch")


def timing_samples(row, key, count):
    values = np.asarray(row[key], dtype=np.float64)
    require(values.shape == (count,) and np.isfinite(values).all() and (values > 0).all(),
            f"invalid {key} timing sample count/values")
    return values


def checked_summary(recorded, values, clusters=None):
    calculated = summary(values, clusters)
    require(set(recorded) == set(calculated), "summary fields mismatch")
    for key, value in calculated.items():
        if isinstance(value, str):
            require(recorded[key] == value, f"summary {key} mismatch")
        else:
            require(np.shape(recorded[key]) == np.shape(value), f"summary {key} shape mismatch")
            require(np.allclose(recorded[key], value, rtol=1e-12, atol=1e-12),
                    f"summary {key} differs from raw samples")
    return calculated


def numerical_detail(record, positions, elements=None, exact=True):
    require(record.get("finite") is True, "nonfinite numerical control")
    require(isinstance(record.get("exact"), bool), "invalid numerical equality flag")
    require(record.get("positions") == positions, "numerical position count mismatch")
    count = record.get("elements", 0)
    require(isinstance(count, int) and count > 0 and count % positions == 0,
            "invalid numerical element count")
    if elements is not None:
        require(count == elements, "numerical element count mismatch")
    require(isinstance(record.get("top1_equal"), int)
            and 0 <= record["top1_equal"] <= positions, "invalid top1 count")
    require(all(np.isfinite(record.get(key, np.nan)) and record[key] >= 0
                for key in ("max_abs", "rms")), "invalid numerical diagnostics")
    require(record["rms"] <= record["max_abs"] + 1e-12, "RMS exceeds maximum absolute error")
    if exact or record["exact"]:
        require(record["exact"] is True and record["max_abs"] == 0
                and record["rms"] == 0 and record["top1_equal"] == positions,
                "required exact numerical control failed")


def validate_model(manifest, rows, numerical, fixture, profile, protocol, protocol_hash, harness_hash):
    check_manifest(manifest, fixture, profile, "model", protocol_hash, harness_hash)
    require(manifest.get("numerical_passed") is True, "model numerical control failed")
    block_count = protocol["performance_blocks"]
    require(block_count == 2, "noise calculation requires exactly two performance blocks")
    expected = [("prefill", prefix, block) for block in range(block_count)
                for prefix in protocol["prefill_tokens"]]
    expected += [("decode", protocol["decode_prefix_tokens"], block) for block in range(block_count)]
    exact_keys(rows, lambda r: (r["kind"], r["prefix_tokens"], r["block"]), expected, "performance")
    trials = protocol["trials_per_block"]
    for row in rows:
        if row["kind"] == "prefill":
            samples = {key: timing_samples(row, key, trials) for key in
                       ("prefill_ms", "last_prompt_token_ms", "ttft_proxy_ms", "gpu_event_prefill_ms")}
            require(np.allclose(samples["ttft_proxy_ms"], samples["prefill_ms"]
                                + samples["last_prompt_token_ms"], rtol=1e-12, atol=1e-12),
                    "TTFT proxy does not match its raw components")
            for key in ("prefill_ms", "ttft_proxy_ms", "gpu_event_prefill_ms"):
                row["summary"][key] = checked_summary(row["summary"][key], samples[key])
        else:
            count = protocol["decode_tokens"]
            require(row["tokens_per_trial"] == count, "decode request length mismatch")
            times = timing_samples(row, "token_ms", trials * count)
            timing_samples(row, "gpu_event_ms", trials * count)
            requests = timing_samples(row, "request_ms", trials)
            groups = times.reshape(trials, count)
            require(np.allclose(requests, groups.sum(axis=1), rtol=1e-12, atol=1e-12),
                    "decode request durations do not match token samples")
            row["summary"] = checked_summary(row["summary"], times, groups)
            row["request_summary"] = checked_summary(row["request_summary"], requests)
    exact_keys(numerical, lambda r: r["prefix"], protocol["numerical_prefix_tokens"], "numerical")
    for row in numerical:
        require(row.get("passed") is True and row.get("poison_verified") is True
                and row.get("restored_storage_exact") is True, "restore detail failure")
        require(row["suffix_tokens"] == protocol["numerical_suffix_tokens"], "suffix length mismatch")
        require(isinstance(row["snapshot_bytes"], int) and row["snapshot_bytes"] > 0,
                "invalid snapshot size")
        require(bool(row["snapshot_sha256"]) and all(re.fullmatch(r"[0-9a-f]{64}", h)
                for h in row["snapshot_sha256"]), "invalid snapshot hashes")
        kinds = ["suffix_repeat"] * protocol["numerical_repeats"] + ["suffix_restore"]
        if row["prefix"] <= protocol["full_prefill_logits_up_to"]:
            kinds += ["prefill_repeat"] * protocol["numerical_repeats"]
        exact_keys(row["comparisons"], lambda c: c["kind"], kinds, "numerical comparisons")
        for check in row["comparisons"]:
            positions = row["prefix"] if check["kind"] == "prefill_repeat" else row["suffix_tokens"]
            numerical_detail(check, positions, positions * fixture["reference_expectations"]["vocab_size"])


def validate_logits(bundle, prefix, protocol, numerical, fixture):
    expected = {"ids", "suffix"}
    if prefix <= protocol["full_prefill_logits_up_to"]:
        expected.add("prefill")
    require(set(bundle.files) == expected, "raw numerical arrays mismatch")
    count = prefix + protocol["numerical_suffix_tokens"]
    ids = np.array([[1000 + ((i * 37 + protocol["seed"]) % 29000) for i in range(count)]], dtype=np.int64)
    require(bundle["ids"].dtype == np.int64 and np.array_equal(bundle["ids"], ids),
            "raw numerical input tokens mismatch")
    for key in expected - {"ids"}:
        logits = bundle[key]
        positions = prefix if key == "prefill" else protocol["numerical_suffix_tokens"]
        require(logits.dtype == np.float32 and logits.ndim == 3 and logits.shape[:2] == (1, positions)
                and logits.shape[-1] == fixture["reference_expectations"]["vocab_size"]
                and np.isfinite(logits).all(), "invalid raw logits")
        for check in numerical["comparisons"]:
            if (check["kind"] == "prefill_repeat") == (key == "prefill"):
                numerical_detail(check, positions, int(logits.size))


def profiled_plan(symbols):
    """Classify the executed plan and trellis rates from profiled kernel symbols."""
    packed = any(re.match(r"void exl3_gem[mv]_kernel<", s) for s in symbols)
    reconstruct = any(s.startswith("void reconstruct_kernel<") for s in symbols)
    fused = any(s.startswith("void reconstruct_had_kernel<") for s in symbols)
    plans = [name for name, seen in (("packed", packed), ("reconstruct", reconstruct),
                                     ("fused_reconstruct", fused)) if seen]
    rates = {int(m.group(1)) for s in symbols
             for m in [re.match(r"void (?:exl3_gem[mv]|reconstruct(?:_had)?)_kernel<(\d+),", s)] if m}
    return (plans[0] if len(plans) == 1 else None), rates


def validate_kernels(manifest, rows, fixture, label, protocol, protocol_hash, harness_hash):
    mode = "synthetic" if label == "synthetic" else "kernels"
    check_manifest(manifest, fixture, "optimized", mode, protocol_hash, harness_hash)
    require(manifest.get("kernel_repeat_passed") is True, "kernel manifest control failed")
    synthetic = {f"synthetic-{k}-{n}-K{rate}": (k, n, rate)
                 for k, n in protocol["kernel_synthetic_shapes"]
                 for rate in protocol["kernel_synthetic_rates"]}
    names = synthetic if mode == "synthetic" else protocol["kernel_real_keys"]
    exact_keys(rows, lambda r: (r["name"], r["rows"]),
               [(name, count) for name in names for count in protocol["kernel_rows"]], "kernel")
    dimensions = {}
    for row in rows:
        shape = (row["k"], row["n"], row["K"])
        require(all(isinstance(x, int) and x > 0 for x in shape[:2]), "invalid kernel dimensions")
        if mode == "synthetic":
            require(shape == synthetic[row["name"]], "synthetic shape/rate mismatch")
        else:
            expected = fixture["reference_expectations"]["real_kernel_shapes"][row["name"]]
            require(shape == tuple(expected[key] for key in ("k", "n", "K")),
                    "fixture kernel shape/rate mismatch")
            require(str(float(row["K"])) in fixture["per_tensor_bits"], "invalid fixture kernel rate")
        require(dimensions.setdefault(row["name"], shape) == shape, "kernel shape/rate changed across rows")
        expected_path = "packed" if row["rows"] <= 144 else ("fused_reconstruct" if row["rows"] >= 1024 else "reconstruct")
        require(row["path"] == expected_path, "kernel execution plan mismatch")
        for key in ("repeat", "capture_check", "versus_reconstruction"):
            numerical_detail(row[key], row["rows"], row["rows"] * row["n"], key != "versus_reconstruction")
        samples = timing_samples(row, "graph_replay_us", protocol["kernel_repeats"])
        row["summary"] = checked_summary(row["summary"], samples)
        events = row["profile"]["events"]
        require(bool(events), "missing CUDA events")
        for event in events.values():
            require(isinstance(event["calls"], int) and event["calls"] > 0
                    and np.isfinite(event["duration_us"]) and event["duration_us"] > 0,
                    "invalid CUDA event")
        plan, rates = profiled_plan(events)
        require(plan == row["path"], "profiled kernels contradict the declared execution plan")
        require(float(row["K"]).is_integer() and rates == {int(row["K"])},
                "profiled kernel rate does not match the trellis rate")
        duration = row["profile"]["active_device_us"]
        invocations = row["profile"].get("profiled_invocations")
        require(invocations == protocol["kernel_replays_per_sample"], "profiled invocation count mismatch")
        require(np.isfinite(duration) and duration > 0
                and np.isclose(duration, sum(e["duration_us"] for e in events.values()) / invocations, rtol=1e-12),
                "CUDA activity total mismatch")

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--results",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    here = Path(__file__).parent
    model_protocol = json.loads((here / "model-protocol.json").read_text())
    kernel_protocol = json.loads((here / "protocol.json").read_text())
    model_protocol_hash, kernel_protocol_hash = (digest(here / name) for name in ("model-protocol.json", "protocol.json"))
    model_runner = kernel_runner = digest(here / "measure.py")
    require(all(re.fullmatch(r"[0-9a-f]{64}", h) for h in (model_runner, kernel_runner)), "invalid runner hash pin")
    fixtures = {f["repository"].rsplit("-", 1)[1]: f for f in json.loads((here / "pins.json").read_text())["fixtures"]}
    result={"recorded_on":"2026-09-22","scope":"External reference; no native llmpalooza performance or paging claim",
            "models":[],"kernels":[],"cross_profile":[],"raw_receipts":[],"kernel_symbols":[]}
    result["expected_runner_sha256"] = {"model": model_runner, "kernels": kernel_runner}
    symbols={}
    for fixture in ("4.0bpw","4.5bpw"):
        for profile in ("optimized","attention_eager"):
            folder=args.results/f"accepted-{fixture}-{profile}"
            manifest=json.loads((folder/'manifest.json').read_text())
            rows=json.loads((folder/'performance.json').read_text())
            numerical=json.loads((folder/'numerical.json').read_text())
            validate_model(manifest, rows, numerical, fixtures[fixture], profile, model_protocol,
                           model_protocol_hash, model_runner)
            grouped={}
            for row in rows:grouped.setdefault((row['kind'],row['prefix_tokens']),[]).append(row)
            performance=[]
            for (kind,prefix),blocks in grouped.items():
                blocks.sort(key=lambda b: b["block"])
                if len(blocks)!=2 or [b['block'] for b in blocks]!=[0,1]:raise ValueError('missing performance block')
                if kind=='prefill':
                    pre=[v for b in blocks for v in b['prefill_ms']]
                    ttft=[v for b in blocks for v in b['ttft_proxy_ms']]
                    data={"kind":kind,"prefix_tokens":prefix,"prefill_ms":summary(pre),"ttft_proxy_ms":summary(ttft),
                          "tokens_per_second":prefix*1000/float(np.median(pre)),
                          "prefill_noise":noise(*[b['summary']['prefill_ms'] for b in blocks]),
                          "ttft_noise":noise(*[b['summary']['ttft_proxy_ms'] for b in blocks]),
                          "first_observed_pair_ms":[b['first_observed_pair_ms'] for b in blocks]}
                else:
                    times=[v for b in blocks for v in b['token_ms']]
                    requests=[v for b in blocks for v in b['request_ms']]
                    count=blocks[0]['tokens_per_trial']
                    groups=np.asarray(times).reshape(-1,count)
                    data={"kind":kind,"prefix_tokens":prefix,"token_ms":summary(times,groups),
                          "request_ms":summary(requests),"tokens_per_trial":count,
                          "tokens_per_second":count*1000/float(np.median(requests)),
                          "noise":noise(*[b['request_summary'] for b in blocks])}
                data['torch_peak_allocated_bytes']=max(b['memory']['torch_peak_allocated'] for b in blocks)
                data['torch_peak_reserved_bytes']=max(b['memory']['torch_peak_reserved'] for b in blocks)
                performance.append(data)
            account=manifest['vram_accounting'][0]
            selected={k:account[k] for k in ('weights_arena_used','weights_arena_tail','weights_direct','weights','cache','statics','allocated','reserved')}
            result['models'].append({"fixture":fixture,"profile":profile,"load_ms":manifest['load_ms'],
                    "harness_sha256":manifest['harness_sha256'],"protocol_sha256":manifest['protocol_sha256'],
                    "numerical":[{"prefix":r['prefix'],"suffix_tokens":r['suffix_tokens'],"passed":r['passed'],
                                 "snapshot_bytes":r['snapshot_bytes'],"comparisons":r['comparisons']} for r in numerical],
                    "performance":performance,"tracked_memory":selected,"max_rss_bytes":manifest['max_rss_bytes'],
                    "untimed_boundary_host_ranges":host_ranges([r['memory']['host'] for r in rows]),
                    "memory_note":"CUDA mem_get_info and upstream non_torch include node-wide unified-memory use/page cache, not this process's non-Torch overhead; no summed CPU/GPU physical budget claimed"})
            for name in ('manifest.json','performance.json','numerical.json'):
                path=folder/name
                result['raw_receipts'].append({'path':str(path.relative_to(args.results)),'sha256':digest(path),'bytes':path.stat().st_size})
        for prefix in model_protocol["numerical_prefix_tokens"]:
            opt=args.results/f"accepted-{fixture}-optimized"/f"numerical-{prefix}.npz"
            eager=args.results/f"accepted-{fixture}-attention_eager"/f"numerical-{prefix}.npz"
            with np.load(opt,allow_pickle=False) as a,np.load(eager,allow_pickle=False) as b:
                for bundle, profile in ((a, "optimized"), (b, "attention_eager")):
                    details = json.loads((args.results/f"accepted-{fixture}-{profile}"/"numerical.json").read_text())
                    detail = next(r for r in details if r["prefix"] == prefix)
                    validate_logits(bundle, prefix, model_protocol, detail, fixtures[fixture])
                if not np.array_equal(a['ids'],b['ids']):raise ValueError('cross-profile input mismatch')
                result['cross_profile'].append({'fixture':fixture,'prefix':prefix,
                        **{key:compare(a[key],b[key]) for key in a.files if key!='ids'}})
            for path in (opt,eager):result['raw_receipts'].append({'path':str(path.relative_to(args.results)),'sha256':digest(path),'bytes':path.stat().st_size})
    for label in ('4.0bpw','4.5bpw','synthetic'):
        folder=args.results/f'final-kernels-{label}'
        manifest=json.loads((folder/'manifest.json').read_text())
        rows=json.loads((folder/'kernels.json').read_text())
        validate_kernels(manifest, rows, fixtures["4.0bpw" if label == "synthetic" else label], label,
                         kernel_protocol, kernel_protocol_hash, kernel_runner)
        for row in rows:
            if not row['repeat']['exact'] or not row['capture_check']['exact']:raise ValueError('kernel check failure')
            if not row['profile']['events'] or row['profile']['active_device_us']<=0:raise ValueError('missing device activity')
            event_ids=[]
            for name,event in row['profile']['events'].items():
                idx=symbols.setdefault(name,len(symbols))
                event_ids.append({'symbol':idx,**event})
            samples=row['graph_replay_us'];split=len(samples)//2
            result['kernels'].append({'fixture':label,'name':row['name'],'k':row['k'],'n':row['n'],'K':row['K'],
                    'rows':row['rows'],'path':row['path'],'graph_replay_us':row['summary'],
                    'noise':noise(summary(samples[:split]),summary(samples[split:])),
                    'active_device_us':row['profile']['active_device_us'],'device_events':event_ids,
                    'profiled_invocations':row['profile']['profiled_invocations'],
                    'repeat':row['repeat'],'capture_check':row['capture_check'],
                    'versus_reconstruction':row['versus_reconstruction'],
                    'torch_allocated_before_bytes':row['torch_allocated_before'],
                    'torch_peak_allocated_bytes':row['memory']['torch_peak_allocated'],
                    'harness_sha256':manifest['harness_sha256'],'protocol_sha256':manifest['protocol_sha256']})
        for name in ('manifest.json','kernels.json'):
            path=folder/name;result['raw_receipts'].append({'path':str(path.relative_to(args.results)),'sha256':digest(path),'bytes':path.stat().st_size})
    result['kernel_symbols']=list(symbols)
    result['all_required_numerical_controls_passed']=True
    result['kernel_case_count']=len(result['kernels'])
    result['unstable_kernel_cases']=[{k:r[k] for k in ('fixture','name','rows','noise')} for r in result['kernels'] if not r['noise']['stable_for_native_gate']]
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('validated',len(result['models']),'model profiles and',len(result['kernels']),'kernel cases')


if __name__=='__main__':main()
