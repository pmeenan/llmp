#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Bounded external ExLlamaV3 numerical and resident-performance reference.

Run each profile in a fresh process. All raw logits/timings go outside Git.
"""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import time

import numpy as np
import torch

from fetch import digest
from checks import compare, summary


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n")


def tensor_hash(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def host_observation():
    mem = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        if key in ("MemTotal", "MemAvailable", "Cached", "SwapFree"):
            mem[key + "_bytes"] = int(value.split()[0]) * 1024
    status = Path("/proc/self/status").read_text().splitlines()
    for line in status:
        if line.startswith(("VmRSS:", "VmHWM:")):
            key, value = line.split(":", 1)
            mem[key + "_bytes"] = int(value.split()[0]) * 1024
    result = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,clocks.sm,clocks.mem,power.draw,utilization.gpu",
                             "--format=csv,noheader,nounits"], capture_output=True, text=True)
    mem["gpu_tempC_smMHz_memMHz_watts_util_pct"] = result.stdout.strip()
    return mem


def memory():
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    return {"torch_allocated": torch.cuda.memory_allocated(),
            "torch_reserved": torch.cuda.memory_reserved(),
            "torch_peak_allocated": torch.cuda.max_memory_allocated(),
            "torch_peak_reserved": torch.cuda.max_memory_reserved(),
            "cuda_free": free, "cuda_total": total, "host": host_observation()}


def tokens(count, seed):
    return torch.tensor([[1000 + ((i * 37 + seed) % 29000) for i in range(count)]], dtype=torch.long)


def timed(fn):
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    wall = time.perf_counter()
    start.record()
    value = fn()
    end.record()
    end.synchronize()
    return value, (time.perf_counter() - wall)*1000, start.elapsed_time(end)


def numerical(model, cache, protocol, output):
    results = []
    capacity = protocol["cache_capacity"]
    def params(p):
        return {"attn_mode": "flash_attn", "cache": cache, "past_len": p, "batch_shape": (1, capacity),
                "pinned_staging": bool(p) and protocol["pinned_decode_staging"]}
    def zero():
        for tensor in cache.get_all_tensors():
            tensor.zero_()
        torch.cuda.synchronize()
    def suffix(ids, p):
        out = []
        for i in range(protocol["numerical_suffix_tokens"]):
            logits = model.forward(ids[:, p+i:p+i+1], params(p+i))
            out.append(logits.float().cpu().numpy().copy())
        return np.concatenate(out, axis=1)
    for prefix in protocol["numerical_prefix_tokens"]:
        ids = tokens(prefix + protocol["numerical_suffix_tokens"], protocol["seed"])
        def begin():
            zero()
            if prefix <= protocol["full_prefill_logits_up_to"]:
                return model.forward(ids[:, :prefix], params(0)).float().cpu().numpy().copy()
            model.prefill(ids[:, :prefix], params(0))
            torch.cuda.synchronize()
            return None
        pre = begin()
        tensors = cache.get_all_tensors()
        snapshot = [t.cpu().clone() for t in tensors]
        hashes = [tensor_hash(t) for t in snapshot]
        ref = suffix(ids, prefix)
        np.savez(output / f"numerical-{prefix}.npz", ids=ids.numpy(), suffix=ref,
                 **({"prefill": pre} if pre is not None else {}))
        comparisons = []
        for _ in range(protocol["numerical_repeats"]):
            repeated_pre = begin()
            if pre is not None:
                comparisons.append({"kind": "prefill_repeat", **compare(pre, repeated_pre)})
            repeated = suffix(ids, prefix)
            comparisons.append({"kind": "suffix_repeat", **compare(ref, repeated)})
        torch.cuda.synchronize()
        for t in tensors:
            t.fill_(float("nan"))
        torch.cuda.synchronize()
        poison_verified = all(bool(torch.isnan(t).all()) for t in tensors)
        for target, saved in zip(tensors, snapshot, strict=True):
            target.copy_(saved)
        torch.cuda.synchronize()
        restored_hashes = [tensor_hash(t.cpu()) for t in tensors]
        comparisons.append({"kind": "suffix_restore", **compare(ref, suffix(ids, prefix))})
        record = {"prefix": prefix, "suffix_tokens": protocol["numerical_suffix_tokens"],
                  "snapshot_bytes": sum(t.numel()*t.element_size() for t in snapshot),
                  "snapshot_sha256": hashes, "poison_verified": poison_verified,
                  "restored_storage_exact": restored_hashes == hashes, "comparisons": comparisons}
        record["passed"] = poison_verified and restored_hashes == hashes and all(c["exact"] and c["finite"] for c in comparisons)
        results.append(record)
        write_json(output / "numerical.json", results)
        print("numerical", prefix, record["passed"], flush=True)
    return results


def performance(model, cache, protocol, output):
    capacity = protocol["cache_capacity"]
    def params(p):
        return {"attn_mode": "flash_attn", "cache": cache, "past_len": p,
                "batch_shape": (1, capacity), "last_tokens_only": 1,
                "pinned_staging": bool(p) and protocol["pinned_decode_staging"]}
    records = []
    for block in range(protocol["performance_blocks"]):
        for prefix in protocol["prefill_tokens"]:
            ids = tokens(prefix + 1, protocol["seed"])
            def prefill():
                return model.prefill(ids[:, :prefix], params(0))
            def last():
                logits = model.forward(ids[:, prefix:], params(prefix))
                return logits.argmax(-1).item()
            first_start = time.perf_counter()
            prefill(); last(); torch.cuda.synchronize()
            first_ms = (time.perf_counter()-first_start)*1000
            for _ in range(protocol["warmup_calls"]):
                prefill(); last()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            pre_ms, last_ms, gpu_ms = [], [], []
            for _ in range(protocol["trials_per_block"]):
                _, w, g = timed(prefill)
                _, l, _ = timed(last)
                pre_ms.append(w); last_ms.append(l); gpu_ms.append(g)
            row = {"kind": "prefill", "block": block, "prefix_tokens": prefix,
                   "first_observed_pair_ms": first_ms,
                   "prefill_ms": pre_ms, "last_prompt_token_ms": last_ms,
                   "ttft_proxy_ms": [a+b for a,b in zip(pre_ms,last_ms,strict=True)],
                   "gpu_event_prefill_ms": gpu_ms, "memory": memory()}
            row["summary"] = {name: summary(row[name]) for name in ("prefill_ms", "ttft_proxy_ms", "gpu_event_prefill_ms")}
            row["prefill_tokens_per_second"] = prefix*1000/row["summary"]["prefill_ms"]["median"]
            records.append(row)
            write_json(output / "performance.json", records)
            print("prefill", block, prefix, row["prefill_tokens_per_second"], flush=True)
        prefix = protocol["decode_prefix_tokens"]
        count = protocol["decode_tokens"]
        ids = tokens(prefix+count, protocol["seed"])
        samples, gpu_samples, request_ms, request_samples = [], [], [], []
        torch.cuda.reset_peak_memory_stats()
        for _ in range(protocol["trials_per_block"]):
            model.prefill(ids[:, :prefix], params(0)); torch.cuda.synchronize()
            trial = []
            for i in range(count):
                def step():
                    return model.forward(ids[:, prefix+i:prefix+i+1], params(prefix+i)).argmax(-1).item()
                _, w, g = timed(step)
                trial.append(w); gpu_samples.append(g)
            samples.extend(trial); request_ms.append(sum(trial)); request_samples.append(trial)
        row = {"kind": "decode", "block": block, "prefix_tokens": prefix,
               "tokens_per_trial": count, "token_ms": samples, "gpu_event_ms": gpu_samples,
               "request_ms": request_ms, "summary": summary(samples, request_samples), "request_summary": summary(request_ms),
               "tokens_per_second": count*1000/float(np.median(request_ms)), "memory": memory()}
        records.append(row)
        write_json(output / "performance.json", records)
        print("decode", block, row["tokens_per_second"], flush=True)
    return records


def profile_cuda(fn):
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    events = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    grouped = {}
    for e in events:
        item = grouped.setdefault(e.name, {"calls": 0, "duration_us": 0.0})
        item["calls"] += 1
        item["duration_us"] += e.time_range.elapsed_us()
    if not events or sum(e.time_range.elapsed_us() for e in events) <= 0:
        raise RuntimeError("CUDA profiler returned no positive device activity")
    return {"active_device_us": sum(e.time_range.elapsed_us() for e in events),
            "events": grouped}


def kernels(model, protocol, output, synthetic):
    from exllamav3.modules.quant.exl3 import LinearEXL3
    torch.manual_seed(protocol["seed"])
    specs = []
    if synthetic:
        specs = [(f"synthetic-{k}-{n}-K{rate}", k, n, rate)
                 for k,n in protocol["kernel_synthetic_shapes"] for rate in protocol["kernel_synthetic_rates"]]
    else:
        specs = [(name, None, None, None) for name in protocol["kernel_real_keys"]]
    rows_out = []
    for name, k, n, rate in specs:
        if synthetic:
            trellis = torch.randint(-32768, 32768, (k//16,n//16,16*rate), dtype=torch.int16, device="cuda")
            suh = (torch.randint(0,2,(k,),device="cuda")*2-1).half()/np.sqrt(k)
            svh = (torch.randint(0,2,(n,),device="cuda")*2-1).half()
            linear = LinearEXL3(None,k,n,suh=suh,svh=svh,trellis=trellis,
                                mcg=torch.tensor(0,dtype=torch.int32,device="cuda"),key=name)
        else:
            module = model.find_module(name)
            linear = module.inner
            k, n, rate = linear.in_features, linear.out_features, linear.K
        for rows in protocol["kernel_rows"]:
            x = torch.randn((1,rows,k),device="cuda",dtype=torch.float16)*.1
            def operation():
                return linear.forward(x, {})
            for _ in range(protocol["warmup_calls"]): operation()
            torch.cuda.synchronize()
            ref = operation(); again = operation()
            eager_cpu = ref.float().cpu().numpy().copy()
            same = compare(eager_cpu,again.float().cpu().numpy())
            recon = linear.reconstruct_hgemm(x,None)
            cross = compare(ref.float().cpu().numpy(),recon.float().cpu().numpy())
            del ref,again,recon
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            before = torch.cuda.memory_allocated()
            graph = torch.cuda.CUDAGraph()
            invocations = protocol["kernel_replays_per_sample"]
            a = torch.cuda.Event(enable_timing=True, external=True)
            b = torch.cuda.Event(enable_timing=True, external=True)
            with torch.cuda.graph(graph):
                a.record()
                for invocation in range(invocations):
                    value = operation()
                    if invocation == invocations - 1:
                        captured = value
                    del value
                b.record()
            for _ in range(protocol["warmup_calls"]): graph.replay()
            torch.cuda.synchronize()
            capture_check = compare(eager_cpu,captured.cpu().float().numpy())
            elapsed = []
            for _ in range(protocol["kernel_repeats"]):
                graph.replay()
                b.synchronize()
                elapsed.append(a.elapsed_time(b)*1000/invocations)
            trace = profile_cuda(graph.replay)
            trace["profiled_invocations"] = invocations
            trace["active_device_us"] /= invocations
            record = {"name":name,"k":k,"n":n,"K":rate,"rows":rows,
                      "path": "packed" if rows<=144 else ("fused_reconstruct" if rows>=1024 else "reconstruct"),
                      "repeat":same,"capture_check":capture_check,"versus_reconstruction":cross,"graph_replay_us":elapsed,
                      "summary":summary(elapsed),"profile":trace,
                      "torch_allocated_before":before,"memory":memory()}
            rows_out.append(record)
            write_json(output / "kernels.json",rows_out)
            print("kernel",name,rows,record["summary"]["median"],flush=True)
            del graph,captured,x
            gc.collect()
        del linear
    return rows_out


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--model",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--protocol",type=Path,default=Path(__file__).with_name("protocol.json"))
    parser.add_argument("--pins",type=Path,default=Path(__file__).with_name("pins.json"))
    parser.add_argument("--profile",choices=("optimized","attention_eager"),default="optimized")
    parser.add_argument("--mode",choices=("smoke","model","kernels","synthetic"),default="model")
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    protocol=json.loads(args.protocol.read_text())
    inherited = [k for k in os.environ if k.startswith("EXL3_")]
    if inherited:
        raise ValueError(f"undeclared inherited EXL3 overrides: {inherited}")
    for key,value in protocol["profiles"][args.profile].items(): os.environ[key]=value
    os.environ["EXL3_BC_ATTN_TRACE"]="1"
    torch.set_num_threads(protocol["threads"])
    torch.set_num_interop_threads(1)
    torch.manual_seed(protocol["seed"])
    from exllamav3 import Config,Model,Cache,Tokenizer
    from exllamav3.util.memory import vram_accounting
    fixture=next(f for f in json.loads(args.pins.read_text())["fixtures"] if f["repository"].endswith(args.model.name))
    for row in fixture["metadata_files_verified"]+[{"filename":fixture["filename"],"sha256":fixture["published_sha256"]}]:
        if digest(args.model/row["filename"])!=row["sha256"]: raise ValueError("fixture hash mismatch")
    config=Config.from_directory(str(args.model))
    model=Model.from_config(config)
    cache=Cache(model,max_num_tokens=protocol["cache_capacity"],max_batch_size=1)
    t=time.perf_counter();model.load(device="cuda:0");torch.cuda.synchronize();load_ms=(time.perf_counter()-t)*1000
    tokenizer=Tokenizer.from_config(config)
    text="<|im_start|>user\nExplain why the sky is blue.<|im_end|>\n<|im_start|>assistant\n"
    token_ids=tokenizer.encode(text,encode_special_tokens=True)
    manifest={"profile":args.profile,"mode":args.mode,"protocol_sha256":digest(args.protocol),
              "harness_sha256":digest(Path(__file__)),
              "fixture":fixture["repository"],"revision":fixture["revision"],"weight_sha256":fixture["published_sha256"],
              "host":platform.platform(),"python":platform.python_version(),"torch":torch.__version__,
              "cuda":torch.version.cuda,"device":torch.cuda.get_device_name(),"load_ms":load_ms,
              "environment":{k:v for k,v in os.environ.items() if k.startswith(("EXL3_","CUDA_","TORCH_","OMP_"))},
              "tokenizer_probe":{"utf8_sha256":hashlib.sha256(text.encode()).hexdigest(),"ids":token_ids.tolist()},
              "packages":{d.metadata["Name"]:d.version for d in importlib.metadata.distributions()},
              "after_load":memory()}
    write_json(args.output/"manifest.json",manifest)
    if args.mode=="smoke":
        ids=tokens(32,protocol["seed"])
        params={"attn_mode":"flash_attn","cache":cache,"past_len":0,"batch_shape":(1,protocol["cache_capacity"])}
        logits=model.forward(ids,params)
        print("SMOKE",tuple(logits.shape),bool(torch.isfinite(logits).all()),flush=True)
        if not bool(torch.isfinite(logits).all()):raise ValueError("nonfinite smoke logits")
    elif args.mode=="model":
        # First-use and warm performance are measured before numerical probes warm these shapes.
        performance(model,cache,protocol,args.output)
        checks=numerical(model,cache,protocol,args.output)
        manifest["numerical_passed"]=all(row["passed"] for row in checks)
    else:
        checks=kernels(model,protocol,args.output,args.mode=="synthetic")
        manifest["kernel_repeat_passed"]=all(row["repeat"]["exact"] and row["repeat"]["finite"]
                                             and row["capture_check"]["exact"] and row["capture_check"]["finite"] for row in checks)
    manifest["after_measurement"]=memory()
    manifest["vram_accounting"]=[r.as_dict() for r in vram_accounting(model,cache)]
    manifest["max_rss_bytes"]=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    write_json(args.output/"manifest.json",manifest)
    if manifest.get("numerical_passed") is False or manifest.get("kernel_repeat_passed") is False:
        raise SystemExit("FAILED required numerical/repeat controls; diagnostics preserved")
    print("DONE",args.output,flush=True)


if __name__=="__main__":
    with torch.inference_mode():
        main()
