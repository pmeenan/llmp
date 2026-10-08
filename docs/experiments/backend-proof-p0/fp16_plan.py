#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Records the FP16 toolchain bridge's executed plan (reference only).

Turns an Nsight Systems SQLite export, cuBLAS and cuBLASLt logs, cuobjdump
SASS and the bridge's compile_commands.json into one JSON "executed plan":
per llama_decode chunk, the ordered input copies, kernel launches (name, grid,
block, dynamic shared memory, launch API), memsets and memcpys; every cuBLAS
call with the heuristic's algorithm and the kernels it launched; each
kernel's build identity (translation unit, NVCC flags, SASS SHA-256); and the
GGML pool high-water mark per chunk from the instrumented copy
(fp16-pool-peak.patch, fp16_pool_peak.sh). Repeated launch runs (the 24
layers) are folded into {"repeat": N, "body": [...]} blocks, and identical
chunk sequences are stored once with their occurrences.

Capture, per arm (control|heldout x fused|unfused), in RUN_DIR:
  env CUDA_DISABLE_PTX_JIT=1 GGML_CUDA_DISABLE_GRAPHS=1 [GGML_CUDA_DISABLE_FUSION=1] \
      CUBLAS_LOGINFO_DBG=1 CUBLAS_LOGDEST_DBG=logs/ARM-cublas.log \
      CUBLASLT_LOG_LEVEL=5 CUBLASLT_LOG_FILE=logs/ARM-cublaslt.log \
      nsys profile --trace=cuda --sample=none --cpuctxsw=none --export=sqlite -o nsys/ARM \
      fp16_reference MODEL out/ARM cuda TRAJECTORY [IDS] > logs/ARM.out 2> logs/ARM.err
  and the instrumented build without nsys or logging:
      POOL_BUILD/fp16_reference MODEL poolout/ARM cuda ... 2> logs/pool-ARM.err
SASS hashes (one JSON line per function; run where the binaries are):
  cuobjdump -sass fp16_reference | fp16_plan.py sass-hash --label executable >> sass.jsonl
  for o in .../ggml-cuda.dir/**/*.cu.o: cuobjdump -sass $o | fp16_plan.py sass-hash --label TU >> sass.jsonl
Then:
  fp16_plan.py plan --run-dir RUN_DIR --compile-commands BUILD/compile_commands.json \
      --sass sass.jsonl --arm control-fused=SHA256 ... --out fp16-plan.json
"""
import argparse
import collections
import hashlib
import json
import os
import re
import sqlite3
import sys

VOCAB = 151936
ROW_BYTES = VOCAB * 4

# --------------------------------------------------------------------------- SASS


def sass_hash(label, stream):
    """Hashes each function's SASS: normalized instruction text and raw encodings."""
    offset = re.compile(r"^\s*/\*([0-9a-f]{4,})\*/\s*(.*?)\s*;?\s*(/\* (0x[0-9a-f]{16}) \*/)?\s*$")
    encoding = re.compile(r"^\s*/\* (0x[0-9a-f]{16}) \*/\s*$")
    label_line = re.compile(r"^\s*(\.L_x_\d+):\s*$")
    name = None
    text, codes, count = [], [], 0

    def flush():
        if name is not None:
            print(json.dumps({
                "label": label, "function": name,
                "text_sha256": hashlib.sha256("\n".join(text).encode()).hexdigest(),
                "encoding_sha256": hashlib.sha256("\n".join(codes).encode()).hexdigest(),
                "instructions": count}))

    for line in stream:
        if line.lstrip().startswith("Function : "):
            flush()
            name = line.split("Function : ", 1)[1].strip()
            text, codes, count = [], [], 0
            continue
        if line.startswith("Fatbin ") or line.startswith("\t\t...."):
            flush()
            name = None
            continue
        if name is None:
            continue
        m = offset.match(line)
        if m:
            text.append(" ".join(m.group(2).split()))
            if m.group(4):
                codes.append(m.group(4))
            count += 1
            continue
        m = encoding.match(line)
        if m:
            codes.append(m.group(1))
            continue
        m = label_line.match(line)
        if m:
            text.append(m.group(1) + ":")
    flush()


# --------------------------------------------------------------------------- nsys


def load_nsys(path):
    db = sqlite3.connect(path)
    strings = dict(db.execute("select id, value from StringIds"))
    api = {cid: strings[n] for cid, n in db.execute("select correlationId, nameId from CUPTI_ACTIVITY_KIND_RUNTIME")}
    flags = {sid: flag for sid, flag in db.execute("select streamId, flag from TARGET_INFO_CUDA_STREAM")}
    flag_names = {i: n for i, _, n in db.execute("select * from ENUM_CUPTI_STREAM_TYPE")}
    events = []
    kernels = {}
    for row in db.execute(
            "select correlationId, streamId, demangledName, mangledName, gridX, gridY, gridZ, blockX, blockY, blockZ, "
            "dynamicSharedMemory, staticSharedMemory, registersPerThread, localMemoryPerThread "
            "from CUPTI_ACTIVITY_KIND_KERNEL"):
        cid, sid, dn, mn, gx, gy, gz, bx, by, bz, dyn, static, regs, local = row
        mangled = strings[mn] if mn is not None else strings[dn]
        launch = re.sub(r"_v\d+$", "", api.get(cid, "?"))
        info = kernels.setdefault(mangled, {
            "demangled": strings[dn], "static_shared": static, "registers": regs,
            "local_per_thread": local, "launch_api": set()})
        info["launch_api"].add(launch)
        for key, val in (("static_shared", static), ("registers", regs), ("local_per_thread", local)):
            if info[key] != val:
                raise SystemExit(f"{mangled}: {key} varies")
        events.append({"cid": cid, "stream": sid, "type": "kernel", "kernel": mangled,
                       "grid": [gx, gy, gz], "block": [bx, by, bz], "dyn": dyn, "api": launch})
    kinds = {1: "HtoD", 2: "DtoH", 8: "DtoD"}
    mem = {0: "pageable", 1: "pinned", 2: "device"}
    for cid, sid, nbytes, kind, src, dst in db.execute(
            "select correlationId, streamId, bytes, copyKind, srcKind, dstKind from CUPTI_ACTIVITY_KIND_MEMCPY"):
        events.append({"cid": cid, "stream": sid, "type": "memcpy", "kind": kinds.get(kind, str(kind)),
                       "src": mem.get(src, str(src)), "dst": mem.get(dst, str(dst)), "bytes": nbytes})
    for cid, sid, nbytes, value in db.execute(
            "select correlationId, streamId, bytes, value from CUPTI_ACTIVITY_KIND_MEMSET"):
        events.append({"cid": cid, "stream": sid, "type": "memset", "bytes": nbytes, "value": value})
    events.sort(key=lambda e: e["cid"])
    return events, kernels, {s: flag_names.get(f, str(f)) for s, f in flags.items()}


def is_cublas_kernel(name, api):
    return api.startswith("cuLaunchKernel") or name.startswith(("nvjet", "void cublasLt::", "void cutlass::")) \
        or "cublas" in name


# --------------------------------------------------------------------------- cuBLAS logs


def parse_cublas(path):
    calls, setup = [], []
    block = None
    for line in open(path, errors="replace"):
        if line.startswith("I! cuBLAS"):
            if block:
                (calls if "Gemm" in block["function"] or "gemm" in block["function"] else setup).append(block)
            m = re.search(r"function \S+ (\w+)\(", line)
            block = {"function": m.group(1), "args": {}}
        elif block and line.startswith("i!  "):
            m = re.match(r"i!  (\w+): type=([^;]*); val=(.*)$", line.rstrip("\n"))
            if m:
                key, typ, val = m.groups()
                if val.startswith("POINTER"):
                    if key in ("workspace", "handle", "streamId", "A", "B", "C", "Aarray", "Barray", "Carray",
                               "alpha", "beta", "mode"):
                        continue
                    val = "pointer"
                else:
                    val = re.sub(r"\(\d+\)$", "", val)
                    if re.fullmatch(r"-?\d+", val):
                        val = int(val)
                block["args"][key] = val
        elif block and line.startswith("i!Process="):
            m = re.search(r"MathMode=(\w+)", line)
            if m:
                block["math_mode"] = m.group(1)
    if block:
        (calls if "Gemm" in block["function"] or "gemm" in block["function"] else setup).append(block)
    return calls, setup


def parse_cublaslt(path):
    out = []
    pref = results = None
    for line in open(path, errors="replace"):
        m = re.search(r"\]\[(Api|Info|Trace)\]\[(cublasLt\w+)\] ?(.*)$", line.rstrip("\n"))
        if not m:
            continue
        level, fn, rest = m.groups()
        if fn.endswith("AlgoGetHeuristicForStream") and level == "Api":
            pref = re.search(r"preference=\[([^\]]*)\]", rest).group(1)
        elif fn.endswith("AlgoGetHeuristicForStream") and level == "Info":
            results = rest.strip()
        elif fn.endswith("Matmul") and level == "Trace":
            rec = {"function": fn, "preference": pref, "heuristic": results}
            for key in ("Adesc", "Bdesc", "Cdesc", "Ddesc", "computeDesc", "algo"):
                rec[key] = re.search(key + r"=\[([^\]]*)\]", rest).group(1)
            rec["workspace_bytes"] = int(re.search(r"workSpaceSizeInBytes=(\d+)", rest).group(1))
            rec["beta"] = re.search(r" beta=(\S+)", rest).group(1)
            out.append(rec)
            pref = results = None
    return out


# --------------------------------------------------------------------------- run logs


def parse_llama_log(path):
    keep = []
    pats = [r"model buffer size", r"output buffer size", r"KV buffer size", r"llama_kv_cache: size =",
            r"compute buffer size", r"graph nodes", r"graph splits", r"n_ctx +=", r"n_batch +=", r"n_ubatch +=",
            r"flash_attn +=", r"VMM:", r"available_memory_kb"]
    seen = set()
    for line in open(path, errors="replace"):
        line = line.rstrip("\n")
        if any(re.search(p, line) for p in pats) and line not in seen and "available_memory_kb" not in line:
            seen.add(line)
            keep.append(line.strip())
    return keep


def parse_pool_log(path):
    out = []
    for line in open(path, errors="replace"):
        m = re.search(r"llmp_pool: nodes=(\d+) last_ne1=(\d+) allocs=(\d+) peak_used=(\d+) pool_size=(\d+) "
                      r"used_after=(\d+)", line)
        if m:
            out.append(dict(zip(("nodes", "rows", "allocs", "peak_used", "pool_size", "used_after"),
                                map(int, m.groups()))))
    return out


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


# --------------------------------------------------------------------------- plan


def fold(tokens, max_len=160):
    """Folds consecutive repeats of any block into {"repeat": n, "body": [...]}."""
    out, i, n = [], 0, len(tokens)
    while i < n:
        best = (0, 1, 1)  # covered, length, repeats
        for length in range(1, min(max_len, (n - i) // 2) + 1):
            reps = 1
            while i + (reps + 1) * length <= n and tokens[i + reps * length:i + (reps + 1) * length] == \
                    tokens[i:i + length]:
                reps += 1
            if reps > 1 and length * reps > best[0]:
                best = (length * reps, length, reps)
        if best[0]:
            _, length, reps = best
            out.append({"repeat": reps, "body": fold(tokens[i:i + length], max_len)})
            i += length * reps
        else:
            out.append(tokens[i])
            i += 1
    return out


def summarize(events, roles):
    groups = collections.Counter()
    for e in events:
        role = roles.get(e["stream"], str(e["stream"]))
        if e["type"] == "kernel":
            key = ("kernel", role, e["kernel"][:60])
        elif e["type"] == "memcpy":
            key = ("memcpy", role, f'{e["kind"]} {e["src"]}->{e["dst"]}', e["bytes"])
        else:
            key = ("memset", role, e["bytes"], e["value"])
        groups[key] += 1
    return [{"op": list(k), "count": c} for k, c in sorted(groups.items(), key=lambda x: (-x[1], str(x[0])))]


def build_arm(args, arm, expected, kernel_table, cublas_table):
    run = args.run_dir
    events, kernels, flags = load_nsys(os.path.join(run, "nsys", arm + ".sqlite"))
    for name, info in kernels.items():
        entry = kernel_table.setdefault(name, dict(info, launch_api=set()))
        entry["launch_api"] |= info["launch_api"]
    summary = json.load(open(os.path.join(run, "out", arm, "summary.json")))
    chunks = summary["chunks"]
    logits_sha = sha256_file(os.path.join(run, "out", arm, "logits.f32le"))

    ends = [i for i, e in enumerate(events) if e["type"] == "memcpy" and e["kind"] == "DtoH"
            and e["dst"] == "pinned" and e["bytes"] % ROW_BYTES == 0]
    if len(ends) != 3 * len(chunks):
        raise SystemExit(f"{arm}: {len(ends)} logit copies for {3 * len(chunks)} chunks")
    rows_seen = [events[i]["bytes"] // ROW_BYTES for i in ends]
    if rows_seen != chunks * 3:
        raise SystemExit(f"{arm}: logit copy rows {rows_seen} do not match chunks")

    legacy, setup_calls = parse_cublas(os.path.join(run, "logs", arm + "-cublas.log"))
    lt = parse_cublaslt(os.path.join(run, "logs", arm + "-cublaslt.log"))
    if len(legacy) != len(lt):
        raise SystemExit(f"{arm}: {len(legacy)} cuBLAS GEMM calls but {len(lt)} cuBLASLt matmuls")
    pool = parse_pool_log(os.path.join(run, "logs", "pool-" + arm + ".err"))
    pool_sha = sha256_file(os.path.join(run, "poolout", arm, "logits.f32le"))
    if len(pool) != 3 * len(chunks):
        raise SystemExit(f"{arm}: {len(pool)} pool records for {3 * len(chunks)} chunks")

    cublas_iter = iter(zip(legacy, lt))
    sequences, seq_index, occurrences, setups = [], {}, [], []
    start = 0
    for n, end in enumerate(ends):
        evaluation, chunk = divmod(n, len(chunks))
        rows = chunks[chunk]
        seg = events[start:end + 1]
        start = end + 1
        compute_stream = events[end]["stream"]
        first = next(i for i, e in enumerate(seg) if e["stream"] == compute_stream)
        j = first
        while j > 0 and seg[j - 1]["type"] == "memcpy" and seg[j - 1]["kind"] == "HtoD" \
                and seg[j - 1]["src"] == "pinned" and seg[j - 1]["stream"] != compute_stream:
            j -= 1
        pre, inputs, compute = seg[:j], seg[j:first], seg[first:]
        roles = {compute_stream: "compute"}
        for e in seg:
            if e["stream"] != compute_stream:
                flag = flags.get(e["stream"], "?")
                roles.setdefault(e["stream"], "per_thread" if flag == "Default stream" else f"other:{flag}")
        if pre:
            phase = "context creation" if chunk == 0 else \
                "state save, context recreation and restore" if evaluation == 2 else "between chunks"
            if evaluation == 0 and chunk == 0:
                phase = "model load and context creation"
            setups.append({"evaluation": evaluation + 1, "before_chunk": chunk, "phase": phase,
                           "ops": summarize(pre, roles)})
        # tokens
        tokens, calls = [], []
        for e in inputs:
            tokens.append(["memcpy", e["kind"], e["src"], e["dst"], e["bytes"], roles[e["stream"]]])
        k = 0
        while k < len(compute):
            e = compute[k]
            owner_cublas = e["type"] == "kernel" and is_cublas_kernel(e["kernel"], e["api"])
            starts_group = owner_cublas or (e["type"] == "memset" and k + 1 < len(compute)
                                            and compute[k + 1]["type"] == "kernel"
                                            and is_cublas_kernel(compute[k + 1]["kernel"], compute[k + 1]["api"]))
            if starts_group:
                group = []
                while k < len(compute) and (
                        (compute[k]["type"] == "kernel" and is_cublas_kernel(compute[k]["kernel"], compute[k]["api"]))
                        or (compute[k]["type"] == "memset" and not group)):
                    group.append(compute[k])
                    k += 1
                try:
                    call, ltrec = next(cublas_iter)
                except StopIteration:
                    raise SystemExit(f"{arm}: more cuBLAS kernel groups than cuBLAS calls")
                a = call["args"]
                launched = []
                for g in group:
                    if g["type"] == "kernel":
                        launched.append(["kernel", g["kernel"], g["grid"], g["block"], g["dyn"], g["api"]])
                    else:
                        launched.append(["memset", g["bytes"], g["value"], roles[g["stream"]]])
                key = json.dumps([call["function"], a, call.get("math_mode"), ltrec, launched], sort_keys=True)
                if key not in cublas_table:
                    cublas_table[key] = {
                        "id": len(cublas_table), "function": call["function"], "math_mode": call.get("math_mode"),
                        "args": a, "cublaslt": ltrec,
                        "launches": launched, "count": collections.Counter()}
                cid = cublas_table[key]["id"]
                cublas_table[key]["count"][arm] += 1
                for g in group:
                    if g["type"] == "kernel":
                        tokens.append(["kernel", g["kernel"], g["grid"], g["block"], g["dyn"], f"cublas#{cid}"])
                    else:
                        tokens.append(["memset", g["bytes"], g["value"], roles[g["stream"]], f"cublas#{cid}"])
                calls.append(cid)
                continue
            if e["type"] == "kernel":
                tokens.append(["kernel", e["kernel"], e["grid"], e["block"], e["dyn"], roles[e["stream"]]])
            elif e["type"] == "memcpy":
                tokens.append(["memcpy", e["kind"], e["src"], e["dst"], e["bytes"], roles[e["stream"]]])
            else:
                tokens.append(["memset", e["bytes"], e["value"], roles[e["stream"]]])
            k += 1
        key = json.dumps([len(inputs), tokens])
        if key not in seq_index:
            seq_index[key] = len(sequences)
            sequences.append({"id": len(sequences), "rows": rows, "input_copies": len(inputs), "tokens": tokens,
                              "cublas_calls": calls, "pool": set(), "occurrences": []})
        s = sequences[seq_index[key]]
        p = pool[n]
        if p["rows"] != rows:
            raise SystemExit(f"{arm}: pool record {n} rows {p['rows']} != {rows}")
        s["pool"].add((p["peak_used"], p["allocs"]))
        n_past = sum(chunks[:chunk])
        s["occurrences"].append([evaluation + 1, chunk, n_past])
        occurrences.append({"evaluation": evaluation + 1, "chunk": chunk, "rows": rows, "n_past": n_past,
                            "sequence": s["id"], "pool_peak_used": p["peak_used"],
                            "pool_committed_after": p["pool_size"], "pool_allocs": p["allocs"]})
    try:
        next(cublas_iter)
        raise SystemExit(f"{arm}: unmatched cuBLAS calls remain")
    except StopIteration:
        pass

    per_eval = collections.defaultdict(list)
    for o in occurrences:
        per_eval[o["evaluation"]].append((o["sequence"], o["pool_peak_used"], o["pool_allocs"]))
    evaluations_identical = per_eval[1] == per_eval[2] == per_eval[3]
    for s in sequences:
        s["pool_peak_used"] = sorted({p for p, _ in s["pool"]})
        s["pool_allocs"] = sorted({a for _, a in s["pool"]})
        del s["pool"]
        s["launches"] = sum(1 for t in s["tokens"] if t[0] == "kernel")
        s["plan"] = fold(s.pop("tokens"))
        s["occurrence_count"] = len(s["occurrences"])
        first_eval = [o[1:] for o in s["occurrences"] if o[0] == 1]
        s["first_evaluation_chunks"] = [[c, p] for c, p in first_eval]
        del s["occurrences"]
    return {
        "arm": arm,
        "trajectory": summary["trajectory"],
        "fusion": not arm.endswith("unfused"),
        "environment": {"CUDA_DISABLE_PTX_JIT": "1", "GGML_CUDA_DISABLE_GRAPHS": "1",
                        **({"GGML_CUDA_DISABLE_FUSION": "1"} if arm.endswith("unfused") else {})},
        "summary": summary,
        "profiled_logits_sha256": logits_sha,
        "expected_logits_sha256": expected,
        "profiled_bit_identical": logits_sha == expected,
        "pool_run_logits_sha256": pool_sha,
        "pool_run_bit_identical": pool_sha == expected,
        "dumped_evaluation": "logits.f32le holds the first (fresh) of the harness's three evaluations; the "
                             "plan covers all three",
        "evaluations_identical": evaluations_identical,
        "chunk_sequence_first_evaluation": [x[0] for x in per_eval[1]],
        "streams": {"compute": "the context's non-blocking stream (one per context); ggml compute, cuBLAS and "
                               "the logits copy",
                    "per_thread": "cudaStreamPerThread (nsys: default stream): input set_tensor copies, buffer "
                                  "clears, state save/restore, each followed by a synchronize",
                    "other:Non-blocking stream": "the model loader's upload stream"},
        "cublas_handle_setup": [{"function": c["function"], "args": c["args"], "math_mode": c.get("math_mode")}
                                for c in setup_calls if c["function"] != "cublasGetMathMode"][:5],
        "cublas_handle_setup_counts": dict(collections.Counter(c["function"] for c in setup_calls)),
        "llama_log": parse_llama_log(os.path.join(run, "logs", arm + ".err")),
        "sequences": sequences,
        "chunks_first_evaluation": [dict((k, v) for k, v in o.items() if k != "evaluation")
                                    for o in occurrences if o["evaluation"] == 1],
        "setup_phases": setups,
    }


def compile_identity(path):
    flags, per_tu = {}, {}
    for e in json.load(open(path)):
        if "/ggml-cuda/" not in e["file"] or not e["file"].endswith(".cu"):
            continue
        cmd = e["command"].split()
        tu = e["file"].split("/ggml/src/ggml-cuda/", 1)[1]
        cleaned = []
        skip = False
        for part in cmd[1:]:
            if skip:
                skip = False
                continue
            if part in ("-o", "-c"):
                skip = True
                continue
            if part.startswith("-ccbin=") or part.startswith("-I") or part.startswith("/"):
                continue
            if part == "-isystem":
                skip = True
                continue
            cleaned.append(part)
        key = " ".join(cleaned)
        fid = flags.setdefault(key, len(flags))
        per_tu[tu] = fid
    return {"flag_sets": [{"id": i, "nvcc_flags": k} for k, i in flags.items()], "tu_flag_set": per_tu}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sass-hash")
    s.add_argument("--label", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--compile-commands", required=True)
    p.add_argument("--sass", required=True, help="JSON lines from sass-hash")
    p.add_argument("--arm", action="append", required=True, help="NAME=EXPECTED_LOGITS_SHA256")
    p.add_argument("--library", action="append", default=[], help="NAME=SHA256 of a runtime library")
    p.add_argument("--note", action="append", default=[])
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "sass-hash":
        sass_hash(a.label, sys.stdin)
        return

    kernel_table, cublas_table = {}, {}
    arms = []
    for spec in a.arm:
        name, expected = spec.split("=", 1)
        arms.append(build_arm(a, name, expected, kernel_table, cublas_table))

    build = compile_identity(a.compile_commands)
    build["tu_count"] = len(build["tu_flag_set"])
    sass = collections.defaultdict(list)
    for line in open(a.sass):
        r = json.loads(line)
        sass[r["function"]].append(r)
    kernels = []
    order = sorted(kernel_table, key=lambda n: (is_cublas_kernel(kernel_table[n]["demangled"],
                                                                 min(kernel_table[n]["launch_api"])),
                                                kernel_table[n]["demangled"]))
    index = {n: i for i, n in enumerate(order)}
    for name in order:
        info = kernel_table[name]
        cub = is_cublas_kernel(info["demangled"], min(info["launch_api"]))
        entry = {"id": index[name], "owner": "cublas" if cub else "ggml", "mangled": name,
                 "demangled": info["demangled"], "registers": info["registers"],
                 "static_shared": info["static_shared"], "local_per_thread": info["local_per_thread"],
                 "launch_api": sorted(info["launch_api"])}
        recs = sass.get(name, [])
        exe = [r for r in recs if r["label"] == "executable"]
        tus = [r for r in recs if r["label"] != "executable"]
        if not cub:
            entry["translation_units"] = sorted({r["label"] for r in tus})
            entry["nvcc_flag_set"] = sorted({build["tu_flag_set"].get(r["label"]) for r in tus},
                                            key=lambda x: (x is None, x))
            texts = sorted({r["text_sha256"] for r in exe})
            codes = sorted({r["encoding_sha256"] for r in exe})
            entry["sass_text_sha256"] = texts[0] if len(texts) == 1 else texts
            entry["sass_encoding_sha256"] = codes[0] if len(codes) == 1 else codes
            entry["sass_instructions"] = sorted({r["instructions"] for r in exe})
            entry["copies_in_executable"] = len(exe)
            entry["tu_sass_matches_executable"] = bool(exe) and {r["encoding_sha256"] for r in tus} <= \
                {r["encoding_sha256"] for r in exe}
        kernels.append(entry)

    def renumber(node):
        if isinstance(node, dict):
            node["body"] = [renumber(x) for x in node["body"]]
            return node
        if node[0] == "kernel":
            node = list(node)
            node[1] = index[node[1]]
        return node

    used = {tu for k in kernels for tu in k.get("translation_units", [])}
    build["tu_flag_set"] = {tu: f for tu, f in build["tu_flag_set"].items() if tu in used}
    for arm in arms:
        for s in arm["sequences"]:
            s["plan"] = [renumber(x) for x in s["plan"]]
    calls = []
    for c in sorted(cublas_table.values(), key=lambda c: c["id"]):
        c = dict(c)
        c["count"] = dict(c["count"])
        c["launches"] = [[l[0], index[l[1]], *l[2:]] if l[0] == "kernel" else l for l in c["launches"]]
        calls.append(c)
    out = {
        "description": "Executed plan of the FP16 toolchain bridge (llama.cpp b29c606e2 built with the llmpalooza SDK) "
                       "for the four P0 arms, from nsys CUDA traces, cuBLAS/cuBLASLt logs, cuobjdump SASS and "
                       "an instrumented pool-peak copy. Reference only.",
        "token_format": {
            "kernel": "[\"kernel\", kernel id, grid, block, dynamic shared bytes, stream role or cublas#call]",
            "memcpy": "[\"memcpy\", kind, src memory, dst memory, bytes, stream role]",
            "memset": "[\"memset\", bytes, value, stream role, (cublas#call)]",
            "repeat": "{\"repeat\": n, \"body\": [...]}: the body runs n times in a row"},
        "notes": a.note,
        "libraries": dict(x.split("=", 1) for x in a.library),
        "build": build,
        "kernels": kernels,
        "cublas_calls": calls,
        "arms": arms,
    }
    with open(a.out, "w") as f:
        f.write(dump(json.loads(json.dumps(out, default=lambda o: sorted(o) if isinstance(o, set) else o))))
        f.write("\n")


def dump(node, level=0):
    """JSON with one line per leaf list or small object, so launch tokens stay on one line."""
    pad, inner = " " * level, " " * (level + 1)
    flat = json.dumps(node)
    if not isinstance(node, (list, dict)) or len(flat) <= 100 or (
            isinstance(node, list) and all(not isinstance(x, (list, dict)) for x in node)):
        return flat
    if isinstance(node, list):
        return "[\n" + ",\n".join(inner + dump(x, level + 1) for x in node) + "\n" + pad + "]"
    return "{\n" + ",\n".join(inner + json.dumps(k) + ": " + dump(v, level + 1) for k, v in node.items()) + \
        "\n" + pad + "}"


if __name__ == "__main__":
    main()
