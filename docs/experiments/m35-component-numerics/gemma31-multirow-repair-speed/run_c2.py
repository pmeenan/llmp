# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0

import hashlib, importlib.util, json, math, os, pathlib, re, statistics, subprocess, sys, time

B = pathlib.Path(__file__).resolve().parent
P = json.loads((B / "speed-pins.json").read_text())
OUT = pathlib.Path(P["output_root"])

def sha(path):
    with pathlib.Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()

def write(path, value):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        json.dump(value, f, indent=2)
        f.write("\n")

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def helper():
    assert sha(P["helper"]["path"]) == P["helper"]["sha256"]
    return load(P["helper"]["path"], "c2_speed_gate")

def quality_gate():
    q = load(P["quality_preflight"]["path"], "c2_quality_preflight")
    assert sha(P["quality_preflight"]["path"]) == P["quality_preflight"]["sha256"]
    # This authenticates original A in its original namespace and current R in
    # the repaired namespace. The reviewer-issued receipts are required.
    checked = q.check("r")
    qr = q.require_manual_review("r")
    pair_path = pathlib.Path(P["quality_pair"]["path"])
    assert sha(pair_path) == P["quality_pair"]["sha256"]
    pair = json.loads(pair_path.read_text())
    assert pair["pass"] is True
    assert sha(P["manual_reviews"]["r_path"]) == P["manual_reviews"]["r_sha256"]
    assert qr["arm_valid"] is True
    assert checked["reference_a"]["manual_review_approved"] is True
    assert checked["reference_a"]["manual_review_sha256"] == P["manual_reviews"]["original_a_sha256"]
    assert checked["registration"] == P["quality_registration"]["commit"]
    assert sha(P["quality_registration"]["path"]) == P["quality_registration"]["sha256"]
    assert sha(P["manual_reviews"]["r_path"]) == P["manual_reviews"]["r_sha256"]
    return {"check": checked, "manual_r": qr, "pair_sha256": sha(pair_path)}

def guard():
    reg = json.loads(pathlib.Path(P["registration"]["path"]).read_text())
    assert reg["enabled"] is True and reg["reviewed_registration"] is True
    assert reg["preparation_only"] is False and re.fullmatch(r"[0-9a-f]{40}", reg["commit"])
    for path, digest in reg["files"].items():
        assert sha(pathlib.Path(P["registration"]["root"]) / path) == digest
    assert sha(__file__) == P["executor_sha256"]
    assert sha(B / "speed-node.toml") == P["config_sha256"]
    assert sha(P["helper"]["path"]) == P["helper"]["sha256"]
    assert sha(P["runtime"]["path"]) == P["runtime"]["sha256"]
    assert sha(P["input"]["corpus_path"]) == P["input"]["corpus_sha256"]
    q = quality_gate()
    env = {k: v for k, v in os.environ.items()
           if k.startswith(("LLMP", "LD_", "CUDA", "NVIDIA"))}
    assert not env, env
    return {"registration": reg["commit"], "executor_sha256": sha(__file__),
            "runtime_sha256": P["runtime"]["sha256"], "quality_gate": q,
            "environment": env}

def same_guard(before, after):
    b = json.loads(json.dumps(before)); a = json.loads(json.dumps(after))
    left, right = b["quality_gate"]["check"], a["quality_gate"]["check"]
    left_ldd, right_ldd = left.pop("ldd"), right.pop("ldd")
    assert b == a
    assert re.sub(r"0x[0-9a-f]+", "ADDRESS", left_ldd) == re.sub(
        r"0x[0-9a-f]+", "ADDRESS", right_ldd)

def out(pair, arm):
    return OUT / f"pair-{pair}-{arm}"

def argv(pair, arm):
    g = helper()
    order = P["orders"][pair - 1]
    assert arm in order
    name = {"a": "old", "r": "new"}[arm]
    args = [sys.executable, "-B", P["helper"]["path"], "run", "--family", "gemma31",
            "--cells", "2", "--runtime", P["runtime"]["path"], "--label", name,
            "--out", str(out(pair, arm)), "--block", str(pair), "--position", str(order.index(arm)),
            "--config-template", str(B / "speed-node.toml"), "--corpus",
            P["input"]["corpus_path"], "--port", "18114"]
    for key, value in sorted(P["runtime_env"][arm].items()):
        args += ["--runtime-env", key + "=" + value]
    return args

def audit(pair, arm):
    g = helper()
    d = out(pair, arm)
    paths = list((d / "runs").glob("*.json"))
    assert len(paths) == 1
    path = paths[0]
    r = json.loads(path.read_text())
    assert r["error"] is None and r["problems"] == []
    assert r["stop"] == {"signal": "TERM", "returncode": 0}
    assert r["facts"]["gpu_compute_apps"] == []
    assert r["runtime_sha256"] == P["runtime"]["sha256"]
    assert r["config_sha256"] == P["config_sha256"]
    assert r["config_template"] == (B / "speed-node.toml").read_text()
    assert r["runtime_env"] == P["runtime_env"][arm]
    assert r["family_spec"]["cells"] == [2] and r["family_spec"]["repeats"] == 1
    assert r["label"] == {"a": "old", "r": "new"}[arm]
    assert r["block"] == pair and r["position"] == P["orders"][pair - 1].index(arm)
    assert len(r["passes"]) == 2
    identities = []
    for index, row in enumerate(r["passes"]):
        assert row["warmup"] == (index == 0) and set(row["cells"]) == {"C2"}
        cell = row["cells"]["C2"]
        assert cell["width"] == 2 and cell["prompts"] == [2 * index, 2 * index + 1]
        assert cell["problems"] == [] and len(cell["requests"]) == 2
        for req in cell["requests"]:
            usage = req["usage"]
            assert req["error"] is None and req["finish"] == "length" and req["chunks"] == 128
            assert usage["completion_tokens"] == 128
            assert usage["total_tokens"] == usage["prompt_tokens"] + 128
            assert usage["prompt_tokens_details"]["cached_tokens"] == 0
            assert len(req["chunk_times"]) == 128
            assert all(math.isfinite(x) for x in req["chunk_times"])
        identities.append({"warmup": row["warmup"], "prompts": cell["prompts"],
                           "usage": [x["usage"] for x in cell["requests"]]})
    metrics = g.run_metrics(r, __import__("dataclasses").replace(
        g.FAMILIES["gemma31"], cells=(2,), repeats=1))
    expected = {"C2.prefill_s", "C2.decode_tok_s", "C2.wall_tok_s", "peak_gib"}
    assert set(metrics) == expected and all(math.isfinite(x) and x > 0 for x in metrics.values())
    result = {"valid": True, "pair": pair, "arm": arm, "record": str(path),
              "record_sha256": sha(path), "runtime_log_sha256": sha(next((d / "work").rglob("runtime.log"))),
              "metrics": metrics, "request_identity": identities}
    write(d / "audit.json", result)
    return result

def pair(pair):
    a = json.loads((out(pair, "a") / "audit.json").read_text())
    r = json.loads((out(pair, "r") / "audit.json").read_text())
    assert a["valid"] and r["valid"] and a["request_identity"] == r["request_identity"]
    ratios = {"C2.prefill_s": r["metrics"]["C2.prefill_s"] / a["metrics"]["C2.prefill_s"],
              "C2.decode_tok_s": a["metrics"]["C2.decode_tok_s"] / r["metrics"]["C2.decode_tok_s"],
              "C2.wall_tok_s": a["metrics"]["C2.wall_tok_s"] / r["metrics"]["C2.wall_tok_s"]}
    result = {"pair": pair, "valid": True, "slower_candidate_ratios": ratios,
              "within_1_10": all(x <= 1.10 for x in ratios.values()),
              "arm_audit_sha256": {x: sha(out(pair, x) / "audit.json") for x in ("a", "r")}}
    write(OUT / f"pair-{pair}.json", result)
    if pair == 3:
        rows = [json.loads((OUT / f"pair-{i}.json").read_text()) for i in (1, 2, 3)]
        medians = {k: statistics.median(x["slower_candidate_ratios"][k] for x in rows)
                   for k in ratios}
        write(OUT / "result.json", {"valid": True, "pairs": rows, "median_ratios": medians,
                                     "pass": all(x <= 1.10 for x in medians.values())})
    return result

if __name__ == "__main__":
    os.umask(0o077)
    assert os.uname().nodename == "spark-56f5"
    n, arm = int(sys.argv[1]), sys.argv[2]
    assert n in (1, 2, 3) and arm in ("a", "r")
    order = P["orders"][n - 1]
    assert arm in order
    index = order.index(arm)
    if n > 1:
        first = json.loads((OUT / "pair-1.json").read_text())
        assert first["within_1_10"] and (OUT / f"pair-{n-1}.json").is_file()
    if n == 3:
        assert (OUT / "pair-2.json").is_file()
    if index:
        assert json.loads((out(n, order[0]) / "audit.json").read_text())["valid"]
    before = guard()
    d = out(n, arm)
    d.mkdir(mode=0o700, parents=True, exist_ok=False)
    write(d / "preflight.json", before)
    args = argv(n, arm)
    write(d / "launch.json", {"argv": args, "executor_sha256": sha(__file__), "start_unix": time.time(),
                              "collector_ld_library_path": P["runtime"]["ld_library_path"]})
    try:
        with (d / "collector-output.txt").open("xb") as f:
            collector_env = dict(os.environ)
            collector_env["LD_LIBRARY_PATH"] = P["runtime"]["ld_library_path"]
            rc = subprocess.run(args, env=collector_env, stdout=f,
                                stderr=subprocess.STDOUT).returncode
        write(d / "returncode.json", {"returncode": rc})
    finally:
        after = guard()
        write(d / "postflight.json", after)
        same_guard(before, after)
    assert rc == 0
    result = audit(n, arm)
    print(json.dumps(result), flush=True)
    if index:
        result = pair(n)
        print(json.dumps(result), flush=True)
        if n == 1:
            assert result["within_1_10"], "registered first-pair stop"
