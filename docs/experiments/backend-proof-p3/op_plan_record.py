#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""The native EXL3 operation plan records for P3's trajectories (reference only).

External reference tooling for the backend proof's P3; it does not implement llmpalooza inference and
shares no code with native's planner. It reuses P0's ../backend-proof-p0/exl3_op_plan.py unchanged
(probe, sass and build) with one more phase kind: the trajectory "prefix 1,023 then 16 steps"
takes its first step at position 1,023 with N = 1,024 attended positions, so K is padded to 1,024,
a single-token kind P0's record (exl3-op-plan.json) lacks. The approved rule requires a
reference-only record of an unrecorded kind before native runs it.

  probe (in the reference container, run_container.sh; GPU): P0's probe with PHASES extended by
      ("step", 1023), under EXL3-G (--arm G: EXL3_GEMV=0, P0's frozen tune-40-gemvoff /
      tune-45-gemvoff) or EXL3-O (--arm O: EXL3_GEMV unset, upstream's default heuristic, P0's
      frozen tune-40 / tune-45). Every other option is P0's probe's. The probe compares each phase's
      logits with the reference ggml_ops arm (--reference), which runs EXL3-G: under O the prefills
      must still be equal (the GEMV serves at most eight rows), the single-token steps need not be.

  build: P0's build over both fixtures' probes, then:
      - the padding text covers the new kind;
      - G: every entry of P0's seven kinds (operations, order, launches with kernels compared by
        identity, not by id, linear paths, dtypes, shape parameters) must equal P0's record, and
        the comparison is written into the record (p0_record_comparison);
      - O: every prefill's logits must equal the G reference's; the record's differences from the
        G record (--g-record) are written into it (g_record_comparison);
      - the tuning cache must be unchanged by each probe and be the arm's frozen P0 cache.

      op_plan_record.py build --arm G --probe 4.0bpw=probe-40-G.json 4.5bpw=probe-45-G.json \\
          --sass sass.jsonl --res-usage res.txt --out exl3-op-plan-g.json
      op_plan_record.py build --arm O --probe ... --g-record exl3-op-plan-g.json --out exl3-op-plan-o.json

  compare A.json B.json: the mechanical differences between two records (kernels by identity).

The SASS inputs are P0's: cuobjdump -sass / -res-usage of the shim build's libggml-cuda.so.0.24.0,
hashed with ../backend-proof-p0/fp16_plan.py sass-hash.
"""

import argparse
import base64
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
P0 = HERE.parent / "backend-proof-p0"
sys.path.insert(0, str(P0))
import exl3_op_plan  # noqa: E402

NEW_KIND = ("step", 1023)
PHASES = tuple(sorted(exl3_op_plan.PHASES + (NEW_KIND,), key=lambda k: (k[0] != "prefill", k[1])))
FROZEN = {("G", "4.0bpw"): "tune-40-gemvoff", ("G", "4.5bpw"): "tune-45-gemvoff",
          ("O", "4.0bpw"): "tune-40", ("O", "4.5bpw"): "tune-45"}
PADDING = ("Every phase, prefill and single-token alike, attends K and V padded to Npad = N rounded up to a "
           "multiple of 256 (FATTN_KQ_STRIDE; llama.cpp pads its cache the same way). The recorded phase "
           "kinds cover N = 32, 144, 145 (Npad 256), 1,023 and 1,024 (Npad 1,024), and single-token steps "
           "with N 33-161 (Npad 256), N = 1,024 (Npad 1,024: the first step after the 1,023-row prefix, "
           "at position 1,023) and N 1,025-1,040 (Npad 1,280), all the trajectories reach.")
# Per phase kind, the entries compared between records (fixtures' entries per fixture).
KIND_KEYS = ("kind", "rows", "first_position", "attended", "padded_kv_length", "shape_parameters", "embedding",
             "layer_order", "layer_launches", "output", "output_order")
FIXTURE_KEYS = ("linear_launches", "lm_head_launches", "linear_paths", "K_by_linear")
GLOBAL_KEYS = ("model", "tensors", "operations", "rope", "kv_write", "launch_format")


def digest(path):
    with open(path, "rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def frozen_sha256(name):
    caches = json.loads((P0 / "results.json").read_text())["exl3"]["tuning_caches"]
    return hashlib.sha256(base64.b64decode(caches[name]["base64"])).hexdigest()


# --------------------------------------------------------------------------- probe


def probe(arm, rest):
    exl3_op_plan.PHASES = PHASES
    if arm == "O":
        # P0's probe sets EXL3_GEMV=0 before loading the model; ExLlamaV3 reads it at every call
        # (exl3_gemv.cu, getenv), so unsetting it here gives upstream's default for every forward.
        install = exl3_op_plan.install

        def install_gemv_on(*args):
            os.environ.pop("EXL3_GEMV", None)
            return install(*args)
        exl3_op_plan.install = install_gemv_on
    sys.argv = [str(P0 / "exl3_op_plan.py"), "probe", *rest]
    exl3_op_plan.main()
    out = Path(rest[rest.index("--output") + 1])
    record = json.loads(out.read_text())
    if arm == "O" and "EXL3_GEMV" in record["environment"]:
        raise SystemExit("EXL3-O probe ran with EXL3_GEMV set")
    record["p3"] = {"arm": arm, "phases": [list(k) for k in PHASES],
                    "op_plan_record_sha256": digest(Path(__file__))}
    out.write_text(json.dumps(record) + "\n")


# --------------------------------------------------------------------------- compare


def normalized(record):
    """The record's comparable entries, with kernel ids replaced by the kernel's identity."""
    identity = {k["id"]: json.dumps({f: v for f, v in k.items() if f != "id"}, sort_keys=True)
                for k in record["kernels"]}

    def resolve(node):
        if isinstance(node, list):
            if len(node) == 4 and isinstance(node[0], int) and isinstance(node[1], list):
                return [identity[node[0]], *node[1:]]
            return [resolve(x) for x in node]
        if isinstance(node, dict):
            return {k: resolve(v) for k, v in node.items()}
        return node

    kinds = {}
    for pk in record["phase_kinds"]:
        entry = {k: resolve(pk[k]) for k in KIND_KEYS}
        for fixture, part in pk["fixtures"].items():
            entry.update({f"{fixture}/{k}": resolve(part[k]) for k in FIXTURE_KEYS})
        kinds[pk["phase_kind"]] = entry
    common = {k: record[k] for k in GLOBAL_KEYS}
    common["attention"] = {k: v for k, v in record["attention"].items() if k != "padding"}
    common["bias_add.owner"] = record["bias_add"]["owner"]
    return common, kinds


def differences(a, b):
    """Mechanical differences between records a and b: [(where, a's value, b's value)]."""
    out = []

    def walk(where, x, y):
        if isinstance(x, dict) and isinstance(y, dict):
            for key in sorted(set(x) | set(y)):
                walk(f"{where}.{key}" if where else key, x.get(key, "<absent>"), y.get(key, "<absent>"))
        elif isinstance(x, list) and isinstance(y, list) and len(x) == len(y):
            for i, (u, v) in enumerate(zip(x, y)):
                walk(f"{where}[{i}]", u, v)
        elif x != y:
            out.append((where, x, y))
    (ca, ka), (cb, kb) = normalized(a), normalized(b)
    walk("", ca, cb)
    for kind in sorted(set(ka) | set(kb)):
        if kind not in ka or kind not in kb:
            out.append((f"phase kind {kind}", "present" if kind in ka else "<absent>",
                        "present" if kind in kb else "<absent>"))
        else:
            walk(f"phase kind {kind}", ka[kind], kb[kind])
    return out


def short(value):
    text = json.dumps(value)
    return text if len(text) <= 160 else text[:157] + "..."


def summary(diffs):
    """The differences, with kernel identities shortened to their names."""
    def name(value):
        if isinstance(value, list):
            return [name(v) for v in value]
        if isinstance(value, str) and value.startswith("{") and '"name"' in value:
            return json.loads(value)["name"]
        return value
    return [{"where": w, "a": name(x), "b": name(y)} for w, x, y in diffs]


# --------------------------------------------------------------------------- build


def build(args):
    exl3_op_plan.PHASES = PHASES
    exl3_op_plan.load_res_usage(args.res_usage)
    probes = dict(item.split("=", 1) for item in args.probe)
    problems = []
    for fixture, path in probes.items():
        rec = json.loads(Path(path).read_text())
        want = frozen_sha256(FROZEN[(args.arm, fixture)])
        if rec.get("p3", {}).get("arm") != args.arm:
            problems.append(f"{fixture}: the probe is not an EXL3-{args.arm} probe")
        if not rec["tune_cache_before_sha256"] == rec["tune_cache_after_sha256"] == want:
            problems.append(f"{fixture}: tuning cache not the frozen {FROZEN[(args.arm, fixture)]} or changed")
        if not all(c["cases"] == c["equal"] for c in rec["checks"].values()):
            problems.append(f"{fixture}: a moved operation differs from upstream's")
        for phase in rec["phases"]:
            equal = phase["checked_run_logits_equal_reference"] and phase["profiled_logits_equal_reference"]
            if not equal and (args.arm == "G" or phase["kind"] == "prefill"):
                problems.append(f"{fixture} {phase['kind']} {phase['p']}: logits differ from the G reference arm")
    if problems:
        raise SystemExit("\n".join(problems))
    with tempfile.TemporaryDirectory() as tmp:
        out, own = Path(tmp) / "record.json", []
        for fixture, path in probes.items():
            rec = json.loads(Path(path).read_text())
            for phase in rec["phases"]:
                phase["calls"] = own_calls(phase)
            (Path(tmp) / fixture).write_text(json.dumps(rec))
            own.append(f"{fixture}={Path(tmp) / fixture}")
        exl3_op_plan.build(argparse.Namespace(probe=own, sass=args.sass, fp16_plan=args.fp16_plan,
                                              res_usage=args.res_usage, out=out))
        record = json.loads(out.read_text())
    record["recorded_on"] = "2026-09-27"
    record["tool"] = ("../backend-proof-p3/op_plan_record.py probe and build (P0's exl3_op_plan.py with the "
                      "phase kind step 1023 added); fp16_plan.py sass-hash")
    record["attention"]["padding"] = PADDING
    probe_recs = {fx: json.loads(Path(p).read_text()) for fx, p in probes.items()}
    for fx, rec in probe_recs.items():
        record["provenance"]["probes"][fx]["p3"] = rec["p3"]
        record["provenance"]["probes"][fx]["environment"] = rec["environment"]
    if args.arm == "G":
        p0 = json.loads(args.p0_record.read_text())
        diffs = differences(p0, record)
        added = [f"{k} {p}" for k, p in PHASES if (k, p) not in recorded_kinds(p0)]
        corrected = [d for d in diffs if "/K_by_linear" in d[0]]
        unexpected = [d for d in diffs if d[0] not in {f"phase kind {a}" for a in added} and d not in corrected]
        record["p0_record_comparison"] = {
            "record": "../backend-proof-p0/exl3-op-plan.json",
            "rule": "every entry of the global tables and of each of P0's phase kinds, per fixture, kernels "
                    "compared by identity (every field but the id)",
            "phase_kinds_added": added, "differences": summary(unexpected),
            "K_by_linear_corrected": summary(corrected),
            "K_by_linear_note": "P0's probe appends the calls of a single-token kind's unprofiled prefill to the "
                                "previous phase's call list, so P0's K_by_linear of that phase also lists the "
                                "prefill's linears (step 32 lists gate_proj and up_proj, which a single-token step "
                                "runs as the gate_up multi-linear). This build keeps each phase's own calls (those "
                                "under its profiled 'phase' range); launches were never affected."}
        if unexpected:
            print(json.dumps(summary(unexpected), indent=1))
            raise SystemExit(f"{len(unexpected)} differences from P0's record")
    else:
        record["interpretation"] = (
            "The native EXL3-O operation plan, as a record: EXL3-G's plan (exl3-op-plan-g.json) run with "
            "ExLlamaV3's GEMV on (EXL3_GEMV unset, upstream's default) and the frozen GEMV-on caches tune-40 / "
            "tune-45. The same probe as EXL3-G's: the ggml_ops arm plus GGML's embedding and MLP residual add and "
            "ExLlamaV3's bias add on the reconstruction path, both fixtures, eight phase kinds. Its prefill "
            "logits equal the EXL3-G ggml_ops arm's (cuBLAS 13.8.0.4) bit for bit; single-token steps are "
            "compared with that arm and not required to equal it (evidence). Reference only.")
        record["pins"]["profile"] = ("EXL3-O: EXL3_GEMV unset (upstream's default heuristic), EXL3_HGEMM_F16ACC=0, "
                                     "EXL3_BC_ATTN=0, frozen tune-40 / tune-45 (copies, unchanged)")
        for part in record["provenance"]["probes"].values():
            part["passed_meaning"] = ("P0's probe flag, which also requires every phase's logits to equal the EXL3-G "
                                      "reference arm's: false here because the single-token steps differ (evidence)")
        g = json.loads(args.g_record.read_text())
        record["g_record_comparison"] = {"record": args.g_record.name, "differences": summary(differences(g, record))}
    from fp16_plan import dump
    args.out.write_text(dump(record) + "\n")
    print("wrote", args.out, len(record["kernels"]), "kernels,", len(record["phase_kinds"]), "phase kinds")


def own_calls(phase):
    """The phase's own labelled calls: those under its profiled 'phase' range. P0's probe also appends
    the unprofiled prefill of the next single-token kind to the stored list; those have other roots."""
    calls = {c["seq"]: c for c in phase["calls"]}

    def root(c):
        while c["parent"] is not None:
            c = calls[c["parent"]]
        return c
    return [c for c in phase["calls"] if root(c)["category"] == "phase"]


def recorded_kinds(record):
    return {(pk["phase_kind"].split()[0], int(pk["phase_kind"].split()[1])) for pk in record["phase_kinds"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("probe", help="P0's probe options follow --arm")
    p.add_argument("--arm", choices=("G", "O"), required=True)
    b = sub.add_parser("build")
    b.add_argument("--arm", choices=("G", "O"), required=True)
    b.add_argument("--probe", nargs="+", required=True, help="FIXTURE=probe.json")
    b.add_argument("--sass", type=Path, required=True)
    b.add_argument("--res-usage", type=Path, required=True)
    b.add_argument("--fp16-plan", type=Path, default=P0 / "fp16-plan.json")
    b.add_argument("--p0-record", type=Path, default=P0 / "exl3-op-plan.json")
    b.add_argument("--g-record", type=Path, default=HERE / "exl3-op-plan-g.json")
    b.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("compare")
    c.add_argument("a", type=Path)
    c.add_argument("b", type=Path)
    args, rest = parser.parse_known_args()
    if args.cmd == "probe":
        return probe(args.arm, rest)
    if rest:
        parser.error(f"unrecognized arguments: {rest}")
    if args.cmd == "build":
        return build(args)
    diffs = summary(differences(json.loads(args.a.read_text()), json.loads(args.b.read_text())))
    for d in diffs:
        print(d["where"], "|", short(d["a"]), "->", short(d["b"]))
    print(len(diffs), "differences")
    return 1 if diffs else 0


if __name__ == "__main__":
    sys.exit(main())
