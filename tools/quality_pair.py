#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Pair native per-target losses with retained reference losses, against pre-registered bounds.

The quality evidence of M3.6's final pass (docs/experiments/m36-baselines/README.md, "The
gate's tools"; the revised quality policy and its audit protocol,
docs/experiments/m36-numerical-audit/README.md): the same teacher IDs, the same history, the
full-vocabulary negative log-likelihood of each actual target, a reference of the same
quantized checkpoint, and acceptance bounds fixed before the run they govern.

    quality_pair.py pair --candidate FILE --reference FILE --prereg PREREG.json
                         [--report OUT.json] [--json]
    quality_pair.py extract --candidate FILE --out LOSSES.jsonl

A *loss file* is one of
  * `losses.jsonl`: one `{"document": D, "position": P, "target": ID, "nll": X}` a line
    (collect-qwen-reference.py's output for a reference implementation, and what `extract`
    writes for a native record, so that it can be retained as a later reference);
  * a `speed_gate.py quality` record (`FAMILY-quality-LABEL.json`) with its sibling
    score-response file, which holds every target loss with its token ID (document 0, the
    position in the scored prompt; the file's SHA-256 is verified);
  * a `k1_quality.py` arm record (`FAMILY-k1-ENCODING.json`): per-token scores and
    `ids_sha256` (SHA-256 of the JSON list of its target IDs) from `prefix_tokens` on, without
    the IDs themselves, so it pairs only with a reference that has them, and the digest of the
    reference's IDs over the same positions must equal it.

The pairing keys every loss by (document, position) and refuses anything but a loss for each
of the same targets: different key sets, a target ID that differs at any position (a
tokenizer difference), a non-finite or positive log-probability, a duplicate, a failed record.
The pre-registration is a JSON object

    {"format": "llmp-quality-prereg/1", "name": "...", "reference_sha256": "...",   (optional)
     "bounds": {"expected_targets": N, "min_targets": N,
                "max_ppl_increase_pct": X,        candidate PPL over the reference's, percent
                "max_abs_ppl_delta_pct": X, "max_abs_nll_delta": X,
                "max_abs_doc_ppl_delta_pct": X,   every document's
                "max_mean_abs_diff": X, "max_abs_diff": X,     per-target |NLL difference|
                "min_agreement": F, "agreement_tolerance": X}}  fraction of targets within X

At least one bound; an unknown key is refused. Exit status: 0 every bound holds, 1 a bound
fails, 2 the inputs are refused. The report lists the aggregate NLL and PPL of both sides,
their delta, per-document PPL deltas, mean and max absolute differences (and where the
largest is), the agreement, how many targets are lower and higher, and each bound's result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import sys

sys.dont_write_bytecode = True

PREREG_FORMAT = "llmp-quality-prereg/1"
REPORT_FORMAT = "llmp-quality-pair/1"
CAP = 256 << 20  # a response with every target loss is a few MiB; refuse an absurd file
BOUNDS = ("expected_targets", "min_targets", "max_ppl_increase_pct", "max_abs_ppl_delta_pct",
          "max_abs_nll_delta", "max_abs_doc_ppl_delta_pct", "max_mean_abs_diff",
          "max_abs_diff", "min_agreement", "agreement_tolerance")


class Refused(Exception):
    """The inputs cannot be paired (exit 2)."""


class Losses:
    """Target losses keyed by (document, position)."""

    def __init__(self, source: str, kind: str):
        self.source = source
        self.kind = kind
        self.rows: dict[tuple[int, int], tuple[int | None, float]] = {}
        self.ids_sha256: str | None = None  # k1: the digest of the target IDs, as recorded

    def add(self, document: int, position: int, target: int | None, nll: float) -> None:
        key = (document, position)
        if key in self.rows:
            raise Refused(f"{self.source}: duplicate loss for document {document} "
                          f"position {position}")
        if isinstance(nll, bool) or not isinstance(nll, (int, float)) or not math.isfinite(nll) \
                or nll < 0:
            raise Refused(f"{self.source}: document {document} position {position}: loss {nll!r} "
                          "is not a finite nonnegative number (log-probability must be <= 0)")
        self.rows[key] = (target, float(nll))


def read_bounded(path: pathlib.Path) -> bytes:
    try:
        with path.open("rb") as f:
            raw = f.read(CAP + 1)
    except OSError as e:
        raise Refused(f"{path}: {e}") from e
    if len(raw) > CAP:
        raise Refused(f"{path}: larger than {CAP} bytes")
    return raw


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def token_id(label, where: str) -> int:
    """A target ID from a prompt-logprob key or an echoed `token_id:N` label."""
    text = label[len("token_id:"):] if isinstance(label, str) and label.startswith(
        "token_id:") else label
    if not isinstance(text, str) or not text.isascii() or not text.isdecimal() or \
            text != str(int(text)):
        raise Refused(f"{where}: {label!r} is not a canonical token ID")
    return int(text)


def from_jsonl(path: pathlib.Path, raw: bytes) -> Losses:
    losses = Losses(str(path), "losses")
    for number, line in enumerate(raw.decode().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            document, position, target, nll = (row["document"], row["position"], row["target"],
                                               row["nll"])
        except (ValueError, KeyError, TypeError) as e:
            raise Refused(f"{path}:{number}: not a loss row ({e})") from e
        if not all(type(v) is int for v in (document, position, target)):
            raise Refused(f"{path}:{number}: document, position and target must be integers")
        losses.add(document, position, target, nll)
    if not losses.rows:
        raise Refused(f"{path}: no losses")
    return losses


def from_speed_gate(path: pathlib.Path, record: dict) -> Losses:
    if record.get("error"):
        raise Refused(f"{path}: the quality run failed: {record['error']}")
    member = record.get("score_response")
    if not isinstance(member, dict) or not isinstance(member.get("file"), str) or \
            pathlib.Path(member["file"]).name != member["file"]:
        raise Refused(f"{path}: no sibling score_response file (a record from before per-target retention keeps only the aggregate; rerun `speed_gate.py quality`)")
    raw = read_bounded(path.parent / member["file"])
    if sha256(raw) != member.get("sha256"):
        raise Refused(f"{path}: the score response changed since it was recorded")
    try:
        choice = json.loads(raw)["choices"][0]
        rows = choice["prompt_logprobs"]
    except (ValueError, KeyError, IndexError, TypeError) as e:
        raise Refused(f"{path}: the score response has no prompt_logprobs ({e})") from e
    echo = (choice.get("logprobs") or {}).get("tokens")
    if not isinstance(rows, list) or not rows or rows[0] is not None:
        raise Refused(f"{path}: prompt_logprobs must start with the unscored first token")
    losses = Losses(str(path), "speed_gate")
    for position in range(1, len(rows)):
        row = rows[position]
        if not isinstance(row, dict) or len(row) != 1:
            raise Refused(f"{path}: position {position}: a row must hold exactly its target")
        label, entry = next(iter(row.items()))
        target = token_id(label, f"{path}: position {position}")
        if echo is not None and (position >= len(echo) or
                                 token_id(echo[position], f"{path}: echo {position}") != target):
            raise Refused(f"{path}: position {position}: the echoed token and the scored target "
                          "disagree")
        try:
            logprob = entry["logprob"]
        except (KeyError, TypeError) as e:
            raise Refused(f"{path}: position {position}: no logprob") from e
        if isinstance(logprob, bool) or not isinstance(logprob, (int, float)):
            raise Refused(f"{path}: position {position}: logprob is not numeric")
        losses.add(0, position, target, -float(logprob))
    return losses


def from_k1(path: pathlib.Path, record: dict) -> Losses:
    if record.get("error"):
        raise Refused(f"{path}: the K1 arm failed: {record['error']}")
    scores, prefix = record.get("scores"), record.get("prefix_tokens")
    if not isinstance(scores, list) or not scores or type(prefix) is not int or \
            not isinstance(record.get("ids_sha256"), str):
        raise Refused(f"{path}: not a K1 arm record (scores, prefix_tokens, ids_sha256)")
    losses = Losses(str(path), "k1")
    losses.ids_sha256 = record["ids_sha256"]
    first = max(prefix, 1)  # the passage's rows, as k1_quality.score_rows takes them
    for offset, score in enumerate(scores):
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise Refused(f"{path}: score {offset} is not numeric")
        losses.add(0, first + offset, None, -float(score))
    return losses


def load(path_name: str) -> tuple[Losses, str]:
    """(losses, file SHA-256)."""
    path = pathlib.Path(path_name)
    raw = read_bounded(path)
    digest = sha256(raw)
    if path.suffix == ".jsonl":
        return from_jsonl(path, raw), digest
    try:
        record = json.loads(raw)
    except ValueError as e:
        raise Refused(f"{path}: neither a .jsonl loss file nor JSON ({e})") from e
    if isinstance(record, dict) and "score_response" in record:
        return from_speed_gate(path, record), digest
    if isinstance(record, dict) and "scores" in record:
        return from_k1(path, record), digest
    if isinstance(record, dict) and str(record.get("format", "")).startswith("llmp-k1-quality"):
        raise Refused(f"{path}: a combined K1 record keeps no per-token scores; "
                      "use an arm record (FAMILY-k1-ENCODING.json)")
    raise Refused(f"{path}: not a speed_gate quality record, a K1 arm record or a loss file")


def ids_digest(ids: list[int]) -> str:
    """k1_quality's digest of target IDs: SHA-256 of the JSON list."""
    return hashlib.sha256(json.dumps(ids).encode()).hexdigest()


def pair(candidate: Losses, reference: Losses) -> list[tuple[tuple[int, int], int | None, float, float]]:
    """[(key, target, candidate nll, reference nll)] in key order, or Refused."""
    keys_c, keys_r = set(candidate.rows), set(reference.rows)
    # A K1 arm has no IDs and scores only its passage: the reference is read over its
    # positions, and its target IDs must hash to the arm's digest.
    if candidate.ids_sha256 is not None or reference.ids_sha256 is not None:
        if candidate.ids_sha256 is not None and reference.ids_sha256 is not None:
            raise Refused("both sides are K1 arm records, which keep no target IDs: "
                          "k1_quality.py compares two arms itself")
        scores_side, other = (candidate, reference) if candidate.ids_sha256 else (reference,
                                                                                 candidate)
        keys_other = set(other.rows)
        wanted = set(scores_side.rows)
        if not wanted <= keys_other:
            raise Refused(f"{other.source} lacks {len(wanted - keys_other)} of the "
                          f"{len(wanted)} targets of {scores_side.source}")
        ids = []
        for key in sorted(wanted):
            target = other.rows[key][0]
            if target is None:
                raise Refused(f"{other.source}: no target ID at {key} to check the digest")
            ids.append(target)
        if ids_digest(ids) != scores_side.ids_sha256:
            raise Refused(f"the target IDs of {other.source} at {scores_side.source}'s "
                          f"{len(ids)} positions do not hash to its recorded ids_sha256: "
                          "the sides scored different tokens")
        keys_c = keys_r = wanted
    elif keys_c != keys_r:
        only_c, only_r = sorted(keys_c - keys_r), sorted(keys_r - keys_c)
        raise Refused(f"the sides score different targets: {len(only_c)} only in the candidate "
                      f"(first {only_c[:3]}), {len(only_r)} only in the reference "
                      f"(first {only_r[:3]})")
    rows = []
    for key in sorted(keys_c):
        tc, nc = candidate.rows[key]
        tr, nr = reference.rows[key]
        if tc is not None and tr is not None and tc != tr:
            raise Refused(f"target IDs differ at document {key[0]} position {key[1]}: "
                          f"candidate {tc}, reference {tr} (a tokenizer or history difference)")
        rows.append((key, tc if tc is not None else tr, nc, nr))
    return rows


def aggregate(nlls: list[float]) -> tuple[float, float]:
    mean = math.fsum(nlls) / len(nlls)
    return mean, math.exp(mean)


def pct(ratio: float) -> float:
    return (ratio - 1.0) * 100.0


def compare(rows: list, tolerance: float | None) -> dict:
    cand = [r[2] for r in rows]
    ref = [r[3] for r in rows]
    diffs = [c - r for c, r in zip(cand, ref)]
    c_nll, c_ppl = aggregate(cand)
    r_nll, r_ppl = aggregate(ref)
    largest = max(range(len(rows)), key=lambda i: abs(diffs[i]))
    mean_diff = math.fsum(diffs) / len(diffs)
    sd = math.sqrt(math.fsum((d - mean_diff) ** 2 for d in diffs) / (len(diffs) - 1)) \
        if len(diffs) > 1 else math.nan
    documents = {}
    for document in sorted({r[0][0] for r in rows}):
        mine = [(c, r) for (k, _, c, r) in rows if k[0] == document]
        dc, dr = aggregate([c for c, _ in mine]), aggregate([r for _, r in mine])
        documents[document] = {"targets": len(mine), "candidate_nll": dc[0],
                               "reference_nll": dr[0], "ppl_delta_pct": pct(dc[1] / dr[1])}
    result = {
        "targets": len(rows),
        "candidate": {"mean_nll": c_nll, "perplexity": c_ppl},
        "reference": {"mean_nll": r_nll, "perplexity": r_ppl},
        "nll_delta": c_nll - r_nll,
        "nll_delta_se": sd / math.sqrt(len(diffs)) if len(diffs) > 1 else math.nan,
        "ppl_delta_pct": pct(c_ppl / r_ppl),
        "mean_abs_diff": math.fsum(abs(d) for d in diffs) / len(diffs),
        "max_abs_diff": abs(diffs[largest]),
        "max_abs_diff_at": {"document": rows[largest][0][0], "position": rows[largest][0][1],
                            "target": rows[largest][1]},
        "candidate_lower": sum(d < 0 for d in diffs),
        "candidate_higher": sum(d > 0 for d in diffs),
        "identical": sum(d == 0 for d in diffs),
        "documents": documents,
    }
    if tolerance is not None:
        result["agreement"] = {"tolerance": tolerance,
                               "fraction": sum(abs(d) <= tolerance for d in diffs) / len(diffs)}
    return result


def load_prereg(path_name: str) -> tuple[dict, str]:
    path = pathlib.Path(path_name)
    raw = read_bounded(path)
    try:
        prereg = json.loads(raw)
    except ValueError as e:
        raise Refused(f"{path}: not JSON ({e})") from e
    if not isinstance(prereg, dict) or prereg.get("format") != PREREG_FORMAT:
        raise Refused(f"{path}: not a {PREREG_FORMAT} pre-registration")
    bounds = prereg.get("bounds")
    if not isinstance(bounds, dict) or not bounds:
        raise Refused(f"{path}: no bounds")
    unknown = sorted(set(bounds) - set(BOUNDS))
    if unknown:
        raise Refused(f"{path}: unknown bounds {unknown}")
    for key, value in bounds.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or \
                not math.isfinite(value) or value < 0:
            raise Refused(f"{path}: bound {key} must be a finite nonnegative number")
    if "min_agreement" in bounds and "agreement_tolerance" not in bounds:
        raise Refused(f"{path}: min_agreement needs an agreement_tolerance")
    if set(bounds) <= {"agreement_tolerance"}:
        raise Refused(f"{path}: agreement_tolerance alone bounds nothing")
    return prereg, sha256(raw)


def check_bounds(result: dict, bounds: dict) -> list[dict]:
    """Each bound's outcome: {"bound", "limit", "observed", "ok"}."""
    checks = []

    def check(name, observed, ok):
        checks.append({"bound": name, "limit": bounds[name], "observed": observed, "ok": ok})

    n = result["targets"]
    if "expected_targets" in bounds:
        check("expected_targets", n, n == bounds["expected_targets"])
    if "min_targets" in bounds:
        check("min_targets", n, n >= bounds["min_targets"])
    if "max_ppl_increase_pct" in bounds:
        check("max_ppl_increase_pct", result["ppl_delta_pct"],
              result["ppl_delta_pct"] <= bounds["max_ppl_increase_pct"])
    if "max_abs_ppl_delta_pct" in bounds:
        check("max_abs_ppl_delta_pct", abs(result["ppl_delta_pct"]),
              abs(result["ppl_delta_pct"]) <= bounds["max_abs_ppl_delta_pct"])
    if "max_abs_nll_delta" in bounds:
        check("max_abs_nll_delta", abs(result["nll_delta"]),
              abs(result["nll_delta"]) <= bounds["max_abs_nll_delta"])
    if "max_abs_doc_ppl_delta_pct" in bounds:
        worst = max(abs(d["ppl_delta_pct"]) for d in result["documents"].values())
        check("max_abs_doc_ppl_delta_pct", worst, worst <= bounds["max_abs_doc_ppl_delta_pct"])
    if "max_mean_abs_diff" in bounds:
        check("max_mean_abs_diff", result["mean_abs_diff"],
              result["mean_abs_diff"] <= bounds["max_mean_abs_diff"])
    if "max_abs_diff" in bounds:
        check("max_abs_diff", result["max_abs_diff"],
              result["max_abs_diff"] <= bounds["max_abs_diff"])
    if "min_agreement" in bounds:
        fraction = result["agreement"]["fraction"]
        check("min_agreement", fraction, fraction >= bounds["min_agreement"])
    return checks


def run_pair(candidate_path: str, reference_path: str, prereg_path: str) -> dict:
    prereg, prereg_sha = load_prereg(prereg_path)
    candidate, candidate_sha = load(candidate_path)
    reference, reference_sha = load(reference_path)
    expected = prereg.get("reference_sha256")
    if expected is not None and expected != reference_sha:
        raise Refused(f"{reference_path}: SHA-256 {reference_sha} is not the pre-registered "
                      f"reference {expected}")
    bounds = prereg["bounds"]
    rows = pair(candidate, reference)
    result = compare(rows, bounds.get("agreement_tolerance"))
    checks = check_bounds(result, bounds)
    return {"format": REPORT_FORMAT, "name": prereg.get("name"),
            "prereg": {"file": prereg_path, "sha256": prereg_sha},
            "candidate": {"file": candidate_path, "kind": candidate.kind,
                          "sha256": candidate_sha},
            "reference": {"file": reference_path, "kind": reference.kind,
                          "sha256": reference_sha},
            "result": result, "bounds": checks, "pass": all(c["ok"] for c in checks)}


def print_report(report: dict) -> None:
    r = report["result"]
    print(f"{report.get('name') or 'quality pairing'}: {r['targets']} paired targets")
    print(f"  candidate {report['candidate']['file']} ({report['candidate']['kind']})")
    print(f"  reference {report['reference']['file']} ({report['reference']['kind']})")
    print(f"  mean NLL  candidate {r['candidate']['mean_nll']:.9f}  "
          f"reference {r['reference']['mean_nll']:.9f}  delta {r['nll_delta']:+.9f}")
    print(f"  PPL       candidate {r['candidate']['perplexity']:.6f}  "
          f"reference {r['reference']['perplexity']:.6f}  delta {r['ppl_delta_pct']:+.4f}%")
    print(f"  |diff|    mean {r['mean_abs_diff']:.6f}  max {r['max_abs_diff']:.6f} at document "
          f"{r['max_abs_diff_at']['document']} position {r['max_abs_diff_at']['position']}")
    if "agreement" in r:
        print(f"  agreement {100 * r['agreement']['fraction']:.3f}% within "
              f"{r['agreement']['tolerance']}")
    print(f"  lower {r['candidate_lower']}, higher {r['candidate_higher']}, "
          f"identical {r['identical']}")
    for document, d in r["documents"].items():
        print(f"  document {document}: {d['targets']} targets, PPL delta "
              f"{d['ppl_delta_pct']:+.4f}%")
    for c in report["bounds"]:
        print(f"  bound {c['bound']:28} {'ok  ' if c['ok'] else 'FAIL'} observed "
              f"{c['observed']:.6g}, limit {c['limit']:.6g}")
    print("PASS" if report["pass"] else "FAIL")


def cmd_pair(args: argparse.Namespace) -> int:
    try:
        report = run_pair(args.candidate, args.reference, args.prereg)
    except Refused as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2
    if args.report:
        pathlib.Path(args.report).write_text(json.dumps(report, indent=2, allow_nan=False)
                                             + "\n")
    if args.json:
        print(json.dumps(report, indent=2, allow_nan=False))
    else:
        print_report(report)
    return 0 if report["pass"] else 1


def cmd_extract(args: argparse.Namespace) -> int:
    try:
        losses, _ = load(args.candidate)
        if any(t is None for t, _ in losses.rows.values()):
            raise Refused(f"{args.candidate}: keeps no target IDs, so it is not a reference")
    except Refused as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2
    try:
        with open(args.out, "x") as sink:  # a retained file is never silently replaced
            for (document, position), (target, nll) in sorted(losses.rows.items()):
                sink.write(json.dumps({"document": document, "position": position,
                                       "target": target, "nll": nll}, allow_nan=False) + "\n")
    except FileExistsError:
        print(f"refused: {args.out} exists", file=sys.stderr)
        return 2
    print(f"{len(losses.rows)} losses to {args.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("pair")
    p.add_argument("--candidate", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--prereg", required=True)
    p.add_argument("--report", help="also write the full report as JSON")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_pair)
    p = sub.add_parser("extract", help="a native record's losses as a retainable losses.jsonl")
    p.add_argument("--candidate", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_extract)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
