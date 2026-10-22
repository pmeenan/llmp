#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Run inside the pinned image, only after reviewed input freeze and preregistration."""
import argparse
import json
import os
from pathlib import Path

from contract import IMAGE, PHASES, RECIPE, encoded, frozen, metadata, select_cases, sha
from source_adapter import SourceAdapter


def helpers_sha():
    root = Path(__file__).parent
    files = sorted(root.glob("*.py")) + [root / "tensorfold-source-hashes.txt"]
    return sha(encoded({p.name: sha(p.read_bytes()) for p in files}))


def validation_identity(protocol_sha, prereg_commit):
    return {"protocol_sha256": protocol_sha, "prereg_commit": prereg_commit,
            "image": IMAGE, "helpers_sha256": helpers_sha()}


def require_validation(path, identity):
    if path is None:
        raise ValueError("source adapter validation must pass before quality collection")
    result = json.loads(path.read_bytes())
    if result.get("identity") != identity or result.get("passed") is not True:
        raise ValueError("adapter validation is absent, stale or failed")
    checks = result.get("checks", [])
    if len(checks) != 2 or [x.get("prompt_tokens") for x in checks] != [513, 1024]:
        raise ValueError("adapter controls are incomplete")
    for check in checks:
        if (check.get("head_1_vs_16_exact") is not True or type(check.get("normal_first_token")) is not int
                or check.get("normal_first_token") != check.get("adapter_first_token")):
            raise ValueError("adapter control did not preserve source arithmetic")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--prereg-commit", required=True)
    parser.add_argument("--phase", choices=["validate", "short", "depth", "behavior"], required=True)
    parser.add_argument("--case", help="one frozen behavior case per bounded job")
    parser.add_argument("--repeat", type=int, choices=[1, 2], default=1)
    parser.add_argument("--validation", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    spec, all_ids, behavior, protocol_sha = frozen(args.protocol, args.inputs, args.prereg_commit)
    cases = select_cases(args.phase, args.case, behavior)
    metadata(args.checkpoint)
    identity = validation_identity(protocol_sha, args.prereg_commit)
    if args.phase != "validate":
        require_validation(args.validation, identity)
    os.umask(0o077)
    args.out.mkdir(parents=True, exist_ok=False)
    receipt = {"identity": identity, "phase": args.phase, "repeat": args.repeat,
               "ids_sha256": spec["ids_sha256"], "behavior_sha256": spec["behavior_sha256"],
               "recipe": RECIPE, "completed": False}
    (args.out / "receipt.json").write_bytes(encoded(receipt))
    adapter = None
    validation = None
    try:
        adapter = SourceAdapter(args.checkpoint)
        receipt["source"] = adapter.identity
        receipt["capacity"] = adapter.engine.capacity_plan
        if args.phase == "validate":
            checks = adapter.validate(all_ids)
            validation = {"identity": identity, "passed": True, "checks": checks}
        elif args.phase == "behavior":
            from tokenizers import Tokenizer
            tokenizer = Tokenizer.from_file(str(args.checkpoint / "tokenizer.json"))
            records = adapter.behavior(cases, tokenizer)
            payload = encoded(records)
            (args.out / "responses.json").write_bytes(payload)
            receipt["responses_sha256"] = sha(payload)
        else:
            count, first, expected, document = PHASES[args.phase]
            records, details = adapter.score(all_ids[:count], first, document)
            if len(records) != expected:
                raise ValueError("collected target count differs from preregistration")
            payload = b"".join(encoded(row) + b"\n" for row in records)
            (args.out / "losses.jsonl").write_bytes(payload)
            receipt.update(details)
            receipt.update(targets=len(records), losses_sha256=sha(payload))
        receipt["completed"] = True
    except BaseException as error:
        receipt["failure"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        try:
            if adapter is not None:
                adapter.close()
        except BaseException as error:
            receipt['completed'] = False
            receipt['close_failure'] = f'{type(error).__name__}: {error}'
            raise
        finally:
            (args.out / "receipt.json").write_bytes(encoded(receipt))
    if validation is not None:
        (args.out / "validation.json").write_bytes(encoded(validation))


if __name__ == "__main__":
    main()
