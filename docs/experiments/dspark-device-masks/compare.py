#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0
"""Check four DSpark block-mask payload directories in off/on/on/off order.

This reads completed benchmark output; successful supervised process retirement,
source/library identities and kernel/boot guards remain separate requirements.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct


def require(condition, message):
    if not condition:
        raise SystemExit(message)


def read_arm(path, device):
    data = json.loads((path / "spec.json").read_text())
    require(not data["problems"] and data["weights_loaded"] and not data["graphs"]["refused"],
            f"{path}: incomplete model work")
    results = data["results"]
    wave = next(row for row in results if row["check"] == "wave")
    work = next(row for row in results if row["check"] == "draft-mask-work")
    payloads = sorted((row for row in results if row["check"] == "wave-payload"),
                      key=lambda row: (row["form"], row["owner"]))
    require([(row["form"], row["owner"]) for row in payloads] ==
            [("solo", 0), ("solo", 1), ("wave", 0), ("wave", 1)],
            f"{path}: missing owner payload")
    require(wave["slots"] == 2 and wave["speculative"] and wave["share"] == 4 and
            wave["widths"].get("2", 0) > 0 and wave["exact_mismatches"] == 0 and
            wave["discard_stale_bytes"] == 0 and wave["discard_rerun_exact"] and
            wave["left_unchanged"], f"{path}: incomplete joined/rollback/peer proof")
    require(work["device_masks"] == device and
            (work["bound_mask_plan_nodes"] > 0) == device and
            (work["constructed_host_mask_bytes"] == 0) == device,
            f"{path}: requested mask policy was not selected")
    for key in ("draft_graphs", "target_graphs", "wave_graphs"):
        require(work[key][1] > 0 and work[key][2] > 0,
                f"{path}: missing capture/replay for {key}")
    for row in payloads:
        require(row["vocab"] == 129280 and row["heads"] == 1 + sum(row["verify_rows"]) and
                len(row["tokens"]) == 1 + sum(row["kept"]) and
                len(row["verify_rows"]) == len(row["kept"]) == len(row["drafts"]),
                f"{path}: incomplete head/history geometry")
        require(len(row["prompt_ids"]) == 254 and
                all(0 <= value < 129280 for value in row["prompt_ids"] + row["tokens"]) and
                all(len(values) == 3 and all(0 <= value < 129280 for value in values)
                    for values in row["drafts"]) and
                all(1 <= kept <= rows <= 4
                    for rows, kept in zip(row["verify_rows"], row["kept"])),
                f"{path}: invalid IDs/draft history")
        require(row["form"] != "solo" or len(row["tokens"]) == 32,
                f"{path}: wrong solo output count")
        require(all(len(row[key]) == 64 for key in ("target_sha256", "draft_state_sha256")),
                f"{path}: missing initialized-state hash")
        name = row["head_file"]
        require(Path(name).name == name, f"{path}: head path is not local")
        raw = (path / name).read_bytes()
        require(len(raw) == row["heads"] * row["vocab"] * 4 and
                hashlib.sha256(raw).hexdigest() == row["head_sha256"] and
                all(math.isfinite(value) for value, in struct.iter_unpack("<f", raw)),
                f"{path}: incomplete/nonfinite/corrupt full heads")
    budget = {key: data[key] for key in ("activations_bytes", "pool_bytes", "host_input_bytes")}
    budget.update({key: work[key] for key in ("observation_charge_bytes", "plan_floor_bytes")})
    identity = {key: wave[key] for key in ("solo_tokens", "wave_tokens", "waves", "widths")}
    identity.update({key: work[key] for key in ("draft_graphs", "target_graphs", "wave_graphs")})
    metrics = {"solo_seconds": wave["solo_seconds"], "wave_seconds": wave["wave_seconds"],
               "joined_mean_ms": wave["width_ms"]["2"]}
    require(all(math.isfinite(value) and value > 0 for value in metrics.values()),
            f"{path}: invalid timing")
    return payloads, identity, budget, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arms", nargs=4, type=Path, metavar="ARM")
    args = parser.parse_args()
    arms = [read_arm(path, device) for path, device in
            zip(args.arms, (False, True, True, False))]
    require(all(arm[:2] == arms[0][:2] for arm in arms),
            "Heads/proposals/kept/history/state/work differ across mask policies")
    require(arms[0][2] == arms[3][2] and arms[1][2] == arms[2][2],
            "Repeated same-policy setup budgets differ")
    report = {"complete_finite_payloads": 16, "heads_per_arm":
              [row["heads"] for row in arms[0][0]], "metrics": {}}
    for key in arms[0][3]:
        off = [arms[0][3][key], arms[3][3][key]]
        on = [arms[1][3][key], arms[2][3][key]]
        report["metrics"][key] = {
            "off": off, "on": on,
            "time_change_pct": 100 * (statistics.mean(on) / statistics.mean(off) - 1),
            "off_bookend_change_pct": 100 * (off[1] / off[0] - 1),
        }
    report["setup_delta_device_minus_host"] = {
        key: arms[1][2][key] - arms[0][2][key] for key in arms[0][2]
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
