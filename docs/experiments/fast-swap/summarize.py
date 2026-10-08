# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Reduce baseline.py's raw session and swap directories to the checked-in JSON.

Usage: summarize.py OUTPUT.json LABEL=RAW_DIRECTORY...
"""
import json
from pathlib import Path
import statistics
import sys

GIB = 1 << 30


def rounded(value, digits=2):
    return None if value is None else round(value, digits)


def session(directory):
    raw = json.loads((directory / "session.json").read_text())
    out = {"load_ready_s": rounded(raw.get("load", {}).get("ready_s")),
           "load_first_token_s": rounded(raw.get("load", {}).get("first_token_s")),
           "peak_memavailable_drop_gib": rounded(raw.get("peak_drop_gib")),
           "baseline_memavailable_gib": rounded(raw["baseline_available_bytes"] / GIB)}
    measure = raw.get("measure")
    if measure:
        out["prefill"] = {
            target: {"prompt_tokens": statistics.median(r["prompt_tokens"] for r in row["runs"]),
                     "n": len(row["runs"]),
                     "client_tok_s": rounded(row["median_client_tok_s"], 1),
                     "client_tok_s_range": [rounded(min(r["client_tok_s"] for r in row["runs"]), 1),
                                            rounded(max(r["client_tok_s"] for r in row["runs"]), 1)],
                     "engine_tok_s": rounded(row["median_engine_tok_s"], 1)}
            for target, row in measure["prefill"].items()}
        out["decode"] = {}
        for variant, prompts in measure["decode"].items():
            out["decode"][variant] = {}
            for name, row in prompts.items():
                entry = {"n": len(row["runs"]),
                         "completion_tokens": [r["completion_tokens"] for r in row["runs"]],
                         "client_tok_s": rounded(row["median_client_tok_s"]),
                         "client_tok_s_range": [rounded(min(r["client_tok_s"] for r in row["runs"])),
                                                rounded(max(r["client_tok_s"] for r in row["runs"]))],
                         "engine_tok_s": rounded(row["median_engine_tok_s"])}
                drafts = [r for r in row["runs"] if r.get("draft_n")]
                if drafts:
                    entry["draft_acceptance"] = rounded(
                        sum(r["draft_n_accepted"] for r in drafts) / sum(r["draft_n"] for r in drafts), 3)
                stats = [r["engine_stats"] for r in row["runs"] if (r.get("engine_stats") or {}).get("drafted")]
                if stats:  # TensorFold's per-reply block: drafted and accepted tokens, rounds
                    entry["draft_acceptance"] = rounded(
                        sum(s["accepted"] for s in stats) / sum(s["drafted"] for s in stats), 3)
                    entry["tokens_per_round"] = rounded(statistics.median(
                        r["completion_tokens"] / r["engine_stats"]["rounds"] for r in row["runs"]
                        if (r.get("engine_stats") or {}).get("rounds")))
                metrics = row.get("spec_metrics_delta") or {}
                drafted = sum(v for k, v in metrics.items() if "num_draft_tokens_total" in k)
                accepted = sum(v for k, v in metrics.items()
                               if "num_accepted_tokens_total" in k and "per_pos" not in k)
                if drafted:
                    entry["draft_acceptance"] = rounded(accepted / drafted, 3)
                out["decode"][variant][name] = entry
    return out


def swap(directory):
    raw = json.loads((directory / "swap.json").read_text())
    keep = ("a_context_tokens", "a_save_s", "a_stop_s", "b_start_to_ready_s", "a_to_b_first_token_s",
            "b_save_s", "b_stop_s", "a_start_to_ready_s", "a_restore_s", "b_to_a_first_token_s")
    cycles = []
    for row in raw["cycles"]:
        entry = {k: rounded(row[k]) if isinstance(row[k], float) else row[k] for k in keep}
        timings = row.get("b_to_a_timings") or {}
        entry["b_to_a_prompt_tokens_processed"] = timings.get("prompt_n")
        entry["b_to_a_cache_tokens"] = timings.get("cache_n")
        entry["a_saved_bytes"] = (row.get("a_save") or {}).get("n_written")
        entry["a_restored_tokens"] = (row.get("a_restore") or {}).get("n_restored")
        entry["min_memavailable_gib"] = rounded(
            min(row["a_to_b_min_available_bytes"], row["b_to_a_min_available_bytes"]) / GIB)
        cycles.append(entry)
    return {"cycles": cycles}


def image(directory):
    raw = json.loads((directory / "image.json").read_text())
    first, plain = raw["runs"][0], raw["runs"][1:]
    return {"spec": raw["spec"], "torch": raw["torch"],
            "load_process_start_to_loaded_s": rounded(raw["load"]["process_start_to_loaded_s"]),
            "load_to_cuda_s": rounded(raw["load"]["to_cuda_s"]),
            "process_start_to_first_step_output_s": rounded(first["process_start_to_first_step_output_s"]),
            "first_request_first_step_output_s": rounded(first["first_step_output_s"]),
            "step_s_median": rounded(first["step_s_median"], 3),
            "step_s_range": [rounded(v, 3) for v in first["step_s_min_max"]],
            "instrumented_generation_s": rounded(first["generation_s"]),
            "plain_generation_s": [rounded(r["generation_s"]) for r in plain],
            "pixels_identical_across_runs": raw["pixels_identical"],
            "pixels_sha256": first["pixels_sha256"],
            "cuda_max_allocated_gib": rounded(raw["cuda_max_allocated_bytes"] / GIB),
            "peak_memavailable_drop_gib": rounded(raw["peak_drop_bytes"] / GIB)}


def main():
    output, *pairs = sys.argv[1:]
    result = {}
    for pair in pairs:
        label, _, path = pair.partition("=")
        directory = Path(path)
        if (directory / "swap.json").exists():
            result[label] = swap(directory)
        elif (directory / "image.json").exists():
            result[label] = image(directory)
        else:
            result[label] = session(directory)
    Path(output).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
