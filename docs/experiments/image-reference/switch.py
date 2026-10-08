# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Short-context text/image/text comparison using the existing reference router."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reference-aba"))
from experiment import Router, Telemetry, sha, check, dump, system_counters


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--image-container", required=True)
    parser.add_argument("--image-root", default="/experiment")
    parser.add_argument("--port", type=int, default=18081)
    parser.add_argument("--device", default="nvme0n1")
    parser.add_argument("--mode", action="append", choices=("resident", "restore", "recompute"))
    parser.add_argument("--reference-pixels")
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    pins = json.loads((Path(__file__).resolve().parents[1] / "reference-aba" / "artifacts.json").read_text())
    model = args.models / pins["models"]["A"]["filename"]
    check(sha(model) == pins["models"]["A"]["sha256"], "Text model integrity mismatch")
    summary = []
    image_digest = args.reference_pixels
    for mode in args.mode or ("resident", "restore", "recompute"):
        directory = args.output / mode
        directory.mkdir(mode=0o700)
        router = Router(args, pins, directory)
        try:
            router.start()
            router.load("A")
            prompt = "".join(f"Record {i}: the storage shelf contains blue boxes.\n" for i in range(48))
            prompt += "\nQuestion: What color are the boxes?\nAnswer:"
            tokens = router.tokens("A", prompt, add_special=True)
            first = router.complete("A", tokens, predict=32, cache_prompt=False)
            saved = router.save("A")
            suffix = router.tokens("A", "\nQuestion: Name the color again.\nAnswer:")
            continuation = tokens + first["tokens"] + suffix
            check(len(continuation) + 32 < 1024, "Short-context coverage exceeded")
            expected = router.complete("A", continuation, predict=32)
            control = expected
            if mode == "recompute":
                # Match the fresh execution context used after the image,
                # rather than re-prefilling in a previously occupied slot.
                router.unload("A")
                router.load("A")
                control = router.complete("A", continuation, predict=32, cache_prompt=False)
            dump(directory / "controls.json", {"resident": expected, "matched_configuration": control})
            # Return to exactly the saved state after obtaining the resident control.
            router.restore("A", saved)
            with Telemetry(directory) as telemetry:
                counters_before = system_counters(args.device)
                outward = time.monotonic()
                if mode == "restore":
                    saved = router.save("A")
                if mode != "resident":
                    router.unload("A")
                image_output = f"{args.image_root}/{args.output.name}-{mode}"
                command = router.docker + ["exec", args.image_container,
                    "timeout", "--signal=TERM", "--kill-after=30", "1800", "python3",
                    f"{args.image_root}/measure.py", f"{args.image_root}/model", image_output,
                    "--size", "512", "--steps", "4", "--plain"]
                with (directory / "image.log").open("w") as stream:
                    try:
                        subprocess.run(command, check=True, stdout=stream,
                                       stderr=subprocess.STDOUT, timeout=1860)
                    except BaseException:
                        # This must be a dedicated experiment container. Killing
                        # docker exec alone does not terminate its GPU process.
                        router.cmd("stop", "--time", "30", args.image_container,
                                   stdout=subprocess.DEVNULL)
                        raise
                outward_seconds = time.monotonic() - outward
                raw = subprocess.run(router.docker + ["exec", args.image_container,
                    "cat", image_output + "/result.json"], check=True, capture_output=True, text=True)
                image_report = json.loads(raw.stdout)
                digest = image_report["output"]["pixels_sha256"]
                if image_digest is None:
                    image_digest = digest
                check(digest == image_digest, "Image differs between switch modes")
                returning = time.monotonic_ns()
                if mode != "resident":
                    router.load("A")
                if mode == "restore":
                    router.restore("A", saved)
                actual = router.complete("A", continuation, predict=32, cache_prompt=mode != "recompute")
                dump(directory / "actual.json", actual)
                check(actual["tokens"] == control["tokens"], "Text differs from matched-configuration control")
                prompt_n = actual["final"]["timings"]["prompt_n"]
                if mode != "recompute":
                    pending = len(continuation) - saved["n_saved"]
                    check(pending <= prompt_n <= pending + 1, "Continuation re-prefilled beyond suffix/tail")
                counters_after = system_counters(args.device)
            row = {"mode": mode, "outward_seconds_including_image_process": outward_seconds,
                   "return_first_token_seconds": (actual["first_token_ns"] - returning) / 1e9,
                   "prompt_n": prompt_n, "continuation_tokens": len(continuation),
                   "saved": saved, "text_matches": True, "image_matches": True,
                   "text_matches_resident": actual["tokens"] == expected["tokens"],
                   "control_matches_resident": control["tokens"] == expected["tokens"],
                   "image": image_report, "memory": telemetry.summary(),
                   "system_counter_deltas": {k: counters_after[k] - counters_before[k] for k in counters_before}}
            summary.append(row)
            dump(args.output / "results.json", summary)
            print(json.dumps({k: v for k, v in row.items() if k not in ("image", "saved")}), flush=True)
        finally:
            router.stop()


if __name__ == "__main__":
    os.umask(0o077)
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(128 + signum))
    main()
