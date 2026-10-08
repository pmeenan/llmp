#!/bin/sh
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
#
# Runs one FP16 arm of llmp_fp16_exec (benchmarks/fp16_exec.cc) on a Spark
# for the FP16 Tier E gate, in the gate's order (backend-proof.md):
#   run_native.sh plan   BUILD ARTIFACT HARNESS WORK ARM
#   run_native.sh logits BUILD ARTIFACT HARNESS WORK ARM
#   run_native.sh peak   BUILD ARTIFACT HARNESS WORK ARM
# BUILD is a deployed cross build, ARTIFACT the installed FP16 fixture,
# HARNESS a copy of docs/experiments (backend-proof-p0 and -p2), WORK the
# working directory holding control-tokens.txt (the bridge's tokens.txt)
# and heldout-ids.i64le, and ARM one of control-fused, control-unfused,
# heldout-fused, heldout-unfused.
#
# plan:   the recorded run under nsys (--trace=cuda) with cuBLAS's and
#         cuBLASLt's logs, the binary's SASS hashes (cuobjdump), then
#         plan_compare.py convert and compare against fp16-plan.json. Exits
#         with plan_compare's status; only 0 lets the logits be compared.
# logits: the same run's logits SHA-256 against the arm's recorded one.
# peak:   the coarse memory check (D-085): a separate run of the census
#         harness (no nsys, no logs, no recording) under peak_memory.sh,
#         which prints its peak memory for comparison with the bridge's.
#         (The census rules this stage ran before D-085 are in Git history.)
set -u
STAGE=$1 BUILD=$2 ARTIFACT=$3 HARNESS=$4 WORK=$5 ARM=$6
TRAJECTORY=${ARM%-*}
case ${ARM#*-} in fused) FUSION=on ;; unfused) FUSION=off ;; *) echo "unknown arm $ARM" >&2; exit 2 ;; esac
case $TRAJECTORY in
  control) TOKENS=$WORK/control-tokens.txt ;;
  heldout) TOKENS=$WORK/heldout-ids.i64le ;;
  *) echo "unknown arm $ARM" >&2; exit 2 ;;
esac
BIN=$BUILD/benchmarks/llmp_fp16_exec
CUOBJDUMP=${CUOBJDUMP:-/usr/local/cuda/bin/cuobjdump}
P0=$HARNESS/backend-proof-p0
P2=$HARNESS/backend-proof-p2
OUT=$WORK/$ARM
if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then
  echo "another process is using the GPU" >&2
  exit 3
fi
export CUDA_DISABLE_PTX_JIT=1
case $STAGE in
  plan)
    rm -rf "$OUT" && mkdir -p "$OUT"
    env CUBLAS_LOGINFO_DBG=1 CUBLAS_LOGDEST_DBG="$OUT/cublas.log" \
        CUBLASLT_LOG_LEVEL=5 CUBLASLT_LOG_FILE="$OUT/cublaslt.log" \
      nsys profile --trace=cuda --sample=none --cpuctxsw=none --export=sqlite -o "$OUT/trace" \
      "$BIN" --artifact "$ARTIFACT" --trajectory "$TRAJECTORY" --tokens "$TOKENS" \
             --fusion $FUSION --out "$OUT/run" --record > "$OUT/run.log" 2>&1 || {
      echo "the run failed:" >&2; tail -5 "$OUT/run.log" >&2; exit 1; }
    "$CUOBJDUMP" -sass "$BIN" | python3 -B "$P0/fp16_plan.py" sass-hash --label executable > "$OUT/sass.jsonl"
    python3 -B "$P2/plan_compare.py" convert "$OUT/run/recording.jsonl" \
      --cublas-log "$OUT/cublas.log" --cublaslt-log "$OUT/cublaslt.log" \
      --sass "$OUT/sass.jsonl" --nsys "$OUT/trace.sqlite" --out "$OUT/native.json" || exit 1
    python3 -B "$P2/plan_compare.py" compare --reference "$P0/fp16-plan.json" --arm "$ARM" \
      --native "$OUT/native.json"
    ;;
  logits)
    python3 -B - "$P0/fp16-plan.json" "$ARM" "$OUT/run/summary.json" "$OUT/run/logits.f32le" <<'EOF'
import hashlib, json, sys
record, arm, summary, logits = sys.argv[1:]
want = next(a for a in json.load(open(record))["arms"] if a["arm"] == arm)["expected_logits_sha256"]
got = hashlib.sha256(open(logits, "rb").read()).hexdigest()
s = json.load(open(summary))
assert s["logits_sha256"] == got, "the summary's hash is not the file's"
print(f"{arm}: native {got}, bridge {want}: {'BIT-IDENTICAL' if got == want else 'DIFFERENT'}; "
      f"repeat differences {s['repeat_bit_differences']}")
sys.exit(0 if got == want and s["repeat_bit_differences"] == 0 else 1)
EOF
    ;;
  peak)
    # The coarse memory check (D-085): the census harness (its readings
    # settle 50 ms) under peak_memory.sh. Run the bridge's fp16_census for
    # the same arm next to it, and compare the two peaks (backend-proof.md).
    rm -rf "$OUT-peak" && mkdir -p "$OUT-peak"
    CENSUS_SETTLE_MS=50 sh "$P2/peak_memory.sh" "native $ARM" -- "$BIN" --artifact "$ARTIFACT" \
      --trajectory "$TRAJECTORY" --tokens "$TOKENS" --fusion $FUSION --out "$OUT-peak" --census
    ;;
  *) echo "unknown stage $STAGE" >&2; exit 2 ;;
esac
