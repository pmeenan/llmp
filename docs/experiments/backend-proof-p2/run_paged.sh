#!/bin/sh
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
#
# Runs one FP16 arm of llmp_fp16_paged (benchmarks/fp16_paged.cc) on a
# Spark for oracle rungs 4 and 5 (backend-proof.md): the fixture paged into
# device VMM through the landing zone (D-081), with RESTORES restored
# evaluations (weights evicted, their backing released and paged back in at
# the profile's restore point, relocated every second time).
#   run_paged.sh plan    BUILD ARTIFACT HARNESS WORK ARM [RESTORES]
#   run_paged.sh logits  BUILD ARTIFACT HARNESS WORK ARM
#   run_paged.sh threads BUILD ARTIFACT HARNESS WORK ARM [RESTORES]
#   run_paged.sh loads   BUILD ARTIFACT HARNESS WORK [ROUNDS]
# BUILD, ARTIFACT, HARNESS, WORK and ARM are run_native.sh's. RESTORES
# defaults to 2, ROUNDS to 3.
#
# plan:    the run with every lane driven from the recording thread
#          (--lanes inline --record) under nsys (--trace=cuda), with cuBLAS's
#          and cuBLASLt's logs and the binary's SASS hashes, converted and
#          compared with fp16-plan.json by plan_compare.py. Exits with its
#          status; only 0 lets the logits be compared. Every evaluation,
#          restored ones included, must match the record's chunks; the
#          page-ins' copies fall between chunks, where only copies and
#          memsets may run.
# logits:  that run's first evaluation against the arm's recorded SHA-256,
#          and every later evaluation (the repeat and the restored ones)
#          bit-identical to it.
# threads: a run with each lane on its own thread, as a program wires them
#          (no recording): the same checks, and the page-ins' times.
# loads:   page-in throughput (BP-P6, reported): the device weights loaded
#          and evicted 5 times per process, for four variants interleaved
#          over ROUNDS rounds: through the zone with managed backing (the
#          runtime's path), through the zone with backing mapped once, and
#          read in place into host VMM (D-034's path) with managed or
#          premapped backing. Each process starts once no other process has
#          held the GPU for 10 s; the count of others at its start is
#          printed. Needs no ARM.
set -u
STAGE=$1 BUILD=$2 ARTIFACT=$3 HARNESS=$4 WORK=$5
if [ "$STAGE" = loads ]; then
  ROUNDS=${6:-3}
  export CUDA_DISABLE_PTX_JIT=1
  idle() {
    quiet=0
    while [ $quiet -lt 10 ]; do
      if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then quiet=0; else quiet=$((quiet + 2)); fi
      sleep 2
    done
  }
  mkdir -p "$WORK/loads"
  round=1
  while [ $round -le "$ROUNDS" ]; do
    for variant in "device managed" "device premapped" "host managed" "host premapped"; do
      set -- $variant
      idle
      others=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
      OUT=$WORK/loads/$1-$2-$round
      "$BUILD/benchmarks/llmp_fp16_paged" --artifact "$ARTIFACT" --trajectory control \
        --tokens "$WORK/control-tokens.txt" --fusion on --out "$OUT" --load-only 5 \
        --weights "$1" --backing "$2" > "$OUT.log" 2>&1 || { echo "the run failed" >&2; exit 1; }
      python3 -B -c "
import json, sys
d = json.load(open(sys.argv[1]))
print(sys.argv[2], ' '.join(f\"{d['read_bytes'] / l['seconds'] / 1e9:.2f}\" for l in d['loads']), 'GB/s;',
      sys.argv[3], 'other GPU processes at start')" "$OUT/loads.json" "round $round $1 $2:" "$others"
    done
    round=$((round + 1))
  done
  exit 0
fi
ARM=$6 RESTORES=${7:-2}
TRAJECTORY=${ARM%-*}
case ${ARM#*-} in fused) FUSION=on ;; unfused) FUSION=off ;; *) echo "unknown arm $ARM" >&2; exit 2 ;; esac
case $TRAJECTORY in
  control) TOKENS=$WORK/control-tokens.txt ;;
  heldout) TOKENS=$WORK/heldout-ids.i64le ;;
  *) echo "unknown arm $ARM" >&2; exit 2 ;;
esac
BIN=$BUILD/benchmarks/llmp_fp16_paged
CUOBJDUMP=${CUOBJDUMP:-/usr/local/cuda/bin/cuobjdump}
P0=$HARNESS/backend-proof-p0
P2=$HARNESS/backend-proof-p2
OUT=$WORK/paged-$ARM
export CUDA_DISABLE_PTX_JIT=1

check() {  # SUMMARY LOGITS: the first evaluation against the record, the rest against it
  python3 -B - "$P0/fp16-plan.json" "$ARM" "$1" "$2" <<'EOF'
import hashlib, json, sys
record, arm, summary, logits = sys.argv[1:]
want = next(a for a in json.load(open(record))["arms"] if a["arm"] == arm)["expected_logits_sha256"]
got = hashlib.sha256(open(logits, "rb").read()).hexdigest()
s = json.load(open(summary))
assert s["logits_sha256"] == got, "the summary's hash is not the file's"
diffs = s["bit_differences_from_first"]
cover = s["coverage"]
print(f"{arm} ({s['lanes']}, {s['evaluations']} evaluations, {s['restores']} restored, relocate "
      f"{s['relocate']}): native {got}, bridge {want}: {'BIT-IDENTICAL' if got == want else 'DIFFERENT'}; "
      f"later evaluations' bit differences {diffs}; coverage {cover['tensors']} tensors, "
      f"{cover['violations']} outside")
sys.exit(0 if got == want and not any(diffs) and cover["violations"] == 0 else 1)
EOF
}

case $STAGE in
  plan)
    # The recording and the nsys trace are this process's alone, so other
    # GPU work on the host does not change them (it may change timings).
    rm -rf "$OUT" && mkdir -p "$OUT"
    env CUBLAS_LOGINFO_DBG=1 CUBLAS_LOGDEST_DBG="$OUT/cublas.log" \
        CUBLASLT_LOG_LEVEL=5 CUBLASLT_LOG_FILE="$OUT/cublaslt.log" \
      nsys profile --trace=cuda --sample=none --cpuctxsw=none --export=sqlite -o "$OUT/trace" \
      "$BIN" --artifact "$ARTIFACT" --trajectory "$TRAJECTORY" --tokens "$TOKENS" \
             --fusion $FUSION --out "$OUT/run" --restores "$RESTORES" --relocate \
             --lanes inline --record > "$OUT/run.log" 2>&1 || {
      echo "the run failed:" >&2; tail -5 "$OUT/run.log" >&2; exit 1; }
    "$CUOBJDUMP" -sass "$BIN" | python3 -B "$P0/fp16_plan.py" sass-hash --label executable > "$OUT/sass.jsonl"
    python3 -B "$P2/plan_compare.py" convert "$OUT/run/recording.jsonl" \
      --cublas-log "$OUT/cublas.log" --cublaslt-log "$OUT/cublaslt.log" \
      --sass "$OUT/sass.jsonl" --nsys "$OUT/trace.sqlite" --out "$OUT/native.json" || exit 1
    python3 -B "$P2/plan_compare.py" compare --reference "$P0/fp16-plan.json" --arm "$ARM" \
      --native "$OUT/native.json"
    ;;
  logits)
    check "$OUT/run/summary.json" "$OUT/run/logits.f32le"
    ;;
  threads)
    rm -rf "$OUT-threads" && mkdir -p "$OUT-threads"
    "$BIN" --artifact "$ARTIFACT" --trajectory "$TRAJECTORY" --tokens "$TOKENS" --fusion $FUSION \
           --out "$OUT-threads" --restores "$RESTORES" --relocate --lanes threads \
           > "$OUT-threads.log" 2>&1 || {
      echo "the run failed:" >&2; tail -5 "$OUT-threads.log" >&2; exit 1; }
    check "$OUT-threads/summary.json" "$OUT-threads/logits.f32le"
    ;;
  *) echo "unknown stage $STAGE" >&2; exit 2 ;;
esac
