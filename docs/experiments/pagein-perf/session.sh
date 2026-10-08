#!/bin/sh
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
#
# The reported page-in session (README.md), on a Spark:
#   session.sh AFTER BEFORE PROBE FILE ARTIFACT TOKENS OUT ROUNDS [MAXLOAD]
# AFTER and BEFORE are spark-native build directories (this change, and the
# commit before it with the harness and the scheduler's page-in observer
# added); PROBE is dmabuf_probe built with probe_offset.patch; FILE a
# pattern file of at least 9 GiB written by `dmabuf_probe create`; ARTIFACT
# the FP16 artifact's installed directory and TOKENS its control token file
# (backend-proof-p2). Each process starts once no compute process has been
# on the GPU and the 1-minute load has stayed below MAXLOAD (default 1.0)
# for 10 s; the conditions at its start go on its result lines in
# OUT/results.txt. Reads start 4 KiB into FILE, as an artifact's chunks do.
set -u
AFTER=$1 BEFORE=$2 PROBE=$3 FILE=$4 ART=$5 TOK=$6 OUT=$7 ROUNDS=$8 MAXLOAD=${9:-1.0}
mkdir -p "$OUT"
{ hostname; uname -r; nvidia-smi --query-gpu=name,driver_version,persistence_mode --format=csv,noheader
  sha256sum "$AFTER/benchmarks/llmp_pagein_bench" "$BEFORE/benchmarks/llmp_pagein_bench" "$PROBE" \
    "$AFTER/benchmarks/llmp_fp16_paged" "$BEFORE/benchmarks/llmp_fp16_paged"
  lsblk -d -o NAME,MODEL | grep nvme; } > "$OUT/manifest.txt"
idle() {
  quiet=0
  while [ $quiet -lt 10 ]; do
    apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
    load=$(cut -d' ' -f1 /proc/loadavg)
    if [ "$apps" = 0 ] && awk -v l="$load" -v m="$MAXLOAD" 'BEGIN{exit !(l < m)}'; then
      quiet=$((quiet + 2))
    else
      quiet=0
    fi
    sleep 2
  done
}
cond() {
  echo "$(date -u +%T) load=$(cut -d' ' -f1 /proc/loadavg)" \
    "apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" \
    "gpu=$(nvidia-smi --query-gpu=pstate,clocks.sm,temperature.gpu --format=csv,noheader | tr -d ' ')"
}
bench() {  # TAG BUILD ARGS...: 8 GiB loaded 5 times
  tag=$1 build=$2
  shift 2
  idle
  c=$(cond)
  "$build/benchmarks/llmp_pagein_bench" --file "$FILE" --gib 8 --offset 4096 --loads 5 "$@" \
    > "$OUT/tmp.txt" 2>&1 || { echo "FAIL $tag" >> "$OUT/results.txt"; cat "$OUT/tmp.txt" >> "$OUT/errors.txt"; return; }
  grep '^load,' "$OUT/tmp.txt" | sed "s|^|$tag,r$round,|;s|\$|,$c|" >> "$OUT/results.txt"
}
probe() {  # TAG MODE DEPTH: 8 GiB restored 4 times
  idle
  c=$(cond)
  PROBE_OFFSET=4096 "$PROBE" restore "$FILE" "$2" "$3" 4 8 > "$OUT/tmp.txt" 2>&1 ||
    { echo "FAIL $1" >> "$OUT/results.txt"; return; }
  grep '^restore,' "$OUT/tmp.txt" | grep -v negative | sed "s|^|$1,r$round,|;s|\$|,$c|" >> "$OUT/results.txt"
}
fp16() {  # TAG BUILD WEIGHTS BACKING: the FP16 artifact's device weights loaded 6 times
  idle
  c=$(cond)
  CUDA_DISABLE_PTX_JIT=1 "$2/benchmarks/llmp_fp16_paged" --artifact "$ART" --trajectory control \
    --tokens "$TOK" --fusion on --out "$OUT/fp16" --load-only 6 --weights "$3" --backing "$4" \
    > "$OUT/tmp.txt" 2>&1 || { echo "FAIL $1" >> "$OUT/results.txt"; return; }
  python3 -B -c "
import json, sys
d = json.load(open(sys.argv[1]))
for i, l in enumerate(d['loads']):
    print(f\"{sys.argv[2]},load,{i + 1},{d['read_bytes'] / l['seconds'] / 1e9:.3f},{l['seconds']:.4f},{sys.argv[3]}\")
" "$OUT/fp16/loads.json" "$1,r$round" "$c" >> "$OUT/results.txt"
}
round=1
while [ "$round" -le "$ROUNDS" ]; do
  for d in 4 2 8; do
    s=$((2 * d))
    probe probe-land-ce-d$d land-ce $d
    bench after-zone-premapped-d$d "$AFTER" --mode zone --backing premapped --depth $d --slots $s
    probe probe-inplace-d$d hvmm-inplace $d
    bench after-inplace-premapped-d$d "$AFTER" --mode inplace --backing premapped --depth $d --slots $s
    bench after-zone-managed-d$d "$AFTER" --mode zone --backing managed --depth $d --slots $s
    bench before-zone-premapped-d$d "$BEFORE" --mode zone --backing premapped --depth $d --slots $s
    bench before-zone-managed-d$d "$BEFORE" --mode zone --backing managed --depth $d --slots $s
    bench before-inplace-premapped-d$d "$BEFORE" --mode inplace --backing premapped --depth $d --slots $s
  done
  for v in "device managed" "device premapped" "host premapped"; do
    set -- $v
    fp16 "after-fp16-$1-$2" "$AFTER" "$1" "$2"
    fp16 "before-fp16-$1-$2" "$BEFORE" "$1" "$2"
  done
  round=$((round + 1))
done
echo done >> "$OUT/results.txt"
