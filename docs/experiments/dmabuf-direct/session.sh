#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
# The reported session, on the Spark: PROBE FILE OUTDIR. Runs as a user with
# sudo (the udmabuf device is root:kvm). Before each process it waits until no
# compute process is on the GPU and the 1-minute load is below 0.5, and it
# records the conditions. The udmabuf size limit is raised to 1 GiB for the
# session and restored afterwards. Each restore process runs four passes;
# the report drops pass 0 as a warm-up.
set -euo pipefail
if [[ $# != 3 ]]; then
  echo "Usage: $0 PROBE PATTERN_FILE OUTDIR" >&2
  exit 2
fi
probe=$1 file=$2 out=$3
mkdir -p "$out"
limit=/sys/module/udmabuf/parameters/size_limit_mb
old_limit=$(cat "$limit")
trap 'echo "$old_limit" | sudo tee "$limit" >/dev/null' EXIT
echo 1024 | sudo tee "$limit" >/dev/null

idle() {
  while :; do
    apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
    load=$(cut -d' ' -f1 /proc/loadavg)
    if [[ $apps == 0 ]] && awk -v l="$load" 'BEGIN{exit !(l < 0.5)}'; then break; fi
    sleep 5
  done
  printf '%s load=%s %s\n' "$(date -u +%FT%TZ)" "$load" \
    "$(nvidia-smi --query-gpu=pstate,clocks.sm,temperature.gpu,clocks_throttle_reasons.active --format=csv,noheader)" \
    >> "$out/conditions.txt"
}
run() {  # name, then the probe's arguments
  local name=$1; shift
  idle
  echo "== $name: $*" >> "$out/conditions.txt"
  sudo "$probe" "$@" > "$out/$name.txt" 2>&1
}

{ uname -r; nvidia-smi --query-gpu=name,driver_version,persistence_mode --format=csv,noheader
  sha256sum "$probe"; cat "$limit"; } > "$out/manifest.txt"
run info info
run feasibility feasibility "$file" "$(dirname "$file")"
run udmabuf-4k udmabuf "$file" 4k
for r in 0 1 2; do run "micro-4k-r$r" micro "$file" 4k 10 "$r"; done
for p in 0 1 2; do
  for depth in 2 4 8; do
    for mode in hvmm-inplace land-ce land-sm udmabuf-4k; do
      run "restore-$mode-d$depth-p$p" restore "$file" "$mode" "$depth" 4 8
    done
  done
done
run cost-4k cost 4k 1000
