#!/bin/sh
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
#
# The coarse memory check (D-085, backend-proof.md "Memory and workspace"):
# one process's peak memory, read from MemAvailable (on the GB10 it includes
# the driver's free memory), sampled about every 20 ms:
#   peak_memory.sh LABEL -- COMMAND [ARG...]
# prints "LABEL base=KiB min=KiB peak_mib=N exit=STATUS samples=N". The base
# is the median over the second before COMMAND starts, and peak_mib is how far
# MemAvailable fell below it while COMMAND ran. Other work on the host moves
# it too: run on an idle host and compare with the reference run next to it.
# Exits with COMMAND's status.
set -u
[ $# -ge 3 ] && [ "$2" = "--" ] || { echo "usage: peak_memory.sh LABEL -- COMMAND..." >&2; exit 2; }
LABEL=$1
shift 2
OUT=$(mktemp -d) || exit 2
(
  while [ ! -e "$OUT/stop" ]; do
    awk '/^MemAvailable:/ { print $2; exit }' /proc/meminfo
    sleep 0.02
  done
) > "$OUT/samples" &
SAMPLER=$!
sleep 1
BASE=$(sort -n "$OUT/samples" | awk '{ a[NR] = $1 } END { print a[int((NR + 1) / 2)] }')
START=$(wc -l < "$OUT/samples")
"$@" > "$OUT/run.log" 2>&1
STATUS=$?
touch "$OUT/stop"
wait $SAMPLER
MIN=$(tail -n +"$START" "$OUT/samples" | sort -n | head -1)
echo "$LABEL base=$BASE min=$MIN peak_mib=$(((BASE - MIN) / 1024)) exit=$STATUS samples=$(wc -l < "$OUT/samples")"
[ $STATUS = 0 ] || tail -3 "$OUT/run.log" >&2
rm -rf "$OUT"
exit $STATUS
