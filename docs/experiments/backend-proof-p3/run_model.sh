#!/bin/sh
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
#
# P3 part 2 on a Spark: the native EXL3 model runs and their judgements, for
# one fixture and arm.
#
#   run_model.sh 40|45 G|O
#
# In order, each step's exit status printed:
#   plan    llmp_exl3_exec --record under nsys, one evaluation; the trace
#           exported; cuobjdump's SASS hashed; op_plan_compare.py against
#           exl3-op-plan-ARM.json (the plan gate: exit 0 before anything is
#           judged);
#   rung3   llmp_exl3_exec, two evaluations (the repeat must equal);
#   capture llmp_exl3_exec --capture (Tier C's captures; its logits must
#           equal rung3's, RE-010: pack_run.py records the comparison and
#           tierc_check.py fails the run on it), packed and scored;
#   ops     EXL3-G only: llmp_exl3_exec --record-ops, then op_tier_e.py
#           with rung3's logits as the uninstrumented control (Tier E,
#           operation level);
#   paged   llmp_exl3_paged, two restores, relocated (rungs 4 and 5):
#           every evaluation equal, and rung 4's logits equal rung3's.
#
# BUILD is the spark-native build (default this worktree's), W the raw
# outputs (default ~/.local/share/llmp/p3b-20260927, with plans/ from
# model_plan.py and P0's held-out IDs). The reference steps run in the
# reference container through run_container.sh.
set -u
short=$1 arm=$2
here=$(cd "$(dirname "$0")" && pwd)
: "${BUILD:=$here/../../../build/spark-native}"
: "${W:=$HOME/.local/share/llmp/p3b-20260927}"
: "${STORE:=$HOME/.local/share/llmp/artifact-layout-20260922/installed}"
: "${IDS:=$HOME/.local/share/llmp/p0-20260925/heldout-ids.i64le}"
case $short in
  40) fixture=4.0bpw art=6e96e499ea9b859ea726327d4f82fb1739cb116ce4e9d3541d1ba67506efb4b4 ;;
  45) fixture=4.5bpw art=00d77caf056c45b6fa05ebdbeff8b38da79d0556a2fd4f1b661e7e21ef9ed83a ;;
  *) echo "fixture 40 or 45" >&2; exit 2 ;;
esac
record=$here/exl3-op-plan-$(echo "$arm" | tr GO go).json
exec_bin=$BUILD/benchmarks/llmp_exl3_exec
paged_bin=$BUILD/benchmarks/llmp_exl3_paged
plan=$W/plans/plan-$short-$arm.txt
out=$W/native
mkdir -p "$out"
cd "$out"
common="--artifact $STORE/$art --fixture $fixture --arm $arm --plan $plan --ids $IDS"
n=$short-$arm

rm -rf "plan-$n" "plan-$n.nsys-rep" "plan-$n.sqlite"
nsys profile -t cuda -o "plan-$n" $exec_bin $common --out "plan-$n" --evaluations 1 --record \
  > "plan-$n.log" 2>&1
r=$?
nsys export --type sqlite -o "plan-$n.sqlite" "plan-$n.nsys-rep" >> "plan-$n.log" 2>&1
/usr/local/cuda-13.0/bin/cuobjdump -sass "$exec_bin" \
  | python3 -B "$here/../backend-proof-p0/fp16_plan.py" sass-hash --label native > "sass-$n.jsonl"
python3 -B "$here/op_plan_compare.py" --record "$record" --fixture "$fixture" \
  --nsys "plan-$n.sqlite" --sass "sass-$n.jsonl" "plan-$n/record.jsonl" > "gate-$n.txt" 2>&1
echo "plan $n run $r gate $?"

rm -rf "rung3-$n"
$exec_bin $common --out "rung3-$n" --evaluations 2 > "rung3-$n.log" 2>&1
echo "rung3 $n $?"

rm -rf "capture-$n" "packed-$n"
$exec_bin $common --out "capture-$n" --evaluations 1 --capture > "capture-$n.log" 2>&1
r=$?
same=0
for f in rung3-$n/logits-*.npy; do
  cmp -s "$f" "capture-$n/$(basename "$f")" || same=1
done
echo "capture $n run $r logits-equal-rung3 $same"
artifacts="--mount type=bind,src=$STORE,dst=/artifacts,readonly"
GPUS="" DOCKER_EXTRA="$artifacts" sh "$here/run_container.sh" pack_run.py "/out/native/capture-$n" \
  "/out/native/packed-$n" --ids /p0/heldout-ids.i64le --uninstrumented "/out/native/rung3-$n" \
  --artifact "/artifacts/$art" > "pack-$n.log" 2>&1
echo "pack $n $?"
GPUS="" DOCKER_EXTRA="$artifacts" sh "$here/run_container.sh" tierc_check.py --fixture "$fixture" \
  --oracle "/p0/oracle/oracle-$fixture.npz" --json "/out/native/tierc-$n.json" \
  --artifact "/artifacts/$art" "/out/native/packed-$n" > "tierc-$n.txt" 2>&1
echo "tierc $n $?"

if [ "$arm" = G ]; then
  rm -rf "ops-$n" "opsrun-$n"
  $exec_bin $common --out "opsrun-$n" --evaluations 1 --record-ops "ops-$n" > "ops-$n.log" 2>&1
  echo "ops-record $n $?"
  DOCKER_EXTRA="--mount type=bind,src=$STORE,dst=/artifacts,readonly -e GGML_OPS_LIB=/p0/ggmlops/cuda134b/libggml_shim.so" \
    sh "$here/run_container.sh" op_tier_e.py --artifact "/artifacts/$art" --ids /p0/heldout-ids.i64le \
    --record "/harness/backend-proof-p3/exl3-op-plan-g.json" --uninstrumented "/out/native/rung3-$n" \
    "/out/native/ops-$n" > "tiere-$n.txt" 2>&1
  echo "tiere $n $?"
fi

rm -rf "paged-$n"
$paged_bin $common --out "paged-$n" --restores 2 --relocate > "paged-$n.log" 2>&1
r=$?
same=0
for f in rung3-$n/logits-*.npy; do
  cmp -s "$f" "paged-$n/$(basename "$f")" || same=1
done
echo "paged $n run $r logits-equal-rung3 $same"
