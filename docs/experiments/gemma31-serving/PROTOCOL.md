<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Scalar route reproduction

Use a separate tree based on the recorded implementation base and the approved
prepared artifacts from [the report](README.md). Native inference executes from
those artifacts, never directly from the GGUF. Synchronize source bytes with
`rsync -rlpc --exclude=.git --exclude=/build`; preserving timestamps can leave
stale objects. Use the pinned SDK and declared prepared sources.

Run the CPU profile contract and both-profile native model controls through
the installed Spark supervisor:

```sh
~/.local/bin/spark-job start --gpu --name gemma-serving-check --timeout 600 \
  --stop-on-fail -- bash -lc \
  'mise run test -- spark-native --locked -- -R GemmaServingProfile\|Gemma4ServingGpu'
~/.local/bin/spark-job wait gemma-serving-check
```

The parameterized test names identify Gemma26 and Gemma31. Every model target
uses the existing CTest GPU resource lock. Tests retain unproven native owners
until process exit. A success result requires completion, not a log marker.

The durable stdlib harness starts one native runtime process with two
ordinary-profile entries, context 4,096, prefill cap 128, twelve scalar
owners and the existing one-second model-turn setting. It writes private configuration, state files, raw runtime logs and
aggregate JSON beneath a new external directory. It terminates and waits for
the child before returning; forced retirement timeout fails the result.

```sh
~/.local/bin/spark-job start --gpu --name gemma-serving-http --timeout 600 \
  --stop-on-fail -- bash -lc \
  'python3 docs/experiments/gemma31-serving/http_control.py \
    --output "$HOME/.cache/gemma-serving-http-new" \
    --installed "$HOME/.local/share/llmp/m3-artifacts" \
    --runtime build/spark-native/src/runtime/llmp-runtime'
~/.local/bin/spark-job wait gemma-serving-http
```

Each case records its failure and allows later bounded controls to finish.
The overall result fails if any case fails, runtime exit is nonzero, or either
switch direction lacks a positive source-specific pause delta. `--switches-only`
runs the two switching controls alone. HTTP literal completions
remain non-streaming; pending stream controls use chat SSE. Standard API
clients require no session extension. The normalization oracle covers the
actual templates through Conversation controls, not an HTTP tool route.

These are own-engine continuity controls. They do not run a reference engine,
select optimization policies or measure a batch speedup.
