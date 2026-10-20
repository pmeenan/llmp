<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# M3.5 R1–R13 closeout verification

Final verification of `main` after the M3.6 post-gate review commits
(`c1dcc8c1`, `fa1f7cad`, `235ff8e4`, `7283b964`) and the M3.5 review follow-up
slices (`R1`–`R12`, commits `aba73a61` through `aa92bd1f`), compared directly
against the frozen M3.6 final-table candidate `53c2610b` (`final-d373a1bf`,
[m36-final](../m36-final/README.md#final-candidate-53c2610b-tree-d373a1bf-the-verification)).

## Candidate and environment

`main` at `aa92bd1f97686090b927996ba182d12cfe3aded3` (`[M3.5] Test coverage and
maintainable frozen expectations`, source manifest
`cc25f6c1313a1f4fae8384cd04e9b3d00d1ff160918db518a7264e87b5ede5ce`), built with
`spark-native` (SDK `aarch64-2c4eb5b2f68f3fa5`) on both Sparks and installed at
`~/.local/share/llmp/m36-candidates/final-r13-aa92bd1f`:

| Host | `llmp-runtime` SHA-256 | `llmp_gemma_engine_greedy` SHA-256 | `IDENTITY.sha256` SHA-256 |
| --- | --- | --- | --- |
| `spark` (`spark-c4e2`) | `3ee65a9b4c41fa2ad017c1b007d040600b7e3e1fcba6a1c8fd097d5966170480` | `67659b7d378aaa32fec36963f5fdd4f9a5ae782768dc3dfed7acb791c1b8341f` | `674864b4b06b575b3cf94f47043527bf7bb068ffca2ff7bc20312ca24350fa93` |
| `spark-b` (`spark-56f5`) | `0f717bae995ff82fb568fca3d59fea89556c243766cffb090e4f48ab3bc06498` | `0764dc5edfbaac875f327740d3ac2967556a46fc8ef5a18b3eb92763da161424` | `82c6abd417609ef78a8bc0345f509ddd77f65983ebea98a23034f73641cb99fc` |

Old arm for the 2-block ABBA speed confirmations: each host's installed
`~/.local/share/llmp/m36-candidates/final-d373a1bf` (`53c2610b`; `spark`
runtime `55fd8e9d…`, `spark-b` runtime `66e343ee…`). Both arms serve the
published format-v1 store (`~/.local/share/llmp/m36-v1-artifacts`) with
identical per-family configurations (`configs/*.toml.in`). For Qwen-Image-2.1,
the old arm serves its pinned pre-R9 composition `2170db12…` and the new arm
serves its format-independent post-R9 composition `ecd27c74…` (same source
checkpoint `Qwen/Qwen-Image-2.1@790c9263`, same pixel pin `93356574…`).

## Check set

All checks ran under `spark-job` supervision with zero failures:

| Check | Host | Result |
| --- | --- | --- |
| `cmake --build --preset spark-native` | `spark`, `spark-b` | ok (`0` errors, `0` warnings) |
| `tools/llmp_boundaries.py` | workstation | `684 files, 0 problem(s)` |
| Shared suite, `ctest --preset spark-native -LE 'models\|timing'` | `spark` (`spark-b`) | **3,375 / 3,375 passed** (`3,366 / 3,366` at R11 on `spark-b`), `0` failed, `0` skipped |
| CPU-only guardrail (`LLMP_CUDA=OFF` in `build/spark-cpu`, `ctest -LE 'models\|timing'`) | `spark-b` | **2,789 / 2,789 passed**, `0` failed (`40.49 s`) |
| Exclusive `models\|timing` suite, `ctest --preset spark-native -L 'models\|timing'` | `spark` | **216 / 216 passed** (`214` passed, `2` env-gated full-checkpoint publication tests skipped when run without raw source checkpoints; the 19 Qwen2/EXL3/FP16 fixture tests previously skipped on `53c2610b` now run and pass via `tests/support/v1_frozen.h`) |

## Tier-1 speed confirmation against the M3.6 final table (`53c2610b`)

Two-block ABBA (`old/new/new/old` × 2, primed per session) comparing
`final-d373a1bf` (`53c2610b`, old) against `final-r13-aa92bd1f` (`aa92bd1f`,
new). Slowdown is $100 \cdot (\exp(\bar{\Delta}) - 1)$ where $\Delta =
\ln(\text{new}/\text{old})$ for times and $\ln(\text{old}/\text{new})$ for
rates; negative values mean `aa92bd1f` is faster than `53c2610b`. The right-most
column notes the cumulative comparison against the frozen M3 baseline
(`1facc957-fixed`) from [m36-final](../m36-final/README.md#speed).

| Family (host) | Cell / metric | `53c2610b` (old) | `aa92bd1f` (new) | $\Delta$ vs `53c2610b` | Peak memory ($\Delta$ vs `53c2610b`) | Status vs `1facc957-fixed` |
| --- | --- | ---: | ---: | ---: | --- | --- |
| **Gemma 2 2B** (`spark`) | C1 prefill / decode / wall<br>C2 prefill / decode / wall | 1.351 s / 62.60 / 37.88<br>2.051 s / 104.91 / 57.28 | 1.354 s / 63.27 / 38.10<br>2.019 s / 107.34 / 58.44 | +0.20% / **−1.06%** / −0.57%<br>**−1.55%** / **−2.27%** / **−1.98%** | 6.031 GiB (+0.189 GiB) | All cells faster or flat (decode −1.6% to −2.9% vs M3) |
| **Gemma 3 4B QAT** (`spark`) | C1 prefill / decode / wall<br>C2 prefill / decode / wall | 0.665 s / 75.31 / 54.40<br>1.187 s / 135.99 / 83.69 | 0.670 s / 75.93 / 54.64<br>1.167 s / 138.96 / 85.53 | +0.62% (+4.2 ms) / **−0.82%** / −0.43%<br>**−1.69%** / **−2.14%** / **−2.14%** | 4.798 GiB (+0.092 GiB) | Decode −1.9% to −4.8% faster than M3; C1 prefill +0.4% vs M3 |
| **Gemma 4 26B-A4B** (`spark`) | C1 prefill / decode / wall<br>C4 prefill / decode / wall | 2.186 s / 48.09 / 26.52<br>9.225 s / 102.66 / 36.88 | 2.184 s / 49.42 / 26.93<br>9.185 s / 104.06 / 37.10 | −0.09% / **−2.69%** / **−1.53%**<br>−0.44% / **−1.34%** / −0.59% | 20.533 GiB (**−0.264 GiB**) | Every metric faster than both `53c2610b` and M3 (−0.49 GiB vs M3) |
| **Gemma 4 31B** (`spark`) | C1 prefill / decode / wall<br>C4 prefill / decode / wall | 9.239 s / 9.89 / 5.79<br>45.506 s / 30.76 / 8.43 | 9.206 s / 10.03 / 5.85<br>45.225 s / 32.03 / 8.54 | −0.35% / **−1.42%** / −0.98%<br>−0.62% / **−3.97%** / **−1.19%** | 26.265 GiB (**−1.685 GiB**) | Every metric faster than both `53c2610b` and M3 (**−1.69 GiB** vs M3) |
| **Gemma 4 31B assistant** (`spark`) | `unit32` greedy (`assistant_gate.py`) | 1.321 s | 1.280 s | **−3.15%** (bound [−4.31%, −1.98%]) | — | **−6.88% faster** than M3 (`1.374 s → 1.280 s`) |
| **Qwen-Image-2.1** (`spark`) | `generation_s` (1024², 40 steps, pin `93356574…`) | 33.289 s | 33.305 s | **+0.05%** | 34.344 GiB (+0.037 GiB) | Matches `53c2610b` and M3 (+0.18% vs M3), pixel-identical |
| **Qwen3.8 native + MTP** (`spark-b`) | C1 prefill / decode / wall<br>C2 prefill / decode / wall<br>C4 prefill / decode / wall | 3.182 s / 48.37 / 22.00<br>7.023 s / 67.77 / 23.47<br>15.611 s / 79.24 / 24.02 | 3.154 s / 48.62 / 22.16<br>6.989 s / 66.56 / 23.36<br>15.800 s / 78.90 / 23.59 | **−0.89%** / −0.52% / −0.74%<br>−0.49% / +1.81% / +0.48%<br>+1.21% / +0.44% / +1.82% | 86.298 GiB (**−4.297 GiB**) | **Every cell still faster than M3** (C1 prefill −2.98%, C2 decode −4.03%, C4 decode −2.50%, C4 wall −1.98%); R7 cut MTP activation peak by **4.30 GiB** |
| **Qwen3.8 GGUF** (`spark-b`) | C1 prefill / decode / wall<br>C2 prefill / decode / wall<br>C4 prefill / decode / wall | 6.506 s / 30.99 / 12.07<br>13.634 s / 48.14 / 13.52<br>28.387 s / 65.04 / 14.13 | 6.730 s / 30.88 / 11.80<br>13.632 s / 48.73 / 13.58<br>28.432 s / 66.35 / 14.19 | +3.44% / +0.34% / +2.25%<br>−0.01% / **−1.21%** / −0.46%<br>+0.16% / **−1.98%** / −0.42% | 62.108 GiB (**−4.436 GiB**) | C2 decode **−14.1% faster** and C4 decode **−5.0% faster** than M3; R7 cut activation peak by **4.44 GiB** |
| **DeepSeek V4 Flash, plain** (`spark-b`) | C1 prefill / decode / wall<br>C2 prefill / decode / wall<br>C4 prefill / decode / wall | 8.163 s / 32.99 / 10.65<br>17.812 s / 37.81 / 10.37<br>35.289 s / 51.13 / 11.37 | 8.160 s / 33.08 / 10.67<br>17.710 s / 37.97 / 10.42<br>35.352 s / 51.02 / 11.35 | −0.03% / −0.27% / −0.11%<br>−0.57% / −0.42% / −0.52%<br>+0.18% / +0.22% / +0.19% | 110.368 GiB (+0.111 GiB) | All 9 metrics within ±0.57% of `53c2610b` and faster than M3 |
| **DeepSeek V4 Flash, DSpark** (`spark-b`) | C1 prefill / decode / wall<br>C2 prefill / decode / wall<br>C4 prefill / decode / wall | 8.156 s / 32.90 / 10.65<br>17.704 s / 46.30 / 11.15<br>36.356 s / 54.71 / 11.49 | 8.164 s / 33.07 / 10.66<br>17.731 s / 48.47 / 11.19<br>36.757 s / 62.79 / 11.35 | +0.09% / −0.53% / −0.11%<br>+0.15% / **−4.46%** / −0.36%<br>+1.10% / **−12.88%** / +1.25% | 110.472 GiB (+0.357 GiB) | C2 decode **+7.7% faster** and C4 decode **+18.5% faster** (`53.01 → 62.79 tok/s`) than M3; C4 wall reflects RE-059 physical page placement |

## Generality canary: Llama 3.2 1B Instruct (`R11`, `b003c9a5`)

Imported and served from `bartowski/Llama-3.2-1B-Instruct-GGUF@067b946c` (`Q8_0`
artifact `fae45ab2…`, `Q4_K_M` artifact `6a83d84f…`) through the parametrized
dense-decoder mapping (`src/spec/mappings/qwen2.cc`) and `kDefaultServingRecipe`
with zero Llama-specific C++ outside the mapping table (`spark-b`, compared
against `llama.cpp` v0.6.0 `d8123504` on the same GGUFs; 924-token teacher-forced
corpus for NLL/PPL; 3,880-token uncached prompts and 128 generated tokens for
speed):

| Variant | Teacher-forced NLL (`llmp` vs `llama.cpp`) | PPL $\Delta$ | C1 decode (`llmp` vs `llama.cpp`) | C4 decode (`llmp` vs `llama.cpp`) | Peak memory |
| --- | --- | ---: | --- | --- | ---: |
| `Llama-3.2-1B-Instruct-Q8_0` | `4.260130` vs `4.258337` | **+0.18%** | **150.0** vs 144.6 tok/s (**+3.8%**) | **448.2** vs 413.7 tok/s (**+8.3%**) | 1.38 GiB |
| `Llama-3.2-1B-Instruct-Q4_K_M` | `4.292454` vs `4.294595` | **−0.21%** | **212.7** vs 198.4 tok/s (**+7.2%**) | **589.9** vs 518.1 tok/s (**+13.9%**) | 0.92 GiB |

## Swap and switch tables (`spark`, `tools/swap_table.py`)

Both the 32-row M3 production swap table and the 24-row Gemma near-8K switch
tables pass all checks (`exact`, `state_exact`, restored 8K state and
continuation hashes, kept and replayed CUDA graphs, image pixel pin `93356574…`,
prepared swaps $\le 20\text{ s}$, first-use swaps $\le 40\text{ s}$). Notably,
the `image -> deepseek` first-use swap improved from **27.354 s** on `53c2610b`
to **8.733 s** on `aa92bd1f`, and `qwen3.8 <-> image` peak memory dropped by
**4.62 GiB** (`89.01 → 84.37 GiB`):

```text
swap table m3: 32 of 32 rows
  qwen3.8 <-> deepseek   A->B, first use (8192 context tokens)            9.067 s  peak  111.56 GiB (119788527616 B)
  qwen3.8 <-> deepseek   B->A, first use (8192 context tokens)            6.084 s  peak  111.46 GiB (119683579904 B)
  qwen3.8 <-> deepseek   A->B, prepared (8192 context tokens)             8.405 s  peak  111.44 GiB (119658835968 B)
  qwen3.8 <-> deepseek   B->A, prepared (8192 context tokens)             6.045 s  peak  111.48 GiB (119703277568 B)
  qwen3.8 <-> deepseek   A->B, prepared (0 context)                       8.385 s  peak  111.47 GiB (119689375744 B)
  qwen3.8 <-> deepseek   B->A, prepared (0 context)                       6.205 s  peak  111.47 GiB (119693836288 B)
  qwen3.8 <-> image      A->B, first use (8192 context tokens)            3.946 s  peak   84.27 GiB (90484264960 B)
  qwen3.8 <-> image      B->A, first use (8192 context tokens)            6.059 s  peak   84.36 GiB (90579255296 B)
  qwen3.8 <-> image      A->B, prepared (8192 context tokens)             3.794 s  peak   84.30 GiB (90517200896 B)
  qwen3.8 <-> image      B->A, prepared (8192 context tokens)             5.971 s  peak   84.37 GiB (90593382400 B)
  qwen3.8 <-> image      A->B, prepared (0 context)                       3.803 s  peak   84.37 GiB (90593611776 B)
  qwen3.8 <-> image      B->A, prepared (0 context)                       6.058 s  peak   84.13 GiB (90328563712 B)
  deepseek <-> image     A->B, first use (8192 context tokens)            4.002 s  peak  111.06 GiB (119245180928 B)
  deepseek <-> image     B->A, first use (8192 context tokens)            8.378 s  peak  111.39 GiB (119606517760 B)
  deepseek <-> image     A->B, prepared (8192 context tokens)             3.829 s  peak  111.40 GiB (119611289600 B)
  deepseek <-> image     B->A, prepared (8192 context tokens)             8.225 s  peak  111.38 GiB (119588564992 B)
  deepseek <-> image     A->B, prepared (0 context)                       3.807 s  peak  111.38 GiB (119588728832 B)
  deepseek <-> image     B->A, prepared (0 context)                       8.685 s  peak  111.33 GiB (119537586176 B)
  deepseek <-> qwen3.8   A->B, first use (8192 context tokens)            6.304 s  peak  111.39 GiB (119599820800 B)
  deepseek <-> qwen3.8   B->A, first use (8192 context tokens)            8.389 s  peak  111.79 GiB (120034308096 B)
  deepseek <-> qwen3.8   A->B, prepared (8192 context tokens)             6.172 s  peak  111.81 GiB (120052891648 B)
  deepseek <-> qwen3.8   B->A, prepared (8192 context tokens)             8.268 s  peak  111.80 GiB (120042926080 B)
  deepseek <-> qwen3.8   A->B, prepared (0 context)                       6.149 s  peak  111.81 GiB (120054710272 B)
  deepseek <-> qwen3.8   B->A, prepared (0 context)                       8.757 s  peak  111.77 GiB (120013156352 B)
  image <-> deepseek     A->B, first use (image)                          8.733 s  peak  110.79 GiB (118964080640 B)
  image <-> deepseek     B->A, first use (image)                          3.824 s  peak  110.80 GiB (118972755968 B)
  image <-> deepseek     A->B, prepared (image)                           8.325 s  peak  110.82 GiB (118990352384 B)
  image <-> deepseek     B->A, prepared (image)                           3.722 s  peak  110.82 GiB (118990352384 B)
  image <-> qwen3.8      A->B, first use (image)                          6.161 s  peak   83.40 GiB (89549594624 B)
  image <-> qwen3.8      B->A, first use (image)                          3.692 s  peak   83.41 GiB (89556480000 B)
  image <-> qwen3.8      A->B, prepared (image)                           6.048 s  peak   83.41 GiB (89559924736 B)
  image <-> qwen3.8      B->A, prepared (image)                           3.658 s  peak   83.41 GiB (89559924736 B)
worst LLM<->LLM row: 9.067 s (qwen3.8 <-> deepseek, A->B, first use (8192 context tokens), peak 111.56 GiB)
worst prepared LLM<->LLM row: 8.757 s (deepseek <-> qwen3.8, B->A, prepared (0 context), peak 111.77 GiB)
largest tracked peak: 111.81 GiB
lowest MemAvailable: 4.77 GiB
PASS
```

Gemma near-8K switch cases (`spark`, 6 rows each, 24 rows total):

| Pair | Worst first use | Worst prepared | Largest peak | Lowest `MemAvailable` | Verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| `gemma2 <-> qwen38` | 6.068 s | 5.877 s | 76.39 GiB | 40.16 GiB | **PASS** |
| `gemma3 <-> qwen38` | 6.031 s | 5.851 s | 76.21 GiB | 40.35 GiB | **PASS** |
| `gemma26 <-> qwen38` | 6.052 s | 5.863 s | 76.17 GiB | 40.40 GiB | **PASS** |
| `gemma31 <-> qwen38` | 6.155 s | 5.961 s | 77.27 GiB | 39.29 GiB | **PASS** |
