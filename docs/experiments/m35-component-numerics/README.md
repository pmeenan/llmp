<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# M3.5 component numerical qualification

Status: pre-registration for the implementation checkpoint, 2026-10-22.
No new quality or speed result is recorded here. U4 remains unchecked until the
integrated native build, focused/shared checks, independent review/challenge and
required device/component transfer runs pass. D-123 remains binding.

## Contracts and existing evidence

`components::NumericalSite` declares the arithmetic requirement, operation,
weight roles or activation input, source encodings and semantic parameters.
Admission joins those declarations to `runtime/numerical_evidence.cc`, then binds
each grant to the actual source/spec identity, stage, component and dimensions.
An approval receipt, container name or component kind does not supply permission.
For example, sigmoid per-stream HC normalization and Sinkhorn normalization do
not share a numerical site simply because both use `hc.mix`.

The existing reports are provenance for GB10 evidence. None supplies sm86
qualification. These records do not claim a newly measured checkpoint result.

| Existing report | Source encoding/domain retained by the checkpoint |
| --- | --- |
| [Native numerical audit](../m36-numerical-audit/qwen-modelopt-v1.md) | Sigmoid HC F32 norms, GDN BF16 products, prepared NVFP4/MXFP8 products, NVFP4 expert intermediates; existing owner/row domains |
| [Qwen GGUF vector record](../qwen38-gguf/README.md) | Q6_K, Q8_0, IQ2_S, IQ4_NL vector arithmetic at rows 1–4 |
| [Final M3.6 record](../m36-final/README.md) | Accepted DeepSeek artifact's dense Q4_K/Q5_K/Q6_K/Q8_0 and selected-expert IQ2_XS/IQ3_XXS/MXFP4 arithmetic at rows 1–5 |
| [Image fusion record](../m36-runtime-fusion/README.md) | BF16 weight products and F32 SwiGLU, stateless attention and adaptive-normalization carry sites; additional transfer checks below still owed |

The final DeepSeek encoding inventory was read from the accepted artifact
`698b1cd8696f75355d8a130b71369099c2c5d1d241932330565d33480cbc0bfd` on Spark.
Its `index.json` SHA-256 is
`4f84772b3cfa5912e6a5ece00f438b9b8ac330679dd848a1896810f7d48a8e80`, and
`spec.json` SHA-256 is
`ddaaa1d38bd6a501b9ef891bc0f639a70f41b19ff85a85674be5d77470e7a8e8`.
This read enumerated metadata; it was not a new inference or quality run.

The image report measured 1024², 25 retained text rows and width 4096. Those
numbers remain provenance, never selection conditions. Image numerical grants
follow declared component semantics at the actual generated graph dimensions.
The algorithm-qualified attention row class covers positive row counts with
exact head K/N and output encoding. Its real D128, mask, layout, ISA and resource
requirements stay in capabilities. Joined BF16 carry still has the kernel's
width 8192 bound. These are algorithm limits, not benchmark geometry gates.

`TunedShape::all_rows` is explicit, defaults false and has canonical rows 1.
It changes eligibility across positive rows, not measured speed ranking. Exact
shape tuples retain their existing meaning. Ownership validation, backend
selection identity, component-source keys and execution-contract identity include
the class flag. U6's measured shape-specific ranks remain separate data.

## Bounded verification recipe

Every Spark job uses the installed supervisor with a timeout of at most 600 s.
No source sync or job starts until the orchestrator assigns host ownership.
Build with the pinned wrapper; it selects the configured SDK's CMake/CTest rather
than an unrelated executable on PATH.

The U4 target union is `llmp`, `execution_test`, `execution_contract_test`,
`components_test`, `components_spatial_test`, `component_admission_test`,
`planner_test`, `ggml_spatial_test`, `ggml_lower_test`, `image_ops_gpu_test`,
`image_fusion_gpu_test` and `image_generation_gpu_test`. `planner_report_test.cc`
is part of `planner_test`, not a standalone target. The integrated wave adds
U5/U6 targets selected by their owners. Rebuild any affected linked executables
before believing the shared inventory; do not accept stale binaries.

Run targeted CTest suites for component numerical declarations, component
admission, spatial stage references, execution capabilities, owned device facts,
execution-contract identity and planner reports, then the shared
`-LE 'models|timing'` set on the final integrated source. Read every skipped
result with `-V`. Synthetic `sm_86` tests prove missing-evidence fallback/reporting;
they do not earn numerical support for that device. Run only affected model
areas as exclusive jobs, split between both Sparks. CPU-only and the full model
sweep remain milestone-closeout checks unless an actual failure requires them.

## Frozen inputs and comparison arms

This protocol must be reviewed and committed before any governed comparison.
It includes the frozen 512 input receipt below. Input preparation ran first
and performed no model inference.

The checkpoint is `Qwen/Qwen-Image-2.1` revision
`790c92633540aa0cb11d9abf19eb46d861714758`, with every source file pinned by
[image-reference/pins.json](../image-reference/pins.json). On Spark the source is
`~/.local/share/llmp/models/Qwen/Qwen-Image-2.1@790c9263`. Use the installed
reference image by immutable ID
`sha256:700d6668940ef01c8ae3fca507618544f514cb63aabcbe245c43cd1092f48781`
(read-only inventory verified 2026-10-22), diffusers
`8b3c707ebd3ec4881f4190cf42931da07eaf3b65`, PyTorch 2.14.0+cu130;
[requirements.txt](../image-reference/requirements.txt) pins its Python packages.
Use BF16 weights, disable CUDA matmul/cuDNN TF32, and retain BF16 reduced-precision
reduction enabled. Record actual driver and GPU facts with the receipt.

Both native arms use the same installed artifacts under
`~/.local/share/llmp/artifacts`. Manifest digests equal their IDs:

| Role | Artifact/composition ID | Spec SHA-256 |
| --- | --- | --- |
| Composition | `95314e538fbe1251c23b46e1bb52f86dd34cfbf7f7509e80849472c2e2faa53a` | `ef266ac1b35803f731982a027d31e3656eb31ddd7fde94f4d549bc70ef70a88a` |
| Text | `32fa6658867f6b8daceba8a01d76d63ed42c43bf232dc3a40c5e61841345756f` | `b815577c4e8d31fd70a4b8be682c5d590f83ada9f4bb1c299b5a249ef6818f2d` |
| DiT | `eb8cea718276e66a7f9f26dd60f4f3c87768100fc99e48017f5c47f96500308c` | `c43b66a636b5d4a099032d200c287757ad0ddb5cb33a400f61bea74bfa64e675` |
| VAE | `3785dfca2a2999377f83236b1086baf9783fb1cd01b1bc5af095342abd825727` | `fe40715ad7d84131f6561a024740a592cdd0ed2c35889ac4d264bdf9334c6a68` |

The component index hashes, in that order, are
`d26a01d51576225995ad4b87405ea7846ce5b2f3361d19df614fcce6072408ee`,
`d47e662718b6c0a5c83119aef376d8fa9e425eb10b11606daab9acb2fa671838`, and
`7c84ef9445a334645d83276c4774b43853ab82e303c1f7e6976fb25c5f30cf96`.
These metadata digests were read on Spark, not inferred from an approval name.
If import changes any identity, amend and review the registration before use.

The comparison arms have distinct jobs:

- **A, production baseline:** the main revision immediately before U4, including
  the independently completed U5a/U6/U10 and startup-payment slices:
  `5ec811a4413982630c05e7b6c2448f51f1191883`.
- **B, U4 candidate:** A plus the reviewed U4 patch; record the source tree/patch
  digest and binary SHA-256 before running. Both arms use identical fixture
  tooling and request settings. No numerical-policy override in either arm.
- **P, primitive diagnostic:** the unfused primitive graph in component tests.
  It isolates arithmetic error, and is never the speed baseline.
- **R, independent quality:** pinned diffusers BF16, including its native text
  encoding, scheduler, denoising trajectory and VAE. Native A/B both consume R's
  initial noise, but produce their own remaining intermediates.

| Case | Prompt | Geometry / steps / noise |
| --- | --- | --- |
| Historical source control | `A red ceramic teapot on a plain wooden table, soft daylight, no text.` | 1024 square, 40 steps, seed 42 |
| Transfer control | `A blue glass vase beside a yellow book, viewed from above.` | 512 square, 2 steps, seed 43 |

Both use `true_cfg_scale=1.0`, no condition image and prefix KV caching enabled.
The historical control uses all 39 reference token IDs (14 system tokens dropped)
and the retained `native/ref1/reference.json`, SHA-256
`fdf34f49865f4eadc86baf0f9f94d4a498377ac7ae1d8f8eb0c5843c11564edd`.
Its packed BF16 noise is `[4096,64]`, SHA-256
`eb7333893ca0409a100955e5305482bb9de8225ee3c21d33fd01fd6c16c042bd`.
Do not compare a two-step native image against this forty-step reference.

The new case uses [reference_inputs.py](reference_inputs.py), which imports the
existing native-reference `Pipeline` and `Store` without invoking its historical
fixed-pixel `main`. `--inputs-only` freezes the exact token IDs, kept row count,
packed BF16 noise, timesteps and sigmas. [The frozen input receipt](inputs.json)
was produced without model inference on `spark` by `m35-u4-freeze-inputs`. Its
SHA-256 is `da41873073df85454cafbab6ea683cc0f18321734354334e72e4fcfbbe018d78`:
35 token IDs, 14 system tokens dropped, 21 retained rows, and packed noise
shape `[1024,64]`. The receipt records every token and each raw input hash;
the raw files remain outside Git. The generator SHA-256 is
`4703ce969c93d62a12f1cf8eb23a8be21be98413e83bc4c7b1398db5a0534d85`.
The full reference run must reproduce every input hash and match a separate call to the
official `QwenImage21Pipeline` pixel for pixel. No existing 1024 pixel digest is
substituted for this check.

Input preparation on the assigned Spark uses the installed supervisor's exclusive
job and the exact image ID. With an existing private external parent directory
`OUT`, a checkout `TREE` containing this harness, and the checkpoint directory
`MODEL`, its single step is:

```sh
docker run --rm --gpus all --network none --user "$(id -u):$(id -g)" \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -v "$MODEL:/model:ro" -v "$TREE/docs/experiments:/exp:ro" -v "$OUT:/out" \
  sha256:700d6668940ef01c8ae3fca507618544f514cb63aabcbe245c43cd1092f48781 \
  /exp/m35-component-numerics/reference_inputs.py /model /out/inputs --inputs-only
```

The supervisor invocation is `spark-job start --gpu --timeout 600 --steps FILE`;
use the installed `~/.local/bin/spark-job`, then wait for its exit status. A
separate post-registration reference job uses `/out/reference` and omits
`--inputs-only`. The reference inputs' file hashes must match the frozen input
receipt before native comparison. The exact image is installed on both Sparks (the same image ID was verified
after transfer and load on `spark-b`); do not substitute another image.

The scheduler is the source's `scheduler/scheduler_config.json`, SHA-256
`5895f3a167c14a967fe9ac70c64924ae5acc79799e0679fd12907e594a713cd1`.
Use the pinned pipeline's `calculate_shift` from the packed latent row count and
that config; feed `numpy.linspace(1.0,1/steps,steps)` to `retrieve_timesteps`,
then `set_begin_index(0)` and its Euler scheduler step. The saved sigma/timestep
arrays govern both native arms. The seed names PyTorch CUDA's generator; native
code reads the frozen bytes rather than pretending to implement that generator.

## Transfer controls, fixed before new runs

Every job is supervised and at most 600 seconds; split preparation, reference,
quality and each speed pair into separate jobs. Record source/spec/implementation
identities, actual selected plans, row counts, encodings, device facts and paid
memory. Keep raw captures outside Git; retain aggregate results here.

1. **Component controls.** Run `ImageFusionGpu` and `ImageOpsGpu` against their
   reference decompositions, including small-width and source-sized blocks.
   Exercise non-reference rows 17, 77 and 264 on eligible D128 attention,
   products, GLU and modulation, widths 256/4096, BF16 carry width 8192 and
   refusal above its real 8192 bound. Each required case must run and select the
   candidate; a skipped or unselected case is missing evidence. Existing
   relative-RMS limits remain 5e-3 for oracle flows/decoder and 6e-3 for
   source-shaped fused-versus-primitive flows. Preserve stricter per-operation
   assertions; do not loosen limits after a result.
2. **Complete image quality.** Both complete cases above must reach PSNR at
   least 32 dB over RGB8 and luma SSIM at least 0.98 against R. Use [compare_image.py](compare_image.py) on the completed RGBA8 export and
   reference directory; it uses the existing image-gguf 11-by-11 Gaussian luma
   SSIM; perfect equality passes PSNR despite its serialized null infinity.
   Native token IDs must match R exactly. Also run the existing 2048-by-512
   request/replacement diagnostic for stage ownership and actual non-reference
   attention rows; its finite output alone is not source-quality evidence.
3. **Speed.** Compare production A/B on each complete case, on the same host,
   same artifact and same frozen inputs. Each native process runs two generations: use the first as its warm-up
   and the second as the measured generation. Invoke processes in AB, BA, AB
   order (three pairs), so every measured process has identical warm-up work. Disable tensor/pixel export and debug taps while timing.
   Report every pair and the median of B/A generation-time ratios; a median
   above 1.10 fails D-085. Include stage times, load/ready times separately and
   complete host/device peaks. A materially unequal request or paid work
   invalidates the pair. Correctness eligibility is not a fastest-shape claim.
4. **LLM transfer.** Run `ComponentsModelsGpu` Gemma2/Gemma3 controls with their
   existing per-logit absolute limit 1.0 and top-token near-tie limit 0.25 on
   both sides. Run affected native and selected-expert controls with their
   existing assertions. Capture selected arithmetic operations, operand/output
   encodings and shapes for A/B on the already supported Qwen native, Qwen
   GGUF and DeepSeek domains. Metadata/receipt digest changes alone do not
   demonstrate changed arithmetic. Identical executed arithmetic and passing
   controls reuse existing source-quality evidence; do not schedule a fresh
   all-family corpus sweep merely because evidence plumbing changed.

   When executed arithmetic changes, run the affected family's quality controls
   before promotion. For Gemma2/3, Qwen GGUF and DeepSeek, use
   `speed_gate.py quality` on A/B with the **same current artifact/config in
   both arms** (never the tool's historical old-v0/new-v1 defaults), then pair
   losses by target ID using `quality_pair.py`. The corpus SHA-256 is
   `55381941fdb19f99a88c5170f25d7cbc3e091d7d839a8776ac491a593c978ceb`,
   `passage(corpus,100003,4000)`; retain both `QUALITY_PROMPTS` greedy controls
   as the tool defines them. Apply the numeric bounds from
   [pair-gemma2](../m36-final/prereg/pair-gemma2.json),
   [pair-gemma3](../m36-final/prereg/pair-gemma3.json),
   [pair-qwen38-gguf](../m36-final/prereg/pair-qwen38-gguf.json), and
   [pair-deepseek](../m36-final/prereg/pair-deepseek.json): respectively
   1070/1070/996/966 targets, absolute PPL delta at most 0.5%, mean absolute
   per-target loss delta at most 0.05 nats, maximum at most 1.0, and at least
   95% within 0.1 nats. These are adopted bounds with the U4 A/B arm identities
   above, not a claim that the old registration's v0/v1 arms were rerun.

   Native Qwen's independent acceptance remains the four-document, 1016-target
   Mia pairing in [qwen38-mia.json](../m36-final/prereg/qwen38-mia.json), using
   its exact pinned reference losses and corpus: absolute aggregate PPL delta
   at most 1%, each document at most 4%, mean absolute per-target difference
   at most 0.25 nats, maximum at most 5.0. Its ordinary 996-target A/B pairing
   is informational, as the final M3.6 registration specifies. Missing retained
   inputs/reference is a prerequisite to restore, never permission to replace
   these controls with a generic NLL tolerance. Existing accepted exceptions
   remain attached to their existing arithmetic; newly changed arithmetic
   does not inherit them silently.

   Native owner/row bounds and GGUF vector rows 1–4/1–5 remain their existing
   domains. A proposed broader domain requires a separately registered control
   before promotion. If any LLM arithmetic changes, add the affected workload's
   matched A/B speed screen with the same 10% rule before closing U4.
5. **Discrete device.** No workstation GPU run is authorized by this protocol.
   Real sm86 qualification requires the owner's explicit GPU instruction and
   its own `gpu-discrete` evidence. GB10 results never transfer by architecture
   number, container or component kind. Its plans retain exact fallback and
   name the missing numerical evidence and device arch.

A failed control, wrong selected plan, non-finite output, changed pin, missing
reference or skipped required case stops promotion of the affected domain.
Repair and rerun that case; never hide missing evidence behind a shape whitelist.

## Native fixture and run preparation

The 512 input receipt is frozen. The native fixture built on `spark-b` against
production baseline A, before U4 source was synchronized. Its executable is
`build/spark-native/tests/unit/m35-baseline-image-generation`, SHA-256
`efd4793a9595a8f8109717105c6509e83d4ae20679a786c1f2db33dbdea1245e`.
The identical fixture source SHA-256 is
`bf460f8ca5687ca63d7714d3dbbc0996b1343485b30c568307fab5f657b9c460`.
The private baseline receipt records dynamic dependencies. Verify those dependencies
before running A after the candidate build; preserve the baseline runtime.
Verify the reference fixture inside the pinned container. The native `SameSourceMigrationScreen` accepts
`LLMP_IMAGE_NUMERICAL_FIXTURE=REFERENCE_DIRECTORY` and verifies the receipt
inputs, exact native token IDs and the source-derived scheduler (finite values
within 1e-6 relative, with scale floor 1), then runs its declared geometry/steps.
`LLMP_IMAGE_GENERATION_NOISE` must name that directory's `latents_init.bf16` and
`LLMP_IMAGE_GENERATION_STORE` the pinned artifact store. Set
`LLMP_IMAGE_SCREEN_ARM=shared`; set `LLMP_IMAGE_SCREEN_OUTPUT` only for quality.
Use `--gtest_filter=ImageGenerationGpu.SameSourceMigrationScreen` and export
GoogleTest XML. The first of its two generations warms the arm; the second is
the measured generation. Stage wall times use the text-to-denoiser input-callback boundary and
the completed FirstOutput/Finish boundaries, and include host dispatch. The
fixture moves FirstOutput to the last denoising step, making Finish VAE-only;
all stages already copy their outputs, so the completed generation is unchanged. They do not claim
CUDA-only timing. Both arms receive the same fixture patch.
Review/build that fixture change and this completed registration, then commit the
registration separately before quality/speed comparisons. No result is implied
by the metadata inventory or by the input-only run.

## Prospective LLM operand capture amendment (2026-10-22)

This amendment fixes the diagnostic arms and capture recipe **before** their
first governed capture. It changes no input, numerical domain, quality bound or
speed threshold above. The image arms remain frozen separately. No LLM capture
or quality result is claimed by this amendment.

Both instrumented LLM arms run on Spark A (`spark-c4e2`, GB10, driver
580.178.04), from individual snapshots in
`/home/pmeenan/src/llmp/build/spark-native/src/runtime/`:

| Arm | Snapshot | SHA-256 |
| --- | --- | --- |
| A | `m35-u4-baseline-diagnostic-runtime` | `405c5930accdc009ad8e824d2fcdeff00f063877f0d936c2b1a5dda6b28e519e` |
| B | `m35-u4-candidate-diagnostic-runtime` | `e47f5b30150e9e8a357fa204c423de10ab706c8c8a7f02f0bc48887b5b2a04af` |

A is base `5ec811a4413982630c05e7b6c2448f51f1191883` plus the generic diagnostic
patch SHA-256 `f7b9304a4ae0c29b6f7bab9490c7c2336103225a0b9392e13a3953c99e4e3230`.
B is that same base plus the frozen U4 source patch
`08c82f5832358a8c68489bfbb2b1a48757e71ec9ca5331293b459e8da3332a30` and the
**identical** diagnostic patch. The seven diagnostic files match byte for byte;
the complete source-manifest difference is exactly the U4 patch's 56 paths.
Both native runtime/lowering/model-runner builds passed, and each arm passed
five focused descriptor/completion regressions on A. The build used SDK
`aarch64-2c4eb5b2f68f3fa5`, Clang 22.1.8, NVCC 13.4.92 and cuBLAS 13.8.0.4.
The remote build receipt's retained Git checkout identity is not the source
identity; the base and patch identities above define these arms.

[llm-capture-pins.json](llm-capture-pins.json) freezes the actual generated
`speed_gate.config_template` text for `qwen38`, `qwen38-gguf` and `deepseek`,
the speed/quality helper hashes, runtime identities, target/drafter IDs, and observed
manifest/index/spec hashes. The current helper ignores its old/new label when
generating configurations: both labels select the same current target artifacts.
The launcher checks that generated text equals the registration before starting.
The DeepSeek target for this capture is `d2046349e81da7019c9d787c2eb92521b7a3f4ca9d09c398448da1a99f8eef3f`;
`698b1cd8…` above identifies the historical encoding inventory, not this launch.
Its index/spec hashes agree with that historical inventory. The accepted Qwen
MTP and DSpark drafter artifacts have no optional `spec.json`; the pins record
null for that observed absence. This is not a claim that all artifacts were
re-imported or that the separate drafter-spec migration has completed.

The source Qwen tokenizer and chat template are pinned by their actual hashes
and checkpoint directory in the same JSON. The default corpus is
`~/.local/share/llmp/corpus/m3-reference-corpus.txt`, verified on A as the symlink
to `m36-baselines/inputs/corpus-1facc957.md`: 817275 bytes, SHA-256
`55381941fdb19f99a88c5170f25d7cbc3e091d7d839a8776ac491a593c978ceb`.
The two unchanged `QUALITY_PROMPTS` each retain `max_tokens=128`; scoring
retains `passage(corpus,100003,4000)`. The launcher captures these already
registered controls together with the plan inventory; native Qwen's ordinary
A/B scores remain informational, with its independent Mia acceptance unchanged.

[capture.py](capture.py) imports the pinned `tools/speed_gate.py` only after
parsing arguments. `--repo` permits an externally staged repository; its default
is derived from this checked-in script. Runtime hashes are fixed in code with
no command-line override. Every attempt requires a new exact `--out` directory,
including retries after failure. The service adds only `--debug-plans` to the
ordinary quality launch; readiness is bounded at 420 seconds and retirement at
20 seconds, inside a required exclusive `spark-job --gpu --timeout 600` job.
One family/arm runs per supervised job, which is then waited on. For example,
a job step after this amendment is reviewed and committed is:

```sh
/home/pmeenan/.local/bin/mise exec -- python3 /home/pmeenan/src/llmp/docs/experiments/m35-component-numerics/capture.py --family qwen38 --arm A --runtime /home/pmeenan/src/llmp/build/spark-native/src/runtime/m35-u4-baseline-diagnostic-runtime --out /home/pmeenan/.local/share/llmp/m35-u4-capture/qwen38-A-ATTEMPT
```

Use the matching B snapshot for B and distinct fresh directories for each
family/arm. A nonzero supervisor exit, incomplete receipt, failed greedy reply,
missing scored targets or unsuccessful runtime retirement does not qualify.
The retained score prompt must equal the exact generated frozen passage in both
bytes and recorded hash; the response hash and exact sibling filenames must
match. The pinned `quality_pair.from_speed_gate` helper checks canonical actual
target IDs, echo agreement and finite nonpositive log-probabilities. Capture
additionally requires one choice, an unscored null first row followed by exactly
996 nonnull target rows for native/GGUF Qwen or 966 for DeepSeek, and an equally
long complete echoed-ID list. `usage.prompt_tokens` equals that full row count,
`usage.completion_tokens` is zero, and the receipt's target count agrees.
The ordered actual target-ID digest is retained for later A/B alignment; a
single arm's digest is not a tokenizer-identity claim. These are the existing
registered target counts; native Qwen's 996-target comparison remains
informational. Six score-validation regressions cover complete responses and
partial, null, invalid-probability, echo, usage, prompt and hash failures.
The exact family/label determines the receipt and runtime-log paths; no glob
chooses a prior result. Capture metadata retains helper, launcher, pin, input,
config, runtime, parser, quality receipt and raw-log identities. Raw logs,
requests, replies and score arrays remain outside Git.

[plan_reports.py](plan_reports.py) produces **physical plan inventories backed
by successful-use markers, not a kernel execution trace**. Installed inventory
is associated with model plus installed serial; only completed uses count, so
unused lookahead is excluded. Transient/private plans stream their paid
inventory at successful completion and are scoped by model plus completion
sequence. Each descriptor preserves concrete dtype, dimensions, byte strides,
view offset/source, semantic encoding/layout where available, and ordered
source/output identity including repeated and null source slots. Launch reads
refer to backing storage; operated source slots retain actual view descriptors.

The parser retains logical selections, product arithmetic and all logical
provenance records, and validates the logical selection summary/count. It also
validates exact descriptor, graph-step, root and routed-group counts,
unique contiguous descriptor/step ordinals, every referenced descriptor, ordered
recipe access multiplicity, and routed copy/selected-descriptor metadata. The
routed arena's descriptor count covers the **whole physical inventory**, not
just its selected-descriptor subset. Recipes describe installed launch structure;
this report does not claim observed expert/routed IDs or dynamic launch counts.
Eleven parser regressions cover missing complete recipes, unreferenced or
duplicate descriptors, missing metadata, repeated/null slots, routed omissions,
private completions, model scopes and unsuccessful/unknown completions.

Count checks detect inconsistent omissions, not coordinated editing of both
records and their declared counts or deletion of a whole completed workload.
Successful quality receipts, exact request provenance and raw-log hashes remain
necessary. Compare actual node operations, product arithmetic, operated
source/output encodings and geometry/order, and implementation bindings.
Retain identity provenance, but eligibility/evidence/schema-digest churn alone
is not changed arithmetic; genuine implementation changes cannot be discarded
as metadata. Ambiguous or incomplete capture fails closed. The original quality
and matched-speed obligations above apply if arithmetic changes.
