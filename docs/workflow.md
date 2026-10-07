<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Development workflow

How AI agents and the human developer collaborate on this repository.
Complements the root `AGENTS.md` rules (especially: commits need the user's
authorization; the main agent has standing authorization to commit
completed plan tasks).

jitLLM is a single-developer project that is meant to be consumed externally
(D-016). The process is sized for that: heavier than a personal project,
lighter than a team with maintainers. The default path from idea to commit is
**one agent builds, a second agent reviews, the human commits** (or asks
the main agent to).

## The loop

1. **Build.** One agent implements the task (scope from
   [plan.md](plan.md)), adds or updates tests for any behaviour change, runs
   one check set on a Spark (D-084: `mise run test -- spark-native
   --locked` on `spark-b`, plus the light local steps the change
   touches; D-061's workstation tiers only where D-084 requires them),
   and writes a
   handoff note: what changed, what was verified, on which host, and what
   was not run and why. The note goes in the agent's final message, for the
   commit; the docs themselves carry at most a one-line provenance stamp.
2. **Review.** A separate agent with fresh context reviews the whole
   uncommitted diff against the handoff note. It hunts real defects — data
   loss or corruption, invariant violations, security, broken behaviour,
   claims in docs the code doesn't back — not style or ceremony. Findings are
   file:line claims ranked by severity. The reviewer fixes what it finds (or
   hands back to the builder for anything larger), re-verifies with
   targeted tests and the Spark check set (not the full tiers), and
   reports a review note the same way. A clean review is a valid result and
   is stated as such.
3. **Commit.** The human commits or authorizes the main agent to commit
   (D-075), with both notes and the diff available for review at whatever
   depth the change warrants. The owner's standing authorization covers
   completed plan tasks of the current milestone; otherwise the main agent needs a direct request
   for the change at hand. No other agent commits.

The human may explicitly waive step 2 for a specific trivial change (a typo, a
doc-only status update). Agents never waive it themselves.

## Blast-radius changes get the heavy path by default

Some areas are where an externally consumed runtime earns or loses trust. A
change that touches any of them gets, in addition to the loop above, an
adversarial challenge pass — a reviewer whose brief is to break it: construct
the input, race, cancellation, or failure that violates a pager invariant,
corrupts an artifact, or escapes a bound — followed by fix/verify rounds until
the challenge comes back clean or finds only low-severity issues, which are
fixed without another round (D-084).

- Memory manager, catalog, reservation/lease logic, eviction — anything the
  pager invariants in [architecture.md](architecture.md) govern.
- CUDA VMM mapping, physical pool, staging, and completion tracking.
- On-disk formats: artifact schema, spill files, anything a user's disk holds.
- Import of untrusted checkpoints; any parser of external input.
- Management API authentication, binding, logging defaults, spill protection.
- License and provenance records; the copyleft-disabled build profile.
- Public interfaces: management API, CLI, configuration schema, versioning.

The human can also ask for the heavy path on anything else; agents don't
downgrade a heavy-path change to the light loop on their own.

## Ground rules

- **Commits need the user's authorization** (D-075): the main agent has
  standing authorization to commit completed plan tasks of the current
  milestone until the owner withdraws it (owner, 2026-10-04); otherwise a direct request covers the change at hand only.
  Only the main agent commits, and only reviewed, checked work. Subagents
  never commit, and no agent pushes, tags, amends or rewrites history.
  Without authorization the working tree is the handoff.
- **One commit per completed task.** A plan task stays on its worktree
  through build, review, challenge, fixes, checks and docs, then lands as
  one commit. Separate commits are only for pre-registrations that a
  protocol (D-079) needs fixed before the run it governs.
- **Focused optimization checks.** During M3's kernel optimization run,
  use operand/model correctness and matched performance A/Bs. Do not rerun
  the full swap-performance table for arithmetic-only changes. Rerun the
  affected swap checks when paging, saved state or graph lifetime changes
  require them, and the full table at the milestone gate (owner,
  2026-09-30). A completed table remains evidence for the unchanged swap
  implementation; it is not part of every optimization test set.
  *Owner override, 2026-10-01:* defer unit suites and the routine full-check
  cycle during experimental comparisons until the source of a difference
  is understood and a change is chosen for adoption. Build the needed
  benchmark target and verify the intended inputs, configuration, paid
  work, output and successful completion. Model admission and supervised
  retirement still apply. Check the resulting production implementation
  before landing it; diagnostic experiments do not each require a full
  unit suite.
  *Owner clarification, 2026-10-01:* screen a candidate first with one
  representative shape or prompt and a short bookended A/B. Expand quality,
  state and context controls when the result warrants adoption or leaves a
  decision unresolved. Do not make full ladders, repeated historical archive
  verification or production check cycles prerequisites to this first screen.
  *Owner override, 2026-10-05:* while closing the Gemma performance gaps,
  use focused correctness, lifetime and matched performance checks for reviewed
  optimization commits. Defer the full regression suite until the performance
  changes are settled; do not run it for each incremental optimization.
- **Don't hand off broken.** Checks pass before you end your turn; if they
  don't, say so plainly instead of papering over it. Skipped or disabled
  tests are called out by name.
- **Check cadence (D-084).** Work lands in slices, each checked once on
  its final state. The per-slice set runs on a Spark: `mise run test --
  spark-native --locked` on `spark-b` (full build and every test, GPU
  included), plus locally the light steps the change touches (clang-format
  on changed files, the REUSE and header checks when files are added, the
  tools/ tests it affects). The workstation tiers (`check`, `check:full`,
  `check:spark`) take 30–60 minutes each, so under D-085 they run only
  before a package ships, to chase a finding, or for a change that needs
  the Linux host, and then only the needed target: x86-64 or CPU-only builds, qemu, the reference container,
  packaging, toolchain or source-lock changes, or sanitizer-only
  behaviour. Review and challenge rounds iterate on the Spark set; a round
  that finds only low-severity issues fixes them without another round.
  Never run two check tiers on one tree at once.
  *Owner override, 2026-09-29:* during the current M3 optimization run,
  keep per-slice checks on the Sparks and defer all x86 workstation checks
  until the implementations are settled at the end. Checked, reviewed
  slices may commit in the meantime; no package ships before the owed checks.
- **Nothing over 10 minutes by default (D-085).** Any command, run,
  session or batch expected to take more than 10 minutes of wall time,
  waits included, runs only when its result is needed now: it decides a
  question in front of us, chases a bug, directly checks what a change
  touches, or precedes something shipping. It is never part of a regular
  cadence, a review round or a milestone gate by default. State its
  expected time and why it is needed before starting it. Run the
  narrowest form, and debug backwards if something surfaces later.
- **Say which checks ran where.** The workstation tiers' native builds and
  CPU tests run on the workstation, including AArch64 CPU tests under
  qemu-user (D-061). The per-slice set, and anything that needs a Spark
  (GPU, VMM, RDMA/NCCL, ARM concurrency, target I/O, performance,
  distributed), runs on `spark` or `spark-b` (see environment.md); when it
  was not run, the note says so rather than implying it passed.
- **Evidence, not adjectives.** A performance or capability claim in a note
  or doc carries the measurement and its provenance (host, driver, toolkit,
  artifact, policy) or is not made.
- **Aggregate experiment results in Git; raw output outside it.** Keep the
  measured latency/throughput tables, sample counts, conditions, limitations,
  conclusions, and enough provenance to interpret or repeat the experiment.
  Reusable harnesses and dependency pins stay in the repository. Validate raw
  samples, histograms, logs, traces, and telemetry in external scratch; do not
  add them to Git or make the checked-in report depend on a raw-result bundle.
  Captured inputs for benchmark replay also stay external; keep their verified
  identities and retrieval/supply instructions with the replay harness.
- **Tests travel with behaviour.** A behaviour change without a test needs a
  stated reason in the handoff note.
- **At most about two streams of work at once** (D-084), each in its own
  worktree and its own copy on a Spark (synced with `rsync -rlpc
  --exclude=.git --exclude=/build`). Check `git status` first; if there
  are changes you didn't make, you're iterating on in-flight work, not
  starting fresh.
  When handing a warm Spark build between worktrees or bases, synchronize
  sources before building. The checksum comparison transfers changed bytes,
  and without timestamp preservation those files receive fresh modification
  times so the build sees the changes. Never use `-a` or `-t`: preserving an
  older source timestamp can leave a newer object from the previous tree
  in place even though its contents differ.
- **Scratch files stay out of the tree.**
- **Clean up when a milestone closes** (owner, 2026-10-07). Once a
  milestone's exit is accepted, delete its work files on every host that
  was used (the workstation and both Sparks): raw experiment outputs,
  logs, traces and snapshots under `~/.local/share/jitllm`, scratch
  directories (`~/scratch`, `/tmp` work trees, session scratchpads),
  Spark source and build copies, finished git worktrees, and finished job
  records (`spark-job gc`). Raw outputs need no archive; the committed
  aggregates are the record, and reports' paths to deleted outputs stay
  as history. Keep what later work uses: prepared artifacts, pinned
  source checkpoints and references, frozen oracles, captured replay
  inputs, test inputs, the SDK and the other standing stores under
  `~/.local/share/jitllm` (`sdk/`, `models/`, `references/`,
  `reference-models/`, `ucd/`, `chat-templates/`), and the next
  milestone's work. When unsure, list the file for the owner. Model
  checkpoints no longer needed locally go to the NAS (`/mnt/llm`) rather
  than being deleted. A worktree with uncommitted or unmerged work is
  listed for the owner, or its changes saved as a patch, never silently
  dropped.
- **Notes stay out of the docs.** Handoff and review notes live in the final
  message and the commit, not in the documents they describe. Process detail
  in a design doc costs every future reader and goes stale on commit.
- **Fix the docs the change makes wrong** (status paragraph, plan checkbox,
  affected doc, support matrix) in the same change. Docs that describe
  capability are release artifacts; overclaiming is a defect the reviewer
  flags.
- **Keep the upstream log current.** A change that logs a rough edge in a
  third-party component, or adds or changes a patch under
  `third_party/patches/`, adds or updates that project's
  [upstream](upstream/README.md) entry in the same change.
- **Public surfaces are decisions.** Changing an on-disk format, the
  management API, the CLI, or configuration semantics gets a
  [decisions.md](decisions.md) entry and a version bump per D-062.
- **External pull requests never run locally** (D-061). Agents may read an
  external PR's diff but never check it out, build or test it on the
  workstation or the Sparks; its checks wait for hosted CI.

## Long runs on the Sparks

Every Spark build, test, lint or measurement job runs through the installed
`~/.local/bin/spark-job start --gpu`, then is waited on with the same installed
tool. Detached runs longer than a minute also use it. Use the installed
supervisor, not a source tree's `tools/spark-job`. Its supervisor records
how every job ended (exit code, signal, timeout, kill) and `status` reports
a supervisor that died without a record as `lost`, so a crash ends a wait
instead of hanging it. No `nohup bash q.sh &`, and no loops that wait for a
marker line in a log or for a process name to disappear: a crashed step
never writes the marker.

    ssh spark-b 'cd ~/src/X && ~/.local/bin/spark-job start --gpu --name ctx-part1 --timeout 600 --steps rungs-part1.txt'
    ssh spark-b '~/.local/bin/spark-job wait ctx-part1'    # as a background task

- **Start** with `--gpu` and a timeout of at most 600 seconds by default
  (the process group gets SIGTERM, then SIGKILL). Split longer batches into
  separately supervised and waited jobs. D-085's needed-now exception above
  still applies to a justified longer run; state its expected time and why
  it is needed before starting it. `--gpu` jobs on a host are
  mutually exclusive: a second one waits for the first to end (shown as
  "waiting for the GPU" in `status` and `busy`, logging whom it waits for),
  then starts by itself; its `--timeout` counts from when it gets the GPU,
  and `--no-wait` refuses at once instead. Builds, tests and lint jobs use
  the same `--gpu` lock even when they do no GPU work;
  checking `busy` first is no longer enough on its own, since two agents
  can both see the host free. A queue is a `--steps` file,
  one command per line; a failed step is recorded and the queue moves on
  (`--stop-on-fail` to stop, `--step-timeout` for a stalled rung).
  The child inherits the GPU lock: a lost supervisor does not admit
  overlapping work while that child survives. Admission also waits for a
  lost job's surviving process group, even when a wrapper's descendants
  close inherited descriptors. `busy` reports such orphaned work and waiting
  jobs; `kill` retires a lost job's remaining process group.
- **Wait** with `~/.local/bin/spark-job wait NAME` run as a background task;
  it exits 0 only for success and prints the failed steps and the log's
  tail otherwise. `status`, `tail` and `kill` cover the rest.
- **Harnesses that loop over cases** (context rungs, prompts, model pairs)
  catch each case's error, record it with the case, and continue, so one
  bad case (an HTTP 400, an out-of-memory) costs one row, not the run.
- **Hand a Spark over by checking `~/.local/bin/spark-job busy`** (exit 1
  while a `--gpu` job or any GPU compute process runs; it also lists `--gpu` jobs waiting),
  not by messages alone. The lock covers `spark-job` jobs only: GPU work
  started outside it shows only in `busy`'s nvidia-smi listing.
- **Profiler completion is separate from application completion.** A
  bounded Nsight CLI capture can end while its target survives in a
  separate process group (RE-039). Keep long validation directly
  supervised; collect profiles separately with explicit target/output
  lifetime. A profiler's successful exit cannot qualify an unfinished
  model result.
- **Clean up when finishing:** kill your own jobs that are still running,
  and run `~/.local/bin/spark-job gc` to remove old finished ones.
