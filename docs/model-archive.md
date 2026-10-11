<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Model archive

The long-term store of source checkpoints, its catalog, and the tools that
write and read it (D-114). The archive holds Hugging Face repositories as
downloaded, one directory per pinned commit, each with a manifest that
records every file's identity from the Hub; a catalog beside them relates
quantizations and versions of the same model and records what was imported
from each checkpoint and where it is installed. The importer reads a
checkpoint straight from the archive, or from a copy staged on a Spark's
NVMe for repeated testing.

**Status (2026-10-11).** Stored format version 1; checkpoint writer
protocol version 2 (D-115). The interim
Python tools (`tools/archive-model`, `tools/archive-catalog`,
`tools/stage-model`) write and read exactly this format, so downloads can
start now. Native `llmp import SOURCE --store DIRECTORY` now consumes archived or
staged sources within the normalizer's closed domain (Gemma 2/3/4 target
GGUFs); its source and publication checks are described in
[artifact-format.md](artifact-format.md#native-writer). Native `llmp install`
(fetching into the archive) remains in M5 ([plan](plan.md)). The per-host
report (`report-host`, [below](#imports-installs-and-stages)) is not yet built.

The [initial model set](model-archive-initial-set.md) gathers the planned
models, formats, quantizations and companions, with existing copies to
migrate and unresolved selections identified before the first batch.

## Downloader quickstart

For any agent that downloads models (Claude, Gemini, Codex). Run from the
repository root on the workstation.

1. **The token.** Gated repositories need a Hugging Face token. It lives
   in the repository's ignored `/.env` as `HF_TOKEN=...`; `mise run` and
   `mise exec` export it. Never print, echo, log or commit it, and never
   pass it on a command line. The tools read it from the environment only
   and print `<token>` in its place if it ever appears in a message.
2. **Check the catalog first.** Never download what is archived.

       mise run archive:catalog -- show qwen3.8        # inventory lines matching a word
       mise run archive -- --dry-run unsloth/DeepSeek-V4-Flash-0731-GGUF

   `show` reads the compacted catalog (before the first download it
   reports that there is no archive yet, and exits 0)
   (`/mnt/llm/checkpoints/catalog/catalog.md`
   is the same inventory, readable in any viewer). `--dry-run` asks the
   Hub and the archive and prints what would be fetched; `archive-model`
   itself always skips what is present.
3. **Download** a pinned revision (a commit, or a branch or tag that is
   resolved to its commit once), under the owner's shared hold:

       hostlock shared --label "llmp: archive unsloth/DeepSeek-V4-Flash-0731-GGUF" -- \
         mise run archive -- unsloth/DeepSeek-V4-Flash-0731-GGUF@main --include 'UD-Q2_K_XL/*'

   A root that does not exist (no `llmp-archive-root.json`) is an error, so
   an unmounted share is never filled on the local disk; the one run that
   starts a new archive adds `--init-root` (the root's parent, the mount
   point, must already exist; its parents are never created). Before that
   one run, check that the share really is mounted (`findmnt /mnt/llm`
   names `//192.168.0.3/llm`): an unmounted mount point is an empty local
   directory, and `--init-root` would start the archive there. The tools
   also never create the root, or anything above it, while writing.

   A repository holding several GGUF quantizations needs `--include`
   (one or more globs over repository paths) or `--all-variants`; the tool
   refuses otherwise and lists the variants with their sizes. An `--include`
   glob that matches no file is a usage error (exit 2), so a typo never ends
   as an archive of configuration files only. Downloads of tens or hundreds
   of GB take hours: start the command detached (nohup, tmux or your
   harness's background mode) with its output in a log file under your
   scratch directory, and poll that log; a foreground call that your
   harness times out is interrupted and has to be rerun. Configuration,
   tokenizer, license and card files always come along. PyTorch pickles
   (`*.bin`, `*.pt`, `*.pth`) and other frameworks' weights are left out
   unless an `--include` names them.
4. **Resume** by running the same command again. Finished files are kept
   (a file the manifest lists but the share no longer holds, or holds at
   another size, is fetched again), a partial file continues where it
   stopped. Another downloader waits for the archive writer lock before
   reading revision state, so finished files are never duplicated. A run that fails on a file (an error, a signal)
   releases its partial files, so the rerun continues at once; the other
   files of the run are finished and listed in the manifest first. A
   writer killed outright (a second stop signal, SIGKILL, power loss) leaves a partial that the
   next exclusive writer can adopt and resume immediately.
5. **Exit codes.** 0: archived (or already was). 1: an error, printed
   (a hash mismatch deletes the bad copy and names the file; a gated
   repository without a token, a revision that does not exist, a full disk
   or an unmounted share names the cause; rerun after fixing it).
   2: a usage error. 3: another writer holds the archive lock (`--no-wait`);
   rerun later. A second SIGINT or SIGTERM forces process exit with
   `128 + signal`; its ownership directory remains for dead-owner recovery.
   Without an explicit revision, a repository already archived is not
   downloaded again: the newest archived commit is used, and only files
   missing from it are fetched. `--latest` resolves the Hub's `main`
   instead and archives a new commit if it moved.

The archive root defaults to `/mnt/llm/checkpoints` (the owner's NAS;
`--archive` or `LLMP_ARCHIVE` override it). That default is owner
environment, like every host name here; the product reads its location
from configuration ([below](#the-native-importer-and-llmp-install)).

## Where it lives

The archive is the node configuration's `checkpoints` storage role
(D-063), pointed at a long-term store when one is configured (in the
owner's environment `long_term = "/mnt/llm"` and `checkpoints =
"/mnt/llm/checkpoints"`, on the Synology share described in
[environment.md](environment.md#long-term-model-store-2026-09-22)). Without
a long-term store the role defaults to `/var/lib/llmp/checkpoints` and
holds the same layout locally. Only import and install jobs and these
tools touch it; the runtime process never opens it (D-054), so an absent or
hung mount fails a job, never serving.

The share shapes the format: no symbolic links (they fail with
`EOPNOTSUPP`), no direct I/O, rename-over works, names are case-insensitive
and Windows-restricted, and reads were measured at about 118 MB/s per
client (about 225 MB/s for three). Cross-host SMB file locks proved
unreliable in the 2026-10-10 checks; checkpoint writers instead serialize
through an atomic NAS directory creation. Catalog event writers and
readers may still work concurrently.

## Layout

```text
<root>/                                       the `checkpoints` role
  llmp-archive-root.json                      {"format": "llmp-archive-root", "format_version": 1}
  .llmp-writer/owner.json                     active checkpoint writer ownership (protocol 2)
  hf/<org>/<repo>@<commit>/                   one commit of one Hugging Face model repository
    <the repository's files, in its own layout>
    llmp-archive.json                         the revision manifest, written last
    <path>.llmp-partial.<writer>              a file being downloaded (transient)
  catalog/
    events/<event-id>.json                    immutable records, one per change
    packs/pack-<sha256>.json                  immutable packs of events
    catalog.json                              the compacted catalog, a cache
    catalog.md                                the same inventory for people
```

- **One directory per commit**, named by the repository's canonical ID as
  the Hub returns it and the full 40-hex commit. A commit is what the
  manifest's identities describe; a branch moves, a short prefix is
  ambiguous in principle. Versions of a repository sit side by side
  (`<repo>@<a>`, `<repo>@<b>`), and nothing is ever overwritten by a later
  commit. `hf/` names the origin; another origin gets its own top-level
  directory when one is defined, without a new format version.
- **The repository's own file layout** (`hf download --local-dir` style),
  not the Hugging Face cache's `snapshots/` to `blobs/` links, which the
  share cannot hold. A revision may hold a subset of its repository's
  files (one quantization of a GGUF repository); the manifest lists them
  all and says which are present.
- **Names.** Repository IDs match `<org>/<name>` with each part
  `[A-Za-z0-9][A-Za-z0-9._-]{0,95}`. A repository path is refused (that
  file is not archived, and the tool says so) if it is absolute, has an
  empty, `.` or `..` component, a backslash, a control character or one
  of `:*?"<>|`, a component ending in a dot or space, a component over
  255 bytes, if it collides with another path of the revision when case is
  ignored, or if it ends in this format's reserved names
  (`llmp-archive.json` at the root, `.llmp-partial.`, `.llmp-tmp.`).
- **A revision is archived when its manifest exists**, and a file is
  archived when the manifest lists it as present. A directory without a
  manifest is a download in progress or abandoned; readers ignore it and
  `archive-catalog rebuild` lists it.

## Writing a revision

Optional local telemetry uses `LLMP_ARCHIVE_PROGRESS_SOCKET` (a Unix
datagram socket) and `LLMP_ARCHIVE_PROGRESS_SESSION` (the controller's
per-invocation nonce). `llmp_archive_progress.py` sends full snapshots of
the transfer's already-known counters and phase every half second, with
nonblocking sends and no NAS queries. Missing/full receivers drop reports
without interrupting downloads. The initial-library controller pushes
newline-delimited JSON status (operational protocol 1) once per second over
its host-local `status.sock`, only to connected viewers. Each viewer has
one bounded pending frame; a stalled viewer is disconnected. These are
ephemeral status channels, independent of stored archive schemas and the
checkpoint writer ownership protocol. The batch's read-only viewer and
offline behavior are documented in [the initial set](model-archive-initial-set.md).

The writer (`archive-model` now, `llmp install` later) works file by file
inside the final revision directory; nothing is renamed at the directory
level, since a directory rename on SMB fails while any file in it is open
and successive writers may add different subsets of the same revision.

1. **Identity first.** The revision's metadata comes from the Hub's
   revision API (`?blobs=true`): the commit, the card data, the gated flag
   and every file's size, Git blob ID and, for LFS files, SHA-256 (the
   resolve endpoint's `X-Linked-Etag`). A revision already archived reuses
   its manifest's identities, which were the Hub's at its first download.
2. **Each file** downloads to `<path>.llmp-partial.<writer>`, where the
   writer ID is `<host>.<pid>.<random>`, hashing as it writes (SHA-256,
   and the Git blob hash for non-LFS files). At the end the size and hash
   must match the Hub's (a file with neither a SHA-256 nor a Git blob ID
   from the Hub is refused, and the partial must hold exactly the recorded
   size); the file is flushed, then renamed to its final
   name, replacing any copy another writer finished meanwhile (identical
   by the same check). So a file appears under its final name only
   verified, and a final name always holds the recorded bytes. A write the
   share refuses (disk full, share gone) stops the file at once with that
   error and is never retried as a network failure; network failures
   resume with a `Range` request (eight failures in a row; any progress
   resets the count), and a server that ignores the range restarts the
   file. Every write first checks that the root marker is still there, so
   an unmounted share is never recreated on the local disk. A run that
   stops on an error or a signal sets its partial's modification time to
   the epoch (or removes it if empty), so the next run adopts it without
   waiting.
3. **One checkpoint writer, enforced by a lock (protocol 2).** Before
   reading revision state or writing any checkpoint, `archive-model/2`
   atomically creates `<root>/.llmp-writer/` on the NAS and writes its
   `owner.json` (kind `llmp-archive-writer`, format version 2). Exactly one
   host can create this directory. Its ownership spans all `--jobs` file
   workers, manifest and catalog publication. Another downloader waits,
   or exits 3 with `--no-wait`. Cancellation interrupts waits between NAS
   operations; an operation blocked in the filesystem must first return.
   Ownership is removed on normal completion and handled errors after all
   workers finish. The first SIGINT or SIGTERM requests this orderly stop;
   a second terminates the entire process without unwinding cleanup, leaving
   ownership in place. It must not release the guard while a worker can
   still write.

   A per-user host-local `flock` on the fixed
   `/tmp/llmp-archive-writer-<uid>.lock` file serializes local writers and
   recovery attempts,
   including runs using different paths to the same archive. That local
   file stays in place and must never be replaced or removed while in use.
   NAS file locks are never used for exclusion. Recovery requires the
   recorded machine, boot, user, PID and mount namespaces and local mutex
   device/inode
   match this process's environment, and that the owner's PID is absent
   or has a different `/proc/<pid>/stat` start time. A paused process
   remains live. Foreign, rebooted, incomplete or unconfirmed ownership
   is never stolen on a timeout; independently establish that its writer
   has stopped before removing an abandoned ownership directory. The
   owner record contains these identities, PID and start time, never
   credentials. Failure to release ownership fails closed.

   Once ownership is acquired, remaining checkpoint partials may be
   adopted immediately without a 15-minute wait. Their prefixes are
   reread and hashed before a ranged resume. Inline hashing, size check,
   flush and final pathname identity check remain; recurring identity
   checks every 64 MiB are omitted. On this NAS one such pathname query
   stalled for 19.59 seconds while HTTP bytes were already queued.

   **Upgrade:** stop all `archive-model/1` and other checkpoint writers
   before using protocol 2. Old tools do not acquire this lock and must
   never overlap new ones. Existing manifests, events, roots and partial
   names remain readable; no checkpoint is migrated or recopied. Future
   native install must take the same ownership guard (D-115).

   Staging into local scratch retains the original timed claims: a partial
   modified within 15 minutes is busy; an older one is renamed, touched,
   rehashed and resumed. Staging checks pathname identity every 64 MiB and
   before publication. These scratch claims depend on synchronized clocks;
   the archive's exclusive writer ownership does not.
4. **The manifest last.** Once the selected files are final, the writer
   reads the existing manifest, if any, merges (a file's `present` only
   goes from false to true, detection is redone for the variants it
   touched, unknown keys are kept), writes `llmp-archive.json.llmp-tmp.<writer>`,
   flushes it and renames it over `llmp-archive.json`. It then reads the
   manifest back; if a concurrent writer's rename lost its files, it
   merges again (up to five times). Two writers of different subsets of
   one revision therefore converge. Files already final but not listed (a
   writer that died between its file and its manifest) are listed by the
   next writer after a size check: their identities are the Hub's, and
   their bytes are hashed at use.
5. **The catalog.** The writer appends a `revision` event carrying the new
   manifest and compacts the catalog ([below](#the-catalog)).

Every write the archive makes, manifests, events, packs and the catalog,
goes through the same temporary name, flush and rename. A rename that
fails because a reader on another client has the target open (SMB sharing
rules; Linux clients open with delete sharing, others may not) is retried
with backoff.

## The revision manifest

`llmp-archive.json`, UTF-8 JSON with sorted keys, at most 16 MiB. Readers
refuse duplicate keys, non-finite numbers, another `format` or a
`format_version` they do not implement, and every field rule below; they
ignore unknown keys, and writers preserve them.

| Key | Content |
| --- | --- |
| `format`, `format_version` | `"llmp-archive-checkpoint"`, `1` |
| `origin`, `repo`, `commit` | `"hf"`; the canonical repository ID; the 40-hex commit |
| `hub` | What the Hub said at the first download: `requested` (the revision named, such as `main` or an EXL3 repository's `4.0bpw` branch), `last_modified`, `gated` (`false`, `"auto"` or `"manual"`), `private`, `license`, `license_name`, `license_link`, `pipeline_tag`, `library_name`, `base_models` (below), `tags` |
| `files` | Every file of the repository at this commit, sorted by path: `path`, `size`, `git_oid`, `lfs`, `sha256` (the Hub's for LFS files; computed at download, after the Git blob ID matched, for the others, and `null` while such a file is absent), `present`, and for a present file `fetched` `{time, host, tool, via}` (`via`: `hub`, `copy` or `move`) |
| `variants` | Each variant of the revision (below), sorted by ID, with what was detected from its files |
| `archived`, `updated` | UTC times of the first and the latest manifest write |

**Licenses never block** (D-087). The license string and the gated flag
are recorded for information; a gated repository needs the token, nothing
else.

**Base models and relations.** `hub.base_models` lists `{repo, relation,
source}`: the card's `base_model` (a string or a list) and
`base_model_relation`, else the relation the Hub's own
`base_model:<relation>:<repo>` tags state, else `null`. `relation` is one
of `quantized`, `finetune`, `merge`, `adapter` or `null`.

**Variants.** A variant is one loadable set of weights in a revision:

- GGUF: each `*.gguf`, with a split's files (`-00001-of-00003.gguf`) as
  one; its ID is the path without the split suffix and `.gguf`
  (`UD-Q2_K_XL/DeepSeek-V4-Flash-0731-UD-Q2_K_XL`). A file whose name
  starts with `mmproj` has role `projector`, the others `weights`.
- Otherwise each directory holding `*.safetensors` files is one variant,
  ID `*` for the root; a repository with `model_index.json` at its root
  (a diffusers pipeline) is one variant `*`.

Each variant records `files`, `role`, `present` (`true`, `"partial"` or
`false`), `bytes`, and `detected`, read from the files' own headers and
configuration, never from the card alone:

| Key | Content |
| --- | --- |
| `from` | `files` (every file present), `partial` (some), or `names` (none present: only the file name's hint) |
| `container` | `gguf`, `safetensors`, `mlx` (shards whose `__metadata__.format` is `mlx`), `exl3` (`quantization_config.quant_method` `exl3`) or `diffusers` |
| `architecture` | GGUF `general.architecture`; config.json `model_type`, else its first `architectures` entry; model_index.json `_class_name` |
| `scheme` | `gguf:<general.file_type>` (`Q4_K_M`, `IQ3_XXS`, ...); `modelopt:<quant_algo>` (hf_quant_config.json or `quantization_config`); `mlx:<mode>-<bits>bit-g<group>`; `exl3:<bits>bpw`; another `quant_method` as `<method>[:<bits>bit]`; an unquantized checkpoint's dominant dtype (`bf16`) |
| `encodings` | Bytes per GGML tensor type or safetensors dtype, from the headers |
| `bits` | GGUF: bits per weight over every tensor (bytes × 8 / elements); otherwise the scheme's nominal bits (NVFP4 4, MXFP8 8, MLX and EXL3 their configured bits, a dtype's width) |
| `parameters` | Elements over every tensor (GGUF, and unquantized safetensors); `null` where packing hides them |
| `quantizer` | GGUF `general.quantized_by`; ModelOpt's `producer` name and version; `mlx`; `exllamav3`; another `quant_method`; `null` |
| `name_hint` | The quantization a GGUF file name states (`UD-Q2_K_XL`), which a dynamic quantization's `file_type` may not |
| `notes` | Disagreements found: a ModelOpt NVFP4 scheme without `U8` codes and `F8_E4M3` scales, an MLX scheme without `scales`, EXL3 without trellises, a GGUF base model differing from the card's |

Header reads are bounded (a GGUF header at most 256 MiB, a safetensors
header at most 100,000,000 bytes, as the native readers bound them) and
read only headers, a few megabytes per file. Detection describes; it never
decides support, which is the importer's and the admission gate's
(D-109).

## Relations: families, models, variants, revisions

The catalog relates checkpoints in four levels:

- **Revision**: one archived commit of one repository. A repository's
  revisions over time are its versions (a re-upload, a fixed tokenizer,
  an EXL3 repository's bit-rate branches).
- **Variant**: one set of weights in a revision (above), with its
  container, scheme, bits and quantizer.
- **Model**: a repository whose weights are a model of their own: one
  with no base, or a finetune, merge or adapter of one. Every repository
  whose relation to it is `quantized`, followed transitively, is a
  quantization of that model, and its variants are listed under it. A
  repository with no recorded relation that holds a quantized variant
  (any GGUF other than F32/F16/BF16, or a quantization scheme) is taken
  as `quantized` of its single base (`source: "inferred"`).
- **Family**: the models connected by any base-model edge, named by the
  connected component's root (a repository with no base; the first by
  name if a merge gives several). So `Qwen/Qwen3-0.6B-Base`'s family holds
  the finetune `Qwen/Qwen3-0.6B`, whose quantizations
  `Qwen/Qwen3-0.6B-FP8` and `unsloth/Qwen3-0.6B-GGUF` are its variants.

Cards are often incomplete, so the owner can override: a `relation` event
sets a repository's base and relation, a `family` event gives a family a
label or joins a repository to one (DeepSeek V4 Flash and its `0731`
release, if the card does not say so). Overrides are the latest per
repository.

## The catalog

The catalog answers "what is archived, how do these relate, what was made
from it and where is it" from one file, without the Hub and without
walking every directory. It is derived data: the revision manifests are
the truth for checkpoints, the events for everything else.

**Events** are immutable JSON files, `catalog/events/<id>.json`, with ID
`<UTC yyyymmddThhmmssZ>-<host>-<pid>-<random>`, unique by construction, so
writers never contend for a name. Each holds `format`
(`"llmp-archive-event"`), `format_version` 1, `id`, `time`, `host`, `tool`,
`kind` and `body`:

| Kind | Body |
| --- | --- |
| `revision` | `manifest`: the full manifest just written, with `manifest_sha256` and `path` (relative to the root) |
| `revision-removed` | `repo`, `commit`, and `paths` (files removed) or `null` (the whole revision) |
| `import` | `repo`, `commit`, `variant`; `artifact` `{id, format, format_version, importer}` from the artifact's manifest; `host` |
| `install`, `uninstall` | `host`, `artifact_id`, `store` |
| `stage`, `unstage` | `host`, `repo`, `commit`, `path`, `variants` |
| `relation` | `repo`, `base` (`null` removes the override), `relation` |
| `family` | `repo`, `label` |

**Packs** gather events: `catalog/packs/pack-<sha256 of its content>.json`
holds `format` `"llmp-archive-pack"`, `format_version` 1 and `events`
sorted by ID. Packs are immutable too.

**Compaction** (`archive-catalog compact`, and after every write by the
tools):

1. List `events/`, then `packs/`, and read them all (a file that vanished
   since the listing restarts it, at most three times). Events first, so
   that an event another compactor packs and deletes in between is in a
   pack the second listing shows.
2. If there are more than 64 loose events or more than 8 packs, write one
   new pack holding every event read, then delete the loose events and
   packs that were read and are not the new pack itself.
3. Fold every event in (time, ID) order into the catalog and write
   `catalog.json` and `catalog.md`.

*Why this is safe without locks.* Events and packs are never modified.
A file is deleted only by a compactor that read all of it and has already
published, by rename after a flush, a pack containing all of its events;
so every event ever written stays in at least one existing file. Two
compactors may publish overlapping packs; a later compaction merges them,
and events are keyed by ID, so duplicates fold once. A content-addressed
pack name makes two identical packs one file, and no compactor deletes
the name it just published. `catalog.json` and `catalog.md` are caches of
a deterministic fold: a compactor that loses a rename-over race leaves a
catalog a moment stale, never wrong, and the next compaction refreshes
it. Nothing depends on SMB locks or on clocks agreeing beyond ordering
events for "latest" overrides, where hosts' NTP is enough.

**The fold.** Revisions are keyed by (repo, commit): the latest
`revision` event's manifest wins, except that a file present in any
earlier event stays present until a `revision-removed` names it (so a lost
manifest race cannot hide a file). `import` events accumulate by (artifact
ID, host); `install`/`uninstall` and `stage`/`unstage` are the latest per
(host, artifact) and (host, revision); overrides the latest per
repository.

**`catalog.json`** (`format` `"llmp-archive-catalog"`, `format_version`
1) holds `generated`, `events` (the count folded), and sorted arrays:
`revisions` (repo, commit, path, requested, license, gated, archived and
updated times, manifest SHA-256, bytes present and total, base models, its
variants with their detection, imports and the hosts that stage them),
`repositories` (each with its model, relation, family and revisions newest
first), `models` (each with its family, its bases and its variants across
repositories) and `families`, and `installs` (host, artifact ID, time).
`catalog.md` renders it by family, then model, then one table row per
variant and revision, with the installed artifacts by host at the end.
Both carry the time they were generated; neither is edited by hand.

**Rebuild** (`archive-catalog rebuild`): walks `hf/*/*@*/llmp-archive.json`,
validates each manifest, writes a `revision` event for every manifest
whose SHA-256 is not the catalog's, lists directories without a manifest
and partial files older than a day (left for the owner, not deleted), and
compacts. A lost or corrupt `catalog.json` is simply rewritten; an event
or pack that does not parse or validate is moved to `catalog/quarantine/`
and named, and its loss affects only that record. If `catalog/` is lost
entirely, rebuild restores every checkpoint and its relations from the
manifests; imports, installs and stages are restored by each host's report
(below), and overrides must be repeated.

### Imports, installs and stages

`stage-model` records `stage` events. *Not yet built:* `archive-catalog
report-host`, run on a host, which lists its staged checkpoints (by their
manifests) and installed artifacts (by `manifest.json`), matches each
artifact's `source` list (name, size, SHA-256; artifact-format.md) to the
archived revision whose files have those hashes, and writes the `import`,
`install`, `uninstall`, `stage` and `unstage` events that bring the
catalog to what is on the host. The artifact format needs no new field
for this: source hashes identify the revision. The native importer writes
the `import` event itself when it reads from an archive (N4). Host names
in events are owner environment, recorded for the owner's inventory; no
runtime reads them.

## Verification

Every byte used is checked against an identity recorded outside the
bytes themselves (D-009, D-054 as amended by D-114):

| When | Checked against | On mismatch |
| --- | --- | --- |
| Download | The Hub: size, and the LFS SHA-256 (the resolve `X-Linked-Etag`) or the Git blob ID for small files | The partial is deleted, the file named, the tool exits 1; nothing reaches the final name or the manifest. Rerun |
| Copy into the archive from a local directory (migration) | The same Hub identities | The same; the source is left as it was |
| Stage | The manifest: size and SHA-256, hashed while copying | The staged partial is deleted and the file named; the archive is not touched. `archive-model --repair` rehashes that revision's files on the archive and fetches the bad ones again |
| Peer copy | The manifest, rehashed on the peer (`sha256sum -c`) | The stage fails naming the file and the peer gets no manifest, so its copy is not staged; a rerun resends what differs (`rsync -c`) |
| Import | The manifest, hashed in the same pass that copies each byte ([below](#the-native-importer-and-llmp-install)) | No artifact is published; the error names the file |

A manifest is itself the identity record: the Hub's metadata at download.
It lives on the store, which any guest on the LAN could change; trusting
it is trusting the environment (D-073 and D-105 as amended), which the
owner chose. Comparing a manifest's identities with the Hub again is not
built yet; `archive-model --repair` rehashes a revision's files against
its manifest.

## Staging to the Sparks

Repeated testing reads a local copy. `stage-model`, run on the receiving
Spark, copies a revision's selected variants and common files from the
archive to the local model store, hashing while it copies (one NAS read,
about 118 MB/s), then writes the manifest last, byte for byte. With
`--peer`, it then copies the verified local files to the second Spark
over the direct link (`rsync` over SSH with AES-128-GCM to the peer's
direct-link address, environment.md), never a second NAS read, has the
peer rehash them with `sha256sum -c`, and copies the manifest last.

    ssh spark 'cd ~/src/llmp && ~/.local/bin/spark-job start --shared --name stage-dsv4 --timeout 3600 -- \
      ~/.local/bin/mise exec -- tools/stage-model unsloth/DeepSeek-V4-Flash-0731-GGUF \
        --variant "UD-Q2_K_XL/*" --peer spark-b --peer-address 10.100.208.2'

Staging is resumable (a rerun keeps files whose size and hash match), so a
run cut off by its timeout is simply rerun; its expected time is the
selected bytes at about 100 MB/s. It runs as a `--shared` job so it never
overlaps a timing run. A staged copy lives in the owner's store at
`~/.local/share/llmp/models/<org>/<repo>@<first 8 of the commit>/`, the
names the model tests already use; its manifest records the full commit.
`--adopt DIR` verifies an existing local copy against the archived
manifest and writes the manifest into it (the M3 checkpoints, below).

## The native importer and `llmp install`

D-109's rule applies: what a user runs is native, no Python.

- **`llmp import <path>`** (M3.6, N4) takes a revision directory, a
  variant inside it, or one of its files. When the revision has a
  manifest, the importer finds it (the directory, or the nearest parent
  up to the revision root), parses it as untrusted input under the rules
  above, and every source file it reads must be listed present; a file it
  needs that the manifest does not list is refused. Its complete weight-file
  set must equal exactly one variant's weight members; subsets and mixed
  independent weight containers are refused before publication. Without a manifest the
  importer works as it does today, hashing what it reads. From the archive
  the importer reads each needed byte once: it streams each source file in
  order, hashes it, and writes the byte ranges the artifact needs into its
  staging shards in the same pass, comparing the hash with the manifest at
  the end of each file; one mismatch abandons the import before
  publication. That replaces, for archive sources, the separate identity,
  copy and post-write source passes artifact-format.md describes, which
  would read a NAS source three times; the staged artifact is then
  verified locally as before. Files of other variants are never read. A
  staged copy is the same directory on fast storage, read the same way.
  The artifact format does not change: its `source` list already names
  each file's size and SHA-256, which tie it to the revision. After
  publishing, the importer writes an `import` event if the source was in
  an archive.
- **`llmp install <org/repo>[@rev] [--variant ID]`** (M5) resolves the
  revision to its commit, uses the archive's revision if the variant is
  present, otherwise fetches the missing files into the `checkpoints`
  role exactly as [above](#writing-a-revision) (same names, manifest and
  events, so the Python tools and the native install share an archive),
  then imports. Without a configured long-term store the role's default,
  `/var/lib/llmp/checkpoints`, takes the same layout. The archive location
  is never hard-coded: it is the `checkpoints` role (D-063, D-073).
- **HTTPS client: libcurl** (approved by the owner, 2026-10-10; D-114),
  in the install job only, never the runtime. Requests are the revision
  API and resolve URLs with `Range`, following redirects, with the token
  sent to the Hub's host only.
- **The token** comes from `HF_TOKEN` in the job's environment or a
  credential file the configuration names by path (secrets never inline,
  D-063; the key is added with install in M5). It is never logged,
  printed, written to the archive or passed on a command line.

## Prepared artifacts are not archived

D-054 allowed an artifact archive (the `archive` role). The archive holds
source checkpoints only, for now: weights stay in their source encoding
(D-108), so an artifact is about as large as the variant it came from and
copying one from the NAS costs the same read as importing from it; the
artifact format is experimental and changes version within milestones
(D-018, D-113), each change invalidating archived artifacts; artifacts
are per-profile (D-035); and a second Spark gets its artifact by peer
replication over the direct link (D-054), not from the NAS. Reopen when
the artifact format carries compatibility guarantees and an import is
measured to cost much more than a copy, or for nodes without the importer.

## Migrating what exists

Described here, not yet run; each step is resumable and verifies against
the Hub.

1. **The MLX offload** (`/mnt/llm/huggingface-mlx/<org>--<repo>@<commit>/`,
   four Vontra repositories). For each, `archive-model <org>/<repo>@<commit>
   --from <dir> --move --all-variants`: it reads the Hub's metadata, hashes
   each source file (one NAS read), and moves the verified files into
   `hf/<org>/<repo>@<commit>/` by rename on the same share, no data copy.
   Files not in the repository (`SOURCE.txt`, which the manifest
   supersedes) stay in the old directory, which is listed for the owner
   to delete.
2. **The Sparks' checkpoints** (environment.md's
   [M3 model store](environment.md#m3-model-store-2026-09-28) and the
   older `reference-models/`). For each, on `spark`: `archive-model
   <org>/<repo>@<full commit> --from ~/.local/share/llmp/models/<org>/<repo>@<prefix>
   --include ...` copies it to the archive verified against the Hub (one
   local read, one NAS write at about 80 MB/s); then `stage-model --adopt`
   on each Spark writes the manifest into the existing copy, which stays
   where the model tests expect it. `reference-models/` holds files
   without their repository identity; those are archived only once their
   repository and commit are known.

## Worked example

`unsloth/Qwen3-0.6B-GGUF` at one commit, one quantization downloaded.
Hashes shortened.

```text
/mnt/llm/checkpoints/hf/unsloth/Qwen3-0.6B-GGUF@1f2e…c0/
  .gitattributes  README.md  config.json  params
  Qwen3-0.6B-Q4_K_M.gguf
  llmp-archive.json
```

```json
{"archived": "2026-10-10T15:02:11Z", "commit": "1f2e…c0",
 "files": [
  {"git_oid": "52373fe2…", "lfs": false, "path": ".gitattributes", "present": true,
   "sha256": "8a1b…", "size": 1570,
   "fetched": {"host": "workstation", "time": "2026-10-10T15:01:02Z", "tool": "archive-model/1", "via": "hub"}},
  {"git_oid": "a0e1…", "lfs": true, "path": "Qwen3-0.6B-Q4_K_M.gguf", "present": true,
   "sha256": "6c41…", "size": 396705472, "fetched": {"…": "…"}},
  {"git_oid": "77c4…", "lfs": true, "path": "Qwen3-0.6B-Q8_0.gguf", "present": false,
   "sha256": "e3b0…", "size": 639446688}],
 "format": "llmp-archive-checkpoint", "format_version": 1,
 "hub": {"base_models": [{"relation": "quantized", "repo": "Qwen/Qwen3-0.6B", "source": "tags"}],
         "gated": false, "last_modified": "2025-05-02T10:11:12Z", "library_name": "transformers",
         "license": "apache-2.0", "license_link": "https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/LICENSE",
         "license_name": null, "pipeline_tag": "text-generation", "private": false,
         "requested": "main", "tags": ["gguf", "qwen3", "unsloth"]},
 "origin": "hf", "repo": "unsloth/Qwen3-0.6B-GGUF", "updated": "2026-10-10T15:02:11Z",
 "variants": [
  {"bytes": 396705472, "files": ["Qwen3-0.6B-Q4_K_M.gguf"], "id": "Qwen3-0.6B-Q4_K_M",
   "present": true, "role": "weights",
   "detected": {"architecture": "qwen3", "bits": 5.32, "container": "gguf",
                "encodings": {"F32": 205824, "Q4_K": 309657600, "Q6_K": 86842048},
                "from": "files", "name_hint": "Q4_K_M", "notes": [], "parameters": 596049920,
                "quantizer": "Unsloth", "scheme": "gguf:Q4_K_M"}},
  {"bytes": 639446688, "files": ["Qwen3-0.6B-Q8_0.gguf"], "id": "Qwen3-0.6B-Q8_0",
   "present": false, "role": "weights",
   "detected": {"from": "names", "name_hint": "Q8_0", "…": null}}]}
```

The catalog then shows, under family `Qwen/Qwen3-0.6B-Base`, model
`Qwen/Qwen3-0.6B` (a finetune of the base, which is not archived), the
row `unsloth/Qwen3-0.6B-GGUF · Qwen3-0.6B-Q4_K_M · gguf · gguf:Q4_K_M ·
5.32 bits · 397 MB · 1f2e…c0`. An import on `spark` adds that artifact's
ID, format version and importer to the row, and `install` events list it
under `spark`.

## Versions

Each file kind (`llmp-archive-root`, `-checkpoint`, `-event`, `-pack`,
`-catalog`) has its own `format_version`, all 1 today. A reader takes
exactly the versions it implements and refuses others by name; a writer
never rewrites a file whose version it does not implement. A new optional
key, a new event kind, a new origin directory, a new variant rule or a new
detected scheme needs no new version: readers ignore unknown keys and
event kinds (packs keep them), and writers preserve them. A change to a
key's meaning, a name rule or the stored layout needs a new
format version and a decision (D-062: a public on-disk surface). The catalog's
files are derived and may be regenerated at any version.
The checkpoint writer protocol has its own version (2, D-115); it adds
an operational lock without changing the stored version-1 schemas.
Events and manifests are never migrated in place: a new version's writer
adds new files beside the old, and the old stay readable while any tool
reads them.
