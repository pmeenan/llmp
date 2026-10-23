<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Prospective two-reply check

Preparation only. `pins.json` is disabled; do not collect until the corresponding head/state screen has passed and the root enables a separate reviewed receipt. This check does not collect another score teacher, run the reference again, or grant a full-answer claim.

The runtime child receives `LD_LIBRARY_PATH=/tmp/llmp-m35-orchestration/u4-export-pairs-r1/lib` from the pinned flat closure (receipt SHA `344179689778d15f19b78d9313b99fd50874344c8fa2596d24a81722c54021ee`); its ten SONAME hashes are in the disabled pins. The current target is the frozen candidate runtime SHA `95a2238acec146b9f27003804da35f3881becf83d5597feb5c74fcffc0f8be47`. Use one fresh runtime process per model on its assigned host, normal policy with no policy overrides, `debug_plans=false`, and the two sequential registered requests only. Launch through `tools/speed_gate.py`’s existing `Service`; use `greedy_capture.py`’s `collect_reply`/`parse_sse` for the existing OpenAI-compatible streamed HTTP client. The existing full `greedy_capture.py` CLI is not used as-is because its arm enum hard-pins older A/B/C/D runtime hashes and quality-receipt flow.

For each host, `run-replies.py` validates the statically disabled pins and external enablement receipt, verifies the candidate runtime and current artifact metadata, renders the pinned current-equivalent node template into a fresh work directory, starts one service on loopback, then submits request indices 0 and 1 serially. It retains each exact request JSON, raw SSE bytes, parsed text/digest/finish/usage, runtime log, start/stop receipts, and before/after runtime SHA. It requires successful retirement, `finish=length`, 128 completion tokens, and the registered prompt-token count. Candidate responses still require independent semantic review against retained A outputs; response hash equality is recorded but is not a pass requirement. Both reference responses stopped at the fixed 128-token boundary. The adapter calls the existing helpers: `config = config_text(template, share, work/data, 18114)`; `service = speed_gate.Service(runtime, config, work, 18114, {"LD_LIBRARY_PATH": pinned_library_dir})`; `service.wait_ready(420)`; for each pinned request it calls `greedy_capture.collect_reply(18114, body, out / f"reply-{index}.sse")`; then `service.stop(20)`. It does not call `cmd_quality` or `verify_score`.

## Qwen GGUF on Spark B

- Current artifact `b31c469e0fc129765fa91ce541cc1639661aa9bcb1ee5b885b8ffc33b6cc21c8`; config retains context 33792, prefill chunk 4096, four slots. The old template hash is `5ef058ac19c04c837a44f60323352883afa491c43afcd59decbbb94b4b872fe7`; the current-equivalent template changes only the artifact ID and hashes to `5d6402a236371aebd3af7c939874b813b4c209113f2f26dbc7ed5ccc9cf18d96`.
- Tokenizer and chat template remain external files, pinned in `pins.json` by SHA-256. The source-fold bridge says the old/current artifact difference adds only sampling defaults while preserving the exact index and payload.
- Retained reference: `/tmp/llmp-u4-greedy-review/raw.json`, member `qwen38-gguf-A-r2`, raw-container SHA `666e929a516f1c62eb07d8e4125e0944ada4cae5ce8c4c59843eb049cafb7d12`; record SHA `dc8bc6af579459abebe3ac36d32170c43f17978dd0bcbcb5a4ad222e2ba4030e`.
- Reference prompt-token counts are 66 for each request. Candidate reply digests, historical SSE digests, request hashes and full request bodies are recorded per prompt in the pins.
- Fresh output parent: `/home/pmeenan/.local/share/llmp/experiments/m35-u4-replies-r1/qwen38-gguf-candidate`.

## DeepSeek on Spark A

- Current target `01a5e5ba5a911528fd58268406ac6cff032e767fd09a20fd4c9cbf84f2043f13` and drafter `e14bf29f28c408a742ba6ab19879e415b1ec13019c0ed33b87b7802232a0bd90`; retain speculation, plain wave, context 262144, prefill chunk 4096, and four slots. The old template hash is `bd2bc1a663758f20d78956ed994d5276b77653cd7d94d4e6aaf43a0a8677ad67`; current-equivalent template substitutes only those two refreshed artifact IDs and hashes to `0fea53ab8f140e611ccd7fd6dcdedff2747007ff92d5aa97571b30983b7e7731`.
- Tokenizer and chat template are carried by the unchanged embedded GGUF metadata; no external tokenizer/template file is introduced. The sampling refresh changed target sampling defaults only, with payload/index unchanged. The current target and drafter manifest/index/spec hashes are pinned.
- Retained reference: `/tmp/llmp-u4-greedy-review/raw.json`, member `deepseek-A-r1`; record SHA `8030dc57edf5dc1890c645c64461d4c605e271fd7a63d16590aaa7eacbb50330`. The source report binds the raw-container hash, prior quality receipt, and each reply record.
- Reference prompt-token counts are 17 for each request; both historical replies contain reasoning only and end at 128 tokens with empty final content. Candidate replies must receive semantic review; this does not imply completed answers.
- Fresh output parent: `/home/pmeenan/.local/share/llmp/experiments/m35-u4-replies-r1/deepseek-candidate`.

The exact request JSON for both models is embedded in `pins.json`: prompt 0 is “Explain, step by step, how a hash table handles collisions.”; prompt 1 is “Write a short story about a lighthouse keeper who finds a map.” Both use `temperature: 0`, `max_tokens: 128`, `stream: true`, and `stream_options.include_usage: true`. Request model names are family-specific and pinned. No retries, changed prompts, concurrent requests, or reference replacement are permitted.

