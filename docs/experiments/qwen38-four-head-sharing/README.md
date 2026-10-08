<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Four-request full target-head sharing at fixed depth one

Spark B (`spark-56f5`), 2026-10-02. One paid head OFF/ON/OFF C4 screen with
four funded slots and fixed draft depth one in every arm. Selected HC and
ragged sharing, paired general/MTP products, the 47,172-entry draft head,
weights, template, context 33,792 and prefill chunk 4,096 remain unchanged.
Each complete frozen common-v2 request reports 8,256 prompt tokens, zero
cached tokens and 256 completion tokens ending in `length`. An unrelated
one-token weight prime precedes each fresh service and is excluded. Wall
covers buffered HTTP submission through complete replies, including prefill,
generation and queueing.

| Arm | C4 wall, s | Aggregate tok/s | Latencies u0/u1/u2/u3, s |
| --- | ---: | ---: | --- |
| Head OFF before | 37.387063609 | 27.389152855 | 36.477727 / 37.387064 / 36.931407 / 36.250954 |
| Head ON | 35.088279039 | 29.183534446 | 34.293824 / 35.088133 / 34.606729 / 33.799413 |
| Head OFF after | 36.862022510 | 27.779267937 | 36.460465 / 36.861812 / 36.461226 / 35.718656 |

Mean OFF duration divided by ON duration gives **+5.803260%** throughput.
OFF bookend duration movement is **−1.404339%**. ON is faster than both
OFF observations. This is one representative serving screen, not a confidence
interval, production default or broader quality qualification.

| Arm | Completed verify waves | Width 1/2/3/4 counts | Four-head groups | Combined head columns | Paid head packed bytes |
| --- | ---: | --- | ---: | ---: | ---: |
| OFF before | 161 | 9 / 5 / 2 / 145 | 0 | 0 | 0 |
| ON | 161 | 10 / 3 / 3 / 145 | 145 | 1,160 | 26,726,400 |
| OFF after | 159 | 8 / 1 / 5 / 145 | 0 | 0 | 0 |

Every arm has maximum one draft pass, two verify rows per request and eight
verify rows per wave. Counters include the untimed prime. ON actually shares
145 four-request, eight-column full target heads. Each group pays three F32
concatenations totaling 184,320 bytes, one ordinary BF16-weight/F32-input/F32-output
MMF and four bounded output views; the combined output is 7,946,240 bytes.
All four original head selectors and the combined selector are authenticated
as ordinary MMF before redirection and after final placement. Exact original
coverage, one combined product, complete view coverage, independent logits
and argmax consumers are checked. One-row, non-four-active and unsupported
heads retain the existing path. MXFP8 products remain paired at at most eight
columns; no routed or numerical kernel changes are made.

Both arms conservatively fund three times each scalar head input and one
scalar head output per slot, covering the prefix-concat intermediates. The
actual startup envelopes are identical: fixed catalog 7,683,411,756 bytes,
shared activations 6,096,420,864, scratch 884,998,144 and host input 35,651,584.
These are guarded budgets rather than measured peaks. Existing state and
rollback ownership are unchanged; no diagnostic state copies run per token.

All four complete ON public replies preserve OFF text, reasoning, usage and
finish exactly. OFF bookends also repeat exactly. Native generated IDs are
unavailable through this API, so public equality does not establish native
token-ID equality or completed-answer quality. Complete raw/parsed responses
and terminal records remain external.

The private build finishes rc0 in 39 seconds. Actual Ninja dependency records
identify all three linked consumers of the changed wave header: wave plan,
runner and serving. Both arms rebuild all three with the same private layout
and funding flags; emitted dependency files authenticate use of the private
header. Only the ON wave-plan object enables the rewrite. Copied engine and
serving archives replace those members under the qualified 37-input closure;
SDK cuBLAS paths and payloads remain authenticated. The selected warm source,
build and ordinary runtime stay unchanged.

The three-service screen completes rc0; all three services return rc0 and are
reaped, all three clients succeed and all 15 HTTP terminals match complete
responses. Final strong memory/GPU/container/native-model probe reports
117.212 GiB clear, and Spark B reports zero GPU jobs/compute processes. No
matrix, rescue screen, production default, suite or workstation script runs.

Provenance:

- Builder SHA256 `4adef2ee883307d698005045b4d5f668c97297b5ec894a201e8f09b10a3b11ea`;
  build receipt `d749f579feb4377a111d6eb19579f6641f70cc865dd71ef9ae5aba5e01527b68`.
- Controller SHA256 `f4750aa4275bdc84a36aa8bf951048ce090e5c2bbf5b492a93d3bab95f024959`;
  screen receipt `5d502348345b6d7ffeeb237e2d213e7d8d109a495f509b120e238a682ebb9d55`.
- Private source SHA256: wave header `04486b750f171a27615ac60eb9f1287f333193b0d1bb39fa8b4c21e71ca21ea7`,
  wave source `39cdb617710d61fcf3f04e1cbc75b50d6207e554de324b5953f6e1fd4910f8a0`,
  runner `c210e35f7a20358c071617a47861adb850388825b04182a17928b07349722b3c`,
  serving `7893699bbaa6a1a7080bc8f9ab11c49a02850978635b7b3470b02d164aec838f`.
- Same selected qualification `b07fcb45`, target `5802ceb0`, ordinary runtime
  `ec0914c1`, artifact receipt `02fc48d8`, client `4f76adb8`, complete matched-v2
  input receipt `d5a35e63` as the preceding row-budget/reference screens.
- Supervisors `3900831` (build), `3901707` (screen), each bounded at 600 seconds.
- Raw local records: `/home/pmeenan/scratch/m3-qwen-four-heads-records/{build-r1,http-r1}/`.
  Frozen kit and full private profiles remain on B under
  `/home/pmeenan/scratch/m3-qwen-four-heads/`.
- Prior row-budget source, kit and records remain separate and unchanged.
