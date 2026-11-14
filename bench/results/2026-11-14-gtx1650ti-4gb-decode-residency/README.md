# GTX 1650 Ti 4GB: 15.5 / 15.4 tok/s with a resident output head

Third local tuning pass, 14 November 2026. This is an opt-in **decode-focused text-chat profile**,
not a new default, a standard-protocol result, or a universal speedup.
It uses the already prepared ISTA-DASLab Flash-Next IQ3_XXS files. No model was downloaded or replaced.

**The tradeoff matters:** generated output is faster, but the longer input takes more time and
free VRAM is lower. Both comparison arms use HC Q8. This does not establish quality equivalence
with the original BF16 HC weights or with batched prompt processing.

## Hardware, build and method

- GTX 1650 Ti 4096 MiB, 3729 MiB visible to CUDA; Intel i5-10300H, 4 cores / 8 threads;
  64 GB installed RAM, about 62.6 GiB usable; NVMe; Linux 7.3.4-1-cachyos.
- NVIDIA driver 620.70.06, CUDA 13.4.92, GCC 16.2.1. Release, sm_75, portable source build,
  engine version 0.1.59.1. No clock, power, OS or driver settings were changed in this pass.
  Clocks were not locked. The desktop and coding tools remained open.
- Source: `e65fa94f` plus this change. Headline arms use the **same frozen final binary**,
  SHA256 `9a8fb68360b29f46c6a5e028f64d160440c0fdfeb18c5dd2846fbade0ae5c117`.
  See `provenance.json` for file hashes, build settings and the distinction between earlier prototypes.
- Model: Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS, the two existing GGUF shards and `packs/iq3_xxs`.
  Original model revision and complete GGUF hashes were not re-verified in this pass. Download-marker
  hashes, if recorded in provenance, are **not newly computed full-file verification**.
- `tools/bench_gtx1650.py`, three sequential rounds per arm, temperature 0, thinking off,
  non-streaming, prompt checkpoints disabled. Every measured request reused **zero** prompt tokens.
  No warm-up was discarded; model loading is outside request timings. Arms were not randomized or interleaved.
- The two decode cases generate 64 tokens each. The filler/capital case has 231 input tokens
  and generates two (`Paris`); its decode number is not a sustained-throughput test.
- This is a comparison of profiles, not an isolated kernel experiment. The window prompt reader,
  head placement, weight cache, reserve and PCIe share all change. Output text can differ.

## Final three-round results

Medians; throughput is the engine's timing, request latency is client wall time. This is **not TTFT**.

| Metric | `hc-q8` / `baseline_verified` | `hc-q8-decode` / `decode_verified` |
| --- | ---: | ---: |
| English decode, tok/s | 9.2 (8.9–9.2) | **15.5 (15.4–15.5)** |
| Arabic decode, tok/s | 9.2 (9.2–9.3) | **15.4 (15.4–15.5)** |
| English request, seconds | 8.7590 | 6.0429 |
| Arabic request, seconds | 8.7334 | 6.0794 |
| 231-token prompt, tok/s | **33.8** | 22.2 |
| Same filler/capital request, seconds | **7.0540** | 10.5290 |
| Free VRAM at startup, MiB | **277** | 149 — LOW warning |
| Context / KV | 8192 / int8 | 8192 / int8 |
| HC precision | Q8_0 | Q8_0 |
| Drafter | none | none |
| CPU workers | 3 | 3 |

Decode improves about **68%** on these output workloads. The English request is about 31% shorter.
The filler/capital request is about **49% longer**. Do not describe this as an improvement for every request.
The earlier 8.8 tok/s result is historical; the same-day final baseline is 9.2, not 8.8.
All raw request bodies, responses, individual timings and errors are preserved in `raw/*.results.jsonl`.

Free memory is a startup snapshot, not peak memory. The new profile captured its two-token prompt graph
and one-token decode graph after that snapshot. Only 135 MiB was reported free before the latter capture.
The 128 MiB planning reserve is **not** a guaranteed free-memory floor. Other GPU applications may make
this profile fail. There is no claim about a long soak or simultaneous GPU workloads.

## Why it helps

The diagnostic `hcq8_timed` measured roughly 36 ms/window in the host-resident output head.
The head contains about 416 MiB and was read over PCIe for every output row. The attempted GPU-head
configurations could not allocate the batched prompt buffers beside it, even with smaller prompt chunks.

`STRATA_DECODE_ONLY_PROMPT=1` skips initialization of the batched prompt buffers and their cuBLAS handle.
It reads every text prompt through the engine's **existing verify-window reader** instead.
There is no new matrix kernel, vocabulary restriction, extra expert skipping, or new requantization here.
The existing route-tail skip remains on at its inherited setting; it was not made more aggressive.

With the prompt buffers omitted, the full head fits in VRAM. The profile also caches a prefix of the mapped
canonical arena: **158.41 MiB / 261 whole tensors**, unchanged bytes. The diagnostic `decode_timed`
measured about 2.8 ms/window in the now-resident head. Diagnostics have overhead and are not headline runs;
their stage totals also include prompt windows in this mode. Do not add their numbers as an isolated speedup.

Additional settings relative to `hc-q8`:

```text
STRATA_DECODE_ONLY_PROMPT=1
STRATA_WEIGHT_CACHE_MIB=160
STRATA_HEAD_HOST removed; STRATA_HEAD_HOST_OFF=1 (do not silently spill the head)
--short-read 8192
--vram-reserve-mib 128
--pcie-frac 0.5
```

HC Q8, native pooled weights, 8K context, int8 KV, three workers, expert cache 8 and no drafter remain.
`--prefill 128` remains a nominal engine argument, but this mode does not allocate or run that batched path.
Normal prompt checkpoints can still be used. The four previous launcher profiles and their stdout/stderr
were checked byte for byte against `e65fa94f`, with and without `--no-prompt-cache`: **8/8 combinations matched**.

The engine checks the mode's restrictions: one discrete CUDA GPU, no batch, vision, MTP/DFlash2,
peer/helper GPU or pipelined windows, no prompt-buffer borrowing, and a short-read limit covering the context.
Checkpoint-reread and pipeline-switch diagnostics are rejected. If a request nevertheless requires the
batched reader, it returns an error instead of using uninitialized buffers.

## Reproduce the headline comparison

Use a Python environment with Strata's server dependencies and an already prepared model directory.
Build using [docs/GTX1650.md](../../../docs/GTX1650.md). Run each arm from a fresh engine, one at a time:

```sh
/path/to/python tools/run_gtx1650.py --data-dir /path/to/Strata-data \
  --profile hc-q8 --port 18080 --no-prompt-cache
# In a second terminal:
python tools/bench_gtx1650.py --url http://127.0.0.1:18080 \
  --label baseline --rounds 3 --out /tmp/hc-q8-baseline.jsonl
# Stop that server before starting the other arm.
/path/to/python tools/run_gtx1650.py --data-dir /path/to/Strata-data \
  --profile hc-q8-decode --port 18080 --no-prompt-cache
python tools/bench_gtx1650.py --url http://127.0.0.1:18080 \
  --label decode --rounds 3 --out /tmp/hc-q8-decode.jsonl
```

For normal chat omit `--no-prompt-cache`. Unset exported `STRATA_HEAD_HOST` first. The launcher does not
clear every ambient tuning override, so use a clean tuning environment and inspect the generated config.
Settings in `raw/` use `$REPO` / `$DATA` placeholders; substitute real paths before using them directly.

## Experiments and failures

`experiments.json` lists every attempted configuration and its individual short-run results, not only the winner.
Engine logs and settings are in `raw/`. The omitted prefill-pool prototype is retained as a patch in `support/`;
it is **not part of the final engine**.

- The original HC Q8 check gave 9.2 tok/s. Prefix caching with a host head gave about 10.4.
- Raising PCIe share to 0.8 or 1.0 with the host head made decode worse (about 8.3 / 7.3).
- GPU head with prompt chunks 32/64/128, with and without existing compact-scratch switches, failed to start.
- A prototype pooling the prompt allocations saved only about 12 MiB at chunk 128 and did not make the GPU-head
  setup fit. It was removed, not promoted as a performance fix. `CUBLAS_WORKSPACE_CONFIG=:0:0` and `:16:8`
  also failed to make that configuration fit; neither is part of the profile.
- The first window-only attempts kept the 512 MiB reserve and ended with no usable expert-cache slots.
  Lowering the explicit planning reserve was necessary; the failure was not hidden.
- Window-only with no canonical cache gave about 13.1; a 96 MiB cache about 13.6; 160 MiB about 14.6 with automatic
  PCIe selection. A fixed share of 0.5 gave about 15.5; 0.6 was slower. Five/seven-worker trials were inconsistent
  or slower and were not retained. These are one-round exploratory values, not final medians.
- Window-only with the unchanged-HC `cached` profile gave about 6.3 and slower prompts; not recommended as an upgrade.
- `baseline_final`, `decode_final` and `baseline_recheck` are earlier confirmation runs. A guard-only rebuild happened
  around those runs. They are retained, but **excluded from the headline table**. `*_verified` use the same frozen
  final binary named and hashed above.

## Validation

**Passed:**

- `cmake --build build-gtx1650 --target strata mapped_weight_arena_test mapped_weight_cache_test native_dense_ple_key_test -j3`:
  CUDA sm_75 Release build. The final guard-only rebuild is recorded separately in `raw/final-rebuild.log`.
- `ctest --test-dir build-gtx1650 -R '^(mapped_weight_(arena|cache)_test|native_dense_ple_key_test(_pooled)?)$' --output-on-failure`:
  all four existing CUDA allocator/PLE regression tests passed. These are not a language-model quality evaluation.
- `python tools/test_run_gtx1650.py` and `python tools/test_bench_gtx1650.py`: four Python tests passed,
  including the new profile settings and rejection of an exported host-head override.
- Eight old-profile/cache combinations produced byte-identical config files, stdout and stderr against `e65fa94f`.
  See `raw/legacy-check.json` and `support/legacy-check.py`.
- Final same-binary comparison: 18/18 HTTP requests completed, zero prompt reuse and zero draft tokens.
- `decode_verified`: **11/11** small sanity checks passed: arithmetic, multiplication, Arabic, a capital,
  strict JSON, sorting, restricted Python square-function execution, greeting, a tool call, its result,
  and near-limit recall. The latter read **8,169 prompt tokens**, generated seven, reused zero, and returned
  `violet-742`. It took **380.533 seconds** end to end; configured context was 8,192, not reduced.
- Controlled HTTP fixture: old engine, new disabled, and new enabled produced identical 64-token messages
  and byte-identical **43-row prompt log-probability files**. All arms forced the head to host RAM and used
  the same window-only prompt reading, fixed PCIe 0.5, frozen adaptation and no prompt reuse. This isolates
  omission of unused buffers. It is **not** parity against batched reading, all logits, or arbitrary prompts.
  See `raw/parity-summary.json`, responses and `*.logprobs`.
- Normal prompt checkpointing: two streaming turns passed; the second reused **29 tokens** and recalled
  `amber-29`. Raw SSE events and single-request first-token timings are in `raw/decode_live.followup.json`.
  Those two short requests are not a repeated TTFT benchmark.

**Failed / rejected during exploration:** the startup failures and slower trials listed above are preserved,
not counted as successful throughput samples. No final benchmark, sanity, or controlled-fixture request failed.

**Not run / not established:** HIP and Metal compilation/runtime (toolchains unavailable), Windows, other GPUs,
an exact 8,192-position boundary, more than 8K context, parallel calls, a long soak, competing GPU applications,
broad language/coding/tool-quality evaluations, and numerical parity between window and batched prompt reading.
The separate SYCL engine uses its own migrated program file and was not changed or built.
Only the invalid short-read guard is exercised as a negative startup integration check; the other unsupported
combinations are documented code restrictions, not claims of runtime tests on those backends.

## Other projects consulted and comparison limits

- [Captain-Tripps/flash-next-dual-gtx-1070](https://github.com/Captain-Tripps/flash-next-dual-gtx-1070),
  `d2f3baca1774a326fd61a95297ffc055b0edb171`: read its README, particularly the PCIe-share experiments and
  residency/drafter tradeoffs. It uses **two 8 GB cards**, a different CPU, context and drafter.
- [Niko1221/Strata #4920](https://github.com/Niko1221/Strata/pull/4920): GTX 1070 Ti **8 GB**, Ryzen 9 5900XT,
  IQ2_XS, MTP and a different benchmark workload. It is not a same-settings comparison to this laptop.
- [liaoyangr8-ai/wzy-flash-serve](https://github.com/liaoyangr8-ai/wzy-flash-serve),
  `10513b156abc94bcd8ad2948b7da5362e83720c4`: README only; two modified RTX 2080 Ti cards with 22 GB each,
  a different model pack and tensor-core path. Its source snapshot warns that it is not runtime-tested as packaged.

These reports informed what to investigate; no external kernels were copied or benchmarked here.
The user did not supply the specific GTX 1070 15 tok/s report, so its conditions remain unknown.
An older product name does not establish a lower performance ceiling. None of the external speed claims
were reproduced in this pass.
