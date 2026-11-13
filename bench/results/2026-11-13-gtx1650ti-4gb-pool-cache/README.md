# GTX 1650 Ti 4GB: pooled weights and a small exact-byte cache

Second local tuning pass, 13 November 2026. This follows the [first report](../2026-11-13-gtx1650ti-4gb-tuning/README.md).
It reuses the downloaded IQ3_XXS model. No model was downloaded, converted or replaced.
These are exploratory measurements on one laptop, not the standard community benchmark protocol.

## Hardware and method

GTX 1650 Ti 4096 MiB, 3729 MiB visible to the engine; i5-10300H, 4 cores / 8 threads;
about 62.6 GiB usable RAM; NVMe; Linux 7.3.3-1-cachyos; driver 620.70.06.
Build: CUDA 13.4.92, GCC 16.2.1, Release, architecture 75. Engine version: 0.1.59.1.
The desktop and coding tools stayed open. System swap/zram was in use. No clock or power settings were changed.
See [provenance.json](provenance.json) for source revisions, build settings and hashes.
Full GGUF hashes and the revision of the already downloaded model were not recorded.

The headline comparison uses the **same new engine binary** for all three profiles.
Each arm ran three sequential rounds with `tools/bench_gtx1650.py`, temperature 0, thinking off,
non-streaming, and `--prompt-cache 0`. No warm-up was discarded. Every measured request reused zero prompt tokens.
English and Arabic requests generated 64 tokens. The filler/capital case had 231 prompt tokens and generated two.
Request bodies, responses, round timings and summaries are in `raw/*.results.jsonl`.
Arms were not randomized or interleaved. Do not interpret small differences as statistically established gains.

## Three-round results

Medians. Fresh-prompt latency includes both prompt processing and generation; it is not TTFT.

| Metric | `fast` / `fast_new` | `cached` / `cached_final` | `hc-q8` / `hcq8_final` |
|---|---:|---:|---:|
| English decode, tok/s | 5.9 | 6.3 | 8.8 |
| Arabic decode, tok/s | 5.9 | 6.3 | 8.8 |
| English request, seconds | 12.7445 | 12.1200 | 9.1318 |
| Arabic request, seconds | 12.7035 | 12.0869 | 9.0880 |
| 231-token prompt, tok/s | 35.4 | 35.8 | 33.8 |
| Same filler/capital request, seconds | 6.8588 | 6.7818 | 7.0616 |
| Free VRAM after startup, MiB | 256 | 274 | 276 |
| Extra HC requantization | No | No | **Yes: Q8_0** |
| Configured context / KV | 8192 / int8 | 8192 / int8 | 8192 / int8 |

`cached` improved these decode medians by about **7%** over the same-binary baseline.
A separate three-round run of the previous published binary (`fast_old`) gave 5.8 English / 5.9 Arabic tok/s,
consistent with the earlier result. The 5.8–5.9 baseline spread is real run-to-run variation, not a default-path speed claim.
Prompt throughput for `cached` was essentially unchanged.

`hc-q8` improved decode by about **49%**, but it changes HC weight precision and also changes head placement,
expert-cache size and prefill chunking. It is **not an equal-precision or isolated quantization comparison**.
It was slower on the 231-token prompt. It remains a separate opt-in, not the recommended unchanged-precision path.
The available VRAM figures are snapshots, not minimum free memory or peak usage. Even 274–276 MiB is a small margin.

## What changed

Profiling (`timed`) attributed about 99 of 173 ms/token to HC operations reading mapped host weights.
That is a diagnostic with instrumentation overhead, not a benchmark arm for headline numbers.

- `STRATA_NATIVE_POOL=1`: allocate immutable native weights using CUDA's stream-ordered memory pool.
  Synchronize allocations before publishing pointers to nonblocking consumers. Normal cleanup uses `cudaFree`.
  The default still uses `cudaMalloc`. NativeDense rejects this opt-in on unsupported runtimes/backends.
- `STRATA_WEIGHT_CACHE_MIB=N`: copy a prefix of the mapped arena into VRAM before sessions and graphs bind pointers.
  Copy whole tensors only; finish the copy before changing the table. Keep the remainder mapped in RAM.
  `cached` requests 160 MiB and actually copies **159.94 MiB / 53 tensors**. It does not change those bytes or requantize them.
  It requires mapped host weights, one discrete CUDA GPU, and a budget from 0 to 1024 MiB.
- The launcher adds `cached` and `hc-q8`. It does **not** change its default, `stable`, or the existing `fast` settings.
  Generated settings **and stdout** for both existing profiles were compared byte for byte with commit `5d42aa08`.

| New profile | Additional settings relative to `fast` |
|---|---|
| `cached` | `STRATA_NATIVE_POOL=1`, `STRATA_WEIGHT_CACHE_MIB=160` |
| `hc-q8` | Pool on, `STRATA_HC_Q8=1`, `STRATA_HEAD_HOST=1`, expert cache 8, prefill 128 |

Both retain 8K context, int8 KV and three workers. Actual expert-cache slots were 42 for `fast`/`cached`
and 10 for `hc-q8`; the requested `--expert-cache` values are not the final slot counts.
Both retain the existing CPU prefill sharing and route-tail skipping.
The HC Q8 path already existed in the engine; this work makes it fit the tested 8K setup with a larger memory margin.
No claim is made about broad answer quality, vision, coding evaluations or other model families.

## Experiments not selected

| Label | Rounds | Finding |
|---|---:|---|
| `fast_profile` | 1 | Initial previous-binary check, 5.9 English decode tok/s. |
| `timed` | 1 | Profiling isolated the large mapped HC read cost. Do not mix its timing with uninstrumented medians. |
| `async_shim` | 1 | A local `LD_PRELOAD` prototype reduced allocation waste, but decode stayed 5.8 tok/s. Not a production solution. |
| `ngram` | 1 | Suffix/lookup drafting gave about 5.9 tok/s, with low draft acceptance. Not retained. |
| `pool_cache192` | 1 | 6.3 tok/s, but only 242 MiB free and a LOW-memory warning. Prefer the 160 MiB budget. |
| `pool_cache160` | 1 | 6.3 tok/s and 274 MiB free; promoted to a separate three-round test. |
| `pooled_hcq8` | 1 | 8.9 tok/s, but only 182 MiB free. Not retained. |
| `pooled_hcq8_room` | 1 | Expert cache 8 / prefill 128: 8.8 tok/s, 276 MiB free; promoted to a separate three-round test. |

Configurations, results and engine logs are retained, including the less successful variants.
The allocation shim source in `support/` documents an experiment; do not add it to the normal launcher.

## Correctness, context and normal conversation checks

**Passed:**

- Four targeted CUDA tests: `mapped_weight_arena_test`, `mapped_weight_cache_test`,
  `native_dense_ple_key_test`, and its pooled variant. Coverage includes whole-tensor relocation, mixed host/device reads,
  metadata/boundaries, and pooled allocations consumed by a nonblocking graph stream.
- Four Python tests across the launcher and benchmark test scripts. Launcher cases cover all four profiles and legacy-file preservation.
- Controlled old/new-disabled/cached diagnostics matched all **16 window-hash rows** byte for byte.
  This uses a teacher-forced continuation, frozen adaptation and CPU prompt sharing disabled.
  It is one diagnostic, **not general output or sampling-law equivalence**. See `raw/hash-check.json` and `raw/hash_*.hashes`.
- Both new profiles passed **11/11** sanity checks each: arithmetic, Arabic, a capital, strict JSON, sorting,
  a small restricted-AST Python square function, greeting, a real OpenAI-style tool call, consumption of its synthetic result,
  and near-limit recall. This small set is not a quality benchmark.
- Near-limit recall used **8,169 prompt tokens + 7 generated tokens**, zero reused tokens, and returned `violet-742`.
  `cached`: **230.219 s**, 35.7 prompt tok/s, 6.2 decode tok/s.
  `hc-q8`: **247.304 s**, 33.1 prompt tok/s, 8.7 decode tok/s.
  These are one synthetic request each, not a long-context evaluation.
- With normal checkpointing restored, both profiles passed a two-turn streaming conversation and recalled `amber-29`.
  The second turn reused **29 of 55** prompt tokens. The raw SSE events include actual TTFT, but two requests without
  an equivalent baseline are not evidence of a TTFT improvement. `cached` was left serving on localhost:8080.

**Failed setup attempt:** the token-ID diagnostic first used system Python, which lacked `regex`, and did not launch an engine.
It then passed with the already installed Control Python. No dependency was installed to work around this.
No final benchmark request or sanity case failed.

**Not run:** soak tests, parallel load, broad quality/coding evaluations, vision, other GPUs or operating systems,
and HIP/SYCL/Metal builds. Their compilers were unavailable here. This is not cross-backend validation or an upstream review request.
The near-limit test is not an exact 8192-position boundary test.

## References: ideas reviewed, not other engines benchmarked

- [basalt at b422e200](https://github.com/jesdga95/basalt/tree/b422e200dde88d0b781d8c71c6709b9e0a99af0a):
  [README](https://github.com/jesdga95/basalt/blob/b422e200dde88d0b781d8c71c6709b9e0a99af0a/README.md) and
  [adaptive settings](https://github.com/jesdga95/basalt/blob/b422e200dde88d0b781d8c71c6709b9e0a99af0a/docs/configuration/adaptive.md).
  It targets Blackwell sm_120/CUDA 13. Its kernels are not a drop-in solution for this Turing card.
  We reviewed residency, transfer budgets and adaptive settings instead of copying a large-GPU configuration.
- [llamAmpere at a5d8684c](https://github.com/JakeATX/llamAmpere/tree/a5d8684ce8e833fb077f8210495cd1a905ed3a45):
  [MoE arena](https://github.com/JakeATX/llamAmpere/blob/a5d8684ce8e833fb077f8210495cd1a905ed3a45/src/llama-moe-arena.h) and
  [short-follow-up release notes](https://github.com/JakeATX/llamAmpere/blob/a5d8684ce8e833fb077f8210495cd1a905ed3a45/docs/llamampere-v0.6.3.11/RELEASE_NOTES.md).
  We reviewed allocation packing and first-token/follow-up behavior. The engine targets Ampere;
  its architecture-specific kernels and custom model formats were not transplanted.

These statements about the reference projects come from source review, not tests of their runtime or performance.
Both repositories were inspected at the pinned revisions. Neither was built or run, and no alternative model was downloaded.
The Strata changes are a local implementation using CUDA's allocator and Strata's own weight table.

## Reproduce

Build the new engine and tests as described in [docs/GTX1650.md](../../../docs/GTX1650.md).
For each profile, start one server, wait for health, and run the benchmark. Stop it before starting the next.
Use a new result filename every time:

```bash
python tools/run_gtx1650.py --data-dir /path/to/existing/data --profile cached --port 18080 --no-prompt-cache
# Another terminal, from the repository root:
python tools/bench_gtx1650.py --url http://127.0.0.1:18080 --label cached --rounds 3 --out /tmp/cached-new.jsonl
```

Repeat with `--profile fast` and, only when accepting the precision change, `--profile hc-q8`.
For ordinary chat omit `--no-prompt-cache`. The default `stable` and original `fast` remain available.
Check for exported tuning variables such as `STRATA_HEAD_HOST` or `STRATA_HC_Q8` before reproducing a profile.
The launcher does not sanitize all ambient engine settings.

Support scripts preserve the local test method; they are not a new supported launcher.
Paths in them and in raw logs/configs are replaced by `$REPO`, `$DATA`, `$CONTROL`, `$HOME`.
Replace these placeholders before trying to run a support script. The original weights and built binaries are not included.
