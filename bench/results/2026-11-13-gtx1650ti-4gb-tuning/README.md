# GTX 1650 Ti 4 GB: opt-in tuning, not a standard benchmark

Measured locally on 13 November 2026. This compares working configurations of the **same
patched engine and same model files**, not this fork against a new upstream release.
These are small, sequential experiments on a desktop laptop, not a model-quality evaluation
or an implementation of the draft standard benchmark protocol.

## Hardware and model

GTX 1650 Ti 4 GB (sm_75), i5-10300H (4 cores / 8 threads), 64 GB installed RAM,
Linux 7.3.3-1-cachyos, NVIDIA driver 620.70.06, nvcc 13.4.92, GCC 16.2.1.
The engine reports 3,729 MiB total usable VRAM. PCIe supports Gen3 x16; the idle link query
reported Gen1. Startup copy probes varied, roughly 6.6–13.1 GB/s. No power, clock,
huge-page or OS tuning was applied. The graphical desktop and coding-agent tools remained
open. System swap/zram was in use: about 4.1 GiB after testing, not a process-specific figure.

Model: ISTA-DASLab Qwen3.8-Flash-Next GSQ-RCO IQ3_XXS, both original GGUF shards,
with an already prepared native pack. The model repository revision and full GGUF hashes
were not recorded. File sizes, engine build flags, source revision and hashes of the engine,
pack index, expert profile and chat template are in [provenance.json](provenance.json).
No model was downloaded or converted for this experiment.

## Method

The script is [tools/bench_gtx1650.py](../../../tools/bench_gtx1650.py).
Each measured run sends the same three requests, in this order:

1. English CPU/GPU explanation: 44 prompt tokens, 64 generated tokens (length cap).
2. Arabic RAM/VRAM explanation: 45 prompt tokens, 64 generated tokens (length cap).
3. Repeated filler then France's capital: 231 prompt tokens, 2 generated tokens (`Paris`).

Temperature 0, thinking off, non-streaming. Prompt checkpoints were disabled with
`--prompt-cache 0` in **every** comparison arm; every recorded benchmark request has zero
reused tokens. Three rounds per arm unless noted. Model loading is excluded. There is no
separate unreported warm-up: the first request in each arm is included. The expert cache
and automatic PCIe/adaptation policy remain active within an arm; a new server is started
between configurations. These are sequential runs, not randomized/interleaved A/B pairs.
A later one-round baseline restart checks for large time drift.

Prompt and decode rates are the engine's separate timing fields. Total request time is
measured by the client. **Time to first token was not measured.** The 2-token answers are
not a reliable sustained-decode benchmark; use the two 64-token cases for decode comparisons.
Both explanations hit the length cap, so they are not checks of complete-answer quality.

All raw requests, responses, timings, errors, settings and engine logs are under [raw/](raw/).
In configs and logs, `$REPO`, `$DATA`, `$CONTROL` and `$HOME` replace local paths. Metadata
records `api_key: false`; no key was used. Early summary records omit an explicit error count,
but all individual failed requests would still be recorded; the successful benchmark arms
had no HTTP failures.

## Kept configuration

`tools/run_gtx1650.py --profile fast` is **opt-in**. Without the flag the original
configuration and launcher stdout remain byte-identical to commit `0d5c9aae` (checked by
executing both launchers with `--write-config-only` against the same files).

Compared with `stable`, it changes only:

- `STRATA_SM75_PERF=1`: the engine's opt-in Turing kernel series.
- Do not force `STRATA_HEAD_HOST=1`: keep the same output-head bytes in VRAM when they fit.
  The engine can still fall back to host memory on a failed allocation.
- `--prefill 256` instead of 32: read larger prompt chunks in the owned buffers.
- `--short-read 0` instead of the engine's default 64: read short prompts through the batched
  prompt path instead of serial decode windows. This changes arithmetic order and can
  change near-tie tokens.

Both keep 8,192 context, int8 KV, three CPU workers, `--expert-cache 32` (42 actual slots
in these runs), no drafter, no suffix/long lookup, no pictures, and no conversation parking.
The canonical weight arena stays in pinned RAM. No additional HC or KV requantization is
enabled in `fast`. The existing route-tail skip and CPU prompt-share defaults are unchanged.
Normal launches keep prompt checkpointing; `--no-prompt-cache` is an explicit benchmark option.

Startup free VRAM: stable **744 MiB**, final fast **256 MiB**, as reported by the engine.
After the final benchmark, sanity checks and long recall request, NVIDIA's query reported
239 MiB free. These are snapshots, **not peak usage**. On a busy GPU use `stable` if `fast`
does not fit. Neither profile overrides the memory held by other programs.

## Results: baseline versus the final launcher

Three rounds each. Values are medians with the observed range in parentheses.

| Case | Stable prompt tok/s | Fast prompt tok/s | Stable decode tok/s | Fast decode tok/s | Stable total s | Fast total s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| English, 44 + 64 tokens | 8.7 (8.4–8.8) | 23.1 (20.8–23.1) | 4.9 (4.9–5.0) | 5.8 (5.8–5.8) | 18.030 (17.952–18.098) | 12.866 (12.861–13.072) |
| Arabic, 45 + 64 tokens | 8.9 (8.8–9.0) | 23.7 (23.7–23.9) | 4.9 (4.9–5.0) | 5.8 (5.8–5.8) | 18.095 (17.751–18.106) | 12.841 (12.832–12.848) |
| Filler/capital, 231 + 2 tokens | 14.4 (14.4–14.5) | 35.6 (35.6–35.8) | 4.9 (4.9–5.0) | 5.8 (5.8–5.8) | 16.411 (16.391–16.417) | 6.831 (6.800–6.833) |

On these inputs, decode rose about **18%**, the 231-token prompt rate rose **2.47x**, and
its total latency fell **58%**. The two short explanation requests took about **29% less**
time. These are workload-specific results, not a speed promise for every request or card.
The later baseline restart returned 4.9 decode tok/s, 14.4 prompt tok/s on the filler case,
and 16.413 s total on that case (one round only).

## Experiments, including rejected settings

All changes below are relative to the original stable configuration unless described as cumulative.
Exact settings are in each matching `raw/<label>.json`.

| Label | Rounds | English decode median tok/s | 231-token prompt median tok/s | Observation |
| --- | ---: | ---: | ---: | --- |
| `baseline` | 3 | 4.9 | 14.4 | Original working profile |
| `sm75` | 3 | 5.0 | 17.0 | Only Turing series enabled |
| `sm75_pf256` | 3 | 5.0 | 35.3 | Turing + chunk 256; retained |
| `gpuhead` | 3 | 6.0 | 35.3 | Above + head in VRAM; retained |
| `shortread0` | 3 | 5.8 | 35.6 | Above + batched short reads; lower total latency, retained |
| `threadplacement` | 3 | 6.0 | 35.6 | Above + aux CPUs auto / sibling host; small gain, not kept |
| `workers7` | 2 | 5.85 | 35.7 | Short-read arm + 7 workers; no clear gain, not kept |
| `baseline_repeat` | 1 | 4.9 | 14.4 | Restarted baseline near the end |
| `fast_final` | 3 | 5.8 | 35.6 | Actual new launcher, fresh server; headline above |
| `hcq8` | startup failed | — | — | HC Q8 + original 512 MiB reserve left no expert tier |
| `hcq8_128` | startup failed | — | — | HC Q8 + 128 MiB reserve; prompt buffer wanted 71 MiB with 66 MiB free |
| `hcq8_4k` | 1 | 9.1 | 15.1 | HC Q8, 4K context, cache 8, reserve 32; only 60 MiB free; **rejected** |

The Q8 arm requantizes HC weights and reduces context. It is not the same-quality or
same-capacity comparison. It answered the three test inputs, but one round and 60 MiB
of free VRAM are not enough evidence for a usable profile. It is not exposed by the launcher.
No DFlash2, MTP or other model pack was enabled for the accepted tuning.

The borrowed ideas were measured memory budgeting, prompt chunk sizing, explicit thread
placement, and rejecting changes that did not clearly help. The RTX 5090 NVFP4 settings
and dual-GPU patches were not applied: different hardware and workloads.
Sources consulted: [ExTV tuning](https://github.com/ExTV/strata-5090-4070/blob/352dc9f8b087/docs/tuning.md),
[Righteous-AI recipes](https://github.com/Righteous-AI/Strata-Recipes/tree/b7d70b032918).

## Sanity and longer input

The final fast server also passed six checks, each once: `17 + 25 -> 42`, Saudi Arabia's
capital in Arabic, strict `{"name":"apple","count":3}` JSON, `hi`, a real OpenAI-style
`get_weather` tool call for Riyadh, and a second turn consuming its tool result (`35°C`).
See [raw/fast_final.sanity.json](raw/fast_final.sanity.json).

A separate **5,892-token** fresh prompt placed `violet-742` before repeated filler and
asked for it at the end. It returned `violet-742`: 167.846 s total, 35.4 prompt tok/s,
5.8 decode tok/s, 7 generated tokens, zero reused tokens. This is one synthetic recall
check, not a long-context quality score. See [raw/fast_final.long-recall.json](raw/fast_final.long-recall.json).

Generated explanation text varied even between repeated baseline requests. No logit parity,
perplexity, coding benchmark, broad tool-call suite, full 8K prompt, parallel load or
hour-long soak was run. No claim of bitwise output equivalence or unchanged overall
model quality is made. Other cards, operating systems and backends remain untested.

## Validation

Passed: `python tools/test_run_gtx1650.py` (2 tests),
`python tools/test_bench_gtx1650.py` (2 tests), and
`ctest --test-dir build-gtx1650 -R '^mapped_weight_arena_test$' --output-on-failure` (1 CUDA test).
The Python tests cover separate profile files, preserved stable settings, optional prompt-cache
disabling, benchmark error recording and refusal to overwrite results.
The unchanged default config and stdout were also compared byte for byte with the previous launcher.

Failed: the two HC Q8 startups listed above. Not run: the broader workloads and platforms
listed under limitations. No C++/CUDA source was changed during this tuning; the same
source-built CUDA binary was used throughout. No new HIP, SYCL or Metal build was performed.

## Reproduce

Use the build and prepared-model instructions in [docs/GTX1650.md](../../../docs/GTX1650.md).
In one terminal, using Python with the server dependencies:

```sh
/path/to/python tools/run_gtx1650.py --data-dir /path/to/Strata-data \
  --profile stable --no-prompt-cache --port 8080
```

In a second terminal, after `ready`:

```sh
python tools/bench_gtx1650.py --label stable --rounds 3 --out /tmp/gtx-stable.jsonl
```

Stop the server, start it with `--profile fast --no-prompt-cache`, then:

```sh
python tools/bench_gtx1650.py --label fast --rounds 3 --out /tmp/gtx-fast.jsonl
```

The benchmark refuses to overwrite an existing result file and preserves request failures.
For normal chat leave out `--no-prompt-cache`. It is safe to return to `--profile stable`.
