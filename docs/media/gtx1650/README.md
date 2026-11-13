# GTX 1650 Ti monitor screenshots

Actual captures of the local Strata interface at `http://127.0.0.1:8080/#monitor`,
on 13 November 2026. The two builds are kept separate.

| Build / profile | File | Capture time (UTC+03:00) | State |
| --- | --- | --- | --- |
| First / `fast` | [monitor-fast-active.png](monitor-fast-active.png) | 03:31:20 | Light theme, reading a prompt |
| First / `fast` | [monitor-fast-requests.png](monitor-fast-requests.png) | 03:32:05 | Dark theme, completed-request history |
| Second / `cached` | [monitor-cached-active.png](monitor-cached-active.png) | 05:28:02 | Light theme, reading the 44-token English explanation prompt |
| Second / `cached` | [monitor-cached-requests.png](monitor-cached-requests.png) | 05:28:28 | Dark theme, completed English/Arabic explanations at 6.4 decode tok/s |

The new times are the PNG write times immediately after CDP screenshot capture.
**There is no `hc-q8` screenshot here.** The new images show `cached`, with no additional HC requantization.

## Shared capture conditions

- GTX 1650 Ti 4GB, i5-10300H, 64GB installed RAM, Linux.
- Local source-built engine 0.1.59.1, prepared ISTA-DASLab Qwen3.8-Flash-Next IQ3_XXS.
  The version string is the same, but the two builds have different binary hashes.
- Normal prompt checkpointing enabled, localhost port 8080.
- Chromium 156.0.8078.79, isolated temporary profile, software browser rendering (`--disable-gpu`),
  1440 × 1008 viewport at device scale 1. The second capture used headless Chromium.
- Light/dark mode was changed through the actual theme button. No metric values, statuses,
  request rows or chart data were edited. Images are visible-viewport captures, not composites
  or generated illustrations. They were visually checked for credentials, personal paths and unrelated content.
- The capture browser was closed and its temporary profile removed afterward. The model server was left running.

## Second build: `cached`

The server was running `tools/run_gtx1650.py --profile cached --port 8080`, model ID
`qwen3.8-flash-next-iq3_xxs-gtx1650-cached`. The engine binary SHA256 was
`87af0e9e62396beee1a694d5335cbd1cfd15ca4802f86d58454cdf6762ad20b3`.

[Capture metadata](cached-capture.json) records the source commit, actual health response,
PNG hashes, dimensions and write times. [Actual demo requests/responses](cached-capture-requests.json)
record the three requests sent for these images. They use the benchmark tool's filler/capital,
English explanation and Arabic explanation cases, once each, temperature 0 and thinking off.
The explanations were capped at 64 output tokens. `Max tokens` means this cap was reached,
not that an HTTP request failed.

The same session already had two short follow-up checks, visible in the history. Those reused
29 prompt tokens on the second turn; the three new demo requests each reused zero.
The earlier checks are preserved in the second report's
[follow-up evidence](../../../bench/results/2026-11-13-gtx1650ti-4gb-pool-cache/raw/cached_live.followup.json).

| Demo request | Prompt tokens | Generated tokens | Prompt tok/s | Decode tok/s | Wall seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Filler/capital | 231 | 2 | 28.3 | 6.2 | 8.488 |
| English explanation | 44 | 64 | 13.4 | 6.4 | 13.275 |
| Arabic explanation | 45 | 64 | 13.7 | 6.4 | 13.310 |

**These are not additional benchmark rounds.** In particular, 6.4 tok/s in the screenshot
is not a replacement for the three-round **6.3 tok/s** `cached` median. Normal checkpointing
was enabled and the browser was open. Prefill and session-average counters differ from the
fresh-prompt benchmark. Its results remain unchanged in the
[second tuning report](../../../bench/results/2026-11-13-gtx1650ti-4gb-pool-cache/README.md).

## First build: `fast`

The original two images used `tools/run_gtx1650.py --profile fast --port 8080`, with normal
checkpointing enabled. That capture session also sent the filler/capital and English/Arabic
explanation cases once each. The two explanations were capped at 64 output tokens; an earlier
`hi` request is visible. The first build's engine SHA256 was
`4fbe3ea2702acf9d937955b068f211ad8e2d9d7a25410255d21dc21bfbccd977`.

Those original PNGs are unchanged and retained for historical comparison. They are not
benchmark rounds or evidence of the second build's speed. The original controlled results
are in the [first tuning report](../../../bench/results/2026-11-13-gtx1650ti-4gb-tuning/README.md).

بالعربية: أُضيفت صورتان فعليتان للبناء الثاني بإعداد `cached`، مع الاحتفاظ بصور `fast`
القديمة للمقارنة. الرقم 6.4 الظاهر من جلسة تشغيل عادية، لا وسيط القياس 6.3.
لم تُعدّل الأرقام أو الصور، ولا توجد هنا صورة لإعداد `hc-q8`.
