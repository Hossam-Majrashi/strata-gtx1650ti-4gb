# GTX 1650 Ti monitor screenshots

Actual captures of the local Strata web interface at
`http://127.0.0.1:8080/#monitor`, on 13 November 2026.

| File | Capture time (UTC+03:00) | State |
| --- | --- | --- |
| [monitor-fast-active.png](monitor-fast-active.png) | 03:31:20 | Light theme, reading a prompt; GPU, PCIe, RAM and VRAM activity visible |
| [monitor-fast-requests.png](monitor-fast-requests.png) | 03:32:05 | Dark theme, idle after requests; recent request history visible |

## Capture conditions

- GTX 1650 Ti 4GB, i5-10300H, 64GB installed RAM, Linux.
- Local source-built engine 0.1.59.1, prepared ISTA-DASLab Qwen3.8-Flash-Next IQ3_XXS.
- `tools/run_gtx1650.py --profile fast --port 8080`, with normal prompt checkpointing enabled.
- Chromium 156.0.8078.79, an isolated temporary browser profile, software browser rendering
  (`--disable-gpu`), 1440 × 1008 viewport at device scale 1.
- The light/dark setting was changed through the actual theme button. No metric values,
  request statuses or chart data were edited. Screenshots contain the visible viewport;
  they are not composites or generated illustrations.
- Captures were visually checked for credentials, personal paths and unrelated content.
  They show hardware counters and request timings, not chat contents.

The screenshot session sent the benchmark script's capital/filler, English explanation
and Arabic explanation prompts, once each, with temperature 0 and thinking off. The two
explanations were capped at 64 output tokens. `Max tokens` in the request table means the
cap was reached, not that an HTTP request failed. The earlier `hi` request is also visible.

**These requests are not additional rounds of the published three-round comparison.**
Normal checkpointing was enabled, the browser was open, and the counters show live or
session-average values. The comparison deliberately disabled prompt checkpoints in both
arms. Do not read a screenshot's prefill counter as the table's 35.6 tok/s median or as a
new benchmark result. The unchanged source measurements are in the
[full tuning report](../../../bench/results/2026-11-13-gtx1650ti-4gb-tuning/README.md).

بالعربية: الصورتان فعليتان من صفحة المراقبة، وليستا رسمًا توضيحيًا أو تركيبًا للأرقام.
أرقامهما من تشغيل عادي، وليست وسيط جولات جدول الأداء. تم فحصهما قبل الإرفاق.
