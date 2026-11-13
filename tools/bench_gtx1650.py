#!/usr/bin/env python3
"""Repeatable local HTTP smoke benchmark. Does not start/stop a server or tune the engine."""
import argparse
import json
from pathlib import Path
import statistics
import time
import urllib.request

CASES = [
    ("english", "Explain how a CPU and a GPU work together to run a language model. Use clear English and keep writing until you have given at least six useful points.", 64),
    ("arabic", "اشرح بالعربية كيف يعمل نموذج اللغة على الكمبيوتر، وما الفرق بين الرام وذاكرة كرت الشاشة. اكتب شرحًا مفيدًا في ست نقاط.", 64),
    ("prefill", "Read this text carefully.\n" + "The small laptop runs a local language model. " * 22 +
     "\nWhat is the capital of France? Reply with the city name only.", 8),
]


def request(url, body=None, timeout=240):
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def summarize(rows):
    out = {}
    for name, _, _ in CASES:
        samples = [r for r in rows if r["case"] == name and "response" in r]
        out[name] = {
            "runs": len(samples),
            "errors": sum(r["case"] == name and "error" in r for r in rows),
            "decode_tok_s": [r["response"].get("timings", {}).get("predicted_per_second") for r in samples],
            "prompt_tok_s": [r["response"].get("timings", {}).get("prompt_per_second") for r in samples],
            "wall_s": [r["wall_s"] for r in samples],
            "completion_tokens": [r["response"].get("usage", {}).get("completion_tokens") for r in samples],
            "cached_tokens": [r["response"].get("timings", {}).get("cache_n") for r in samples],
        }
        for metric in ("decode_tok_s", "prompt_tok_s", "wall_s"):
            values = [v for v in out[name][metric] if isinstance(v, (int, float))]
            out[name]["median_" + metric] = statistics.median(values) if values else None
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--label", required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True, help="new JSONL file; existing files are never overwritten")
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("--rounds must be positive")
    url = args.url.rstrip("/")
    health = request(url + "/health")
    if not health.get("loaded"):
        parser.error("server has no loaded model")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with args.out.open("x", encoding="utf-8") as output:
        output.write(json.dumps({"type": "metadata", "label": args.label, "health": health,
                                 "rounds": args.rounds, "temperature": 0, "thinking": False}, ensure_ascii=False) + "\n")
        output.flush()
        for round_id in range(args.rounds):
            for name, prompt, max_tokens in CASES:
                body = {"model": health["model"], "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0, "max_tokens": max_tokens,
                        "chat_template_kwargs": {"enable_thinking": False}}
                row = {"type": "sample", "label": args.label, "round": round_id + 1,
                       "case": name, "request": body}
                start = time.monotonic()
                try:
                    row["response"] = request(url + "/v1/chat/completions", body)
                except Exception as error:
                    row["error"] = str(error)
                row["wall_s"] = round(time.monotonic() - start, 4)
                rows.append(row)
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                output.flush()
                timings = row.get("response", {}).get("timings", {})
                print(f"{args.label} {round_id + 1}/{args.rounds} {name}: {row['wall_s']:.2f}s "
                      f"decode={timings.get('predicted_per_second')} tok/s "
                      f"prefill={timings.get('prompt_per_second')} tok/s "
                      f"error={row.get('error', 'none')}", flush=True)
        summary = summarize(rows)
        output.write(json.dumps({"type": "summary", "label": args.label, "cases": summary}, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if any("error" in row for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
