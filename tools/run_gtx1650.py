#!/usr/bin/env python3
"""Experimental single-GPU Flash-Next launcher; reuses prepared IQ3_XXS files, downloads nothing."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="prepared Strata data directory")
    parser.add_argument("--engine", type=Path, default=ROOT / "build-gtx1650" / "strata")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--profile", choices=("stable", "fast"), default="stable",
                        help="stable: original small-VRAM settings; fast: opt-in measured tuning (needs more free VRAM)")
    parser.add_argument("--no-prompt-cache", action="store_true", help="disable prompt checkpoints for fresh-prompt benchmarks")
    parser.add_argument("--write-config-only", action="store_true", help="write config without starting the server")
    args = parser.parse_args()
    data, engine = args.data_dir.expanduser().resolve(), args.engine.expanduser().resolve()
    pack = data / "packs" / "iq3_xxs"
    shards = data / "models" / "IQ3_XXS"
    first = shards / "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf"
    second = shards / "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf"
    for path in (engine, first, second, pack / "dense.bin", pack / "index.txt",
                 pack / "tokenizer" / "tokenizer.json"):
        if not path.is_file():
            parser.error(f"missing {path}; this launcher does not build or download files")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    config = {
        "exe": str(engine), "cwd": str(ROOT), "tokenizer": str(pack / "tokenizer"),
        "model_name": "qwen3.8-flash-next-iq3_xxs-gtx1650", "host": "127.0.0.1",
        "port": args.port, "gpu": 0, "log": str(engine.parent / "gtx1650.engine.log"),
        "env": {"STRATA_WEIGHT_ARENA_HOST": "1", "STRATA_HEAD_HOST": "1"},
        "args": [
            "--pack", str(pack), "--native", str(first), "--ple-gguf", str(second),
            "--expert-profile", str(ROOT / "data" / "expert-profile.bin"),
            "--expert-cache", "32", "--max-context", "8192", "--kv", "int8",
            "--prefill", "32", "--no-prefill-borrow", "--pool-workers", "3",
            "--vram-reserve-mib", "512", "--lookup-max-t", "0", "--suffix-draft", "0",
            "--conversation-cache-mib", "0", "--spec", "2",
        ],
    }
    if args.profile == "fast":
        # Keep the same GGUF bytes: no HC requantization or smaller KV/context.
        config["env"].pop("STRATA_HEAD_HOST")
        config["env"]["STRATA_SM75_PERF"] = "1"
        config["args"][config["args"].index("--prefill") + 1] = "256"
        config["args"].extend(["--short-read", "0"])
        config["model_name"] += "-fast"
        config["log"] = str(engine.parent / "gtx1650-fast.engine.log")
        if "STRATA_HEAD_HOST" in os.environ:
            print("Warning: your exported STRATA_HEAD_HOST forces the output head into RAM; "
                  "unset it to use the measured fast configuration.", flush=True)
    if args.no_prompt_cache:
        config["args"].extend(["--prompt-cache", "0"])
    path = engine.parent / ("gtx1650-fast.json" if args.profile == "fast" else "gtx1650.json")
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    note = "Experimental:" if args.profile == "stable" else "Experimental fast profile:"
    print(f"Config: {path}\n{note} weights in pinned RAM, 8K context, images off.", flush=True)
    if args.write_config_only:
        return 0
    os.chdir(ROOT)
    os.execv(sys.executable, [sys.executable, "-m", "serve.server", "--engine", "strata",
                            "--config", str(path), "--host", "127.0.0.1", "--port", str(args.port)])


if __name__ == "__main__":
    raise SystemExit(main())
