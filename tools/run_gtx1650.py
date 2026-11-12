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
    path = engine.parent / "gtx1650.json"
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"Config: {path}\nExperimental: weights in pinned RAM, 8K context, images off.", flush=True)
    if args.write_config_only:
        return 0
    os.chdir(ROOT)
    os.execv(sys.executable, [sys.executable, "-m", "serve.server", "--engine", "strata",
                            "--config", str(path), "--host", "127.0.0.1", "--port", str(args.port)])


if __name__ == "__main__":
    raise SystemExit(main())
