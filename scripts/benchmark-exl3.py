#!/usr/bin/env python3
"""Benchmark a TabbyAPI/ExLlamav3 endpoint with separate prompt and decode rates."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_PROMPTS = (2048, 8192, 32768, 65536)
DEFAULT_MODEL_DIR = Path("models/qwen38-27b-exl3-3.5bpw")


def post(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    body = json.dumps(payload).encode()
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=1800) as response:
            return json.loads(response.read())
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"{url} failed: {exc}") from exc


def make_prompt(
    tokenizer: Any,
    target_tokens: int,
    nonce: str,
    prompt_file: Path | None = None,
) -> str:
    """Make a raw prompt with exactly target_tokens using the EXL3 tokenizer."""
    if prompt_file:
        task = prompt_file.read_text()
        line = (
            f"Repository context record {nonce}: inspect this existing implementation, "
            "preserve its public behavior, and identify a safe minimal change.\n"
        )
        text = line + task
        while True:
            encoded = tokenizer.encode(text, encode_special_tokens=True, add_bos=False)
            if encoded.shape[-1] >= target_tokens:
                selected = encoded[:, :target_tokens]
                return tokenizer.decode(selected)[0]
            text = line + text

    line = (
        f"qflash EXL3 benchmark nonce {nonce}: The quick brown fox records a stable "
        "throughput sample while preserving varied text for tokenization.\n"
    )
    text = line
    while True:
        encoded = tokenizer.encode(text, encode_special_tokens=True, add_bos=False)
        if encoded.shape[-1] >= target_tokens:
            selected = encoded[:, :target_tokens]
            return tokenizer.decode(selected)[0]
        text += line


def one_run(
    url: str,
    model: str,
    prompt: str,
    output_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float,
) -> dict[str, Any]:
    response = post(
        url.rstrip("/") + "/v1/completions",
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": output_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "ignore_eos": True,
            "stream": False,
            "stream_options": {"include_usage": True},
        },
    )
    usage = response.get("usage") or {}
    return {
        "prompt_actual": int(usage.get("prompt_tokens") or 0),
        "output_actual": int(usage.get("completion_tokens") or 0),
        "prompt_tps": float(usage.get("prompt_tokens_per_sec") or 0),
        "decode_tps": float(usage.get("completion_tokens_per_sec") or 0),
        "prompt_ms": float(usage.get("prompt_time") or 0) * 1000,
        "decode_ms": float(usage.get("completion_time") or 0) * 1000,
        "total_ms": float(usage.get("total_time") or 0) * 1000,
        "cached_tokens": int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0),
        "accepted_draft": int(
            (usage.get("completion_tokens_details") or {}).get("accepted_prediction_tokens") or 0
        ),
        "rejected_draft": int(
            (usage.get("completion_tokens_details") or {}).get("rejected_prediction_tokens") or 0
        ),
    }


def median(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in rows[0]:
        values = [row[key] for row in rows]
        result[key] = statistics.median(values) if isinstance(values[0], (int, float)) else values[-1]
    return result


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://127.0.0.1:8082")
    p.add_argument("--model", default="qwen38-27b-exl3-3.5bpw")
    p.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    p.add_argument("--prompt-file", type=Path, help="append this coding task after generated context")
    p.add_argument("--prompt-tokens", type=int, action="append", dest="prompts")
    p.add_argument("--output-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--min-p", type=float, default=0.0)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--csv", type=Path)
    return p


def main() -> int:
    args = parser().parse_args()
    if args.runs < 1 or args.warmup < 0 or args.output_tokens < 1:
        raise SystemExit("runs and output-tokens must be positive; warmup cannot be negative")

    try:
        with urllib.request.urlopen(args.url.rstrip("/") + "/health", timeout=30) as response:
            if response.status != 200:
                raise RuntimeError(f"/health returned HTTP {response.status}")
        from exllamav3.model import Config
        from exllamav3.tokenizer import Tokenizer

        tokenizer = Tokenizer.from_config(Config.from_directory(str(args.model_dir)))
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ImportError) as exc:
        raise SystemExit(f"benchmark-exl3: setup failed: {exc}") from exc

    prompt_targets = tuple(args.prompts or DEFAULT_PROMPTS)
    fields = [
        "prompt_target", "prompt_actual", "output_actual", "prompt_tps", "decode_tps",
        "prompt_ms", "decode_ms", "total_ms", "cached_tokens", "accepted_draft", "rejected_draft",
    ]
    results: list[dict[str, Any]] = []
    print(",".join(fields), flush=True)
    for prompt_target in prompt_targets:
        for warmup_index in range(args.warmup):
            warmup_prompt = make_prompt(
                tokenizer, prompt_target, f"warmup-{time.time_ns()}-{warmup_index}", args.prompt_file
            )
            one_run(args.url, args.model, warmup_prompt, min(args.output_tokens, 32), args.temperature, args.top_p, args.top_k, args.min_p)

        rows = []
        for run_index in range(args.runs):
            # TabbyAPI reuses a matching prefix. A fresh nonce makes each run a
            # true prefill measurement rather than a cache-reuse measurement.
            prompt = make_prompt(
                tokenizer, prompt_target, f"run-{time.time_ns()}-{run_index}", args.prompt_file
            )
            rows.append(one_run(args.url, args.model, prompt, args.output_tokens, args.temperature, args.top_p, args.top_k, args.min_p))
        row = {"prompt_target": prompt_target, **median(rows)}
        results.append(row)
        print(",".join(str(row[field]) for field in fields), flush=True)

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(results)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"benchmark-exl3: {exc}", file=sys.stderr)
        raise SystemExit(1)
