#!/usr/bin/env python3
"""Measure llama-server prompt and decode throughput over its native API.

This intentionally uses /completion rather than a client-side stopwatch. The
server's timings distinguish prompt processing from generation and report the
actual token counts, which makes results comparable across batch settings.
"""
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


def request(base_url: str, path: str, payload: dict[str, Any]) -> dict[str, Any]:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=1800) as response:
            return json.loads(response.read())
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"{path} failed: {exc}") from exc


def exact_prompt(
    base_url: str,
    target_tokens: int,
    nonce: str,
    prompt_file: Path | None = None,
) -> str:
    """Make a deterministic raw prompt near target_tokens tokens."""
    if target_tokens <= 0:
        return ""

    if prompt_file:
        task = prompt_file.read_text()
        line = (
            f"Repository context record {nonce}: inspect this existing implementation, "
            "preserve its public behavior, and identify a safe minimal change.\n"
        )
        filler = line
        # Keep the task at the end so the model sees an actionable request rather
        # than a truncated instruction when testing a long context.
        while True:
            content = filler + task
            tokenized = request(base_url, "/tokenize", {"content": content})["tokens"]
            if len(tokenized) >= target_tokens:
                return content
            filler += line

    line = (
        f"qflash benchmark nonce {nonce}: The quick brown fox records a stable "
        "throughput sample while preserving enough varied text for tokenization.\n"
    )
    content = line
    while True:
        tokenized = request(base_url, "/tokenize", {"content": content})["tokens"]
        if len(tokenized) >= target_tokens:
            selected = tokenized[:target_tokens]
            return request(base_url, "/detokenize", {"tokens": selected})["content"]
        content += line


def one_run(
    base_url: str,
    prompt: str,
    output_tokens: int,
    seed: int,
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float,
) -> dict[str, Any]:
    response = request(
        base_url,
        "/completion",
        {
            "prompt": prompt,
            "n_predict": output_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "seed": seed,
            "cache_prompt": False,
            "ignore_eos": True,
            "timings_per_token": False,
            "stream": False,
        },
    )
    timings = response.get("timings") or {}
    return {
        "prompt_actual": int(timings.get("prompt_n") or 0),
        "output_actual": int(timings.get("predicted_n") or 0),
        "prompt_tps": float(timings.get("prompt_per_second") or 0),
        "decode_tps": float(timings.get("predicted_per_second") or 0),
        "prompt_ms": float(timings.get("prompt_ms") or 0),
        "decode_ms": float(timings.get("predicted_ms") or 0),
        "cached_tokens": int(response.get("tokens_cached") or 0),
    }


def median(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in rows[0]:
        values = [row[key] for row in rows]
        result[key] = statistics.median(values) if isinstance(values[0], (int, float)) else values[-1]
    return result


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://127.0.0.1:8081", help="llama-server root URL")
    p.add_argument("--prompt-file", type=Path, help="append this actionable task after generated context")
    p.add_argument(
        "--prompt-tokens",
        type=int,
        action="append",
        dest="prompts",
        help="target raw prompt length; repeat for a sweep (default: 2k, 8k, 32k, 64k)",
    )
    p.add_argument("--output-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--min-p", type=float, default=0.0)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--csv", type=Path, help="also write results as CSV")
    return p


def main() -> int:
    args = parser().parse_args()
    if args.runs < 1 or args.warmup < 0 or args.output_tokens < 1:
        raise SystemExit("runs and output-tokens must be positive; warmup cannot be negative")
    prompt_targets = tuple(args.prompts or DEFAULT_PROMPTS)
    base_url = args.url.rstrip("/")
    try:
        with urllib.request.urlopen(base_url + "/health", timeout=30) as response:
            if response.status != 200:
                raise RuntimeError(f"/health returned HTTP {response.status}")
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
        raise SystemExit(f"benchmark-api: /health failed: {exc}") from exc

    fields = [
        "prompt_target", "prompt_actual", "output_actual", "prompt_tps",
        "decode_tps", "prompt_ms", "decode_ms", "cached_tokens",
    ]
    results: list[dict[str, Any]] = []
    print(",".join(fields), flush=True)
    for prompt_target in prompt_targets:
        nonce = f"{time.time_ns()}-{prompt_target}"
        prompt = exact_prompt(base_url, prompt_target, nonce, args.prompt_file)
        for _ in range(args.warmup):
            one_run(
                base_url, prompt, min(args.output_tokens, 32), args.seed,
                args.temperature, args.top_p, args.top_k, args.min_p,
            )
        rows = [
            one_run(
                base_url, prompt, args.output_tokens, args.seed + i,
                args.temperature, args.top_p, args.top_k, args.min_p,
            )
            for i in range(args.runs)
        ]
        row = median(rows)
        row = {"prompt_target": prompt_target, **row}
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
        print(f"benchmark-api: {exc}", file=sys.stderr)
        raise SystemExit(1)
