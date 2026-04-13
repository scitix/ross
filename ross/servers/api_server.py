#!/usr/bin/env python3
"""
A tiny stand-alone HTTP server that can accept `bench_serving` requests
and record their arrival timestamps for offline simulation.

Supported endpoints (streaming SSE-style):
  - POST /generate           (used by `--backend sglang`)
  - POST /v1/completions     (OpenAI completions style)
  - POST /v1/chat/completions

For each request, a JSON line is appended to the log file:
  {"ts": <epoch_seconds>, "path": "<path>", "req_id": <int>, "body": {...}}

This is intentionally minimal: it immediately streams back a dummy response.
"""

import argparse
import asyncio
import json
import os
import time
import urllib.request
from itertools import count
from typing import Any, AsyncGenerator, Dict, Optional

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from transformers import AutoTokenizer
import uvicorn
import subprocess

app = FastAPI()
req_counter = count(1)

log_path: Optional[str] = None
tokenize_url: str = ""
bootstrap_remote_count: int = 0
TOKENIZER = None


def _tokenize_remote(text: str) -> tuple[int, float]:
    payload = json.dumps({"text": text}).encode("utf-8")
    http_req = urllib.request.Request(
        tokenize_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(http_req) as response:
        data = json.loads(response.read().decode("utf-8"))
    return len(data["token_ids"])


def _tokenize_local(text: str) -> tuple[int, float]:
    token_ids = TOKENIZER.encode(text)
    return len(token_ids)


def _resolve_tokenization(rid: int, text: str) -> int:
    return _tokenize_local(text)

def call_bench_serving(model, framework, dataset_name, dataset, isl, osl, rate, num_prompt, req_output, batch_size, random_range_ratio=0.0):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ross_dir = os.path.dirname(script_dir)  # ross/ (parent of servers/)
    log_dir = os.path.join(ross_dir, "log")
    os.makedirs(log_dir, exist_ok=True)
    subprocess.run(
        [os.path.join(ross_dir, "ross_launch_server.sh"), model, framework,
         dataset_name, dataset, isl, osl, rate, num_prompt, req_output,
         os.path.join(log_dir, "launch_server.log"),
         os.path.join(log_dir, "client.log"),
         os.path.join(log_dir, "client_bench.log"),
         batch_size, str(random_range_ratio)],
        text=True,
        check=True,
        cwd=ross_dir,
    )

def _append_log(entry: Dict[str, Any]) -> None:
    if not log_path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


async def _stream_sglang(text: str) -> AsyncGenerator[bytes, None]:
    payload = {"text": text, "meta_info": {"completion_tokens": len(text)}}
    yield f"data: {json.dumps(payload)}\n\n".encode("utf-8")
    await asyncio.sleep(0)
    yield b"data: [DONE]\n\n"

async def _stream_oai(text: str) -> AsyncGenerator[bytes, None]:
    payload = {
        "choices": [
            {
                "index": 0,
                "text": text,
                "finish_reason": "stop",
            }
        ],
        "usage": {"completion_tokens": len(text), "prompt_tokens": 0, "total_tokens": len(text)},
    }
    yield f"data: {json.dumps(payload)}\n\n".encode("utf-8")
    await asyncio.sleep(0)
    yield b"data: [DONE]\n\n"

async def _stream_oai_chat(text: str) -> AsyncGenerator[bytes, None]:
    payload = {
        "choices": [
            {
                "index": 0,
                "delta": {"content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {"completion_tokens": len(text), "prompt_tokens": 0, "total_tokens": len(text)},
    }
    yield f"data: {json.dumps(payload)}\n\n".encode("utf-8")
    await asyncio.sleep(0)
    yield b"data: [DONE]\n\n"

async def _handle_request(request: Request, stream_fn) -> StreamingResponse:
    body = await request.json()
    rid = next(req_counter)
    arrival_ts = time.perf_counter()    
    text = body.get('prompt') or body.get('text') or ''
    prompt_tokens = await asyncio.to_thread(_resolve_tokenization, rid, str(text))
    elapsed = time.perf_counter() - arrival_ts
    entry = {
        'ts': arrival_ts,
        'path': str(request.url.path),
        'req_id': rid,
        'body': body,
        'prompt_tokens': prompt_tokens,
        'tokenize_time': elapsed,
    }
    _append_log(entry)

    return StreamingResponse(stream_fn(str(text)), media_type="text/event-stream")

@app.post("/generate")
async def generate(request: Request):
    return await _handle_request(request, _stream_sglang)

@app.post("/v1/completions")
async def completions(request: Request):
    return await _handle_request(request, _stream_oai)

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    return await _handle_request(request, _stream_oai_chat)

def main():
    parser = argparse.ArgumentParser("simple bench_serving capture server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--log-file", type=str, default='')
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--tokenize-url", type=str, default="http://127.0.0.1:8001/tokenize")
    parser.add_argument("--bootstrap-remote-count", type=int, default=0)
    args = parser.parse_args()

    global log_path, tokenize_url, bootstrap_remote_count, TOKENIZER
    log_path = args.log_file
    tokenize_url = args.tokenize_url
    bootstrap_remote_count = args.bootstrap_remote_count
    if args.model:
        TOKENIZER = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")

if __name__ == "__main__":
    main()
