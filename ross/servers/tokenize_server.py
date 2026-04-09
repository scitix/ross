#!/usr/bin/env python3
import argparse
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn
from pydantic import BaseModel
from transformers import AutoTokenizer

class TokenizeRequest(BaseModel):
    text: str


app = FastAPI()
TOKENIZER: Any = None


def load_tokenizer(model_path: str):
    return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

@app.get("/health")
async def health():
    return JSONResponse({"ok": True})

@app.post("/tokenize")
@app.post("/v1/tokenize")
async def tokenize(req: TokenizeRequest):
    token_ids = TOKENIZER.encode(req.text)
    return JSONResponse(
        {
            "token_ids": token_ids,
            "prompt_tokens": len(token_ids),
        }
    )

def main():
    parser = argparse.ArgumentParser("lightweight tokenize server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    global TOKENIZER
    TOKENIZER = load_tokenizer(args.model)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
