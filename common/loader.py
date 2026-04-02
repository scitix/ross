import os, sys
import json
from typing import List, Tuple, Dict, Any
import logging

def setup_logging(log_file, debug=False):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    if logger.hasHandlers():
        logger.handlers.clear()
    console_fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
    file_fmt = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_fmt)

    if debug:
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8') 
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_fmt)
        logger.addHandler(file_handler)
        
    logger.addHandler(console_handler)

def load_online_requests_w_arrivals(
        frontend_log: str,
        dp: int,
        disaggregation: bool = False,
        trace_configs: Dict[str, Any] = {},
    ):
    from scheduler.request import Request

    input_requests = []
    with open(frontend_log, 'r', encoding='utf-8') as f:
        for line_number, line in enumerate(f):
            line = line.strip()
            if line:
                try:
                    data = json.loads(line)
                    ts, req_id = data.get('ts'), f"req_{data.get('req_id')}"
                    body = data.get('body', {})
                    if 'max_tokens' in body:
                        max_tokens = body['max_tokens']
                    else:
                        if 'sampling_params' not in body or 'max_new_tokens' not in body['sampling_params']:
                            raise RuntimeError(f'Cannot find max_new_tokens in {frontend_log}, line {line_number}')
                        max_tokens = body['sampling_params']['max_new_tokens']

                    if 'prompt' in body:
                        prompt = body['prompt']
                    elif 'text' in body:
                        prompt = body['text']
                    else:
                        raise RuntimeError(f'Cannot resolve prompt text in {frontend_log}, line {line_number}')
                    prompt_tokens = data['prompt_tokens']
                    tokenize_time = data['tokenize_time']
                    input_requests.append((ts, req_id, prompt, max_tokens, prompt_tokens, tokenize_time))
                except json.JSONDecodeError:
                    print(f"Error: Line {line_number + 1} is not a valid JSON")

    # round-robin for prefill_dp > 1
    requests, prompt_len = [], []
    if not input_requests:
        return requests, prompt_len
    base_ts = input_requests[1][0] if len(input_requests) > 1 else input_requests[0][0]
    for idx, (ts, req_id, prompt, max_tokens, prompt_tokens, tokenize_time) in enumerate(input_requests):
        arrive_time = max(0, ts - base_ts)
        assigned = (idx % dp, 0)

        req = Request(request_id=req_id,
            prompt=prompt,
            prompt_tokens=prompt_tokens,
            max_new_tokens=max_tokens,
            dp_rank=assigned,
            disaggregation=disaggregation,
            arrive_time=arrive_time,
        )
        req.tokenize_time = tokenize_time
        requests.append(req)
        
    return requests[1:], prompt_len[1:]
