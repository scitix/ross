import re
import os
import json
from collections import defaultdict
import argparse
import ast
from typing import List, Dict, Any, Optional

PROFILING_KEYS = [
    "gpu_forward_time_ms", "total_time_ms",
    "sched_start_time", "total_sampling_time_ms",
    # scheduling result
    "batch_size", "req_ids",
    "cached_req_ids", "cached_req_preemption", "cached_req_block_ids",
    # seq_lens
    "seq_lens", "prefill_seq_lens", "decode_seq_lens",
]

def _is_main_client_log(file_name: str, request_rate: Any) -> bool:
    rate_suffix = f"rate_{request_rate}.log"
    if not file_name.endswith(rate_suffix):
        return False
    if "monitor" in file_name or "server" in file_name:
        return False
    return "main" in file_name

def parse_log_file(log_file: str) -> List[Dict]:
    """
    Parse SGL log file to extract profiler forward timing entries.
    """
    def get_from_pattern(profiler_pattern):
        data = []

        with open(log_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                # Remove ANSI color codes (like ^[[36m and ^[[0m)
                clean_line = re.sub(r'\x1b\[[0-9;]*m', '', line)
                # Try new profiler format first
                profiler_match = re.search(profiler_pattern, clean_line)
                if profiler_match:
                    try:
                        # Parse the dictionary string
                        dict_str = profiler_match.group(1)
                        timing_data = ast.literal_eval(dict_str)
                        data.append(timing_data)
                        continue
                    except (ValueError, SyntaxError, KeyError) as e:
                        raise ValueError(f"Failed to parse profiler format line {line_num} in {log_file}: {e}") from e
        return data

    # Pattern for new format: [Profiler: Forward Statistics]: {...}
    timing_data = get_from_pattern(r'\s*(\{.*\})')
    return timing_data

def extract_params_from_path(log_file_path: str, disaggregation_mode: bool = False, stage: str = '') -> Dict[str, Optional[int]]:
    """ Get configs from log_path
    """
    assert((disaggregation_mode and stage != '') or not disaggregation_mode)

    iosl_match = re.search(r"isl_(\d+)_osl_(\d+)", log_file_path)
    assert(iosl_match)
    if not disaggregation_mode:
        bpt_match = re.search(r"(\d+):(\d+):(\d+)_batch_(\d+)", log_file_path)
        assert(bpt_match)
        params = {
            "isl": int(iosl_match.group(1)),
            "osl": int(iosl_match.group(2)),
            'batch_size': int(bpt_match.group(4)),
            'dp': int(bpt_match.group(1)),
            'pp': int(bpt_match.group(2)),
            'tp': int(bpt_match.group(3))
        }
    else:
        bpt_match = re.search(r"(\d+):(\d+):(\d+)@(\d+):(\d+):(\d+)_batch_(\d+)", log_file_path)
        assert(bpt_match)
        params = {
            "isl": int(iosl_match.group(1)),
            "osl": int(iosl_match.group(2)),
            'batch_size': int(bpt_match.group(7)),
        }
        parallels = [int(bpt_match.group(i)) for i in range(1, 7)]
        assert(parallels[0] == parallels[3] and parallels[1] == 1 and parallels[4] == 1)
        params.update({
            "dp": parallels[0],
            'pp': 1,
            "tp": parallels[2] if stage == 'prefill' else parallels[5],
        })

    # get rank
    rank_match = re.search(r"rank_dp_(\d+)_tp_(\d+).txt", log_file_path)
    if rank_match:
        params['dp_rank'] = int(rank_match.group(1))
        params['tp_rank'] = int(rank_match.group(2))
    return params

def get_bench_results(log_file: str):
    with open(log_file, 'r') as f:
        lines = f.readlines()
        if not lines:
            return {}
        results = json.loads(lines[0].strip())
    return results

def load_ttft_from_server_log(server_log: str) -> Dict[str, Dict[str, float]]:
    ttft_by_req: Dict[str, Dict[str, float]] = defaultdict(dict)
    stats_pattern = re.compile(
        r"TTFT first token:\s+req_id=(?P<req_id>[0-9a-fA-F-]+).*?\bttft=(?P<ttft>\d+\.\d+)"
    )
    engine_pattern = re.compile(
        r"ENGINE last-mile:\s+req_id=(?P<req_id>[0-9a-fA-F-]+)\s+"
        r"stage=(?P<stage>\S+).*?"
        r"since_arrival=(?P<since_arrival>-?\d+\.\d+).*?"
        r"since_first_token_stats=(?P<since_first_token_stats>-?\d+\.\d+)"
    )
    dynamo_pattern = re.compile(
        r"DYNAMO last-mile:\s+req_id=(?P<req_id>[0-9a-fA-F-]+)\s+"
        r"stage=(?P<stage>\S+).*?"
        r"since_decode_generate_start=(?P<since_decode_generate_start>-?\d+\.\d+)"
        r"(?:.*?since_arrival=(?P<since_arrival>-?\d+\.\d+))?"
        r"(?:.*?since_first_token_stats=(?P<since_first_token_stats>-?\d+\.\d+))?"
    )

    with open(server_log, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            stats_match = stats_pattern.search(line)
            if stats_match:
                req_id = stats_match.group("req_id")
                ttft_by_req[req_id]["stats_first_token_ms"] = (
                    float(stats_match.group("ttft")) * 1000.0
                )
                continue

            engine_match = engine_pattern.search(line)
            if engine_match:
                req_id = engine_match.group("req_id")
                stage = engine_match.group("stage")
                ttft_by_req[req_id][f"engine_{stage}_since_arrival_ms"] = (
                    float(engine_match.group("since_arrival")) * 1000.0
                )
                ttft_by_req[req_id][f"engine_{stage}_since_first_token_stats_ms"] = (
                    float(engine_match.group("since_first_token_stats")) * 1000.0
                )
                continue

            dynamo_match = dynamo_pattern.search(line)
            if dynamo_match:
                req_id = dynamo_match.group("req_id")
                stage = dynamo_match.group("stage")
                ttft_by_req[req_id][f"dynamo_{stage}_since_decode_generate_start_ms"] = (
                    float(dynamo_match.group("since_decode_generate_start")) * 1000.0
                )
                since_arrival = dynamo_match.group("since_arrival")
                if since_arrival is not None:
                    ttft_by_req[req_id][f"dynamo_{stage}_since_arrival_ms"] = (
                        float(since_arrival) * 1000.0
                    )
                since_first_token_stats = dynamo_match.group("since_first_token_stats")
                if since_first_token_stats is not None:
                    ttft_by_req[req_id][
                        f"dynamo_{stage}_since_first_token_stats_ms"
                    ] = float(since_first_token_stats) * 1000.0

    return dict(ttft_by_req)


def get_req_names(log_file: str, server_log: Optional[str] = None) -> Dict[str, str]:
    def extract_prompt_token_ids(log_text: str) -> Optional[List[int]]:
        match = re.search(r'prompt_token_ids=\[(.*?)\]', log_text)
        if match:
            content = match.group(1).strip()
            if not content:
                return []
            return [int(x.strip()) for x in content.split(',')]
        return None

    data = parse_log_file(log_file=log_file)
    new_req_ids = []
    prompt_tokens = dict()
    raw_ttft = load_ttft_from_server_log(server_log) if server_log and server_log != "" else {}
    ttft_by_req = {}
    ttft_stages_by_req = {}
    for idx, line in enumerate(data):
        for req in line["req_ids"]:
            match = re.search(r"req_id=(.*?),", req)
            if match:
                req_id = match.group(1)
                if req_id not in new_req_ids:
                    new_req_ids.append(req_id)
                if req_id in raw_ttft:
                    ttft_stages_by_req[req_id] = raw_ttft[req_id]
                    if "stats_first_token_ms" in raw_ttft[req_id]:
                        ttft_by_req[req_id] = raw_ttft[req_id]["stats_first_token_ms"]

                prompt_token_ids = extract_prompt_token_ids(req)
                prompt_tokens[req_id] = len(prompt_token_ids)

    return new_req_ids, prompt_tokens, ttft_by_req, ttft_stages_by_req

# ==============================================================================
# Main execution block
# ==============================================================================


def calc_execution_timing(rank, data_per_round, req_name = None, prompt_tokens=None, ttft_client=None, f = None):
    result = {
        "pre_time": 0,
        "forward_time": 0,
        'sample_time': 0,
        "post_time": 0,
        "total_time": 0,
        "warmup_time": 0,
    }
    def get_new_req_ids_per_round(req_name, round_data):
        req_name_list = []
        for req in round_data["req_ids"]:
            match = re.search(r"req_id=(.*?),", req)
            if match:
                req_id = match.group(1)
                req_name_list.append(req_id)
        
        return req_name_list


    iter_start = data_per_round[0]['sched_start_time']
    wall_start = 0   
    avail_tokens = []
    ttft, arrive_time = dict(), dict()

    for round, round_data in enumerate(data_per_round):
        raw_req_id_list = get_new_req_ids_per_round(req_name, round_data)
        req_id_list = [req_name[r] for r in raw_req_id_list]
        req_name_list = [req_name[r] for r in round_data['cached_req_ids']]
        if 'req_0' not in req_name_list and 'req_0' not in req_id_list:
            if result["warmup_time"] == 0:
                result["warmup_time"] = round_data['sched_start_time'] - iter_start
            if wall_start == 0:
                wall_start = round_data['sched_start_time']
            # --- Print Profiling Results
            if f is not None:
                print(f"[rank={rank}] Scheduling Round {round + 1}:", file=f)
                print(f"current_time={(round_data['sched_start_time'] - iter_start):.2f}, TS = {round_data['sched_start_time']}", file=f)
                print(f"  - New Request IDs (len={len(round_data['req_ids'])}): {req_id_list}", file=f)
                print(f"  - Cached Request IDs (len={len(round_data['cached_req_ids'])}): {req_name_list}", file=f)
                print(f"  - seq_lens (sum = {sum(round_data['seq_lens'])}): {round_data['seq_lens']}", file=f)
                print(
                    f"  - Pre Time: {(round_data.get('total_time_ms', 0.0) - round_data['gpu_forward_time_ms'] - round_data['total_sampling_time_ms']):.2f} ms\n"
                    f"    Forward Time: {round_data['gpu_forward_time_ms']:.2f} ms\n"
                    f"    Sampling Time: {round_data['total_sampling_time_ms']:.2f} ms", file=f)
                print(f"    Total Time={round_data['total_time_ms']:.2f} ms", file=f)
                print(f"next_round_time={((round_data['gpu_forward_time_ms'] + round_data['total_sampling_time_ms']) / 1000 + round_data['sched_start_time'] - iter_start):.2f}", file=f)
                print("-" * 20, file=f)

            if round < len(data_per_round) - 1:
                total_time = (data_per_round[round + 1]['sched_start_time'] - round_data['sched_start_time']) * 1000
            else:
                total_time = round_data['total_time_ms']
            result['pre_time'] += (total_time - round_data['gpu_forward_time_ms'] - round_data['total_sampling_time_ms']) / 1000
            result['post_time'] += round_data['total_sampling_time_ms'] / 1000
            result['forward_time'] += round_data['gpu_forward_time_ms'] / 1000
            result['sample_time'] += round_data.get('gpu_sample_time_ms', 0) / 1000
            result['total_time'] += total_time / 1000

            if f is not None and ttft_client:     
                for idx,r in enumerate(req_id_list):
                    if r not in arrive_time:
                        arrive_time[r] = round_data['sched_start_time'] - wall_start
                    if r not in ttft:
                        ttft[r] = (result['total_time'] - arrive_time[r]) * 1000
                        print(f"rid={r}, raw_rid={raw_req_id_list[idx]}, arr_time={arrive_time[r]:.5f}, ttft_client={ttft_client[r] / 1000}", file=f)
                   

    return result, avail_tokens

def load_traces(args, debug=False):
    rank_logs = list(args["rank_logs"])
    main_server_log = args["server_log"]
    main_client_log = args["client_log"]

    if not main_server_log or not main_client_log:
        raise RuntimeError("No trace logs found.")
    print(f"[trace] loading client log: {main_client_log}")

    bench_results = get_bench_results(main_client_log)    

    # Load Arrivals
    req_name_set, req_name = [], dict()
    for rank_log in rank_logs:
        req_name_rank, prompt_tokens_rank, ttft_rank, ttft_stages_rank = get_req_names(rank_log, main_server_log)
        req_name_set.extend(req_name_rank)

    for req_id in req_name_set:
        if req_id not in req_name:
            req_name[req_id] = f"req_{len(req_name.values())}"
    bench_results["server_ttft_ms_by_req"] = {}
    bench_results["server_ttft_stages_ms_by_req"] = {}
    for rank_log in rank_logs:
        req_name_rank, _, ttft_rank, ttft_stages_rank = get_req_names(rank_log, main_server_log)
        for req_id in req_name_rank:
            if req_id in ttft_rank and req_id in req_name:
                bench_results["server_ttft_ms_by_req"][req_name[req_id]] = ttft_rank[req_id]
            if req_id in ttft_stages_rank and req_id in req_name:
                bench_results["server_ttft_stages_ms_by_req"][req_name[req_id]] = ttft_stages_rank[req_id]

    max_rank_time, max_rank = 0, 0
    max_decode_data, flog, max_tps = None, None, None
    if debug:
        flog = open('log/vllm_outputs.log', 'w')

    if debug:
        calculated_dp_rank = {}
        for rank_id, rank in enumerate(rank_logs):
            params = extract_params_from_path(rank, disaggregation_mode=False)
            data_per_round = parse_log_file(log_file=rank)
            if not data_per_round:
                continue
            if params['dp_rank'] in calculated_dp_rank:
                continue
            calculated_dp_rank[params['dp_rank']] = 1

            rank_timing, _ = calc_execution_timing(params['dp_rank'], data_per_round, req_name)
            if rank_timing['total_time'] > max_rank_time:
                max_decode_data, max_rank = data_per_round, params['dp_rank']
                max_rank_time = rank_timing['total_time']
                max_tps = f"{params['dp']}:{params['pp']}:{params['tp']}"
        
        if not max_decode_data:
            return bench_results
        timing_data, avail_tokens = calc_execution_timing(max_rank, max_decode_data, req_name, f=flog)
        timing_data.update({ "result_dp_rank": max_rank })
        
        bench_results.update({'tps': max_tps})
        bench_results.update( {"staged_timing_data": timing_data })   

    return bench_results
