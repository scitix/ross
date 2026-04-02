import re
import os
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
                        print(log_file)
                        print(f"Warning: Failed to parse profiler format line {line_num}: {e}")
                        exit(0)
        f.close()
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
            "osl": int(iosl_match.group(3)),
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
    results = 0
    import json
    with open(log_file, 'r') as f:
        lines = f.readlines()
        if not lines:
            return {}
        results = json.loads(lines[0].strip())

    return results

def get_req_names(log_file: str) -> Dict[str, str]:
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
    prompt_tokens = []
    for idx, line in enumerate(data):
        for req in line["req_ids"]:
            match = re.search(r"req_id=(.*?),", req)
            if match:
                req_id = match.group(1)
                if req_id not in new_req_ids:
                    new_req_ids.append(req_id)

                prompt_token_ids = extract_prompt_token_ids(req)
                if prompt_token_ids is not None:
                    prompt_tokens.append(len(prompt_token_ids))

    return new_req_ids, prompt_tokens

# ==============================================================================
# Main execution block
# ==============================================================================


def calc_execution_timing(rank, data_per_round, req_name = None, f = None):
    result = {
        "pre_time": 0,
        "forward_time": 0,
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
        
        return [req_name[r] for r in req_name_list]


    iter_start = data_per_round[0]['sched_start_time']    
    avail_tokens = []
    ttft, arrive_time = dict(), dict()

    for round, round_data in enumerate(data_per_round):
        req_id_list = get_new_req_ids_per_round(req_name, round_data)
        if 'req_0' not in round_data['cached_req_ids'] and 'req_0' not in req_id_list:
            if result["warmup_time"] == 0:
                result["warmup_time"] = round_data['sched_start_time'] - iter_start
            # --- Print Profiling Results
            if f is not None:
                req_name_list = [req_name[r] for r in round_data['cached_req_ids']]

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
            result['total_time'] += total_time / 1000

            if f is not None:     
                for idx,r in enumerate(req_name_list):
                    if r not in arrive_time:
                        arrive_time[r] = round_data['sched_start_time'] - iter_start - result["warmup_time"]
                    if r not in ttft:
                        ttft[r] = (result['total_time'] - arrive_time[r]) * 1000
                        print(f"rid={r}, arr_time={arrive_time[r]:.5f}, ttft={(result['total_time'] - arrive_time[r]):.5f}", file=f)

    return result, avail_tokens

def load_traces(args, debug=False, skip_timing=True):
    rank_logs = list(args["rank_logs"])
    main_server_log = args["server_log"]
    main_client_log = args["client_log"]

    if not main_server_log or not main_client_log:
        raise RuntimeError("No trace logs found.")

    bench_results = get_bench_results(main_client_log)    

    # Load Arrivals
    req_name_set, req_name = [], dict()
    for rank_log in rank_logs:
        req_name_rank, prompt_tokens_rank = get_req_names(rank_log)
        req_name_set.extend(req_name_rank)

    for req_id in req_name_set:
        if req_id not in req_name:
            req_name[req_id] = f"req_{len(req_name.values())}"

    max_rank_time, max_rank = 0, 0
    max_decode_data, flog, max_tps = None, None, None
    if debug:
        flog = open('log/vllm_outputs.log', 'w')

    if not skip_timing:
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
        timing_data, avail_tokens = calc_execution_timing(max_rank, max_decode_data, req_name, flog)
        timing_data.update({ "result_dp_rank": max_rank })
        
        bench_results.update({'tps': max_tps})
        bench_results.update( {"staged_timing_data": timing_data })   

    return bench_results
