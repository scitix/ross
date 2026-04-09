import re
import os
import numpy as np
import json
import argparse
import ast
from typing import List, Dict, Any, Optional

PROFILING_KEYS = [
    "forward_gpu_ms",
    "iteration_start", "total_time_ms", "post_gpu_ms",
    "req_ids", "seq_lens", "extend_input_lens"
]

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
        rank_match = re.search(r"rank_dp_(\d+)_tp_(\d+).txt", log_file_path)
        if rank_match:
            params['dp_rank'] = int(rank_match.group(1))
            params['tp_rank'] = int(rank_match.group(2))
    else:
        bpt_match = re.search(r"(\d+):(\d+):(\d+)@(\d+):(\d+):(\d+)_batch_(\d+)", log_file_path)
        assert(bpt_match)
        params = {
            "isl": int(iosl_match.group(1)),
            "osl": int(iosl_match.group(2)),
            'batch_size': int(bpt_match.group(7)),
        }
        parallels = [int(bpt_match.group(i)) for i in range(1, 7)]
        params.update({
            "dp": parallels[0],
            'pp': 1,
            "tp": parallels[2] if stage == 'prefill' else parallels[5],
        })
        rank_match = re.search(r"prefill_rank_dp_(\d+)_tp_(\d+).txt", log_file_path)
        if rank_match:
            params['prefill_dp_rank'] = int(rank_match.group(1))
            params['prefill_tp_rank'] = int(rank_match.group(2))
        rank_match = re.search(r"decode_rank_dp_(\d+)_tp_(\d+).txt", log_file_path)
        if rank_match:
            params['decode_dp_rank'] = int(rank_match.group(1))
            params['decode_tp_rank'] = int(rank_match.group(2))

    return params

def get_bench_results(log_file: str):
    with open(log_file, 'r') as f:
        lines = f.readlines()
        if not lines:
            return {}
        results = json.loads(lines[0].strip())
    return results

def get_req_names(log_file: str, text_to_name = None, stage_name: str = 'Sched') -> Dict[str, str]:
    pattern = r"\[" + stage_name + r"\] original req ([\s\S]*?): text = @@([\s\S]*?)@@ output_len = (\d+), time=(\d+\.\d+)"
    req_name, new_text_to_name, output_lens = dict(), dict(), dict()
    arrive_time, first_arr = dict(), 0
    cnt = 0
    with open(log_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            match = re.search(pattern, line)
            if match:
                req_id = match.group(1)
                text = match.group(2)
                output_len = match.group(3)
                arr = match.group(4)
                if req_id in req_name:
                    continue
                elif text == 'None':
                    req_name[req_id] = f"req_None_{cnt}"
                    cnt += 1
                elif text_to_name is not None:
                    req_name[req_id] = text_to_name[text]
                else:
                    req_name[req_id] = f"req_{len(req_name.values())}"
                    new_text_to_name[text] = req_name[req_id]

                    if first_arr == 0 and arr and int(output_len) > 32:
                        first_arr = float(arr)

                output_lens[req_id] = output_len
                if req_name[req_id] != 'req_0' and req_name[req_id] != 'req_1':
                    arrive_time[req_name[req_id]] = float(arr) - first_arr

    return req_name, new_text_to_name, output_lens, arrive_time

def calc_execution_timing(rank, data_per_round, req_name, arrive_time, stage_type = '', f = None):
    result = {
        "total_time": 0,
        "wall_time": 0,
    }
    if stage_type != 'decode':
        result.update({
            "prefill_pre_time": 0,
            "prefill_total_time": 0,
            "prefill_forward_time": 0,
            "prefill_post_time": 0,
            "prefill_pre_process_time": 0,
        })
    if stage_type != 'prefill':
        result.update({
            "decode_pre_time": 0,
            "decode_total_time": 0,
            "decode_forward_time": 0,
            "decode_post_time": 0,
            "decode_pre_process_time": 0,
        })
    wall_iter_start = 0
    avail_tokens, diff_dist = [], []
    ttft, ttlt = dict(), dict()
    last_remains = dict()
    for round, round_data in enumerate(data_per_round):
        stage_name = round_data['stage_type']
        req_name_list = [req_name[r] for r in round_data['req_ids']]
        skip_this_round = False
        for rname in req_name_list:
            if rname in ['req_0', 'req_1', 'req_None_0', 'req_None_1']:
                skip_this_round = True
                break
        if not skip_this_round:
            if not wall_iter_start:
                wall_iter_start = round_data['iteration_start']
            result[f'{stage_name}_post_time'] += round_data['post_gpu_ms'] / 1000
            result[f'{stage_name}_forward_time'] += round_data['forward_gpu_ms'] / 1000

            if round < len(data_per_round) - 1:
                total_time = (data_per_round[round + 1]['iteration_start'] - round_data['iteration_start']) * 1000
            else:
                continue
            result[f'{stage_name}_pre_process_time'] += (round_data['pre_sched_ms'] + round_data['pre_process_ms']) / 1000
            result['total_time'] += total_time / 1000
            result[f'{stage_name}_pre_time'] += max(0, (total_time - round_data['forward_gpu_ms'] - round_data['post_gpu_ms']) / 1000)

            process_time = round_data['pre_sched_ms'] + round_data['pre_process_ms'] + round_data['forward_gpu_ms'] + round_data['post_gpu_ms']
            result[f'{stage_name}_total_time'] += process_time / 1000

            iteration_end = round_data['iteration_start'] + process_time / 1000 # s
            result['wall_time'] = iteration_end - wall_iter_start
            sum_seqs = 0
            for idx,r in enumerate(req_name_list):
                ttlt[r] = result['wall_time']
                if r not in ttft:
                    ttft[r] = (result['wall_time'] - arrive_time[r]) * 1000
                    if f is not None and stage_type != 'decode':
                        print(f"rid={r}, arr_time={arrive_time[r]:.5f}, ttft={(result['wall_time'] - arrive_time[r]):.5f}", file=f)
                if r not in last_remains:
                    last_remains[r] = round_data['seq_lens'][idx]
                else:
                    last_remains[r] = round_data['seq_lens'][idx] - last_remains[r]
                sum_seqs += last_remains[r]

            # --- Print Profiling Results
            if f is not None:
                print(f"[{stage_name}] [rank={rank}] Scheduling Round {round + 1}: [ available_tokens={round_data['available_tokens']} ]", file=f)
                print(f"wall_time={(round_data['iteration_start'] - wall_iter_start):.2f}", file=f)
                print(f"  - Request IDs (len={len(round_data['req_ids'])}, sum={sum_seqs}): {req_name_list}", file=f)
                print(f"  - seq_lens: {round_data['seq_lens']}", file=f)
                print(f"  - arrival time: {[arrive_time[r] for r in req_name_list]}", file=f)
                
                print(
                        # f"  - Pre Time: {(total_time - round_data['forward_gpu_ms'] - round_data['post_gpu_ms']):.2f} ms\n"
                        f"  - Pre Time (SCHED+PROCESS): {(round_data['pre_sched_ms'] + round_data['pre_process_ms']):.2f} ms\n"
                        f"    Forward Time: {round_data['forward_gpu_ms']:.2f} ms\n"
                        f"    Sampling Time: {round_data['post_gpu_ms']:.2f} ms", file=f
                )
                print(f"    Total Time={round_data['total_time_ms']:.2f} ms", file=f)                    
                print(f"    current time: {iteration_end - wall_iter_start}", file=f)
                print("-" * 20, file=f)

    return result, avail_tokens

def load_traces(args, debug=False):
    rank_logs = list(args["rank_logs"])
    main_server_log = args["server_log"]
    main_client_log = args["client_log"]

    if not main_server_log or not main_client_log:
        raise RuntimeError("No trace logs found.")
    print(f"[trace] loading client log: {main_client_log}")

    bench_results = get_bench_results(main_client_log)

    req_name, text_to_name, output_lens, arrive_time = get_req_names(main_server_log)
    req_dispatch_info = []
    dp_rank_count, arrival_dict = dict(), dict()
    for rank_id, rank in enumerate(rank_logs):
        params = extract_params_from_path(rank, disaggregation_mode=False)
        data_per_round = parse_log_file(log_file=rank)

        if not data_per_round:
            continue
        for round, round_data in enumerate(data_per_round):
            for rid in round_data['req_ids']:
                if rid in req_name and int(output_lens[rid]) > 8:
                    if req_name[rid] not in arrival_dict:                  
                        arrival_dict[req_name[rid]] = (params['dp_rank'], round_data['iteration_start'])
                        if params['dp_rank'] not in dp_rank_count:
                            dp_rank_count[params['dp_rank']] = 0

                        req_dispatch_info.append((round_data['iteration_start'], params['dp_rank'], req_name[rid]))
                        dp_rank_count[params['dp_rank']] += 1

    if debug:
        flog = open('log/sglang_outputs.log', 'w') if debug else None
        max_rank_time, max_rank = 0, 0
        max_decode_data = None
        calculated_dp_rank = {}
        for rank_id, rank in enumerate(rank_logs):
            params = extract_params_from_path(rank, disaggregation_mode=False)
            data_per_round = parse_log_file(log_file=rank)
            if not data_per_round:
                continue
            if params['dp_rank'] in calculated_dp_rank:
                continue
            else:
                calculated_dp_rank[params['dp_rank']] = 1
            rank_timing, _ = calc_execution_timing(params['dp_rank'], data_per_round, req_name, arrive_time)
            if rank_timing['wall_time'] > max_rank_time:
                max_decode_data, max_rank = data_per_round, params['dp_rank']
                max_rank_time = rank_timing['wall_time']

        if not max_decode_data:
            return bench_results
            
        timing_data, avail_tokens = calc_execution_timing(max_rank, max_decode_data, req_name, arrive_time, f=flog)
        timing_data.update({ "result_dp_rank": max_rank })    
        bench_results.update( {"staged_timing_data": timing_data, } )

    return bench_results
