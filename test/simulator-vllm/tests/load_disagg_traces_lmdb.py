import re
import os
import json
import argparse
import ast
from typing import List, Dict, Any, Optional

from tests.load_traces_lmdb import get_bench_results, get_req_names, parse_log_file, extract_params_from_path, calc_execution_timing
PROFILING_KEYS = [
    "gpu_forward_time_ms", "total_time_ms",
    "sched_start_time", "total_sampling_time_ms",
    # scheduling result
    "batch_size", "req_ids",
    "cached_req_ids", "cached_req_preemption", "cached_req_block_ids",
    # seq_lens
    "seq_lens", "prefill_seq_lens", "decode_seq_lens",
]

# ==============================================================================
# Main execution block
# ==============================================================================

def load_traces(args, debug=False, skip_timing=True, stage_name = '', req_name = None):
    def _calc_execution_timing(rank, data_per_round, req_name, f = None):
        result = {
            "pre_time": 0,
            "forward_time": 0,
            "post_time": 0,
            "total_time": 0,
        }
        if not data_per_round:
            return None, None, None
        req_assignment = dict()
        # req0 = req_name[data_per_round[2]['cached_req_ids'][0]]
        iter_start = data_per_round[0]['sched_start_time']
        avail_tokens = []
        for round, round_data in enumerate(data_per_round):
            for r in round_data['req_ids']:
                match = re.search(r"req_id=(.*?),", r)
                if match:
                    rid = match.group(1)
                    req_assignment[ req_name[rid] ] = (rank, round_data['sched_start_time'])
                else:
                    raise RuntimeError("Should be req_id")

            for r in round_data['cached_req_ids']:
                assert( req_name[r] not in req_assignment or req_assignment[ req_name[r] ][0] == rank )
                if req_name[r] in req_assignment: 
                    continue
                req_assignment[ req_name[r] ] = (rank, round_data['sched_start_time'])

            req_name_list = [req_name[r] for r in round_data['cached_req_ids']]
            # --- Print Profiling Results
            if f is not None and round >= 1: # and req0 not in req_name_list:
                print(f"[rank={rank}] Scheduling Round {round + 1}:", file=f)
                print(f"current_time={(round_data['sched_start_time'] - iter_start):.2f}", file=f)
                print(f"  - New Request IDs (len={len(round_data['req_ids'])})", file=f)
                print(f"  - Cached Request IDs (len={len(round_data['cached_req_ids'])}): {req_name_list}", file=f)
                print(f"  - seq_lens: {round_data['seq_lens']}", file=f)
                # print(f"  - extend_input_lens: {[x[1] for x in round_data['extend_input_lens']]}", file=f)
                
                print(
                    f"  - Pre Time: {(round_data.get('total_time_ms', 0.0) - round_data['gpu_forward_time_ms'] - round_data['total_sampling_time_ms']):.2f} ms\n"
                    f"    Forward Time: {round_data['gpu_forward_time_ms']:.2f} ms\n"
                    f"    Sampling Time: {round_data['total_sampling_time_ms']:.2f} ms", file=f)
                print(f"    Total Time={round_data['total_time_ms']:.2f} ms", file=f)
                print(f"next_round_time={(round_data['sched_start_time'] - iter_start + round_data['gpu_forward_time_ms'] / 1000 + round_data['total_sampling_time_ms'] / 1000):.2f}", file=f)
                print("-" * 20, file=f)

            if round >= 1:
                result['pre_time'] += (round_data['total_time_ms'] - round_data['gpu_forward_time_ms'] - round_data['total_sampling_time_ms']) / 1000
                result['post_time'] += round_data['total_sampling_time_ms'] / 1000
                result['forward_time'] += round_data['gpu_forward_time_ms'] / 1000
                result['total_time'] += round_data['total_time_ms'] / 1000

                # avail_tokens.append(round_data['available_tokens'])

        return result, avail_tokens, req_assignment

    rank_logs = list(args[f"{stage_name}_rank_logs"])
    main_client_log = args[f"client_log"]

    if not main_client_log:
        raise RuntimeError("No trace logs found.")

    bench_results = get_bench_results(main_client_log)
    if not req_name:
        req_name, req_name_set = dict(), []
        for rank_log in rank_logs:
            req_name_rank, _ = get_req_names(rank_log)
            req_name_set.extend(req_name_rank)
        for req_id in req_name_set:
            if req_id not in req_name:
                req_name[req_id] = f"req_{len(req_name.values())}"

    max_rank_time, max_rank = 0, 0
    max_decode_data, flog, max_tps = None, None, None
    if debug:
        flog = open(f'log/vllm_{stage_name}outputs.log', 'w')

    if not skip_timing:
        calculated_dp_rank = {}
        bench_results["staged_timing_data"] = []
        for rank_id, rank in enumerate(rank_logs):
            params = extract_params_from_path(rank, disaggregation_mode=False)
            data_per_round = parse_log_file(log_file=rank)

            if not data_per_round or params['dp_rank'] in calculated_dp_rank:
                continue
            calculated_dp_rank[params['dp_rank']] = 1

            rank_timing, _ = calc_execution_timing(params['dp_rank'], data_per_round, req_name)
            # bench_results["staged_timing_data"].append(rank_timing)
            if rank_timing['total_time'] > max_rank_time:
                max_decode_data, max_rank = data_per_round, params['dp_rank']
                max_rank_time = rank_timing['total_time']
                max_tps = f"{params['dp']}:{params['pp']}:{params['tp']}"
        
        if not max_decode_data:
            return bench_results
        timing_data, _ = calc_execution_timing(max_rank, max_decode_data, req_name, flog)
        timing_data.update({ "result_dp_rank": max_rank })
        
        bench_results.update({'tps': max_tps})
        bench_results.update( {"staged_timing_data": timing_data })   

    return bench_results, req_name

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--log_file", type=str, help=f"Path to the VLLM scheduler log file to be analyzed.")
#     parser.add_argument("--request-rate", type=str, help=f"num_reqs / s")
#     parser.add_argument("--osl", type=int, help=f"osl")
#     parser.add_argument("--disaggregation", action='store_true')

#     args = parser.parse_args()
#     req_name = None
#     prefill_results, req_name, prefill_as = load_traces(args, debug=True, stage_name='prefill', req_name=req_name)
#     vllm_results, req_name, decode_as = load_traces(args, debug=True, stage_name='decode', req_name=req_name)
#     vllm_results.update({ "prefill_phase": prefill_results })
#     print(f"vllm results: {vllm_results}")

#     # Assignment
#     with open('./tests/request_assignment.json', 'w') as fout:
#         req_as = { "prefill": prefill_as, "decode": decode_as }
#         json.dump(req_as, fout)
