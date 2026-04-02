import re
import os
import json
import argparse
import ast
from typing import List, Dict, Any, Optional

from tests.load_traces_lmdb import extract_params_from_path, parse_log_file, get_bench_results, get_req_names, calc_execution_timing

PROFILING_KEYS = [
    "forward_gpu_ms",
    "iteration_start", "total_time_ms", "post_gpu_ms",
    "req_ids", "seq_lens", "extend_input_lens"
]

# ==============================================================================
# Main execution block
# ==============================================================================

def load_traces(args, debug = False, skip_timing = True) -> Dict[str, Any]:
    prank_logs, drank_logs = [], []

    rank_logs = list(args["rank_logs"])
    main_prefill_log, main_decode_log = args["prefill_server_log"], args["decode_server_log"]
    main_client_log = args["client_log"]

    if not main_client_log:
        raise RuntimeError("No trace logs found.")
    
    prank_logs = [log for log in rank_logs if log.find("prefill") != -1]
    drank_logs = [log for log in rank_logs if log.find("decode") != -1]

    bench_results = get_bench_results(main_client_log)    
    prefill_req_name, text_to_prefill_name, _, prefill_arrivals = get_req_names(main_prefill_log, stage_name="Prefill")
    decode_req_name, _, _, _ = get_req_names(main_decode_log, text_to_prefill_name, stage_name="Decode")

    if debug or not skip_timing:
        flog = open('log/sglang_disagg_outputs.log', 'w') if debug else None
        max_rank_time, max_rank, max_rank_id = 0, 0, 0
        max_decode_data = None
        calculated_dp_rank = {}
        for rank_id, rank in enumerate(prank_logs):
            params = extract_params_from_path(rank, disaggregation_mode=True, stage='prefill')
            data_per_round = parse_log_file(log_file=rank)
            if not data_per_round:
                continue
            dp_rank = params['prefill_dp_rank']
            if dp_rank not in calculated_dp_rank:
                calculated_dp_rank[dp_rank] = 1
                rank_timing, _ = calc_execution_timing(dp_rank, data_per_round, prefill_req_name, prefill_arrivals, 'prefill')
        
        calculated_dp_rank = {}
        for rank_id, rank in enumerate(drank_logs):
            params = extract_params_from_path(rank, disaggregation_mode=True, stage='decode')
            data_per_round = parse_log_file(log_file=rank)
            if not data_per_round:
                continue
            dp_rank = params['decode_dp_rank']
            if dp_rank not in calculated_dp_rank:
                calculated_dp_rank[dp_rank] = 1
                rank_timing, _ = calc_execution_timing(dp_rank, data_per_round, decode_req_name, prefill_arrivals, 'decode')
                if rank_timing['decode_total_time'] > max_rank_time:
                    max_decode_data, max_rank, max_rank_id = data_per_round, dp_rank, rank_id
                    max_rank_time = rank_timing['decode_total_time']

        if not max_decode_data:
            return bench_results

        max_prefill_data = parse_log_file(log_file=prank_logs[max_rank_id])
        prefill_timing_data, _ = calc_execution_timing(max_rank, max_prefill_data, prefill_req_name, prefill_arrivals, 'prefill', flog)
        decode_timing_data, _  = calc_execution_timing(max_rank, max_decode_data, decode_req_name, prefill_arrivals, 'decode', flog)
        bench_results['staged_timing_data'] = {
            "result_dp_rank": max_rank,
            "prefill_timing_data": prefill_timing_data,
            "decode_timing_data": decode_timing_data
        }
    return bench_results
