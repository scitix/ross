import re
import os
import argparse
import ast
from typing import List, Dict, Any, Tuple
from collections import defaultdict

PROFILING_KEYS = [
    "forward_gpu_ms",
    "iteration_start", "total_time_ms", "post_gpu_ms",
    "req_ids", "seq_lens", "extend_input_lens", "available_tokens"
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
                        print(log_file)
                        print(f"Warning: Failed to parse profiler format line {line_num}: {e}")
                        exit(0)
        f.close()
        return data

    # Pattern for new format: [Profiler: Forward Statistics]: {...}
    timing_data = get_from_pattern(r'\s*(\{.*\})')

    # print(f"    Parsed {len(timing_data)} timing entries") # from {log_file}")
    # print(f"=" * 20)
    return timing_data

def extract_scheduler_data_from_log(log_file_path: str, req_name: Dict[str, str]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Reads a vLLM scheduler log and extracts detailed information about new and 
    cached requests for each scheduling round.

    Args:
        log_file_path (str): The path to the log file.x

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, where each dictionary 
                                represents one scheduling round and contains 
                                information like new_requests, cached_request_ids, etc.
    """
    data = parse_log_file(log_file=log_file_path)
    prompt_len = dict()
    # cnt = 0
    for round_data in data:
        stage = round_data['stage_type']
        for idx, req_id in enumerate(round_data['req_ids']):
            # if req_id not in req_name:
            #     req_name[req_id] = f'req_None_{cnt}'
            #     cnt += 1
            if req_id not in prompt_len:
                prompt_len[req_name[req_id]] = round_data['seq_lens'][idx]
            else:
                prompt_len[req_name[req_id]] = max(prompt_len[req_name[req_id]], round_data['seq_lens'][idx])
    return data, prompt_len


def get_bench_time(log_file: str, disagg=False):
    time_elapsed = 0
    with open(log_file, 'r') as f:
        lines = f.readlines()
        if not disagg:
            last_line = lines[-1]
            if last_line.find("Throughput of the token generation") == -1:
                # not successfully run
                return 0
            matches = re.findall(r"\d+\.\d+", last_line)
            # Convert the found strings to floating-point numbers
            if len(matches) == 2:
                time_elapsed = float(matches[1])
            else:
                print(log_file)
                print("Could not find exactly two floating-point numbers in the string.")
                exit(0)
        else:
            for line in lines[-30:-1]:
                if line.find("Benchmark duration") == -1:
                    continue
                matches = re.findall(r"\d+\.\d+", line)
                # Convert the found strings to floating-point numbers
                if len(matches) == 1:
                    time_elapsed = float(matches[0])
                else:
                    print(log_file)
                    print("Could not find exactly two floating-point numbers in the string.")
                    exit(0)
    return time_elapsed

def get_req_names(log_file: str, osl: int, text_to_name = None, stage_name: str = 'SCHED') -> Dict[str, str]:
    pattern = r"\[" + stage_name + r"\] original req ([\s\S]*?): text = @@([\s\S]*?)@@ output_len = (\d+)"
    new_text_to_name = {}
    req_name = {}
    cnt = 0
    with open(log_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            match = re.search(pattern, line)
            if match:
                req_id = match.group(1)
                text = match.group(2)
                if req_id in req_name:
                    continue
                elif text == 'None':
                    req_name[req_id] = f"req_None_{cnt}"
                    cnt += 1
                elif text_to_name is not None:
                    assert(text in text_to_name)
                    req_name[req_id] = text_to_name[text]
                else:
                    req_name[req_id] = f"req_{len(req_name.values())}"
                    new_text_to_name[text] = req_name[req_id]
    return req_name, new_text_to_name

# ==============================================================================
# Main execution block
# ==============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Parse a SGL scheduler log to extract details for each scheduling round."
    )
    
    # --- Argument has been updated ---
    # Positional argument is now optional with nargs='?' and has a default value.
    parser.add_argument(
        "--model",
        type=str,
        help=f"Name to Model Files."
    )
    parser.add_argument(
        "--tps",
        type=str,
        help=f"Name to Model Files."
    )
    parser.add_argument(
        "--log_file",
        type=str,
        help=f"Path to the SGL scheduler log file to be analyzed."
    )
    parser.add_argument(
        "--rank_num",
        type=int,
        help=f"number of rank nodes"
    )
    parser.add_argument(
        "--osl",
        type=int,
        help=f"osl"
    )
    args = parser.parse_args()
    prank_logs, drank_logs = [], []
    main_logs = []
    main_log_prefill, main_log_decode = '', ''
    for root, _, files in os.walk(args.log_file):
        for f in files:
            if f.startswith("prefill_rank") and f.endswith(".txt") and root.find('tmp') == -1:
                prank_logs.append(os.path.join(root, f))
            if f.startswith("decode_rank") and f.endswith(".txt") and root.find('tmp') == -1:
                drank_logs.append(os.path.join(root, f))
            if f.startswith("main") and f.endswith(".log.client"):
                main_logs.append(os.path.join(root, f))
            if f.startswith("main") and f.endswith(".log.server.prefill"):
                main_log_prefill = os.path.join(root, f)
            if f.startswith("main") and f.endswith(".log.server.decode"):
                main_log_decode = os.path.join(root, f)

    last_prank = sorted(prank_logs)[-1]
    last_drank = sorted(drank_logs)[-1]
    main_logs = sorted(main_logs)
    bench_time = get_bench_time(main_logs[-1], disagg=True)

    prefill_req_name, text_to_prefill_name = get_req_names(main_log_prefill, args.osl, stage_name="Prefill")
    decode_req_name, _ = get_req_names(main_log_decode, args.osl, text_to_prefill_name, stage_name="Decode")
    # Call the function to extract data
    prefill_data, prompt_len = extract_scheduler_data_from_log(last_prank, prefill_req_name)
    decode_data, output_len = extract_scheduler_data_from_log(last_drank, decode_req_name)

    # Print the results
    assert(prefill_data)

    first_arrival, first_arrival_time = dict(), dict()
    for current_round, round_data in enumerate(prefill_data):
        for req in round_data['req_ids']:
            if req not in first_arrival:
                first_arrival[req] = current_round
                first_arrival_time[req] = round_data['iteration_start']
        
    with open('./tests/arrive_times.log', 'w') as f:
        arrive_time_json = {
            "model_name": args.model,
            "log_file": args.log_file,
            "tps": args.tps,
            "arrive_time": list(first_arrival_time.values()),
            "arrive_round": list(first_arrival.values())
        }
        print(f"{arrive_time_json}", file=f)
    f.close()
    
    fprefill = open('tests/prefill_outputs.log', 'w')
    fdecode = open('tests/decode_outputs.log', 'w')
    def _debug_print(stage_name, data_per_round, req_name, f):
        total_time = 0
        total_forward_time = 0
        total_post_time = 0
        for round, round_data in enumerate(data_per_round):
            # --- Print Profiling Results
            print(f"[{stage_name}] Scheduling Round {round + 1}:", file=f)
            print(f"  - Request IDs (len={len(round_data['req_ids'])}): {[req_name[r] for r in round_data['req_ids']]}", file=f)
            print(f"  - seq_lens (sum={sum(round_data['seq_lens'])}): {round_data['seq_lens']}", file=f)
            print(f"  - extend_input_lens: {[x[1] for x in round_data['extend_input_lens']]}", file=f)
            if 'available_tokens' in round_data:
                print(f"  - available_tokens: {round_data['available_tokens']}", file=f)
            
            print(
                # f"  - Scheduling Time: {round_data['total_time_ms']:.2f} ms\n"
                f"    Forward Time: {round_data['forward_gpu_ms']:.2f} ms\n"
                f"    Sampling Time: {round_data['post_gpu_ms']:.2f} ms", file=f)
            print("-" * 20, file=f)

            total_time += round_data['post_gpu_ms'] + round_data['forward_gpu_ms']
            total_post_time += round_data['post_gpu_ms']
            total_forward_time += round_data['forward_gpu_ms']
        return {
            stage_name + "_total_time": total_time / 1000,
            stage_name + "_forward_time": total_forward_time / 1000,
            stage_name + "_post_time": total_post_time / 1000
        }

    prefill_times = _debug_print("prefill", prefill_data, prefill_req_name, fprefill)
    decode_times = _debug_print("decode", decode_data, decode_req_name, fdecode)

    print([(k, v, output_len[k] if k in output_len else 1, 0, 0) for k, v in prompt_len.items()], file=fprefill)

    sgl_results = {
        "model": args.model,
        "tps": args.tps,
        **prefill_times,
        **decode_times,
        "bench_time": bench_time,
        "diffs": bench_time - decode_times['decode_total_time']
    }
    print(f"sgl results: {sgl_results}")  
