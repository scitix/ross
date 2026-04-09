import re
import pandas as pd
import ast
from typing import List, Dict, Any

def parse_log_file(log_file: str, pattern: str) -> List[Dict]:
    def get_from_pattern(profiler_pattern):
        data = []

        with open(log_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                # Remove ANSI color codes (like ^[[36m and ^[[0m)
                if line.find("vllm no result file") != -1 and skip:
                    data.append({})
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
                        print(clean_line)
                        print(f"Warning: Failed to parse profiler format line {line_num}: {e}")
                        exit(0)
        f.close()
        return data

    timing_data = get_from_pattern(pattern)

    # print(f"    Parsed {len(timing_data)} timing entries") # from {log_file}")
    # print(f"=" * 20)
    return timing_data

def parse_sim_log(log_file: str) -> List[Dict[str, Any]]:
    """Parses the real vLLM profile log file."""
    data = parse_log_file(log_file, r'\[SIM\] result=\s*(\{.*\})')
    return data

def parse_vllm_log(log_file: str) -> List[Dict[str, Any]]:
    """Parses the real vLLM profile log file."""
    data = parse_log_file(log_file, r'vllm results: \s*(\{.*\})')
    return data

if __name__ == "__main__":
    iosls = ["2500_2500", "500_2500", "2500_500"]
    timing_results = []
    
    for iosl in iosls:
        sim_results = parse_sim_log(f'./logs/ross_disagg_output_{iosl}.log')  
        vllm_results = parse_vllm_log(f'./logs/ross_disagg_output_{iosl}.log')
        for idx, vllm_data in enumerate(vllm_results):
            if not vllm_data:
                continue
            vllm_total_time = vllm_data['duration']

            vllm_data['bench_time'] = vllm_total_time
            try:
                vllm_data['tps'] = vllm_data['prefill_phase']['tps'] + "@" + vllm_data['tps']
            except:
                pass
            sim_result = sim_results[idx]
            data_dict = vllm_data
            data_dict.update({
                                # "sim_prefill_forward_time": sim_data['prefill_forward_time'],
                                # "sim_prefill_post_forward_time": sim_data['prefill_post_time'],
                                # "sim_prefill_total_time": sim_data['prefill_total_time'],
                                
                                # "sim_decode_pre_forward_time": sim_data['decode_pre_forward_time'],
                                # 'sim_wall_time': sim_data['wall_time'],
                'sim_bench_time': sim_result['wall_time'],
                                
                'pe': abs(sim_result['wall_time'] - vllm_total_time) / vllm_total_time,
                'ae': abs(sim_result['wall_time'] - vllm_total_time),
                'iosl': iosl
            })
            timing_results.append(data_dict)
                
    df = pd.DataFrame(timing_results)
    
    df = df[["iosl", "model_id", "tps", \
        # "prefill_forward_time", "sim_prefill_forward_time", \
        # "prefill_post_forward_time", "sim_prefill_post_forward_time",
        # "prefill_total_time", "sim_prefill_total_time", \
        "bench_time", "sim_bench_time", \
        # "mean_ttft", "mean_itl", \
        # "sim_mean_ttft", "sim_mean_itl", \
        "pe", "ae", \
        # "decode_forward_time", "sim_decode_forward_time", \
        # "decode_post_forward_time", "sim_decode_post_forward_time", \
        # "decode_total_time", "sim_decode_total_time",
    ]]
    df = df.round(2)
    df.to_csv("./logs/timing_results.csv")
