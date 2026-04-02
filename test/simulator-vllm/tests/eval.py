import re
import pandas as pd
import ast
from typing import List, Dict, Any

def parse_log_file(log_file: str, pattern: str, skip=False) -> List[Dict]:
    def get_from_pattern(profiler_pattern):
        data = []

        with open(log_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                # Remove ANSI color codes (like ^[[36m and ^[[0m)
                if line.find("vllm no result file") != -1 and skip:
                    data.append({})
                clean_line = re.sub(r'\x1b\[[0-9;]*m', '', line.strip())
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
    data = parse_log_file(log_file, r'vllm results: \s*(\{.*\})', skip=True)
    return data

if __name__ == "__main__":
    iosls = ["500_2500", "2500_2500", "2500_500"]
    timing_results = []
    rates = ["inf"] #, "1", "5", "10", "20", "50", "100" ]
    for iosl in iosls:
        sim_results = parse_sim_log(f'./logs/ross_output_{iosl}.log')  
        vllm_results = parse_vllm_log(f'./logs/ross_output_{iosl}.log')
        for idx, vllm_data in enumerate(vllm_results):
            if not vllm_data:
                continue
            # model_name = vllm_data['model']
            assert(vllm_data['duration'] > 0)
            vllm_total_time = vllm_data['duration']

            vllm_data['bench_time'] = vllm_total_time
            sim_result = sim_results[idx]

            sim_result = sim_results[idx]
            vllm_data.update({
                "request_rate": rates[0], # idx
                'sim_bench_time': sim_result['wall_time'],
                                
                'pe': abs(sim_result['wall_time'] - vllm_total_time) / vllm_total_time,
                'ae': abs(sim_result['wall_time'] - vllm_total_time),
                'iosl': iosl
            })
            timing_results.append(vllm_data)

    df = pd.DataFrame(timing_results)
    df = df[["iosl", "model_id", "tps", "request_rate", # "diffs", "sim_pre_forward_time", "forward_time", "sim_forward_time", "post_forward_time", "sim_post_forward_time",
            "bench_time", "sim_bench_time", "pe", "ae"]]
    df = df.round(2)
    df.to_csv("./logs/timing_results.csv")
