import sys
import time
from datetime import datetime
import pandas as pd
import subprocess
from typing import List, Dict, Any, Tuple
from pathlib import Path

from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

import os
import logging

import ross.util_copy as util_copy
from bench_config import BenchmarkConfig, build_parser
import pareto.pareto as pa
from pareto.report import log_final_summary

from common.config import RuntimeConfig
from common.loader import setup_logging

os.environ["OMP_NUM_THREADS"] = "8"
os.environ["NUMEXPR_NUM_THREADS"] = "8"

def search_parallel_configs(num_gpus: int, disaggregation=False, backend='sglang'):
    parallel_configs = []
    for p_dp in (1, 2, 4, 8):
        for p_pp in (1, 2, 4, 8):
            for p_tp in (1, 2, 4, 8):
                if disaggregation:
                    for d_dp in (1, 2, 4, 8):
                        for d_tp in (1, 2, 4, 8):
                            if backend == 'vllm':
                                if p_pp > 1 or p_dp != d_dp:
                                    continue
                            if backend == "sglang":
                                if p_dp * d_tp != d_dp * p_tp:
                                    continue                        
                            num_gpus_used = p_dp * p_pp * p_tp + d_dp * d_tp
                            if num_gpus_used <= num_gpus and num_gpus_used >= num_gpus // 2:
                                parallel_configs.append(f"{p_dp}:{p_pp}:{p_tp}@{d_dp}:1:{d_tp}")
                else:
                    num_gpus_used = p_dp * p_pp * p_tp
                    if num_gpus_used <= num_gpus and num_gpus_used >= num_gpus // 2:
                        parallel_configs.append(f"{p_dp}:{p_pp}:{p_tp}")
    return parallel_configs


def run_single_simulation(task_args):
    """
    Worker function to run a single simulation configuration.
    """
    (model_path,  backend, parallel_config, 
    batch, isl, osl, rate,
    mem_fraction_static, chunked_prefill_size, gpu) = task_args
    if backend == 'sglang':
        scheduler_config = {
            "mem_fraction_static": mem_fraction_static,
            "chunked_prefill_size": chunked_prefill_size
        }
    elif backend == 'vllm':
        scheduler_config = {
            "gpu_memory_utilization": mem_fraction_static,
            "max_num_batched_tokens": chunked_prefill_size
        }
    
    runtime_config = RuntimeConfig(
            batch_size=batch,
            isl=isl,
            osl=osl,
            rate=rate,
            scheduler_config=scheduler_config
        )
    result_df = pa.coloation_pareto(
            model_uri=model_path,
            runtime_config=runtime_config,
            backend=backend,
            parallel_config=parallel_config,
            gpu=gpu,
        )
    return result_df
    # except Exception as e:
    #     util.echo_warn(f"Error in task {task_args}: {e}")
    #     return pd.DataFrame()


def main():
    parser = build_parser()
    args = parser.parse_args()
    assert(len(sys.argv) > 1)

    conf = BenchmarkConfig(args)
    util_copy.echo_line(util_copy.line_width, "-", "🔥 Benchmark Configuration")
    util_copy.echo_info(conf.summary())

    assert(len(conf.backends) == 1 and len(conf.models) == 1 and len(conf.rates) == 1 and len(conf.inputs) == 1 and len(conf.backend_opts) == 1)
    backend = conf.backends[0]
    backend_opt = conf.backend_opts[0]
    model = conf.models[0]
    model_path = util_copy.get_model(model)
    mode = conf.disaggregation_mode
    fixed_prompts = 0

    if "--ross-config" in conf.args[backend][0]:
        conf_args = conf.args[backend][0]["--ross-config"]
        num_gpus = conf_args["num_gpus"]
        if 'fix_prompts' in conf_args:
            fixed_prompts = 1
            
        if backend == 'sglang':
            mem_fraction_list = conf_args["mem_fraction_static"]
            num_batch_tokens_list = conf_args["chunked_prefill_size"]
        elif backend == 'vllm':
            mem_fraction_list = conf_args["gpu_memory_utilization"]
            num_batch_tokens_list = conf_args["max_num_batched_tokens"]
        conf.num_gpu = num_gpus

    start_time = time.perf_counter()

    parallel_configs = search_parallel_configs(conf.num_gpu, disaggregation=conf.disaggregation_mode == 'disaggregation', backend=backend)
    util_copy.echo_info(f"all parallel configs: {parallel_configs}")
    
    util_copy.echo_line(util_copy.line_width, "-", "🔥 Apply Plugins")
    subprocess.run(
        ["../bin/bench_apply_plugin.sh", backend, "online", conf.output],
        text=True,
        check=True
    )
    util_copy.echo_line(util_copy.line_width, "-", "🔥 Get Requests")
    req_output = f"./log/arrive_times_{conf.input['isl'][0]}_{conf.input['osl'][0]}.jsonl"
    if conf.rates[0] != "inf": 
        req_output = f"./log/arrive_times_{conf.input['isl'][0]}_{conf.input['osl'][0]}_{conf.rates[0]}.jsonl"
    subprocess.run(
        ["./ross_launch_server.sh", model, conf.input["path"], str(conf.input["isl"][0]), str(conf.input["osl"][0]), \
                                    str(conf.rates[0]), req_output, \
                                    "./log/launch_server.log", "./log/client.log", "./log/client_bench.log", backend, str(fixed_prompts)],
        text=True,
        check=True
    )
    exit(0)
    tasks = []
    for batch in conf.batches:
        for mem_fraction_statc in mem_fraction_list:
            for chunked_prefill_size in num_batch_tokens_list:
                for parallel_config in parallel_configs:
                    task_args = (
                        model_path, backend, parallel_config,
                        batch, conf.input["isl"], conf.input["osl"], conf.rates[0],
                        mem_fraction_statc, chunked_prefill_size,
                        backend_opt[0],
                    )
                    tasks.append(task_args)

    util_copy.echo_info(f"Total simulations to run: {len(tasks)}")
    
    # Multi-Process Pool: max_workers = by default 64
    max_workers = min(len(tasks), 8)
    util_copy.echo_info(f"Starting execution with {max_workers} workers...")

    results_dfs = []
    # for task in tasks:
    #     res_df = run_single_simulation(task)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {executor.submit(run_single_simulation, task): task for task in tasks}
        
        for future in tqdm(as_completed(future_to_task), total=len(tasks), desc="Processing logs"):
        # for future in concurrent.futures.as_completed(future_to_task):
            res_df = future.result()
            if not res_df.empty:
                results_dfs.append(res_df)

    if results_dfs:
        pareto_df = pd.concat(results_dfs, axis=0, ignore_index=True)
    else:
        pareto_df = pd.DataFrame()
                    
    util_copy.echo_info(f"Simulation Duration(s): {(time.perf_counter() - start_time):.2f}")
    timestamp = datetime.now().strftime("%m%d_%H%M")
    pareto_df.to_csv(f'./log/pareto_{timestamp}_{mode}.csv')

    if not pareto_df.empty:
        pareto_frontier_df = pa.get_pareto_front(pareto_df, "tokens/s/user", "tokens/s/gpu").reset_index(drop=True).reset_index()
        logger.info("=== pareto ===")
        logger.info(pareto_frontier_df)

        pareto_results = { mode: pareto_frontier_df }
        log_final_summary(pareto_results)
    else:
        logger.info("No valid results found.")

if __name__ == "__main__":
    setup_logging('./log/ross_bench.log')
    logger = logging.getLogger(__name__)

    main()