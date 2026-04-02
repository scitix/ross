import sys
import time
from datetime import datetime
import pandas as pd
import subprocess
from typing import List, Dict, Any, Tuple
from pathlib import Path

REPO_BIN = str(Path(__file__).resolve().parents[1]) + "/bin"
sys.path.insert(0, REPO_BIN)

from bench import BenchmarkConfig, build_parser
import ross.util_copy as util_copy
import pareto.pareto as pa
from pareto.report import log_final_summary
from common.config import RuntimeConfig

def setup_server(api_path: str):
    api_server_process = subprocess.Popen(['python', api_path])
    time.sleep(2)
    return api_server_process

def search_parallel_configs(num_gpus: int, disaggregation=False):
    parallel_configs = []
    for p_dp in (1, 2, 4):
        for p_pp in (1, 2, 4, 8):
            for p_tp in (1, 2, 4):
                if disaggregation:
                    for d_dp in (1, 2, 4, 8):
                        for d_tp in (1, 2, 4, 8):
                            if p_dp * d_tp == d_dp * p_tp:
                                num_gpus_used = p_dp * p_pp * p_tp + d_dp * d_tp
                                if num_gpus_used <= num_gpus and num_gpus_used >= num_gpus // 2:
                                    parallel_configs.append(f"{p_dp}:{p_pp}:{p_tp}@{d_dp}:1:{d_tp}")
                else:
                    num_gpus_used = p_dp * p_pp * p_tp
                    if num_gpus_used == num_gpus:
                        parallel_configs.append(f"{p_dp}:{p_pp}:{p_tp}")
    return parallel_configs

def main():
    parser = build_parser()
    args = parser.parse_args()
    assert(len(sys.argv) > 1)

    conf = BenchmarkConfig(args)
    util_copy.echo_line(util_copy.line_width, "-", "🔥 Benchmark Configuration")
    util_copy.echo_info(conf.summary())

    assert(len(conf.backends) == 1 and len(conf.models) == 1)
    backend = conf.backends[0]
    model = conf.models[0]
    model_path = util_copy.get_model(model)
    mode = "Disagg" if conf.disaggregation_mode == "disaggregation" else "Colocate"

    if "--ross-config" in conf.args[backend][0]:
        conf_args = conf.args[backend][0]["--ross-config"]
        num_gpus = conf_args["num_gpus"]
        mem_fraction_static_list = conf_args["mem_fraction_static"]
        chunked_prefill_size_list = conf_args["chunked_prefill_size"]

        conf.num_gpu = num_gpus

    # util.echo_line(util.line_width, "-", "🔥 Get Requests")
    # subprocess.run(
    #     ["./ross_launch_server.sh", "500", "2500", "inf"],
    #     text=True,
    #     check=True
    # )
    start_time = time.perf_counter()

    parallel_configs = search_parallel_configs(conf.num_gpu, disaggregation=conf.disaggregation_mode)
    print(f"all parallel configs: ", parallel_configs)
    pareto_df = pd.DataFrame()

    for input in conf.inputs:
        for batch in conf.batches:
            for mem_fraction_statc in mem_fraction_static_list:
                for chunked_prefill_size in chunked_prefill_size_list:
                    runtime_config = RuntimeConfig(
                        batch_size=batch,
                        isl=input["isl"],
                        osl=input["osl"],
                        scheduler_config={
                            "mem_fraction_static": mem_fraction_statc,
                            "chunked_prefill_size": chunked_prefill_size
                        }
                    )
                    result_df = pa.coloation_pareto(
                            model_uri=model_path,
                            runtime_config=runtime_config,
                            backend=backend,
                            parallel_config_list=parallel_configs,
                    )
                    pareto_df = pd.concat([pareto_df, result_df], axis=0, ignore_index=True)

    print(f"Simulation Duration(s): {(time.perf_counter() - start_time):.2f}")
    print(pareto_df)
    pareto_frontier_df = pa.get_pareto_front(pareto_df, "tokens/s/user", "tokens/s/gpu").reset_index(drop=True).reset_index()
    print("\n=== pareto ===")
    print(pareto_frontier_df)
    timestamp = datetime.now().strftime("%m%d_%H%M")
    pareto_frontier_df.to_csv(f'./log/pareto_frontier_{timestamp}_{mode}.csv')

    pareto_results = { mode: pareto_frontier_df }
    log_final_summary(pareto_results)


if __name__ == "__main__":
    main()