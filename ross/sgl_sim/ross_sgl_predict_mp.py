import os
import time
import sys
import json
import argparse
from pathlib import Path
import pandas as pd

from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from tqdm import tqdm

sys.path.insert(0, ".")
import ross.util_copy as util_copy
from bench_config import etcpath, BenchmarkConfig, build_parser, DEFAULT_RESERVE
from sglang.bench_sglang import find_best_colocate_result_under_constraints, find_best_disagg_result_under_constraints

sys.path.insert(0, '../test/simulator-sgl')
import tests.load_traces_lmdb as load_traces_lmdb
import tests.load_disaggregation_traces_lmdb as load_disaggregation_traces_lmdb

import pandas as pd
import logging
from common.loader import setup_logging
from common.config import InferenceConfig, RuntimeConfig

from typing import Any, Dict, List
from dataclasses import dataclass


os.environ["OMP_NUM_THREADS"] = "8"
os.environ["NUMEXPR_NUM_THREADS"] = "32"


def extract_metrics(sim_result = None, framework_result=None, columns=None, backend: str = None):
    sim_result.update({"framework": "ROSS"})
    row1 = [sim_result.get(col) for col in columns]

    if not framework_result:
        return pd.DataFrame([row1], columns=columns)

    framework_result.update({"framework": backend})
    row2 = [framework_result.get(col) for col in columns]

    return pd.DataFrame([row1, row2], columns=columns)

def run_sim_predict(model: str, parallel: str, runtime_config: RuntimeConfig, platform_perf_yaml: str, gpu: str = '', trace_configs = {}):
    if parallel.find("@") == -1: # COLOCATE
        dp_size, pp_size, tp_size = parallel.split(":")
        summary = find_best_colocate_result_under_constraints(
            model_uri=model,
            inference_config=InferenceConfig(
                dp_size=int(dp_size), pp_size=int(pp_size), tp_size=int(tp_size)
            ),
            runtime_config = runtime_config,
            platform_perf_yaml=platform_perf_yaml,
            trace_configs=trace_configs,
            gpu=gpu
        )
    else: # DISAGGREGATION
        p_config, d_config = parallel.split("@")
        prefill_dp_size, prefill_pp_size, prefill_tp_size = p_config.split(":")
        decode_dp_size,  decode_pp_size,  decode_tp_size = d_config.split(":")

        summary = find_best_disagg_result_under_constraints(
            model_uri=model,
            prefill_inference_config=InferenceConfig(
                dp_size=int(prefill_dp_size), pp_size=int(prefill_pp_size), tp_size=int(prefill_tp_size)
            ),
            decode_inference_config=InferenceConfig(
                dp_size=int(decode_dp_size), pp_size=int(decode_pp_size), tp_size=int(decode_tp_size)
            ),
            runtime_config=runtime_config,
            platform_perf_yaml=platform_perf_yaml,
            trace_configs=trace_configs,
        )
    sim_result_dict = summary.get_result_dict()
    return sim_result_dict


@dataclass
class SimulationTask:
    model: str
    parallel: str
    runtime_config: RuntimeConfig
    hw_yaml: str
    trace_config_dict: Dict[str, Any]
    gpu: str
    debug: bool
    skip_timing: bool

def run_single_sim(task: SimulationTask):
    # 1. EVAL: Load Trace Logs
    trace_result_dict = None
    trace_args = argparse.Namespace(**task.trace_config_dict)
    if task.parallel.find("@") == -1: # COLOCATE
        trace_result_dict = load_traces_lmdb.load_traces(trace_args, task.debug, task.skip_timing)        
    else:
        trace_result_dict = load_disaggregation_traces_lmdb.load_traces(trace_args, task.debug)

    if not trace_result_dict or "duration" not in trace_result_dict:
        return None, trace_result_dict, task.parallel, f"{task.runtime_config.isl[0]}@{task.runtime_config.osl[0]}", task.model, task.gpu, task.runtime_config.rate

    if not task.skip_timing and not task.debug:
        trace_result_dict['duration'] = trace_result_dict['staged_timing_data']['wall_time']

    # 2. Load Sim Results
    sim_result_dict = run_sim_predict(task.model, task.parallel, task.runtime_config, task.hw_yaml, task.gpu)
    if not sim_result_dict:
        sim_result_dict = {'duration': 0}
    
    return sim_result_dict, trace_result_dict, task.parallel, f"{task.runtime_config.isl[0]}@{task.runtime_config.osl[0]}", task.model, task.gpu, task.runtime_config.rate

def main():
    MACHINE_STATS_PREFIX = '/volume/ycao03/SiLLM-OP/test/collector'
    parser = build_parser()

    parser.add_argument('--debug', action='store_true', help='Enable DEBUG print')
    parser.add_argument('--record-path', type=str, default='')
    parser.add_argument('--use-server-time', action='store_true', default='')

    args = parser.parse_args()
    conf = BenchmarkConfig(args)

    setup_logging('./log/ross_sgl_predict.log', args.debug)
    logger = logging.getLogger(__name__)
    
    util_copy.echo_line(util_copy.line_width, "-", "🔥 Benchmark Configuration")
    util_copy.echo_info(conf.summary())

    # start_time = time.perf_counter()
    metric_list = ['framework'] + ['duration', 'mean_ttft_ms', 'std_ttft_ms', 'median_ttft_ms', 'p99_ttft_ms', "mean_tpot_ms", "median_tpot_ms", "std_tpot_ms", "p99_tpot_ms"]

    if args.record_path != '':
        record_dir = str(Path(__file__).resolve().parent)
        if not os.path.exists(record_dir):
            raise RuntimeError(f"{args.record_path} Not Exist!")

        RECORD_COL = ['Platform', 'model', 'parallel', 'iosl', 'rate', 'pe', 'pe_mean_ttft', 'ae_median_ttft', 'pe_tpot', 'ross_duration', 'framework_duration',  'ross_mean_ttft', 'ross_median_ttft', 'ross_tpot',  'framework_mean_ttft', 'framework_median_ttft', 'framework_tpot']
        record_df = []
    
    tasks: List[SimulationTask] = []
    for input in conf.inputs:
        isl = input["isl"]
        osl = input["osl"]
        mem_fraction_static_list = [0.9,]
        chunked_prefill_size_list = [8192,]
        for backend_opts in conf.backend_opts:
            if "--ross-config" in conf.args['sglang'][0]:
                conf_args = conf.args['sglang'][0]["--ross-config"]
                if "mem_fraction_static" in conf_args:
                    mem_fraction_static_list = conf_args["mem_fraction_static"]
                if "chunked_prefill_size" in conf_args:
                    chunked_prefill_size_list = conf_args["chunked_prefill_size"]

            for model_idx, model in enumerate(conf.models):
                for parallel in conf.parallel:
                    for batch in conf.batches:
                        for rate in conf.rates:
                            for chunked_prefill_size in chunked_prefill_size_list:
                                for mem_fraction_static in mem_fraction_static_list:
                                    conf.set_curr('sglang', model, parallel, batch, input)

                                    hw_yaml = f"{MACHINE_STATS_PREFIX}/{backend_opts[0].lower()}/platform_features.yaml"
                                    assert(os.path.exists(hw_yaml) == True)

                                    conf.test_dst = conf.test_dst.replace(conf.gpuname, backend_opts[0])
                                    conf.test_dst = conf.test_dst.replace(conf.backend_info[conf.backends[0]]['version'], backend_opts[1])          
                                    conf.test_dst = conf.test_dst.replace(str(DEFAULT_RESERVE), str(mem_fraction_static))
                                    if not os.path.exists(conf.test_dst):
                                        continue
                                                                            
                                    runtime_config = RuntimeConfig(
                                        batch_size=batch,
                                        isl=isl, osl=osl, rate=rate,
                                        scheduler_config={
                                            "mem_fraction_static": mem_fraction_static,
                                            "chunked_prefill_size": chunked_prefill_size
                                        }
                                    )
                                    trace_config_dict = {
                                            "log_file": conf.test_dst,
                                            "request_rate": rate,
                                            "osl": osl,
                                        }
                                    tasks.append( SimulationTask(model, parallel, runtime_config, hw_yaml, trace_config_dict, backend_opts[0].lower(), args.debug, not (args.debug or args.use_server_time)) )

    num_workers = min(32, os.cpu_count() or 4)
    print(f"Total tasks: {len(tasks)}; num workers: {num_workers}")

    results = []
    # for task in tasks:
    #     sim_result_dict, trace_result_dict, parallel, iosl, model, gpu = run_single_sim(task)
    #     if not trace_result_dict or not sim_result_dict:
    #         continue
    #     df = extract_metrics(sim_result_dict, trace_result_dict, metric_list, 'sglang')
    #     pe = 1.0 * abs(sim_result_dict['duration'] - trace_result_dict['duration']) / trace_result_dict['duration']

    #     sim_p50_ttft, sim_avg_ttft, sim_tpot   = sim_result_dict["median_ttft_ms"], sim_result_dict["mean_ttft_ms"], sim_result_dict["mean_tpot_ms"]
    #     base_p50_ttft, base_avg_ttft, base_tpot = trace_result_dict["median_ttft_ms"], trace_result_dict["mean_ttft_ms"], trace_result_dict["mean_tpot_ms"]

    #     pe_mean_ttft = abs(sim_avg_ttft - base_avg_ttft) / base_avg_ttft * 100
    #     pe_p50_ttft = abs(sim_p50_ttft - base_p50_ttft) / base_p50_ttft * 100
    #     pe_mean_tpot = abs(sim_tpot - base_tpot) / base_tpot * 100

    #     print(f"GPU: {gpu}, model: {model}, parallel: {parallel}, iosl: {iosl}")
    #     print(f"DF: {df}\nPE: {pe}, {pe_mean_ttft}, {pe_p50_ttft}, {pe_mean_tpot}")
                                        
    #     if args.record_path != "":
    #         results.append([gpu, model, parallel, iosl, pe, pe_mean_ttft, pe_p50_ttft, pe_mean_tpot, trace_result_dict['duration'], sim_result_dict['duration'], \
    #         sim_result_dict["mean_ttft_ms"], sim_result_dict["median_ttft_ms"], sim_result_dict["mean_tpot_ms"], trace_result_dict["mean_ttft_ms"], \
    #         trace_result_dict["median_ttft_ms"], trace_result_dict["mean_tpot_ms"]])      

    try:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_to_task = {
                executor.submit(run_single_sim, task): task
                for task in tasks
            }
            for future in tqdm(as_completed(future_to_task), total=len(tasks), desc="Processing logs"):
                try:
                    sim_result_dict, trace_result_dict, parallel, iosl, model, gpu, rate = future.result()
                    if not trace_result_dict or not sim_result_dict:
                        continue
                    df = extract_metrics(sim_result_dict, trace_result_dict, metric_list, 'sglang')
                    pe = 100.0 * abs(sim_result_dict['duration'] - trace_result_dict['duration']) / trace_result_dict['duration']

                    sim_p50_ttft, sim_avg_ttft, sim_tpot   = sim_result_dict["median_ttft_ms"], sim_result_dict["mean_ttft_ms"], sim_result_dict["mean_tpot_ms"]
                    base_p50_ttft, base_avg_ttft, base_tpot = trace_result_dict["median_ttft_ms"], trace_result_dict["mean_ttft_ms"], trace_result_dict["mean_tpot_ms"]

                    ae_p50_ttft = abs(sim_p50_ttft - base_p50_ttft)
                    pe_mean_ttft = 0 if base_avg_ttft == 0 else abs(sim_avg_ttft - base_avg_ttft) / base_avg_ttft * 100
                    pe_mean_tpot = 0 if base_tpot == 0 else abs(sim_tpot - base_tpot) / base_tpot * 100

                    print(f"GPU: {gpu}, model: {model}, parallel: {parallel}, iosl: {iosl}, rate: {rate}")
                    print(f"DF: {df}\nPE: {pe}, {pe_mean_ttft}, {ae_p50_ttft}, {pe_mean_tpot}")

                    if args.record_path != "":
                        results.append([gpu, model, parallel, iosl, rate, pe, pe_mean_ttft, ae_p50_ttft, pe_mean_tpot, sim_result_dict['duration'], trace_result_dict['duration'], \
                        sim_result_dict["mean_ttft_ms"], sim_result_dict["median_ttft_ms"], sim_result_dict["mean_tpot_ms"], trace_result_dict["mean_ttft_ms"], \
                        trace_result_dict["median_ttft_ms"], trace_result_dict["mean_tpot_ms"]])                                                
                except Exception as e:
                    task = future_to_task[future]
                    raise RuntimeError(f"Unexpected error in task {task}: {e}", file=sys.stderr)
    except KeyboardInterrupt:
        print("\nUser interrupted. Exiting...", file=sys.stderr)
        sys.exit(1)

    if args.record_path != "":
        for record_data in results:
            row_df = pd.DataFrame([record_data], columns=RECORD_COL)
            if not row_df.empty:
                record_df.append(row_df)
        if record_df:
            result_record_df = pd.concat(record_df, axis=0, ignore_index=True)
        else:
            result_record_df = pd.DataFrame()

        agg_cols = ['pe', 'pe_mean_ttft', 'ae_median_ttft', 'pe_tpot']
        result_record_df[['avg_pe', 'avg_pe_mean_ttft', 'avg_ae_median_ttft', 'avg_pe_tpot']] = (
            result_record_df.groupby(['Platform', 'model'])[agg_cols].transform('mean')
        )
        result_record_df.to_csv(args.record_path)

if __name__ == "__main__":
    main()