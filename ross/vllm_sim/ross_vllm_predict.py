import os
import time
import sys
import json
import subprocess
import argparse
from pathlib import Path
import pandas as pd

sys.path.insert(0, ".")
from api_server import call_bench_serving
import util
from bench_config import etcpath, BenchmarkConfig, build_parser, DEFAULT_RESERVE
from vllm_sim.bench_vllm import find_best_colocate_result_under_constraints, find_best_disagg_result_under_constraints

sys.path.insert(0, '../test/simulator-vllm')
import tests.load_traces_lmdb as load_traces
import tests.load_disagg_traces_lmdb as load_disagg_traces

import pandas as pd
import logging
from common.loader import setup_logging
from common.config import InferenceConfig, RuntimeConfig
from common.utils import resolve_vllm_log_paths

# import matplotlib.pyplot as plt
# import seaborn as sns

# def plot_and_save_distribution(data_list, save_path='1.png'):
#     plt.figure(figsize=(8, 6))
#     sns.histplot(data_list, kde=True, color='skyblue', edgecolor='black')
    
#     plt.title('Distribution Plot')
#     plt.xlabel('Value')
#     plt.ylabel('Frequency')
#     plt.grid(axis='y', alpha=0.5)
    
#     plt.savefig(save_path)
#     plt.close()
#     print(f"{save_path}")

def extract_metrics(sim_result = None, framework_result=None, columns=None, backend: str = None):
    sim_result.update({"framework": "ROSS"})
    row1 = [sim_result.get(col) for col in columns]

    if not framework_result:
        return pd.DataFrame([row1], columns=columns)

    framework_result.update({"framework": backend})
    row2 = [framework_result.get(col) for col in columns]

    return pd.DataFrame([row1, row2], columns=columns)

def run_sim_predict(model: str, parallel: str, runtime_config: RuntimeConfig, platform_perf_yaml: str, gpu: str, trace_configs = None):
    if parallel.find("@") == -1: # COLOCATE
        dp_size, pp_size, tp_size = parallel.split(":")
        summary = find_best_colocate_result_under_constraints(
            model_uri=model,
            inference_config=InferenceConfig(
                dp_size=int(dp_size), pp_size=int(pp_size), tp_size=int(tp_size)
            ),
            runtime_config = runtime_config,
            platform_perf_yaml=platform_perf_yaml,
            gpu=gpu,
            trace_configs=trace_configs,
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
            gpu=gpu,            
            trace_configs=trace_configs,
        )
    sim_result_dict = summary.get_result_dict()
    return sim_result_dict


def main():
    MACHINE_STATS_PREFIX = '../collector'
    parser = build_parser()

    parser.add_argument('--debug', action='store_true', help='Enable DEBUG print')
    parser.add_argument('--record-path', type=str, default='')
    parser.add_argument('--use-server-time', action='store_true', default='')
    parser.add_argument('--no-comparison', action='store_true', default='')
    parser.add_argument('--no-apply-plugin', action='store_true', default='')

    args = parser.parse_args()
    conf = BenchmarkConfig(args)
    results = []

    setup_logging('./log/ross_vllm_predict.log', args.debug)
    logger = logging.getLogger(__name__)

    util.echo_line(util.line_width, "-", "🔥 Benchmark Configuration")
    util.echo_info(conf.summary())

    # start_time = time.perf_counter()
    metric_list = ['framework'] + ['duration', 'mean_ttft_ms', "mean_tpot_ms", "mean_itl_ms", "throughput"]

    if args.record_path != '':
        record_dir = str(Path(__file__).resolve().parent)
        if not os.path.exists(record_dir):
            raise RuntimeError(f"{args.record_path} Not Exist!")

        if not args.no_comparison:
            RECORD_COL = ['Platform', 'model', 'parallel', 'iosl', 'rate', 'pe', 'pe_mean_ttft', 'pe_mean_tpot', 'pe_mean_itl', 'pe_mean_throughput', \
                        'ross_duration', 'framework_duration',  'ross_mean_ttft', 'framework_mean_ttft', \
                        'ross_mean_tpot', 'framework_mean_tpot', 'ross_mean_itl', 'framework_mean_itl', \
                        'ross_mean_throughput', 'framework_mean_throughput']
        else:
            RECORD_COL = ['Platform', 'model', 'parallel', 'iosl', 'rate', \
                          'duration', 'mean_ttft_ms', "mean_tpot_ms", "mean_itl_ms", "throughput"]
        record_df = []

    if len(conf.args['vllm']) > 0:
        conf_args = conf.args['vllm'][0]
        if "gpu_memory_utilization" in conf_args:
            gpu_memory_utilization_list = conf_args["gpu_memory_utilization"]
        if "max_num_batched_tokens" in conf_args:
            max_num_batched_tokens_list = conf_args["max_num_batched_tokens"]

    for input in conf.inputs:
        dataset, data_path, isl, osl  = input["dataset"], input["path"], input["isl"][0], input["osl"][0]
        datatype           = f"{dataset}_isl_{isl}_osl_{osl}"

        for backend_opts in conf.backend_opts:
            for model_idx, model in enumerate(conf.models):
                p                  = Path(model)
                last               = p.name
                model_name         = p.parent.name if last.startswith("v") else last
                for parallel in conf.parallel:
                    for batch in conf.batches:
                        for rate in conf.rates:
                            conf.set_curr('vllm', model, parallel, batch, input)
                            util.echo_line(util.line_width, "-", f"Test [{conf.cur_test:02}/{conf.num_test * len(conf.backend_opts)}], Model: {model_name}, Backend: {backend_opts}, Parallel: {parallel}")

                            for gpu_memory_utilization in gpu_memory_utilization_list:
                                for max_num_batched_tokens in max_num_batched_tokens_list:

                                    hw_yaml = f"{MACHINE_STATS_PREFIX}/{backend_opts[0].lower()}/platform_features.yaml"
                                    assert(os.path.exists(hw_yaml) == True)

                                    if not args.no_comparison:
                                        conf.test_dst = conf.test_dst.replace(conf.gpuname, backend_opts[0])
                                        conf.test_dst = conf.test_dst.replace(conf.backend_info[conf.backends[0]]['version'], backend_opts[1])          
                                        conf.test_dst = conf.test_dst.replace(str(DEFAULT_RESERVE), str(gpu_memory_utilization))
                                        if not os.path.exists(conf.test_dst):
                                            print(f"Trace Directory Not Found: {conf.test_dst}")
                                            continue
                                        trace_config_dict = resolve_vllm_log_paths(
                                            log_dir=conf.test_dst, gpu=backend_opts[0],
                                            isl=isl, osl=osl, num_prompt=conf.num_prompt,
                                            dataset=dataset, request_rate=str(rate),
                                            disaggregation=parallel.find('@') != -1
                                        )
                                        if not trace_config_dict["client_log"]:
                                            print("Trace Client Log Not Found")
                                            continue
                                        skip_timing = not (args.debug or args.use_server_time)
                                    
                                    if not os.path.exists("arrivals/vllm"):
                                        os.makedirs("arrivals/vllm")
                                    arrival_path = f"arrivals/vllm/arrivals_{model_name}_{datatype}_batch_{batch}_promptnum_{conf.num_prompt}_rate_{rate}.jsonl"
                                    if not os.path.exists(arrival_path):
                                        call_bench_serving(model, 'vllm', dataset, data_path, str(isl), str(osl), 
                                            str(rate), str(conf.num_prompt), arrival_path, str(batch)
                                        )
                                    runtime_config = RuntimeConfig(
                                        batch_size=batch, isl=isl, osl=osl, rate=rate,
                                        scheduler_config={
                                            "gpu_memory_utilization": gpu_memory_utilization,
                                            "max_num_batched_tokens": max_num_batched_tokens
                                        },
                                        arrival_path=arrival_path,
                                    )
                                    if not args.no_comparison:
                                        # 1. EVAL: Load Trace Logs
                                        if parallel.find("@") == -1: # COLOCATE
                                            trace_result_dict = load_traces.load_traces(
                                                trace_config_dict, args.debug, skip_timing
                                            )
                                        else:
                                            req_name = None
                                            prefill_results, req_name = load_disagg_traces.load_traces(
                                                trace_config_dict, debug=args.debug, skip_timing=skip_timing,
                                                stage_name='prefill', req_name=req_name
                                            )
                                            trace_result_dict, req_name = load_disagg_traces.load_traces(
                                                trace_config_dict, debug=args.debug, skip_timing=skip_timing,
                                                stage_name='decode', req_name=req_name
                                            )
                                            trace_result_dict.update({ "prefill_timing_data": prefill_results['staged_timing_data'] })
                            
                                        trace_result_dict["throughput"] = trace_result_dict["output_throughput"]
                                        if args.use_server_time:
                                            trace_result_dict['duration'] = trace_result_dict['staged_timing_data']['total_time']
                                            
                                        if not skip_timing:
                                            print(trace_result_dict['staged_timing_data'])
                                    else:
                                        trace_result_dict = None
                                    
                                    # 2. Load Sim Results
                                    sim_start_t = time.perf_counter()
                                    sim_result_dict = run_sim_predict(model, parallel, runtime_config, hw_yaml, backend_opts[0].lower(), {})
                                    sim_elapsed_s = time.perf_counter() - sim_start_t
                                    if sim_result_dict is None:
                                        continue
                                    if parallel.find("@") != -1:
                                        print(sim_result_dict['decode_phases'])
                                    else:
                                        print(sim_result_dict['timing_phases'])

                                    df = extract_metrics(sim_result_dict, trace_result_dict, metric_list, 'vllm')
                                    if not args.no_comparison:
                                        pe = 100.0 * abs(sim_result_dict['duration'] - trace_result_dict['duration']) / trace_result_dict['duration']

                                        sim_avg_ttft, sim_tpot, sim_itl, sim_throughput = sim_result_dict["mean_ttft_ms"], sim_result_dict["mean_tpot_ms"], sim_result_dict["mean_itl_ms"], sim_result_dict["throughput"]
                                        base_avg_ttft, base_tpot, base_itl, base_throughput = trace_result_dict["mean_ttft_ms"], trace_result_dict["mean_tpot_ms"], trace_result_dict["mean_itl_ms"], trace_result_dict["throughput"]

                                        pe_mean_ttft = 0 if base_avg_ttft == 0 else abs(sim_avg_ttft - base_avg_ttft) / base_avg_ttft * 100
                                        pe_mean_tpot = 0 if base_tpot == 0 else abs(sim_tpot - base_tpot) / base_tpot * 100
                                        pe_mean_itl = 0 if base_itl == 0 else abs(sim_itl - base_itl) / base_itl * 100
                                        pe_mean_throughput = 0 if base_throughput == 0 else abs(sim_throughput - base_throughput) / base_throughput * 100

                                        pe_row = {col: None for col in df.columns}
                                        pe_row.update({
                                            "framework": "PE(%)", "duration": pe, "mean_ttft_ms": pe_mean_ttft,
                                            "mean_tpot_ms": pe_mean_tpot, "mean_itl_ms": pe_mean_itl, "throughput": pe_mean_throughput,
                                        })
                                        df.loc[len(df)] = pe_row
                                    print(f"DF (took {sim_elapsed_s:.3f}s):\n{df}")
                                    if args.record_path != "":
                                        if not args.no_comparison:
                                            results.append([backend_opts[0].lower(), model, parallel, f"{isl}@{osl}", rate, \
                                            pe, pe_mean_ttft, pe_mean_tpot, pe_mean_itl, pe_mean_throughput, \
                                            sim_result_dict['duration'], trace_result_dict['duration'], \
                                            sim_avg_ttft, base_avg_ttft, \
                                            sim_tpot, base_tpot, sim_itl, base_itl, sim_throughput, base_throughput,
                                        ])
                                        else:
                                            results.append([backend_opts[0].lower(), model, parallel, f"{isl}@{osl}", rate,
                                            sim_result_dict['duration'], sim_avg_ttft, sim_tpot, sim_itl, sim_throughput])

    if args.record_path != "":
        for record_data in results:
            row_df = pd.DataFrame([record_data], columns=RECORD_COL)
            if not row_df.empty:
                record_df.append(row_df)
        if record_df:
            result_record_df = pd.concat(record_df, axis=0, ignore_index=True)
        else:
            result_record_df = pd.DataFrame()
        print(result_record_df)
        if not args.no_comparison:
            agg_cols = ['pe', 'pe_mean_ttft', 'pe_mean_tpot', 'pe_mean_itl', 'pe_mean_throughput']
            result_record_df[['avg_pe', 'avg_pe_mean_ttft', 'avg_pe_mean_tpot', 'avg_pe_mean_itl', 'avg_pe_throughput']] = (
                result_record_df.groupby(['Platform', 'model'])[agg_cols].transform('mean')
            )
        result_record_df.to_csv(args.record_path)

if __name__ == "__main__":
    main()
