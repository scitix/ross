#!/usr/bin/env python3
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
import numpy as np
import re
from datetime import datetime

from typing import List, Dict, Tuple
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import r2_score, mean_squared_error

import os, sys
from pathlib import Path

sys.path.append("..")
TEST_ROOT = Path(__file__).resolve().parents[2]
SGL_SIM_ROOT = TEST_ROOT / "simulator-sgl"
BIN_ROOT = Path(__file__).resolve().parents[3] / "ross"

for _path in (TEST_ROOT, SGL_SIM_ROOT, BIN_ROOT):
    path_str = str(_path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from tests.load_traces_lmdb import parse_log_file, extract_params_from_path
import util
from bench_config import BenchmarkConfig, build_parser, DEFAULT_RESERVE

from common.models import BaseModel, get_model
from common.plot import plot_and_save_distribution, plot_x_vs_time
from common.config import InferenceConfig
from common.features import SGLRegressionFeatures, PlatformPerf


def calculate_log_time_diff(filename):
    last_time_stats_time = None
    endpoint_time = None
    
    time_pattern = re.compile(r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})')

    try:
        with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                match = time_pattern.search(line)
                if not match:
                    continue
                
                time_str = match.group(1)
                try:
                    current_time = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    continue

                if "TimeStats: [unified]" in line:
                    last_time_stats_time = current_time
                
                if "Endpoint '/get_server_info'" in line:
                    endpoint_time = current_time
        
        if last_time_stats_time and endpoint_time:
            delta = endpoint_time - last_time_stats_time
            total_seconds = delta.total_seconds()
            
            return abs(total_seconds)
        return -1

    except Exception as e:
        print(f"发生未知错误：{e}")
        exit(1)

def main():
    parser = build_parser()

    args = parser.parse_args()
    conf = BenchmarkConfig(args)
    
    util.echo_line(util.line_width, "-", "🔥 Benchmark Configuration")
    util.echo_info(conf.summary())
    
    logs = []
    for backend_opts in conf.backend_opts:
        if "--ross-config" in conf.args['sglang'][0]:
            conf_args = conf.args['sglang'][0]["--ross-config"]
            if "mem_fraction_static" in conf_args:
                mem_fraction_static_list = conf_args["mem_fraction_static"]
        for model in conf.models:
            for parallel in conf.parallel:
                for batch in conf.batches:
                    for inp in conf.inputs:
                        for mem_fraction_static in mem_fraction_static_list:
                            conf.set_curr('sglang', model, parallel, batch, inp)
                            conf.input = inp

                            conf.test_dst = conf.test_dst.replace(conf.gpuname, backend_opts[0])
                            conf.test_dst = conf.test_dst.replace(conf.backend_info[conf.backends[0]]['version'], backend_opts[1])      
                            conf.test_dst = conf.test_dst.replace(str(DEFAULT_RESERVE), str(mem_fraction_static))
                            if not os.path.exists(conf.test_dst):
                                continue

                            main_log = ""
                            for root, _, files in os.walk(conf.test_dst):
                                for f in files:
                                    if f.find(f"_0.log") != -1:
                                        main_log = os.path.join(root, f)
                            if main_log != "":
                                logs.append({
                                    "log_path": main_log,
                                    "model": model,
                                    "parallel": parallel,
                                    "batch_size": batch,
                                    "iosl": f"{inp['isl'][0]}@{inp['osl'][0]}",
                                })
    assert(logs)
    overheads = []
    for log_items in logs:
        log_path = log_items["log_path"]
        overhead = calculate_log_time_diff(log_path)
        if overhead == -1:
            continue
        overheads.append(overhead)
    
    

if __name__ == "__main__":
    main() 