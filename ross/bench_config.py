#!/usr/bin/python3
# -*- coding: utf-8 -*-

import argparse
import os
import re
import sys
import json
from pathlib import Path


###############################################################################
# Resolve repo structure
###############################################################################
def resolve_repo(arg0: str):
    path = Path(arg0).expanduser().resolve()
    repopath, reponame = str(Path(path).parents[1]), Path(path).parents[1].name
    sys.path.append(str(repopath) + "/ross")
    import util
    sys.modules["util"]           = util
    etcpath = Path("~").expanduser().resolve()
    return str(repopath), str(reponame), str(etcpath) + "/.etc"

repopath, reponame, etcpath = resolve_repo(sys.argv[0])
import util
###############################################################################

DEFAULT_MODELING_DIR = str(Path(repopath) / "modeling")

avail_backends   = ["vllm", "sglang", "tensorrt"]
avail_mode       = ["online", "offline"]
avail_dataset    = ["sharegpt", "random", "repoqa", "aime"]
DEFAULT_MODE     = "online"
DEFAULT_GPU      = "H200"
DEFAULT_MODELS   = "Qwen2.5-72B-Instruct"
DEFAULT_PARALLEL = "1:1:8"
DEFAULT_BATCH    = "64" # per GPU
DEFAULT_RATE     = "inf"
DEFAULT_INPUT    = "sharegpt@0_0"
DEFAULT_RESERVE  = 0.9
DEFAULT_CHUNK_PREFILL_SIZE = 8192
DEFAULT_MODEL_SEARCH_PATHS = ["~/models"]
DEFAULT_RANDOM_RANGE_RATIO = 0.0

class BenchmarkConfig:
    """
    Configuration object built from parsed command-line arguments.
    Stores benchmark settings and provides convenience properties such as
    PD separation enablement, parsed parallelism, etc.
    """

    def __init__(self, args):
        self.pre_init()

        try:
            self.gpuname, _   = util.cuda_info()
        except:
            self.gpuname  = DEFAULT_GPU
        self.num_prompt   = 512

        self.args = {b: [] for b in avail_backends}
        import torch
        num_gpu = torch.cuda.device_count()
        self.num_gpu      = num_gpu

        self.backends     = self._split(args.backend)
        self.platforms    = None
        self.model_specs  = self._split(args.model)
        self.models       = []
        self.mode         = args.mode

        self.disaggregation_mode = "colocation"

        self.parallel     = args.parallel
        self.batches      = self._split_int(args.batch)
        self.rates        = self._split(args.rate)

        self.input        = self._parse_input(args.input)
        self.inputs       = []

        self.output       = args.output
        self.datapath     = args.datapath
        self.random_range_ratio = args.random_range_ratio
        self.model_search_paths = self._as_list(args.model_search_paths) if getattr(args, "model_search_paths", None) else []
        self.config       = str(Path(args.config).expanduser().resolve()) if args.config else None
        self.modeling_dir = args.modeling_dir if hasattr(args, "modeling_dir") else None
        self._parse_args(args.args)

        self.apply_conf()
        self.apply_default()
        self.models = [util.get_model(m, self.model_search_paths) for m in self.model_specs]
        self.post_init()

        self.check()

    def pre_init(self):
        self.args = {}

    def post_init(self):
        self.backend_info = {b: {} for b in self.backends}
        for framework in self.backend_info.keys():
            venv = f"{repopath}/venv/{framework}"
            if framework not in ["vllm", "sglang", "tensorrt"]:
                venv = f"{repopath}/venv/vllm"
            if Path(venv).is_dir():
                version = util.echo_back(f"bash -l -c 'source {venv}/bin/activate && pip show {framework.lower()} | grep \"^Version:\" | cut -d\" \" -f2'", capture=True)
            else:
                if not self.platforms:
                    raise RuntimeError(f"{venv} does not exist!")

                version = self.platforms[0]["version"]
            self.backend_info[framework] = {
                "version": version if version is not None and version != "" else "nover",
                "venv": venv
            }

        self.args = {b: self.args[b] for b in self.backends}
        for backend, params_list in self.args.items():
            for param_dict in params_list:
                venv = self.backend_info[backend]["venv"]
                if venv.find("vllm") != -1:
                    if "gpu_memory_utilization" not in param_dict.keys():
                        param_dict["gpu_memory_utilization"] = [DEFAULT_RESERVE]
                    if "max_num_batched_tokens" not in param_dict.keys():
                        param_dict["max_num_batched_tokens"] = [DEFAULT_CHUNK_PREFILL_SIZE]
                elif venv.find("sglang") != -1:
                    if "mem_fraction_static" not in param_dict.keys():
                        param_dict["mem_fraction_static"] = [DEFAULT_RESERVE]
                    if "chunked_prefill_size" not in param_dict.keys():
                        param_dict["chunked_prefill_size"] = [DEFAULT_CHUNK_PREFILL_SIZE]

        # self.parallel = self._parse_parallel(self.parallel)
        self.parallel = self._split(self.parallel)
        for input in ([self.input] + self.inputs):
            location = ""
            if input["dataset"] in {"sharegpt", "random"}:
                location = f"{self.datapath}/ShareGPT_V3_unfiltered_cleaned_split.json"
            elif input["dataset"] == "repoqa":
                location = f"{self.datapath}/repoqa.jsonl"
            elif input["dataset"] == "aime":
                location = ""  # loaded from HuggingFace, no local file needed
            input["path"] = location

        if location != "":
            if not Path(location).is_file():
                util.echo_erro(f"{location} does not exist!")

        self.cur_test = 0
        self.num_test = sum([len(arg) for arg in self.args.values()]) * \
                        len(self.parallel) * len(self.batches) * len(self.models) * len(self.inputs)
        self.test_backend, self.test_model, self.test_parallel, self.test_batch = None, None, None, None
        self.test_venv, test_extra = None, None
    
    def apply_default(self):
        if len(self.model_specs) == 0:
            self.model_specs = [DEFAULT_MODELS]
        if not self.mode:
            self.mode = DEFAULT_MODE
        if self.parallel is None:
            self.parallel = DEFAULT_PARALLEL
        if len(self.batches) == 0:
            self.batches = self._split_int(DEFAULT_BATCH)
        if len(self.rates) == 0:
            self.rates = self._split(DEFAULT_RATE)
        if self.input is None:
            self.input = self._parse_input(DEFAULT_INPUT)
        if not self.output:
            self.output = str(Path(etcpath).expanduser().resolve())
        if not self.datapath:
            self.datapath =  str(Path(etcpath).expanduser().resolve())
        if not self.modeling_dir:
            self.modeling_dir = DEFAULT_MODELING_DIR
        if self.random_range_ratio is None:
            self.random_range_ratio = DEFAULT_RANDOM_RANGE_RATIO
        if not self.model_search_paths:
            self.model_search_paths = list(DEFAULT_MODEL_SEARCH_PATHS)
        for key in self.args.keys():
            if len(self.args[key]) == 0:
                self.args[key].append({})

    def apply_conf(self):
        if not self.config:
            return 
        conf_path = Path(self.config)
        if not conf_path.is_file():
            raise FileNotFoundError(f"Config file not found: {self.config}")

        with open(conf_path, "r") as f:
            conf = json.load(f)

            if len(self.backends) == 0 and "backend" in conf.keys():
                self.backends = self._as_list(conf["backend"])
            if "platforms" in conf.keys():
                self.platforms = [{"gpu": p["gpu"], "version": p["version"]} for p in conf["platforms"]]
            elif "backend_opts" in conf.keys():
                # backward-compat: old list-of-arrays format [["H200", "0.5.6"]]
                self.platforms = [{"gpu": p[0], "version": p[1]} for p in conf["backend_opts"]]
            if len(self.model_specs) == 0 and "model" in conf.keys():
                self.model_specs = self._as_list(conf["model"])
            if not self.mode and "mode" in conf.keys():
                self.mode = conf["mode"]
            if "disaggregation_mode" in conf.keys():
                self.disaggregation_mode = conf["disaggregation_mode"]
            if self.parallel is None and "parallel" in conf.keys():
                self.parallel = ",".join(self._as_list(conf["parallel"]))
            if "num_prompt" in conf.keys():
                self.num_prompt = conf["num_prompt"]
            if len(self.batches) == 0 and "batch" in conf.keys():
                self.batches = [int(x) for x in self._as_list(conf["batch"])]
            if len(self.rates) == 0 and "rate" in conf.keys():
                self.rates = conf["rate"]
            if self.input is None and "input" in conf.keys():
                self.input = self._parse_input(conf["input"])
                self.inputs = [self.input]
            # add inputs: dataset/iosl list ( only validate first pair )
            if "inputs" in conf.keys():
                for input in conf["inputs"]:
                    self.inputs.append(self._parse_input(input))
                self.input = self.inputs[0]
            if not self.output and "output" in conf.keys():
                self.output = str(Path(conf["output"]).expanduser().resolve())
            if not self.datapath and "datapath" in conf.keys():
                self.datapath = str(Path(conf["datapath"]).expanduser().resolve())
            if self.random_range_ratio is None and "random_range_ratio" in conf.keys():
                self.random_range_ratio = float(conf["random_range_ratio"])
            if not self.modeling_dir and "modeling_dir" in conf.keys():
                self.modeling_dir = str(Path(conf["modeling_dir"]).expanduser().resolve())
            if not self.model_search_paths and "model_search_paths" in conf.keys():
                self.model_search_paths = [
                    str(Path(p).expanduser().resolve()) for p in self._as_list(conf["model_search_paths"])
                ]

            if "ross_extra" in conf.keys():
                self._parse_args(conf["ross_extra"])

    def check(self):
        if len(self.backends) == 0:
            util.echo_erro(f"No engine is specified!")
        if not set(self.backends).issubset(set(avail_backends)):
            util.echo_erro(f"Unrecognized backends: {self.backends}")
        if self.mode not in avail_mode:
            util.echo_erro(f"Unrecognized mode: '{self.mode}'")
        if self.input["dataset"] not in avail_dataset:
            util.echo_erro(f"Unrecognized dataset type: {self.input}")
        if not Path(self.datapath).is_dir():
            util.erro(f"{self.datapath} does not exist!")

    def parse_parallel(self, s, remove_p=None, remove_d=None, sep=False):
        iter_host, iter_gpu = 0, 0
        parts = s.split("@")
        parts = [p.strip() for p in parts]
        per_conf = [s]

        if len(parts) == 1:
            p, iter_host, iter_gpu, assigned = self._parse_triplet(parts[0], iter_host, iter_gpu, remove_p)
            per_conf.append(assigned)
            
        elif len(parts) == 2:
            p, iter_host, iter_gpu, assigned_p = self._parse_triplet(parts[0], iter_host, iter_gpu, remove_p)
            if sep:
                iter_host += 1
                iter_gpu = 0
            d, iter_host, iter_gpu, assigned_d = self._parse_triplet(parts[1], iter_host, iter_gpu, remove_d)
            per_conf.append(assigned_p)
            per_conf.append(assigned_d)

        else:
            raise ValueError(f"Invalid --parallel format: {s}")

        return per_conf

    def _as_list(self, value):
        if isinstance(value, str):
            return self._split(value)

        if isinstance(value, list):
            value = [x for x in value if x not in (None, "")]
            return value

        raise TypeError(
            f"Invalid type for list-like config: expected str (comma-separated) or list, "
            f"but got {type(value).__name__}: {value!r}"
        )

    def _split(self, s):
        value = [x.strip() for x in s.split(",")] if s else []
        value = [x for x in value if x not in (None, "")]
        return value

    def _split_int(self, s):
        return [int(x) for x in s.split(",")] if s else []
    

    def _parse_args(self, extra):
        if not extra:
            return

        if isinstance(extra, str):
            BACKEND = r"[A-Za-z_\-]+"
            ARG = r"(?:[A-Za-z_][A-Za-z0-9_\-]*=[^,]+|--[A-Za-z0-9_\-]+(?:=[^,]+)?)"
            SEG = rf"{BACKEND}@{ARG}(?:,{ARG})*"
            PATTERN = rf"^{SEG}(?:;{SEG})*$"
            pattern = re.compile(PATTERN)

            if bool(pattern.match(extra.strip())):
                for seg in extra.split(";"):
                    alist = seg.split("@")
                    temp = {}
                    for item in alist[1].split(","):
                        parts = item.split("=", 1)
                        if len(parts) == 1:
                            key = parts[0]
                            val = ""
                        else:
                            key, val = parts
                        temp[key] = val
                    self.args[alist[0]].append(temp)
            else:
                util.echo_erro(f"Unrecognized args: {extra}")
        elif isinstance(extra, list):
            for i, temp in enumerate(extra):
                if "backend" not in temp:
                    raise RuntimeError(
                        f"Invalid config at index {i}: each item in 'extra' must contain a 'backend' key. "
                        f"Got: {temp!r}"
                    )

                backend = temp["backend"]
                temp.pop("backend")
                self.args[backend].append(temp)


    def _parse_triplet(self, s, iter_host, iter_gpu, remove_p):
        dp, pp, tp = s.split(":")
        ret = {
            "dp": int(dp),
            "pp": int(pp),
            "tp": int(tp),
            "num_gpu": int(dp) * int(pp) * int(tp)
        }
        if remove_p is not None and remove_p in ret.keys():
            ret["num_gpu"] = ret["num_gpu"] // ret[remove_p] 
        num_gpu, ih, ig, flag  = ret["num_gpu"], 0, 0, False
        all_assigned = []
        for ih in range(iter_host, len(self.workers)):
            assigned = []
            for ig in range(iter_gpu, self.workers[ih][2]):
                assigned.append(ig)
                num_gpu -= 1
                if num_gpu == 0:
                    flag = True
                    break

            item = (ih, self.workers[ih][0], self.workers[ih][1], self.workers[ih][3], assigned)
            all_assigned.append(item)
            if flag == True:
                break
            iter_gpu = 0

        if num_gpu > 0:
            raise RuntimeError("Not enough GPUs to launch the specified configuration!")

        ig += 1
        if ig == self.workers[ih][2]:
            ih, ig = ih + 1,  0
        return ret, ih, ig, all_assigned

    def _parse_len_range(self, text):
        text = text.strip()

        if ":" in text:
            a, b = text.split(":", 1)
            assert int(a) <= int(b)
            return (int(a), int(b))
        else:
            v = int(text)
            return (v, v)

    def _parse_input(self, s):
        if not s:
            return None
        if "@" not in s:
            return {
                "dataset": s.strip(),
                "isl": (500, 500),
                "osl": (100, 100),
            }
        dataset, seq = s.split("@", 1)
        dataset = dataset.strip()
        if "_" not in seq:
            raise ValueError(f"Invalid --input '{s}', missing '_' between isl and osl")
        isl_str, osl_str = seq.split("_", 1)
        return {
            "dataset": dataset,
            "isl": self._parse_len_range(isl_str),
            "osl": self._parse_len_range(osl_str),
        }

    def set_curr(self, backend_, model_, parallel_, batch_, input = None, platform = None):
        self.test_backend  = backend_
        self.test_model    = model_ 
        self.test_parallel = parallel_ 
        self.test_batch    = batch_
        self.test_venv     = self.backend_info[backend_]["venv"]
        self.cur_test      = self.cur_test + 1
        
        target_gpu         = platform["gpu"] if platform else self.gpuname
        version            = platform["version"] if platform else self.backend_info[backend_]["version"]
        if not input:
            dataset, isl, osl  = self.input["dataset"], self.input["isl"], self.input["osl"]
        else:
            dataset, isl, osl  = input["dataset"], input["isl"], input["osl"]
        datatype           = f"{dataset}_isl_{isl[0]}_osl_{osl[0]}"
        p                  = Path(model_)
        last               = p.name
        model_name         = p.parent.name if last.startswith("v") else last
        opt, res           = "opt", DEFAULT_RESERVE
        # if self.mode == "online":
        #     if "--mem-fraction-static" in extra_.keys():
        #         res = extra_["--mem-fraction-static"]
        #     elif "--gpu-memory-utilization" in extra_.keys():
        #         res = extra_["--gpu-memory-utilization"]
        #     if "--disable-cuda-graph" in extra_.keys() or "--enforce-eager" in extra_.keys():
        #         opt = "eag"
        # else:
        #     if "mem_fraction_static" in extra_.keys():
        #         res = extra_["mem_fraction_static"]
        #     elif "gpu_memory_utilization" in extra_.keys():
        #         res = extra_["gpu_memory_utilization"]
        #     if "disable_cuda_graph" in extra_.keys() or "enforce_eager" in extra_.keys():
        #         opt = "eag"
        self.test_dst      = f"{self.output}/{target_gpu}/{self.mode}_{opt}_{res}_{backend_}_{version}/{model_name}_{datatype}_{parallel_}_batch_{batch_}"
        if self.num_prompt != 512:
            self.test_dst = self.test_dst + f"_promptnum_{self.num_prompt}"
        Path(self.test_dst).expanduser().resolve().mkdir(parents=True, exist_ok=True)
    
    def get_curr(self):
        is_moe = False
        with open(f"{self.test_model}/config.json", "r") as f:
            conf = json.load(f)
            for key in conf.keys():
                if key.find("experts") != -1:
                    is_moe = True
        return self.test_backend, self.test_model, self.test_parallel, self.test_batch, self.test_venv, is_moe

    def summary(self):
        """Printable config summary."""
        s = []
        # Backend + version
        s.append(f"Backend      : {self.backends[0]:8}  version={self.backend_info[self.backends[0]]['version']}")
        for i in range(1, len(self.backends)):
            s.append(f"               {self.backends[i]:8}  version={self.backend_info[self.backends[i]]['version']}")
        # Platforms (GPU targets, predict-mode only)
        if self.platforms:
            gpus = ", ".join(f"{p['gpu']} ({p['version']})" for p in self.platforms)
            s.append(f"       Platforms    : {gpus}")
        s.append(f"       Models       : {[Path(m).name for m in self.models]}")
        s.append(f"       Mode         : {self.mode}")
        s.append(f"       Parallel     : {self.parallel}")
        s.append(f"       Batches      : {self.batches}")
        s.append(f"       Rates        : {self.rates}")
        s.append(f"       Inputs       : {[inp.get('dataset','?') + '@' + str(inp.get('isl',('?','?'))[0]) + '_' + str(inp.get('osl',('?','?'))[0]) for inp in self.inputs]}")
        s.append(f"       Output       : {self.output}")
        s.append(f"       Random ratio : {self.random_range_ratio}")
        s.append(f"       Modeling dir : {self.modeling_dir}")
        s.append(f"       Model search : {self.model_search_paths}")
        s.append(f"       Config       : {self.config}")
        # Scheduler args (only the current backend's params)
        for i, key in enumerate(self.args.keys()):
            label = "Scheduler args" if i == 0 else "              "
            s.append(f"       {label} : {key} = {self.args[key]}")
        return "\n".join(s)

def build_parser():
    """Build the base argument parser shared by bench.py and predict scripts."""
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument("--backend",    type=str,   help="Comma-separated backend list")
    parser.add_argument("--mode",       type=str,   choices=avail_mode)
    parser.add_argument("--model",      type=str,   help="Model names, comma-separated")
    parser.add_argument("--parallel",   type=str,   help="dp:pp:tp or dp:pp:tp,dp:pp:tp")
    parser.add_argument("--batch",      type=str,   help="Batch sizes, comma-separated")
    parser.add_argument("--rate",       type=str,   help="Request rates, comma-separated")
    parser.add_argument("--input",      type=str,   help="Dataset input spec")
    parser.add_argument("--output",     type=str,   help="Output directory")
    parser.add_argument("--datapath",   type=str,   help="Absolute path to dataset directory")
    parser.add_argument("--random-range-ratio", type=float, dest="random_range_ratio", help=f"Range ratio for bench_serving random dataset (default: {DEFAULT_RANDOM_RANGE_RATIO})")
    parser.add_argument("--config",       type=str,   help="Path to JSON config file")
    parser.add_argument("--modeling-dir", type=str,   dest="modeling_dir", help=f"Path to modeling directory containing xgboost models (default: {DEFAULT_MODELING_DIR})")
    parser.add_argument("--model-search-paths", type=str, dest="model_search_paths", help="Comma-separated model search roots used to resolve model names")
    parser.add_argument("--args",       type=str,   help="Extra args for engine")
    parser.add_argument("-h", "--help", action="store_true")

    return parser


def show_predict_help():
    """Print help text for ross_predict.py."""
    help_text = f"""
usage: ross_predict.py [options]

ROSS Simulator — offline prediction script.
Runs the ROSS simulator over a sweep of configurations and optionally
compares results against real execution traces.

Select a backend with --backend ({" | ".join(avail_backends)}).

────────────────────────────────────────────────────────────────────────────
SHARED OPTIONS  (see docs/bench_config.md for full reference)
────────────────────────────────────────────────────────────────────────────

  --config FILE            Path to JSON configuration file.
                           Priority: default < config file < command-line args.
                           Recommended for large sweeps; keeps command lines short.

  --backend BACKEND        Backend to simulate: {" | ".join(avail_backends)}  (REQUIRED)

  --model MODELS           Comma-separated model names or full paths.
                           Default: {DEFAULT_MODELS}

  --parallel PARALLEL      Comma-separated parallelism specs.
                           Colocate  : dp:pp:tp           (e.g. 1:1:8)
                           Disaggreg.: dp:pp:tp@dp:pp:tp  (e.g. 1:1:4@1:1:4)
                           Default: {DEFAULT_PARALLEL}

  --batch BATCH_SIZES      Comma-separated max-batch-size values (per GPU).
                           Default: {DEFAULT_BATCH}

  --rate RATES             Comma-separated request rates (req/s).
                           Use "inf" for offline / max-throughput mode.
                           Default: {DEFAULT_RATE}

  --input INPUT_SPEC       Dataset and sequence-length spec.
                           Format: dataset[@isl_osl]
                             dataset  ∈ {{{", ".join(avail_dataset)}}}
                             isl/osl  : integer (e.g. 128); 0 = use the
                                        dataset's natural length, no cap
                           Examples:
                             sharegpt               (use built-in ISL/OSL defaults)
                             sharegpt@0_0           (natural ISL and OSL)
                             repoqa@4096_1024
                             aime@512_8192
                           Default: {DEFAULT_INPUT}

  --output OUTPUT_PATH     Root directory under which --eval looks up real
                           execution traces. Not read in pure forward-
                           prediction mode.
  --datapath PATH          Absolute path to dataset directory.
  --random-range-ratio R   Range ratio forwarded to bench_serving random dataset.
                           Default: {DEFAULT_RANDOM_RANGE_RATIO}
  --model-search-paths     Comma-separated model search roots used to resolve model names.
  --modeling-dir PATH      Path to xgboost model directory.
                           Default: <repo_root>/modeling

  --mode MODE              online | offline  (default: {DEFAULT_MODE})
  --args STRING            Backend scheduler arguments.
                           SGLang: --args "sglang@mem_fraction_static=0.9,chunked_prefill_size=8192"
                           vLLM:   --args "vllm@gpu_memory_utilization=0.9,max_num_batched_tokens=8192"

────────────────────────────────────────────────────────────────────────────
PREDICT-ONLY OPTIONS
────────────────────────────────────────────────────────────────────────────

  --debug                  Enable verbose/DEBUG-level logging.
  --fast                   Use the fast vLLM simulator entry when available.
  --record-path FILE       Write per-configuration PE metrics to a CSV file.
  --eval                   Load real trace logs and compare against simulator results.

  --max-workers INT        Number of parallel worker processes. 0 = auto (default: 0)
  --threads-per-worker INT CPU threads per worker. 0 = auto (default: 0)

  --get-pareto-front       Enumerate parallel configs and compute Pareto front
                           (tokens/s/user vs tokens/s/gpu). Incompatible with --eval.

────────────────────────────────────────────────────────────────────────────
CONFIG FILE EXAMPLE
────────────────────────────────────────────────────────────────────────────

  {{
      "backend": "sglang",
      "model":   "Qwen2.5-72B-Instruct",
      "parallel": "1:1:8",
      "batch":   [32],
      "rate":    ["1", "2", "4", "inf"],
      "input":   "sharegpt@0_0",
      "random_range_ratio": 0.0,
      "model_search_paths": ["~/models"],
      "modeling_dir": "~/modeling",
      "platforms": [{{"gpu": "H200", "version": "0.6.6.post1"}}],
      "ross_extra": [{{"backend": "sglang", "mem_fraction_static": [0.9], "chunked_prefill_size": [8192]}}]
  }}

  python ross_predict.py --config config.json --record-path out.csv

  -h, --help               Show this help message and exit.
"""
    print(help_text)


def build_predict_parser():
    """Build the argument parser for ross_predict.py."""
    parser = build_parser()

    # ── predict-specific arguments ──────────────────────────────────────────
    parser.add_argument("--debug", action="store_true", default=False,
                        help="Enable verbose/DEBUG-level logging.")
    parser.add_argument("--fast", action="store_true", default=False,
                        help="Use the fast vLLM simulator entry when available.")
    parser.add_argument("--record-path", type=str, default="", metavar="FILE",
                        help="Write per-configuration PE metrics to a CSV file.")
    parser.add_argument("--eval", action="store_true", default=False,
                        help="Load real trace logs and compare against simulator results.")
    parser.add_argument("--max-workers", "--max-worker", dest="max_workers", type=int, default=0,
                        help="Number of parallel worker processes. 0=auto.")
    parser.add_argument("--threads-per-worker", type=int, default=0,
                        help="CPU threads per worker process. 0=auto.")
    parser.add_argument("--get-pareto-front", action="store_true",
                        help="Enumerate parallel configs and compute Pareto front.")
    parser.add_argument("--cache-worker-config", action="store_true",
                        help="Cache heavy simulator objects within each worker (SGL only).")

    return parser


def parse_predict_args():
    """Parse sys.argv for ross_predict.py, printing help and exiting if needed."""
    parser = build_predict_parser()
    args = parser.parse_args()

    if args.help or len(sys.argv) <= 1:
        show_predict_help()
        sys.exit(0)

    return args
