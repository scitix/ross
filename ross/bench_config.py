#!/usr/bin/python3
# -*- coding: utf-8 -*-

import argparse
import os
import re
import sys
import json
from pathlib import Path
import subprocess


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

avail_backends   = ["vllm", "sglang", "tensorrt"]
avail_mode       = ["online", "offline"]
avail_dataset    = ["random", "sharegpt", "bigcodebench", "humanevalplus", "repoqa"]
DEFAULT_MODE     = "online"
DEFAULT_GPU      = "H200"
DEFAULT_MODELS   = "Qwen2.5-72B-Instruct"
DEFAULT_PARALLEL = "1:1:8"
DEFAULT_BATCH    = "32" # per GPU
DEFAULT_RATE     = "inf"
DEFAULT_INPUT    = "sharegpt@500_100"
DEFAULT_RESERVE  = 0.9
DEFAULT_CHUNK_PREFILL_SIZE = 8192

def show_help():
    help_text = f"""
usage: bench.py [options]

optional arguments:
  --backend BACKENDS       Comma-separated list of backends ({"/".join(avail_backends)}) to benchmark.
  --model MODELS           Comma-separated list of models (name or full path) (default: {DEFAULT_MODELS}).
  --mode MODE              Benchmark mode: "online" or "offline" (default: {DEFAULT_MODE}).
  
  --parallel PARALLEL      Comma-separated list of parallelism configuration (default: {DEFAULT_PARALLEL}). 
                               Format: conf1,conf2,conf3,...
                               Each conf can be
                               dp:pp:tp           Prefill-decode colocate mode.
                               dp:pp:tp@dp:pp:tp  Prefill-decode disaggregation mode (prefill parallelism, decode parallelism]
  --batch BATCH_SIZES      Comma-separated list of batch sizes (default: {DEFAULT_BATCH}).
  --rate RATE              Comma-separated list of request rates (default: {DEFAULT_RATE}).

  --input INPUT_SPEC       Input dataset type. Format: dataset[@isl_osl], e.g., random@128_128, sharegpt@64:128_100.
                               Where:
                                  - dataset ∈ '{'{avail_dataset}'}'
                                  - isl and osl ([@isl_osl] is optional) can be 
                                    * a single integer (e.g., 128)
                                    * a range in the form X:Y (e.g., 64:256)
                               Default: {DEFAULT_INPUT}
  --output OUTPUT_PATH     Path for benchmark logs and result outputs (default: ~/.etc).
  --datapath PATH          Absolute path to dataset directory (default: ~/.etc).

  --worker WORKER_NAME     Comma-separated list of worker hostname for multi-machine benchmarks.
                           Not recommended to set manually. The runtime system will automatically select the appropriate machine
                           based on ~/.etc/cluster_info: if the requested number of GPUs does not exceed the local machine’s capacity,
                           the local host will be used; otherwise, the system will query ~/.etc/cluster_info for machines with the
                           same GPU type and automatically choose one that satisfies the requirements.

  --env KEY=VALUE          Declare environment variables (repeatable).
                           Examples:
                               OMP_NUM_THREADS=8,NUMEXPR_MAX_THREADS=16
  --args STRING            Extra arguments passed directly to the inference engine. 
                           Format: backend1@key=val,key=val,...;backend2@@key=val,key=val,...
                        
                           Notes:
                               - The backend name appears before the '@'.
                               - The same backend can appear multiple times, allowing you to test multiple argument sets for the same backend.
                               - Arguments after '@' are comma-separated.
                               - Each argument can be:
                                   key=value                                     For offline
                                   --flag=value or --flag (boolean flags)        For online
                           Examples:
                               --args "vllm@reserve=0.9,eager=True"
                               --args 'vllm@--reserve=0.9,--eager'
                               --args "vllm@reserve=0.9,eager=True;vllm@reserve=0.5,eager=False"

  --validation             Validate correctness of outputs.
  --override               Skip runs that already exist in output dir.
  --quiet                  Suppress non-critical output.

  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
  --config FILE            Path to configuration file. 
                           *** Priority: default value < config file < command line args. ***
                           For repeated or large-scale test runs, it is recommended to put frequently used arguments 
                           into the config file instead of typing long command lines every time. This allows running:
                               python bench.py --config FILE
                           to reproduce or simplify benchmark executions.
  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
  
  -h, --help               Show this help message and exit.
"""
    print(help_text)

import json

def generate_config_template(path="config_template.json"):
    """
    Generate a JSON config template containing all available arguments.
    """
    template = {
        "backend":       "vllm,sglang",
        "model":         DEFAULT_MODELS,
        "mode":          DEFAULT_MODE,
        "parallel":      DEFAULT_PARALLEL,
        "batch":         DEFAULT_BATCH,
        "input":         DEFAULT_INPUT,
        "output":        str(Path(etcpath).expanduser().resolve()),
        "datapath":      str(Path(etcpath).expanduser().resolve()),
        "worker":        "",

        # special fields
        "env":           {"OMP_NUM_THREADS": "2"},
        "args":          "",
        "override":      False,
        "quiet":         False,
        "validation":    False,
        "verbose":       False,
    }

    with open(path, "w") as f:
        json.dump(template, f, indent=4)

    print(f"Config template written to {path}")


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
        self.hostname     = util.get_local_hostname()
        self.address      = util.get_local_ip()
        self.username     = "root"
        self.num_prompt   = 512
        self.port_ssh     = util.port_ssh
        self.port_ray     = 55556

        self.args = {b: [] for b in avail_backends}
        self.all_workers  = []
        import torch
        num_gpu = torch.cuda.device_count()
        self.num_gpu      = num_gpu
        with open(etcpath + "/cluster_info", "r") as f:
            for line in f.readlines():
                items = line.strip().split()
                if items[3] == self.gpuname:
                    self.all_workers.append((items[1], items[2], num_gpu, items[3]))
                    if items[1] == self.hostname:
                        self.all_workers[0], self.all_workers[-1] = self.all_workers[-1], self.all_workers[0]

        self.raw          = args
        self.backends     = self._split(args.backend)
        self.backend_opts = None
        self.models       = [util.get_model(m) for m in self._split(args.model)]
        self.mode         = args.mode

        self.disaggregation_mode = "colocation"

        self.workers      = self._split(args.worker)
        self.parallel     = args.parallel
        self.batches      = self._split_int(args.batch)
        self.rates        = self._split(args.rate)

        self.input        = self._parse_input(args.input)
        self.inputs       = []

        self.output       = args.output
        self.datapath     = args.datapath
        self.override     = args.override
        self.quiet        = args.quiet
        self.validation   = args.validation
        self.config       = str(Path(args.config).expanduser().resolve()) if args.config else None
        self.envs.update(self._split_dict(args.env))
        self._parse_args(args.args)

        self.apply_conf()
        self.apply_default()
        self.post_init()

        self.check()

    def pre_init(self):
        self.envs, self.args = {}, {}
        self.envs["OMP_NUM_THREADS"] = 8
        self.envs["NUMEXPR_MAX_THREADS"] = 32

    def post_init(self):
        self.backend_info = {b: {} for b in self.backends}
        for framework in self.backend_info.keys():
            venv = f"{repopath}/venv/{framework}"
            if framework not in ["vllm", "sglang", "tensorrt"]:
                venv = f"{repopath}/venv/vllm"
            if Path(venv).is_dir():
                version = util.echo_back(f"bash -l -c 'source {venv}/bin/activate && pip show {framework.lower()} | grep \"^Version:\" | cut -d\" \" -f2'", capture=True)
            else:
                if not self.backend_opts:
                    raise RuntimeError(f"{venv} does not exist!")

                version = self.backend_opts[0][1]
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

        if len(self.workers) == 0:
            self.workers = [workers for workers in self.all_workers if workers[0] == self.hostname]
        else:
            s1 = set(self.workers)
            s2 = set([item[0] for item in self.all_workers])
            assert s1.issubset(s2), f"Available workers: {s2}, but got: {s1}"
            worker_dict = {item[0]: item for item in self.all_workers}
            self.workers = [worker_dict[host] for host in self.workers]

        # self.parallel = self._parse_parallel(self.parallel)
        self.parallel = self._split(self.parallel)
        for input in ([self.input] + self.inputs):
            location = ""
            if input["dataset"] == "sharegpt":
                location = f"{self.datapath}/ShareGPT_V3_unfiltered_cleaned_split.json"
            elif input["dataset"] == "bigcodebench":
                location = f"{self.datapath}/bigcodebench_complete.jsonl"
            elif input["dataset"] == "humanevalplus":
                location = f"{self.datapath}/humanevalplus.jsonl"
            elif input["dataset"] == "repoqa":
                location = f"{self.datapath}/repoqa.jsonl"
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
        if len(self.models) == 0:
            self.models = [util.get_model(DEFAULT_MODELS)]
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
            if "backend_opts" in conf.keys():
                self.backend_opts = conf["backend_opts"]
            if len(self.models) == 0 and "model" in conf.keys():
                models = self._as_list(conf["model"])
                self.models = [util.get_model(m) for m in models]
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
            if len(self.workers) == 0 and "worker" in conf.keys():
                self.workers = self._as_list(conf["worker"])

            if "env" in conf.keys():
                self.envs.update(conf["env"])
            if "ross_extra" in conf.keys():
                self._parse_args(conf["ross_extra"])
            if not self.validation and "validation" in conf.keys():
                self.validation = conf["validation"]
            if not self.quiet and "quiet" in conf.keys():
                self.quiet = conf["quiet"]
            if not self.override and "override" in conf.keys():
                self.override = conf["override"]

    def check(self):
        if len(self.backends) == 0:
            util.echo_erro(f"No engine is specified!")
        if not set(self.backends).issubset(set(avail_backends)):
            util.echo_erro(f"Unrecognized backends: {self.backends}")
        if self.mode not in avail_mode:
            util.echo_erro(f"Unrecognized mode: '{self.mode}'")
        if self.input["dataset"] not in avail_dataset:
            util.echo_erro(f"Unrecognized dataset type: {self.input}")
        if not Path(self.output).is_dir():
            util.erro(f"{self.output} does not exist!")
        if not Path(self.datapath).is_dir():
            util.erro(f"{self.datapath} does not exist!")

    def kill_procs(self, backend="sglang", venv=None):
        print()
        if backend == "sglang":
            util.echo_back(f"pkill -f sglang_router.launch_router")
            util.echo_back(f"pkill -9 -f 'sglang::'") # ::router
            
            venv = self.backend_info[backend]["venv"]
            for worker in self.workers:
                host_name, _, _, _ = worker
                util.echo_back(f"ssh -p {self.port_ssh} root@{host_name} 'pkill -f sglang.launch_server'")
                util.echo_back(f"ssh -p {self.port_ssh} root@{host_name} 'pkill -9 -f \"{venv}/bin/python3\"'")
        elif backend == "vllm":
            util.echo_back(f"pkill -9 -f 'vllm serve'")
            for worker in self.workers:
                host_name, _, _, _ = worker
                util.echo_back(f"ssh -p {self.port_ssh} root@{host_name} 'pkill -9 -f \"{venv}/bin/python3\"'")
                util.echo_back(f"ssh -p {self.port_ssh} root@{host_name} 'pkill -9 -f \"VLLM::\"'")

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
    
    def _split_dict(self, s):
        return {x.split("=")[0]: x.split("=")[1] for x in s.split(",")} if s else {}

    def _parse_env(self, env_list):
        """Parse repeated --env KEY=VALUE"""
        if not env_list:
            return {}
        env_dict = {}
        for item in env_list:
            if "=" not in item:
                raise ValueError(f"Invalid --env '{item}', expected KEY=VALUE")
            k, v = item.split("=", 1)
            env_dict[k.strip()] = v.strip()
        return env_dict

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

    def set_curr(self, backend_, model_, parallel_, batch_, input = None):
        self.test_backend  = backend_
        self.test_model    = model_ 
        self.test_parallel = parallel_ 
        self.test_batch    = batch_
        self.test_venv     = self.backend_info[backend_]["venv"]
        self.cur_test      = self.cur_test + 1
        
        version            = self.backend_info[backend_]["version"]
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
        self.test_dst      = f"{self.output}/LMDB/{self.gpuname}/{self.mode}_{opt}_{res}_{backend_}_{version}/{model_name}_{datatype}_{parallel_}_batch_{batch_}"
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

    def get_log(self, suffix=""):
        file = f"{self.output}/main_{suffix}.log"
        Path(file).write_text("")
        return file

    def save_log(self, log):
        util.echo_back(f"mv {log} {self.test_dst}")

    def summary(self):
        """Printable config summary."""
        s = []
        s.append(f"Backends     : {self.backends[0]:8}  version={self.backend_info[self.backends[0]]['version']:15} venv={self.backend_info[self.backends[0]]['venv']}")
        for i in range(1, len(self.backends)):
            s.append(f"                      {self.backends[i]:8}  version={self.backend_info[self.backends[i]]['version']:15} venv={self.backend_info[self.backends[i]]['venv']}")
        s.append(f"       Models       : {self.models}")
        s.append(f"       Mode         : {self.mode}")
        s.append(f"       Parallel     : {self.parallel}")
        s.append(f"       Batches      : {self.batches}")
        s.append(f"       Inputs       : {self.inputs}")
        s.append(f"       Output       : {self.output}")
        s.append(f"       Worker       : {self.workers}")
        s.append(f"       All Workers  : {self.all_workers}")
        s.append(f"       Env Vars     : {self.envs}")
        for i, key in enumerate(self.args.keys()):
            if i == 0:
                s.append(f"       Extra Args   : {key:8} = {self.args[key]}")
            else:
                s.append(f"                      {key:8} = {self.args[key]}")
        s.append(f"       Config path  : {self.config}")
        s.append(f"       Override     : {self.override}")
        s.append(f"       Validation   : {self.validation}")
        s.append(f"       Verbose      : {not self.quiet}")

        return "\n".join(s)

def build_parser():
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument("--backend",    type=str,   help="Comma-separated backend list")
    parser.add_argument("--mode",       type=str,   choices=avail_mode)
    parser.add_argument("--model",      type=str,   help="Model names, comma-separated")
    parser.add_argument("--parallel",   type=str,   help="dp:pp:tp or dp:pp:tp,dp:pp:tp")
    parser.add_argument("--batch",      type=str,   help="Batch sizes, comma-separated")
    parser.add_argument("--rate",       type=str,   help="Request rates, comma-separated")
    parser.add_argument("--input",      type=str,   help="Dataset input spec")
    parser.add_argument("--output",     type=str,   help="Output directory")
    parser.add_argument("--worker",     type=str,   help="")
    parser.add_argument("--datapath",   type=str,   help="")
    parser.add_argument("--config",     type=str,   help="")
    parser.add_argument("--env",        type=str,   help="Environment variables KEY=VALUE (repeatable)")
    parser.add_argument("--args",       type=str,   help="Extra args for engine")
    parser.add_argument("--validation", action="store_true")
    parser.add_argument("--override",   action="store_true")
    parser.add_argument("--quiet",      action="store_true")
    parser.add_argument("--verbose",    action="store_true")
    parser.add_argument("--kill",       action="store_true")
    parser.add_argument("-h", "--help", action="store_true")

    return parser

def main():
    global repopath, reponame, etcpath

    # generate_config_template()
    parser = build_parser()
    args = parser.parse_args()

    if args.help or len(sys.argv) <= 1:
        show_help()
        sys.exit(0)

    conf = BenchmarkConfig(args)

    util.echo_line(util.line_width, "-", "🔥 Benchmark Configuration")
    util.echo_info(conf.summary())

    util.echo_line(util.line_width, "-", "🔥 Apply Plugins")
    subprocess.run(
        ["./bench_apply_plugin.sh", "sglang", "online", conf.output],
        text=True,
        check=True
    )

    if args.kill:
        conf.kill_procs()
        return
    
    for input in conf.inputs:
        for backend in conf.backends:
            for model in conf.models:
                for parallel in conf.parallel:
                    for batch in conf.batches:
                        conf.set_curr(backend, model, parallel, batch, input)
                        conf.input = input
                        util.echo_line(util.line_width, "-", f"🫣 Test [{conf.cur_test:02}/{conf.num_test}]")
                        if Path(conf.test_venv).is_dir():
                            eval(f"bench_{backend}.bench_{conf.mode}(conf)")
                        else:
                            util.echo_warn(
                                f"Skipping test: LLM inference engine '{backend}' is not installed "
                                f"(expected at {conf.backend_info[backend]['venv']})."
                            )
                        print()

if __name__ == "__main__":
    main()
