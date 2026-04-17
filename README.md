# ROSS

ROSS is an offline simulator for LLM inference performance. Given a model, a hardware platform, and a workload description, it predicts throughput and latency metrics (TTFT, TPOT, ITL) without running a real inference server. It supports both SGLang and vLLM backends, colocated and disaggregated parallelism, and Pareto-front search over parallel configurations.

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Basic Usage](#basic-usage)
3. [Advanced Features](#advanced-features)
4. [Supported Models](#supported-models)
5. [Citation](#citation)

---

## Getting Started

### Repository layout

```
collector/          Platform profiling scripts + pre-collected hardware profiles
common/             Shared model/feature/config classes used by both simulators
ross/               Main simulator package
  sgl_sim/            SGLang simulator backend
  vllm_sim/           vLLM simulator backend
  pareto/             Pareto-front analysis utilities
  config/             Example JSON config files
  ross_predict.py     Entry point for offline prediction sweeps
modeling/           Pre-trained XGBoost regression models
test/               Unit and integration tests
```

### Installation

ROSS runs on a CPU-only host; no GPU is required for simulation.

```bash
pip install -r requirements-ross.txt
```

This installs the Python dependencies used by the simulator and arrival generation.

Pre-trained regression models trained on H200 and B200 profiling data are loaded at runtime from the directory specified by the `modeling_dir` field in your config file (or via the `--modeling-dir` CLI flag). These models have useful transferability to new GPU platforms, but if you want platform-specific calibration, follow the profiling workflow described in the [Advanced Features](#advanced-features) section and point `modeling_dir` at the resulting model directory.

Pre-trained ROSS modeling artifacts are also published at [CharlesCAOO/ross](https://huggingface.co/CharlesCAOO/ross). You can download them from Hugging Face and point `modeling_dir` to the local directory that contains the extracted files.

### A first prediction

Create a minimal JSON config:

```json
{
    "backend":   "sglang",
    "model":     "Meta-Llama-3.1-70B",
    "parallel":  "1:1:8",
    "batch":     [64],
    "num_prompt": 512,
    "rate":      ["1", "2", "inf"],
    "inputs":    ["sharegpt@0_0"],
    "platforms": [{"gpu": "H200", "version": "0.6.6"}],
    "output":    "~/.etc",
    "datapath":  "~/.etc",
    "model_search_paths": "/path/to/models",
    "modeling_dir":       "/path/to/modeling"
}
```

Run the simulator:

```bash
python ross/ross_predict.py --config my_config.json
```

Output includes per-request and aggregate latency/throughput metrics. Add `--record-path results.csv` to also persist a CSV log.

---

## Basic Usage

### The `ross_predict.py` entry point

`ross_predict.py` is the single entry point for all offline prediction sweeps. Configuration is read from a JSON file and may be overridden per-flag on the command line. Priority order (later wins):

```
built-in default  <  --config FILE  <  command-line flags
```

### Common CLI flags

| Flag                      | Description                                          |
| ------------------------- | ---------------------------------------------------- |
| `--config FILE`           | JSON config file (recommended for sweeps)            |
| `--backend sglang\|vllm`  | Simulator backend                                    |
| `--model`                 | Model name or absolute path                          |
| `--parallel`              | Parallelism spec, e.g. `1:1:8` or `1:1:4@1:1:4`      |
| `--batch`                 | Max batch size per GPU                               |
| `--rate`                  | Comma-separated request rates (`inf` for max tput)   |
| `--input`                 | Dataset + sequence-length spec                       |
| `--modeling-dir`          | Path to XGBoost model directory                      |
| `--model-search-paths`    | Comma-separated model search roots                   |
| `--record-path FILE`      | Write metrics to CSV                                 |
| `--eval`                  | Load trace logs and compute prediction error         |
| `--get-pareto-front`      | Enumerate configs and compute Pareto front           |
| `--debug`                 | Verbose logging                                      |
| `--vllm-src-root DIR`     | Enable the optional vLLM sidecar scheduler              |
| `--compare-vllm-schedule` | Compare simulator and vLLM sidecar schedules per step   |
| `--vllm-result-source`    | Use `sim` or `vllm` timing input for prediction         |

A complete list of CLI flags, JSON config fields, and engine-specific arguments is documented in [`docs/bench_config.md`](docs/bench_config.md).

### Parallelism format

```
# Prefill-decode colocate
dp:pp:tp                       e.g. 1:1:8   (1 DP × 1 PP × 8 TP = 8 GPUs)

# Prefill-decode disaggregation
dp:pp:tp@dp:pp:tp              e.g. 1:1:4@1:1:4 (4 prefill + 4 decode GPUs)

# Multiple configurations (comma-separated)
1:1:8,1:1:4,1:1:4@1:1:4
```

### Input (workload) format

```
dataset[@isl_osl]

dataset  ∈ sharegpt | repoqa | aime
isl, osl = integer (exact length; 0 = use the dataset's natural length, no cap)

Examples:
  sharegpt                 # dataset defaults (ISL≈500, OSL≈100)
  sharegpt@0_0             # use each sample's natural ISL and OSL, no cap
  sharegpt@0_100           # natural ISL, OSL forced to 100
  repoqa@4096_1024         # ISL=4096, OSL=1024
  aime@512_8192            # ISL=512, OSL=8192
```

Setting `isl` or `osl` to `0` tells the dataset loader to keep the original sequence length from the source dataset instead of truncating or padding to a fixed value. This is the recommended setting when you want the simulator to see the workload's real length distribution (e.g. `sharegpt@0_0`).

### Validating against a real trace

When real benchmark traces are available, add `--eval` to compute per-configuration percentage error (PE) for E2E latency, TTFT, TPOT, and ITL against the recorded ground truth:

```bash
python ross/ross_predict.py --config my_config.json --eval
```

---

## Advanced Features

### Pareto-front search over parallelism

Pass `--get-pareto-front` to enumerate all valid parallel configurations (colocated and PD-disaggregated) for a given model and hardware budget and plot the Pareto frontier of *tokens/s/user* vs *tokens/s/GPU*:

```bash
python ross/ross_predict.py --config my_config.json --get-pareto-front
```

A 228-candidate sweep for a 32B model on an 8-GPU B200 cluster completes in about 70 minutes on a CPU-only server — roughly **1,258× cheaper** than exhaustive on-hardware evaluation. `--get-pareto-front` is mutually exclusive with `--eval`.

### Prefill–decode disaggregation

Disaggregated deployments are expressed by splitting the parallelism spec with an `@`:

```bash
python ross/ross_predict.py \
    --config base.json \
    --parallel "1:1:4@1:1:4"
```

ROSS models the KV-cache transfer between the prefill and decode workers as part of the virtual-clock critical path.

### Multi-dimensional sweeps

Any sweep dimension — backend, model, parallelism, batch size, request rate, dataset, platform, engine argument — can be a list. ROSS iterates the Cartesian product and writes one row per cell to the CSV record:

```json
{
    "backend":  ["vllm", "sglang"],
    "model":    ["Qwen2.5-72B-Instruct", "Llama-3.1-70B"],
    "parallel": ["1:1:8", "1:1:4@1:1:4"],
    "batch":    [32, 64],
    "rate":     ["1", "2", "4", "inf"],
    "inputs":   ["sharegpt@500_100", "repoqa@4096_1024"],
    "platforms": [
        {"gpu": "H200", "version": "0.6.6.post1"},
        {"gpu": "B200", "version": "0.7.0"}
    ],
    "model_search_paths": "/path/to/models",
    "modeling_dir":       "/path/to/modeling"
}
```

### Engine-specific arguments

Forward framework-specific knobs through `ross_extra` (config file) or `--args` (CLI). Each entry may contain lists to trigger an inner sweep.

```json
"ross_extra": [
    {
        "backend":              "sglang",
        "mem_fraction_static":  [0.85, 0.9],
        "chunked_prefill_size": [8192, 16384]
    },
    {
        "backend":                "vllm",
        "gpu_memory_utilization": [0.9],
        "max_num_batched_tokens": [8192, 16384]
    }
]
```

CLI equivalent:

```bash
--args "sglang@mem_fraction_static=0.9,chunked_prefill_size=8192"
```

### vLLM sidecar scheduler

For vLLM, ROSS can run an optional sidecar scheduler. You can use it either to:

- compare each scheduling step against the simulator scheduler
- or drive timing features from the vLLM sidecar output

- Sidecar enablement requires `--vllm-src-root /path/to/vllm`
- Step-by-step comparison additionally requires `--compare-vllm-schedule`
- To use sidecar timing for prediction without strict comparison, set `--vllm-result-source vllm`
- CLI-only; not read from JSON config

Example:

```bash
python ross/ross_predict.py \
  --config my_config.json \
  --vllm-src-root /path/to/vllm \
  --vllm-result-source vllm
```

### Parallel workers

Large sweeps are parallelized across CPUs via `--max-workers` and `--threads-per-worker`. Both default to auto-selection based on the host's CPU count and scale linearly with available cores.

### Profiling a new GPU platform

ROSS's data plane is trained from sparse per-platform profiles collected under `collector/`. To target a new GPU, run the provided profiling scripts on that platform (~3–4 wall-clock hours on an 8-GPU node) and retrain the stage-wise regressor. The resulting XGBoost model is then pointed to via `modeling_dir` in your config. The control plane requires no changes because it is reused from the native serving framework.

Step-by-step instructions for profiling a new platform are in [`docs/profiling.md`](docs/profiling.md).

### Stage-level discrepancy analysis

Because ROSS decomposes each iteration into pre/forward/post stages, comparing simulated and measured stage times isolates where a real system deviates from its predictable behavior. This was used to localize a batch-boundary bottleneck in SGLang's TokenizerManager, whose structural fix reduced end-to-end latency by 36% at high concurrency.

---

## Supported Models

ROSS's stage-wise regressor takes **model configuration features** as input rather than per-model kernel calibration, so new models within a supported family typically work out of the box. Validated models include:

| Family      | Variants                                      |
| ----------- | --------------------------------------------- |
| Llama-3.1   | 8B, 70B                                       |
| Qwen2.5     | 72B                                           |
| Qwen3       | 32B, 30B-A3B (MoE), 235B-A22B (MoE)           |
| DeepSeek-V3 | 671B (MoE)                                    |
| gpt-oss     | 20b, 120b                                     |

**Backends:** vLLM, SGLang
**Deployment modes:** colocated, PD-disaggregated
**Hardware:** Trained on NVIDIA H200 and B200 profiles, with practical generalization to new GPUs; for best accuracy on a new platform, profile it with the `collector/` workflow.
**Datasets:** ShareGPT, RepoQA, AIME

Models and platforms outside this list can generally be added by:

1. Placing the HF-format model under your configured `model_search_paths`, and
2. Profiling the target GPU with the `collector/` scripts if you want platform-specific calibration.
