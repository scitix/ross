# ROSS Benchmark Configuration Reference

This document describes every option available to the ROSS benchmark system —
both via command-line arguments and via JSON config files.

**Priority order** (later wins):

```
built-in default  <  config file (--config)  <  command-line argument
```

---

## Table of Contents

1. [Running the scripts](#running-the-scripts)
2. [CLI arguments](#cli-arguments)
   - [Shared arguments](#shared-arguments)
   - [Predict-only arguments](#predict-only-arguments)
3. [JSON config file reference](#json-config-file-reference)
   - [Full field listing](#full-field-listing)
   - [Annotated template](#annotated-template)
4. [Engine-specific extra args (`--args` / `ross_extra`)](#engine-specific-extra-args)
5. [Examples](#examples)

---

```bash
# Quick help:
python ross/ross_predict.py -h

# Run with a config file (recommended for sweeps):
python ross/ross_predict.py --config my_config.json --record-path out.csv
```

---

## CLI arguments

### Shared arguments

These are accepted by `ross_predict.py`.

| Argument                 | Type | Default                  | Description                                                                                                       |
| ------------------------ | ---- | ------------------------ | ----------------------------------------------------------------------------------------------------------------- |
| `--config FILE`        | str  | _(none)_               | Path to a JSON config file. Values inside override built-in defaults; CLI args override the file.                 |
| `--backend BACKENDS`   | str  | _(required)_           | Comma-separated backend names. Allowed values:`vllm`, `sglang`, `tensorrt`.                                 |
| `--model MODELS`       | str  | `Qwen2.5-72B-Instruct` | Comma-separated model names or absolute paths.                                                                    |
| `--mode MODE`          | str  | `online`               | Benchmark mode:`online` or `offline`.                                                                         |
| `--parallel PARALLEL`  | str  | `1:1:8`                | Comma-separated parallelism specs. See[Parallelism format](#parallelism-format).                                     |
| `--batch BATCH_SIZES`  | str  | `64`                   | Comma-separated max-batch-size values (per GPU).                                                                  |
| `--rate RATES`         | str  | `inf`                  | Comma-separated request rates (req/s). Use `inf` for max-throughput / offline mode.                             |
| `--input INPUT_SPEC`   | str  | `sharegpt@0_0`         | Dataset + sequence-length spec. See[Input format](#input-format).                                                    |
| `--output OUTPUT_PATH` | str  | `~/.etc`               | Root directory under which `--eval` looks up real execution traces. Not read in pure forward-prediction mode.     |
| `--datapath PATH`      | str  | `~/.etc`               | Absolute path to the dataset directory. Must exist.                                                               |
| `--args STRING`        | str  | _(none)_               | Engine-specific arguments. See[Engine-specific extra args](#engine-specific-extra-args).                             |
| `--modeling-dir PATH`  | str  | `<repo_root>/modeling` | Path to directory containing xgboost regression models.                                                           |
| `--model-search-paths` | str  | _(none)_               | Comma-separated model search roots used to resolve model names into absolute paths.                              |
| `-h` / `--help`      | flag | —                       | Print help and exit.                                                                                              |

#### Parallelism format

```
# Prefill-decode colocate
dp:pp:tp                  e.g.  1:1:8   (1 DP × 1 PP × 8 TP = 8 GPUs)

# Prefill-decode disaggregation  (prefill_config @ decode_config)
dp:pp:tp@dp:pp:tp         e.g.  1:1:4@1:1:4   (4 prefill GPUs + 4 decode GPUs)

# Multiple configs (comma-separated)
1:1:8,1:1:4,1:1:4@1:1:4
```

#### Input format

```
dataset[@isl_osl]

dataset   ∈ sharegpt | repoqa | aime
isl, osl  = integer (exact length)

Examples:
  sharegpt                  # use dataset defaults (ISL≈500, OSL≈100)
  repoqa@4096_1024          # ISL=4096, OSL=1024
  aime@512_8192             # ISL=512, OSL=8192
```

---

### Predict-only arguments

These flags are available **only** in the `ross_predict.py` script.

| Argument                     | Type | Default           | Description                                                                                                                                   |
| ---------------------------- | ---- | ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `--debug`                  | flag | `false`         | Enable verbose / DEBUG-level logging. Also forces timing data to be printed.                                                                  |
| `--record-path FILE`       | str  | `""` (disabled) | Write per-configuration PE (percentage error) metrics to a CSV file at FILE. When omitted, results are printed to stdout only.                |
| `--eval`                   | flag | `false`         | Load real trace logs and compare against simulator results. When omitted, the simulator runs in pure forward-prediction mode (no PE metrics). |
| `--max-workers INT`        | int  | `0` (auto)      | Number of parallel worker processes.`0` = auto-select based on CPU count.                                                                   |
| `--threads-per-worker INT` | int  | `0` (auto)      | CPU threads per worker process.`0` = auto.                                                                                                  |
| `--get-pareto-front`       | flag | `false`         | Enumerate all valid parallel configs and compute the Pareto front (tokens/s/user vs tokens/s/gpu). Incompatible with `--eval`.              |

---

## JSON config file reference

Pass a config file with `--config path/to/config.json`. Any field left out of
the file falls back to the built-in default. Command-line arguments always
override the file.

### Full field listing

| Field                   | Type                                         | Default                                                   | Notes                                                                                                                                                                                                      |
| ----------------------- | -------------------------------------------- | --------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `backend`             | string or list                               | _(required)_                                            | Same as `--backend`. E.g. `"vllm,sglang"` or `["vllm", "sglang"]`.                                                                                                                                   |
| `platforms`           | list of `{"gpu": …, "version": …}` dicts | _(none)_                                                | Overrides GPU detection + venv version lookup. Used in predict scripts to sweep multiple hardware targets. E.g.`[{"gpu": "H200", "version": "0.6.6.post1"}, {"gpu": "B200", "version": "0.6.6.post1"}]`. |
| `model`               | string or list                               | `"Qwen2.5-72B-Instruct"`                                | Same as `--model`.                                                                                                                                                                                       |
| `mode`                | string                                       | `"online"`                                              | `"online"` or `"offline"`.                                                                                                                                                                             |
| `parallel`            | string or list                               | `"1:1:8"`                                               | Same as `--parallel`. E.g. `"1:1:8,1:1:4"` or `["1:1:8", "1:1:4"]`.                                                                                                                                  |
| `batch`               | string or list                               | `"64"`                                                  | Same as `--batch`. E.g. `"32,64"` or `[32, 64]`.                                                                                                                                                     |
| `rate`                | list                                         | `["inf"]`                                               | Request rates as a list of strings. E.g.`["1", "2", "4", "inf"]`.                                                                                                                                        |
| `input`               | string                                       | `"sharegpt@0_0"`                                    | Same as `--input`. Only the first `input` entry is used for validation.                                                                                                                                |
| `inputs`              | list of strings                              | `[]`                                                    | Additional input specs (same format as `input`). The first element also sets `input`. Allows sweeping multiple datasets / ISL-OSL pairs.                                                               |
| `output`              | string                                       | `"~/.etc"`                                              | Same as `--output`.                                                                                                                                                                                      |
| `datapath`            | string                                       | `"~/.etc"`                                              | Same as `--datapath`.                                                                                                                                                                                    |
| `modeling_dir`        | string                                       | `<repo_root>/modeling`                                  | Same as `--modeling-dir`. Path to directory containing xgboost regression models.                                                                                                                        |
| `model_search_paths`  | string or list                               | _(built-in)_                                            | Same as `--model-search-paths`. Roots used to resolve model names into absolute paths.                                                                                                                    |
| `num_prompt`          | int                                          | `512`                                                   | Number of prompts per benchmark run.                                                                                                                                                                       |
| `disaggregation_mode` | string                                       | `"colocation"`                                          | Disaggregation scheduling mode. Currently informational.                                                                                                                                                   |
| `ross_extra`          | list of dicts                                | `[]`                                                    | Engine-specific arguments in list form. Each dict must have a `"backend"` key. See [Engine-specific extra args](#engine-specific-extra-args).                                                               |

> **Note on `platforms`:** when this field is present, version strings and
> GPU names are taken from the list rather than auto-detected at runtime. This
> is the primary mechanism used by the predict scripts to replay traces
> collected on specific GPU/version combinations.

---

### Annotated template

```jsonc
{
    // ── Basic sweep ──────────────────────────────────────────────────────
    "backend":  "sglang",
    "model":    "Qwen2.5-72B-Instruct",
    "mode":     "online",

    // Parallelism: single spec or list
    "parallel": "1:1:8",
    // "parallel": ["1:1:8", "1:1:4@1:1:4"],

    "batch":    [32],

    // Rates as a list (preferred) or comma-string
    "rate":     ["1", "2", "4", "8", "inf"],

    // ── Dataset ──────────────────────────────────────────────────────────
    // Single dataset
    "input":  "sharegpt@0_0",

    // OR multiple datasets via "inputs" (first also sets "input")
    // "inputs": [
    //     "sharegpt@0_0",
    //     "random@128_128",
    //     "random@1024_512"
    // ],

    // ── Paths ─────────────────────────────────────────────────────────────
    "output":   "/mnt/results",
    "datapath": "/mnt/datasets",

    // ── Hardware targets (predict scripts only) ──────────────────────────
    // Each entry: {"gpu": GPU_name, "version": framework_version}
    "platforms": [
        {"gpu": "H200", "version": "0.6.6.post1"},
        {"gpu": "B200", "version": "0.7.0"}
    ],

    // ── Prompt count ──────────────────────────────────────────────────────
    "num_prompt": 512,

    // ── Engine-specific arguments ─────────────────────────────────────────
    // List form (used in config files). Each entry must have "backend".
    // Values that are lists cause an inner sweep over those values.
    "ross_extra": [
        {
            "backend":              "sglang",
            "mem_fraction_static":  [0.9, 0.85],
            "chunked_prefill_size": [8192]
        }
        // For vLLM:
        // {
        //     "backend":                  "vllm",
        //     "gpu_memory_utilization":   [0.9],
        //     "max_num_batched_tokens":   [8192, 16384]
        // }
    ]
}
```

---

## Engine-specific extra args

Extra arguments are forwarded to the inference engine at launch time. They can
be provided in two ways:

### 1. CLI string form (`--args`)

```
--args "backend@key=val,key=val,...;backend@key=val,..."
```

- Segments are separated by `;`.
- Each segment starts with a backend name followed by `@`.
- The same backend can appear multiple times to define multiple argument sets
  (each set becomes one inner sweep iteration).

```bash
# SGLang — single sweep point
--args "sglang@mem_fraction_static=0.9,chunked_prefill_size=8192"

# SGLang — two sweep points for mem_fraction_static
--args "sglang@mem_fraction_static=0.9,chunked_prefill_size=8192;sglang@mem_fraction_static=0.85,chunked_prefill_size=8192"

# vLLM
--args "vllm@gpu_memory_utilization=0.9,max_num_batched_tokens=8192"
```

### 2. Config file list form (`ross_extra`)

Preferred for large sweeps — values may be lists for an automatic inner sweep.

```json
"ross_extra": [
    {
        "backend":              "sglang",
        "mem_fraction_static":  [0.85, 0.9],
        "chunked_prefill_size": [8192, 16384]
    }
]
```

### SGLang keys

| Key                      | Type  | Default  | Description                                              |
| ------------------------ | ----- | -------- | -------------------------------------------------------- |
| `mem_fraction_static`  | float | `0.9`  | Fraction of GPU memory reserved for the static KV-cache. |
| `chunked_prefill_size` | int   | `8192` | Maximum number of tokens per chunked-prefill batch.      |

### vLLM keys

| Key                        | Type  | Default  | Description                                           |
| -------------------------- | ----- | -------- | ----------------------------------------------------- |
| `gpu_memory_utilization` | float | `0.9`  | Fraction of GPU memory vLLM may use.                  |
| `max_num_batched_tokens` | int   | `8192` | Maximum number of tokens in a single scheduler batch. |

---

## Examples

### Minimal CLI run (simulator, no ground-truth comparison)

```bash
cd ross
python ross_predict.py \
    --backend sglang \
    --model   /path/to/Qwen2.5-72B-Instruct \
    --parallel 1:1:8 \
    --batch   32 \
    --rate    inf \
    --input   sharegpt@500_100 \
    --record-path results.csv
```

### Config-file driven sweep

```bash
cd ross
python ross_predict.py \
    --config configs/sweep_h200.json \
    --record-path results/h200_sweep.csv
```

`configs/sweep_h200.json`:

```json
{
    "backend":  "vllm",
    "model":    "/models/Qwen2.5-72B-Instruct",
    "parallel": "1:1:8",
    "batch":    [32],
    "rate":     ["1", "2", "4", "inf"],
    "input":    "sharegpt@500_100",
    "platforms": [{"gpu": "H200", "version": "0.7.0"}],
    "ross_extra": [
        {
            "backend":                "vllm",
            "gpu_memory_utilization": [0.9],
            "max_num_batched_tokens": [8192]
        }
    ]
}
```

### Disaggregated prefill-decode

```bash
python ross_predict.py \
    --config base.json \
    --parallel "1:1:4@1:1:4" \
    --record-path disagg_results.csv
```
