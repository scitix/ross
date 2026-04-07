# ROSS

ROSS is an offline simulator for LLM inference performance. Given a model, a
hardware platform, and a workload description, it predicts throughput and
latency metrics (TTFT, TPOT, ITL) without running a real inference server.
It supports both SGLang and vLLM backends, colocated and disaggregated
parallelism, and Pareto-front search over parallel configurations.

---

## Repository layout

```
collector/          Platform profiling scripts + pre-collected hardware profiles
common/             Shared model/feature/config classes used by both simulators
ross/               Main simulator package
  sgl_sim/            SGLang simulator backend
  vllm_sim/           vLLM simulator backend
  pareto/             Pareto-front analysis utilities
  config/             Example JSON config files
  ross_predict.py     Entry point for offline prediction sweeps
docs/               Extended documentation
  bench_config.md     Full config file and CLI reference
  profiling.md        How to profile a new GPU platform
modeling/           Pre-trained XGBoost regression models
test/               Unit and integration tests
```

---

## Quick start

### 1. Install dependencies

```bash
pip install xgboost==3.2.0 scikit-learn pandas tqdm plotext
```

### 2. Write a config file

Use one of the templates in `ross/config/` as a starting point:

```json
{
    "backend":  "sglang",
    "model":    "Meta-Llama-3.1-70B",
    "parallel": "1:1:8",
    "batch":    [64],
    "num_prompt": 512,
    "rate":     ["1", "2", "inf"],
    "inputs":   ["sharegpt@0_0"],
    "platforms": [{"gpu": "H200", "version": "0.6.6"}],
    "output":   "~/.etc",
    "datapath": "~/.etc"
}
```

### 3. Run the simulator

```bash
cd <repo_root>
python ross/ross_predict.py --config <your_config.json>
```

Add `--eval` to also load real execution traces and compute prediction error:

```bash
python ross/ross_predict.py --config <your_config.json> --eval
```

Add `--get-pareto-front` to search all valid parallel configs and plot the
Pareto frontier (tokens/s/user vs tokens/s/gpu):

```bash
python ross/ross_predict.py --config <your_config.json> --get-pareto-front
```

### 4. Save results

```bash
python ross/ross_predict.py --config <your_config.json> --record-path results.csv
```

---

## Key CLI flags

| Flag                      | Description                                  |
| ------------------------- | -------------------------------------------- |
| `--config FILE`         | JSON config file (recommended for sweeps)    |
| `--backend sglang\|vllm` | Simulator backend                            |
| `--eval`                | Load trace logs and compute prediction error |
| `--record-path FILE`    | Write metrics to a CSV file                  |
| `--get-pareto-front`    | Enumerate configs and compute Pareto front   |
| `--debug`               | Verbose logging                              |

Full reference: [`docs/bench_config.md`](docs/bench_config.md)

---

## Adding a new GPU platform

See [`docs/profiling.md`](docs/profiling.md) for the step-by-step guide on
collecting kernel and communication performance data for a new hardware target.

Pre-collected profiles for H200 and B200 are already included under
`collector/`.

---

## Supported configurations

| Dimension   | Options                                                                         |
| ----------- | ------------------------------------------------------------------------------- |
| Backend     | SGLang, vLLM                                                                    |
| Parallelism | Tensor parallel, pipeline parallel, data parallel, disaggregated prefill/decode |
| Hardware    | H200, B200 (extensible via `collector/`)                                      |
| Datasets    | ShareGPT, RepoQA, AIME                                                          |
