# Profiling a New Hardware Platform

This guide explains how to collect the raw performance data needed to add a new
GPU to ROSS. The output is a single `platform_features.yaml` file that the
simulator reads at runtime.

---

## Overview

Three types of data are collected, then combined into one feature file:

| Step | Script | Output |
|------|--------|--------|
| 1. GEMM / attention / MoE kernels | `collector/collect.py` | `gemm_perf_*.txt`, `context_attention_perf.txt` |
| 2. NCCL / custom all-reduce comms | `collector/collect_comm.sh` | `nccl_perf.txt`, `comm_perf.txt` |
| 3. Combine + GPU spec → feature file | `collector/extract_platform_features.py` | `platform_features.yaml` |

---

## Prerequisites

### Node preparation

Before collecting data, make sure you own the whole node with no other workloads
running. Enable persistence mode and lock GPU clocks to eliminate frequency
throttling noise:

```bash
sudo nvidia-smi -pm 1
sudo nvidia-smi -lgc <sm_freq>,<mem_freq>
```

Query the supported clock range:
```bash
nvidia-smi -q -i 0   # see "Max Clocks" → SM / Memory
```

Also verify the cooling system is working and GPU temperatures are stable before
starting a long collection run.

### Python environment

```bash
pip install xgboost scikit-learn matplotlib seaborn
pip install cuda-python==12.6 flashinfer-python
```

Install the target inference framework (sglang or vllm) in the same environment.

---

## Step 1 — Collect kernel performance data

```bash
cd <repo_root>/collector
python collect.py --backend sglang   # or --backend vllm
```

Supported backends: `sglang`, `vllm`, `trtllm`.

This benchmarks GEMM, context-attention, and MoE kernels across a sweep of
shapes. On 8 GPUs the full run takes roughly 3–4 hours. Expect some missing
data-points (logged as errors) — the extractor handles gaps gracefully.

Output files (written to the current directory):
- `gemm_perf_<backend>.txt`
- `context_attention_perf.txt`

---

## Step 2 — Collect communication performance data

```bash
cd <repo_root>/collector
bash collect_comm.sh
```

This covers intra-node collective operations (custom all-reduce, NCCL
all-reduce, all-gather, all2all, reduce-scatter).

Output files:
- `nccl_perf.txt`
- `comm_perf.txt` / `custom_all_reduce.txt`

---

## Step 3 — Write a GPU spec file

Create a YAML file describing the hardware. Example for an L40:

```yaml
gpu:
  mem_bw: 864000000000        # 864 GB/s
  mem_capacity: 51539607552   # 48 GB
  float16_tc_flops: 181050000000000   # 181.05 TFLOPS
  int8_tc_flops: 362000000000000      # 362 TFLOPS
  fp8_tc_flops: 362000000000000       # 362 TFLOPS
  power: 300                  # Watts
```

Consult the GPU's datasheet for the correct values.

---

## Step 4 — Generate `platform_features.yaml`

```bash
cd <repo_root>/collector
python extract_platform_features.py \
    --gemm_data   gemm_perf_vllm.txt \
    --attn_data   context_attention_perf.txt \
    --nccl_data   nccl_perf.txt \
    --platform_specs <path/to/gpu_spec.yaml> \
    --output_dir  <gpu_name>/
```

The script writes `platform_features.yaml` into `--output_dir`. Place this
directory under `collector/<gpu_name>/` (e.g. `collector/h200/`) so that
`ross_predict.py` can locate it via the `platforms` field in your config:

```json
"platforms": [{"gpu": "H200", "version": "0.6.6.post1"}]
```

The simulator resolves the yaml path as:
```
<repo_root>/collector/<gpu_lower>/platform_features.yaml
```

---

## Existing profiles

Pre-collected profiles are already checked in for:

| GPU | Path |
|-----|------|
| H200 | `collector/h200/platform_features.yaml` |
| B200 | `collector/b200/platform_features.yaml` |
