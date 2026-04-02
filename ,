### Environment Setup
pip install xgboost==3.2.0 scikit-learn==1.7.2 matplotlib seaborn

### [Optional] For New Hardware Platform：Collect Platform Performance
- The collector uses VLLM to collect platform performance
pip install cuda-python==12.6 flashinfer-python
```
cd <your-path>/SiLLM-OP/test/collector
python collect.py --backend vllm
```
The results are profiled in gemm_perf_vllm_.txt and context_attention_perf.txt.
- NCCL Statistics
```
./collect_comm.sh
```
The result file is nccl_perf.txt
- Manually build SPEC file
The spec file includes the GPU spec information. Take L40 as an example:
```
gpu:
  mem_bw: 864000000000 # 864GB/s
  mem_capacity: 51539607552 # 48GB
  float16_tc_flops: 181050000000000 # 181.05TFLOPS
  int8_tc_flops: 362000000000000 # 362TFLOPS
  fp8_tc_flops: 362000000000000 # 362TFLOPS
  power: 300 # Watt
```
- Generate the Performance Feature File
```
python extract_platform_features.py --gemm_data=<gemm_file> \
  --attn_data=<attn_file> --nccl_data=<nccl_file> \
  --plaform_specs=<spec_file> --output_dir=<outut_path>
```
The result yaml file is platform_features.yaml

Run Simulation
- ROSS uses ross_config.json to load model configs
```
cd <your-path>/ross
cat ross_config.json # xgboost model paths
```
- The simulation configurations are loaded from a JSON file. Template configs are in ross/config directory.
```
cat config/test_sglang.json
```
- Run simulation with given config file.
### SGLang Simulation
```
python sglang/ross_sgl_predict.py --no-comparison --config=<your-config-file>
```
### vLLM Simulation
```
python vllm_sim/ross_vllm_predict.py --no-comparison --config=<your-config-file>
```
Argument Choices:
- --no-comparsion: only run simulation. By default ROSS will also load the result of real traces.
- --debug
