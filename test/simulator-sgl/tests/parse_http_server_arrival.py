import re

ts_list = {"arrive_time": []}
with open('/volume/ycao03/SiLLM-OP/output_sgl_disagg/ycao-02-0/iosl_500_2500/sglang_Qwen3-32B/ds_/volume/ycao03/dataset/ShareGPT_V3_unfiltered_cleaned_split.json_batch_128_pdp1_ppp1_ptp4_ddp1_dtp4/main_pdp1_ppp1_ptp4_ddp1_dtp4.log.server.prefill', 'r') as f:
    lines = f.readlines()
    for line in lines:
        time_match = re.search(r"enter /generate time=(\d+\.\d+)", line.strip())
        if time_match is not None:
            ts = time_match.group(1)        
            ts_list["arrive_time"].append(float(ts))

fout = open('tests/test_arrive_time_batch_128.log', 'w')
print(ts_list, file=fout)