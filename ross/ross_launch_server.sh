#!/bin/bash

color_green="\033[1;32m"
color_yellow="\033[1;33m"
color_purple="\033[1;35m"
color_reset="\033[0m"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${SCRIPT_DIR}/log"

echo_back() {
    local _cmdLog=${1}
    printf "[${color_purple}EXEC${color_reset}] ${_cmdLog}\n"
    eval ${_cmdLog}
}

rm_if_exists() {
    local fpath=$1
    if [ -f "${fpath}" ]; then
        echo_back "rm ${fpath}"
    fi
}

bench_setup_server(){
    local model=$1
    local framework=$2
    local dataset_name=$3
    local dataset=$4
    local isl=$5
    local osl=$6
    local rate=$7
    local num_prompt=$8
    local arrive_json_log=$9
    local launch_server_log=${10}
    local client_log=${11}
    local client_bench_log=${12}
    local batch_size=${13}
    local tokenize_log="${LOG_DIR}/sgl_tokenize_server.log"
    local tokenize_pid=""

    mkdir -p "${LOG_DIR}"

    rm_if_exists "${arrive_json_log}"
    rm_if_exists "${launch_server_log}"
    rm_if_exists "${client_log}"
    rm_if_exists "${client_bench_log}"
    rm_if_exists "${tokenize_log}"

    tokenize_args="cd ${SCRIPT_DIR} && python3 tokenize_server.py --host 127.0.0.1 --port=8001 --model ${model}"
    echo_back "${tokenize_args} > ${tokenize_log} 2>&1 &"
    tokenize_pid=$!
    sleep 2
    
    server_args="cd ${SCRIPT_DIR} && python3 api_server.py --host 127.0.0.1 --port=8000 --log-file ${arrive_json_log} --model ${model} --bootstrap-remote-count ${batch_size}"
    if [[ $framework == 'sglang' ]]; then
        server_args="${server_args} --tokenize-url http://127.0.0.1:8001/tokenize"
    fi
    if [[ $fix == 1 ]]; then
        server_args="${server_args} --fix-prompt ${isl}"
    fi
    echo_back "${server_args} > ${launch_server_log} 2>&1 &"
    sv_pid=$!
    sleep 2
    
    client_args="python3 bench_serving.py --backend=sglang --num-prompts=${num_prompt} --max-concurrency=${batch_size}"
    client_args="${client_args} --model=${model} --dataset-name=${dataset_name} --dataset-path=${dataset}"
    client_args="${client_args} --host=127.0.0.1 --port=8000"
    client_args="${client_args} --sharegpt-prompt-len ${isl} --sharegpt-output-len ${osl} --request-rate=${rate}"
    echo_back "${client_args} --output-file ${client_bench_log} > ${client_log} 2>&1"
    
    pkill -9 -f "python3 tokenize_server.py"
    pkill -9 -f "python3 api_server.py"
}

##########################################################
#################### * Main Process * ####################
##########################################################

bench_setup_server $1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12} ${13}
