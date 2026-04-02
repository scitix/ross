#!/bin/bash

root_dir="/volume/ycao03"

source "${root_dir}/SiLLM-OP/bin/include/file.sh"
source "${root_dir}/SiLLM-OP/bin/include/logging.sh"
source "${root_dir}/SiLLM-OP/bin/bench_util.sh"

saved_model_prefix="${root_dir}/SiLLM-OP/test/modeling/vllm"

train() {
    dataset_name="ShareGPT_V3_unfiltered_cleaned_split.json"

    engine_launcher=$*
    engine_args="${engine_args} --group_split"
    engine_args="${engine_args} --regressor xgb --val_ratio 0.2"

    echo_back "${engine_launcher} ${engine_args}"
}

if [ "$#" -lt 4 ]; then
    usage
    exit 1
fi

COMMAND=$1
shift

if [[ $2 == "" ]]; then
    echo "Specificy Model SAVE_DIR"
    exit 1
fi

config_file=$3

target_args=""
output_dir="${saved_model_prefix}/$2"

if [[ $COMMAND == "train_mix" ]]; then
    launcher="train/forward_trainer_lmdb_mix.py --config=${config_file}"            
elif [[ $COMMAND == "train" ]]; then
    launcher="train/forward_trainer_lmdb.py --config=${config_file}"        
    if [[ $1 == "pre-forward" ]]; then
        launcher="train/pre_trainer_lmdb_global.py --config=${config_file}"
    fi
else
    exit 1
fi

mkdir_if_not_exists ${output_dir}

case "$1" in
    pre-forward)
        target_args="python ${launcher} --target=pre-forward --saved_model_name=pre_forward_trained_models"
        ;;
    forward)
        target_args="python ${launcher} --target=forward --saved_model_name=forward_trained_models"
        ;;
    post-forward)
        target_args="python ${launcher} --target=post-forward --saved_model_name=post_forward_trained_models"
        ;;
    *)
        echo "Usage: $0 train [options]"
        echo ""
        echo "Commands:"
        echo "  pre-forward     Train the ROSS performance prediction model."
        echo "  forward         Run the ROSS simulator with a given configuration."
        echo "  post-forward    Run the ROSS simulator with a given configuration."
        echo ""
        exit 0
        ;;
esac

train ${target_args} "--output_dir=${output_dir}"
