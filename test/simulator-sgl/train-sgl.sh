#!/bin/bash

# setup env
root_dir="/volume/ycao03"
source "${root_dir}/SiLLM-OP/bin/include/file.sh"
source "${root_dir}/SiLLM-OP/bin/include/logging.sh"
source "${root_dir}/SiLLM-OP/bin/bench_util.sh"

saved_model_prefix="${root_dir}/SiLLM-OP/test/modeling/sgl"


host=`hostname`
if [[ -d /scratch/${username} ]]; then
    host=`echo $host | cut -d'-' -f1`
fi

lmdb_path="${root_dir}/.etc/LMDB/H200/online_opt_0.9_sglang_0.5.2"

train() {
    dataset_name="ShareGPT_V3_unfiltered_cleaned_split.json"

    engine_launcher=$*
    engine_args="${engine_args} --group_split"
    engine_args="${engine_args} --val_ratio 0.2"

    echo_back "${engine_launcher} ${engine_args}"
}

usage() {
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  train|train_disagg     Train the ROSS performance prediction model."
    echo "  predict                Run the ROSS simulator with a given configuration."
    echo "  load_output            Extract Memory and Time Profiling Results."
    echo ""
    echo "Use '$0 <command> -h' for more information on a specific command."
}

if [ "$#" -lt 3 ]; then
    usage
    exit 1
fi

# COMMAND=$1
# shift

target_args=""
output_dir="${saved_model_prefix}/$3"

config_file=$4
launcher="train/forward_trainer_lmdb.py --config=${config_file}"            
        
if [[ $1 == "forward" || $1 == "pre-forward" ]]; then
    output_dir="${output_dir}/$2"
fi
case $1 in
    pre-forward)
        target_args="python ${launcher} --target=pre-forward --saved_model_name=pre_forward_trained_models --regressor xgb"
        ;;
    forward)
        target_args="python ${launcher} --target=forward --saved_model_name=forward_trained_models --regressor xgb"
        ;;
    post-forward)
        target_args="python ${launcher} --target=post-forward --saved_model_name=post_forward_trained_models --regressor xgb"
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

train ${target_args} "--stage=$2 --output_dir=${output_dir}"


