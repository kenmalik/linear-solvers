#!/usr/bin/env bash

set -e

function error() {
    echo "error: nsys-profile: $1" >&2
    exit -1
}

data_path=$(realpath ${1:-"test/data/1138_bus.mat"})
data_base=$(basename "$data_path" .mat)
data_dir=$(dirname "$data_path")

if ! [ -f $data_path ]; then
    error "matrix $data_path does not exist"
fi

prec_type=$2
prec_path=${2:+"${data_dir}/${data_base}_${2}.mat"}

if ! [ -f $prec_path ]; then
    error "preconditioner $prec_path does not exist"
fi

b_path="${data_dir}/${data_base}_b.mat"

if ! [ -f $b_path ]; then
    error "b $b_path does not exist"
fi

gpu=$(nvidia-smi --query-gpu name --format csv,noheader | sed 's/NVIDIA //g' | tr ' -' '_')

function profile_variant() {
    local variant=$1
    local extra_args=$2

    local formatted="runtimes-cg-cuda-${data_base}-${prec_type:-"identity"}-${gpu}-${variant}-"

    nsys profile -f true -o $formatted -t cuda,nvtx \
        build/cgrun --real-residual -b $b_path "${data_dir}/${data_base}.mat" $prec_path $extra_args

    nsys stats -o $formatted --force-overwrite true --force-export true -f csv \
        --timeunit ms -r nvtx_sum --filter-nvtx "cg" "${formatted}.nsys-rep"
    nsys stats -o $formatted --force-overwrite true --force-export true -f csv \
        --timeunit ms -r cuda_api_sum --filter-nvtx "pre-cg" "${formatted}.nsys-rep"
    nsys stats -o $formatted --force-overwrite true --force-export true -f csv \
        --timeunit ms -r cuda_api_sum --filter-nvtx "post-cg" "${formatted}.nsys-rep"

    echo nsys profile -f true -o $formatted -t cuda,nvtx build/cgrun --real-residual -b "${data_dir}/${data_base}_b.mat" "${data_dir}/${data_base}.mat" $prec_path $extra_args
}

profile_variant "tc" ""
profile_variant "notc" "--no-tensor-cores"
