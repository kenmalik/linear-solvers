#!/usr/bin/env bash

set -e

data_path=$(realpath ${1:-"test/data/1138_bus.mat"})
data_base=$(basename "$data_path" .mat)
data_dir=$(dirname "$data_path")

prec_type=$2
prec_path=${2:+"${data_dir}/${data_base}_${2}.mat"}

gpu=$(nvidia-smi --query-gpu name --format csv,noheader | sed 's/NVIDIA //g' | tr ' ' '_')

formatted="runtimes-cg-cuda-${data_base}-${prec_type:-"identity"}-${gpu}-"

nsys profile -f true -o $formatted -t cuda,nvtx \
    build/cgrun --real-residual -b "${data_dir}/${data_base}_b.mat" "${data_dir}/${data_base}.mat" $prec_path

nsys stats -o $formatted --force-overwrite true --force-export true -f csv \
    -r nvtx_sum "${formatted}.nsys-rep"
nsys stats -o $formatted --force-overwrite true --force-export true -f csv \
    -r cuda_api_sum --filter-nvtx "pre-cg" "${formatted}.nsys-rep"
nsys stats -o $formatted --force-overwrite true --force-export true -f csv \
    -r cuda_api_sum --filter-nvtx "post-cg" "${formatted}.nsys-rep"

echo nsys profile -f true -o $formatted -t cuda,nvtx build/cgrun --real-residual -b "${data_dir}/${data_base}_b.mat" "${data_dir}/${data_base}.mat" $prec_path