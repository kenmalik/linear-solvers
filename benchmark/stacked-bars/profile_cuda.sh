#!/usr/bin/env bash

set -eufo pipefail

algorithms=("cg" "dr-bcg")

for alg in ${algorithms[@]}; do
    nsys profile --force-overwrite true -t nvtx -o "cuda_${alg}" \
        build/runner/cgrun "${alg}" cuda data/G2_circuit.mat -L data/G2_circuit_ichol.mat
    nsys stats --force-export true --force-overwrite true -r nvtx_sum \
        -o "cuda_${alg}" -f csv "cuda_${alg}.nsys-rep"
done
