#!/usr/bin/env bash

set -eufo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$(realpath "${SCRIPT_DIR}/../build")"
DATA_DIR="$(realpath "${SCRIPT_DIR}/../data")"

algorithms=("cg" "dr-bcg")

for alg in ${algorithms[@]}; do
    nsys profile --force-overwrite true -t cuda,nvtx -o "cuda_${alg}" \
        "${BUILD_DIR}/runner/cgrun" "${alg}" cuda "${DATA_DIR}/G2_circuit.mat" -L "${DATA_DIR}/G2_circuit_ichol.mat"
done
