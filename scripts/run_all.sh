#!/usr/bin/env bash

set -eufo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$(realpath "${SCRIPT_DIR}/../build")"
DATA_DIR="$(realpath "${SCRIPT_DIR}/../data")"

algorithms=("cg" "dr-bcg")
implementations=("mkl" "cuda")

for alg in ${algorithms[@]}; do
    for impl in ${implementations[@]}; do
        "${BUILD_DIR}/runner/cgrun" "$alg" "$impl" "${DATA_DIR}/G2_circuit.mat" "${DATA_DIR}/G2_circuit_ichol.mat" \
            2> "residuals_${impl}_${alg}.txt"
    done
done
