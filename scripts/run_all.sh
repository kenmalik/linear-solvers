#!/usr/bin/env bash

set -eufo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$(realpath "${SCRIPT_DIR}/../build")"
DATA_DIR="$(realpath "${SCRIPT_DIR}/../data")"

algorithms=("cg" "dr-bcg")
implementations=("mkl" "cuda")
block_size=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        -s|--block-size)
            block_size="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

for alg in ${algorithms[@]}; do
    for impl in ${implementations[@]}; do
        if [[ "${alg}" == "dr-bcg" ]]; then
            "${BUILD_DIR}/runner/cgrun" "$alg" "$impl" \
                "${DATA_DIR}/G2_circuit.mat" "${DATA_DIR}/G2_circuit_ichol.mat" \
                -s "$block_size" 2> "residuals_${impl}_${alg}.txt"
        else
            "${BUILD_DIR}/runner/cgrun" "$alg" "$impl" \
                "${DATA_DIR}/G2_circuit.mat" "${DATA_DIR}/G2_circuit_ichol.mat" \
                2> "residuals_${impl}_${alg}.txt"
        fi
    done
done
