#!/usr/bin/env bash

set -eufo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$(realpath "${SCRIPT_DIR}/../build")"
DATA_DIR="$(realpath "${SCRIPT_DIR}/../data")"

algorithms=("cg" "dr-bcg")
implementations=("mkl" "cuda")
block_sizes=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -s|--block-size)
            IFS=',' read -r -a block_sizes <<< "$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

if [[ ${#block_sizes[@]} == 0 ]]; then
    block_sizes=(1 2 4 8 16 32 64)
fi

for alg in ${algorithms[@]}; do
    for impl in ${implementations[@]}; do
        if [[ "${alg}" == "dr-bcg" ]]; then
            for block_size in "${block_sizes[@]}"; do
                "${BUILD_DIR}/runner/cgrun" "$alg" "$impl" \
                    "${DATA_DIR}/G2_circuit.mat" "${DATA_DIR}/G2_circuit_ichol.mat" \
                    --timer-out "${impl}_${alg}_s${block_size}" \
                    -s "$block_size" 2> "residuals_${impl}_${alg}.txt"
            done
        else
            "${BUILD_DIR}/runner/cgrun" "$alg" "$impl" \
                "${DATA_DIR}/G2_circuit.mat" "${DATA_DIR}/G2_circuit_ichol.mat" \
                --timer-out "${impl}_${alg}" 2> "residuals_${impl}_${alg}.txt"
        fi
    done
done
