#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$(realpath "${SCRIPT_DIR}/../build")"

FILENAME="benchmark_$(date +%F-%H-%M-%S)"

"${BUILD_DIR}/benchmark/solver_benchmarks" --benchmark_out="${FILENAME}.json" --benchmark_out_format=json --benchmark_time_unit=ms

jq "[.benchmarks | .[] | {name: .name, time: .real_time, time_unit: .time_unit, iters: .iters}]" < "${FILENAME}.json" \
    | python "${SCRIPT_DIR}/runtime_comparison.py" -o "${FILENAME}.png"
