#!/usr/bin/env bash

set -e

report_name="cg_nvtx"

nsys profile -f true -o $report_name -t cuda,nvtx build/cgrun --real-residual -b test/data/1138_bus_b.mat test/data/1138_bus.mat test/data/1138_bus_ichol.mat

nsys stats -o "cg" --force-overwrite true --force-export true -f csv -r nvtx_sum "${report_name}.nsys-rep"
nsys stats -o "cg" --force-overwrite true --force-export true -f csv -r cuda_api_sum --filter-nvtx "pre-cg" "${report_name}.nsys-rep"
nsys stats -o "cg" --force-overwrite true --force-export true -f csv -r cuda_api_sum --filter-nvtx "post-cg" "${report_name}.nsys-rep"