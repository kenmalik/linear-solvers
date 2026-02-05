#!/usr/bin/env bash

report_name="cg_nvtx"

nsys profile -f true -o $report_name -t nvtx build/cgrun --real-residual -B test/data/1138_bus_b.mat test/data/1138_bus.mat test/data/1138_bus_ichol.mat

nsys stats -o "cg" --force-overwrite true --force-export true -f csv -r nvtx_sum "${report_name}.nsys-rep"