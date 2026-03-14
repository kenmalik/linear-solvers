#!/usr/bin/env bash

set -e

datasets=(1138_bus)
algs=(cg dr-bcg)
impls=(mkl cuda)

files=()
for dataset in ${datasets[@]}; do
    for alg in ${algs[@]}; do
        for impl in ${impls[@]}; do
            fname="${impl}_${alg}_${dataset}.txt"
            build/runner/cgrun $alg $impl -s 16 "data/${dataset}.mat" "data/${dataset}_ichol.mat" 2>> $fname
            files+=($fname)
        done
    done
done

if [ ${#files[@]} -eq 0 ]; then
    echo "No residual files generated" >&2
    exit -1
fi

python test/residual-curves/plot.py ${files[@]}
