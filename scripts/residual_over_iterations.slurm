#!/bin/bash
#SBATCH --job-name="residual_over_iterations"
#SBATCH --partition=gpuA100x4
#SBATCH --mem=20G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --no-requeue
#SBATCH -t 48:00:00

out_dir=${OUT_DIR}

if [ -z "${out_dir}" ]; then
    echo "Warning: Output directory not defined via OUT_DIR env var" >&2
    read -p "Continue? (y/n) " -n 1 -r
    echo
    if ! [[ $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
elif [[ "${out_dir: -1}" != "/" ]]; then
    out_dir="${out_dir}/"
fi

mat_dir=$1
if ! [[ -d "$mat_dir" ]]; then
    echo "Could not find .mat directory: ${1}" >&2
    exit 1
fi

command="${@:2}"
if [[ ${#command[@]} == 0 ]]; then
    echo "No commands provided" >&2
    exit 1
fi

block_sizes=(1 $(seq 2 2 20))

readarray -d "" -t mats < <(find ${mat_dir} -maxdepth 1 -name "*.mat" -print0)

for mat_file in ${mats[@]}; do
    mat_name=$(basename ${mat_file} .mat)
    echo "Now running on ${mat_name}" >&2
    mat_formatted=$(echo "${command}" | sed "s#{MAT}#${mat_file}#g")

    parent_dir="${out_dir}${SLURM_JOB_NAME}_${SLURM_JOB_ID}/${mat_name}"
    mkdir -p "${parent_dir}" || exit 1

    for s in ${block_sizes[@]}; do
        block_formatted=$(echo "${mat_formatted}" | sed "s#{BLOCK}#${s}#g")

        output_base="${parent_dir}/${s}.txt"

        echo "${block_formatted}"
        srun ${block_formatted} 2> "${output_base}"
    done
done