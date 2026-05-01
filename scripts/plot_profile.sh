#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

dir="${1:-.}"

for csv in "$dir"/*.csv; do
    [[ -e "$csv" ]] || continue

    base="$(basename "$csv" .csv)"
    
    # Extract impl and algo from filename: <impl>_<algo>[_<anything>]
    # Supported impls: mkl, cuda   Supported algos: cg, dr-bcg
    if [[ "$base" =~ (mkl|cuda)_(cg|dr-bcg)(_|$) ]]; then
        impl="${BASH_REMATCH[1]}"
        algo="${BASH_REMATCH[2]}"
    else
        echo "Skipping '$csv': could not extract impl/algo from filename" >&2
        continue
    fi

    output="$dir/${base}.png"
    echo "Plotting $csv (impl=$impl, algo=$algo) -> $output"
    python "${SCRIPT_DIR}/stacked_bars.py" "$csv" --impl "$impl" --algo "$algo" -o "$output"
done
