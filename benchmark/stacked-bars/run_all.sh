#!/usr/bin/env bash
set -euo pipefail

dir="${1:-.}"

for csv in "$dir"/*.csv; do
    [[ -e "$csv" ]] || continue

    base="$(basename "$csv" .csv)"
    
    # Extract impl and algo from filename: <anything>_<impl>_<algo>[_<anything>]
    # Supported impls: mkl, cuda   Supported algos: cg, dr-bcg
    if [[ "$base" =~ _(mkl|cuda)_(cg|dr-bcg)(_|$) ]]; then
        impl="${BASH_REMATCH[1]}"
        algo="${BASH_REMATCH[2]}"
    else
        echo "Skipping '$csv': could not extract impl/algo from filename" >&2
        continue
    fi

    output="$dir/${base}.png"
    echo "Plotting $csv (impl=$impl, algo=$algo) -> $output"
    python "$(dirname "$0")/main.py" "$csv" --impl "$impl" --algo "$algo" -o "$output"
done
