#!/usr/bin/env bash

set -eufo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$(realpath "${SCRIPT_DIR}/../build")"

# nvcc-only flags that end up verbatim in compile_commands.json but that
# plain clang (which is what clang-tidy parses each TU with, instead of
# nvcc) doesn't understand. Left in place they abort parsing with
# "unknown argument" errors before any real analysis happens.
static_removed_args=(
    -forward-unknown-to-host-compiler
    -rdc=true
    --extended-lambda
    --expt-relaxed-constexpr
)

# --generate-code=... and -ccbin=... carry values that depend on
# CMAKE_CUDA_ARCHITECTURES / the host compiler, so pull the exact strings
# actually in use out of the compile database instead of hardcoding them.
# Any TU whose command line pulls in MathDx's include path is excluded too:
# cuBLASDx/cuSolverDx headers rely on nvcc-only parsing (arch-gated database
# fragments, __CUDACC_VER_* macros) that plain clang can't handle, and
# clang-tidy re-parses every TU with its own bundled clang instead of nvcc,
# so those files always end in a parse-failure cascade with no useful
# diagnostics.
mapfile -t db_lines < <(python3 - "${BUILD_DIR}/compile_commands.json" <<'PYEOF'
import json
import shlex
import sys

with open(sys.argv[1]) as f:
    db = json.load(f)

removed_args = set()
excluded_files = set()
for entry in db:
    args = entry.get("arguments")
    if args is None:
        args = shlex.split(entry.get("command", ""))
    if any("mathdx" in arg.lower() for arg in args):
        excluded_files.add(entry["file"])
    for arg in args:
        if arg.startswith("--generate-code=") or arg.startswith("-ccbin="):
            removed_args.add(arg)

for arg in sorted(removed_args):
    print(f"removed_arg\t{arg}")
for path in sorted(excluded_files):
    print(f"excluded_file\t{path}")
PYEOF
)

dynamic_removed_args=()
excluded_files=()
for line in "${db_lines[@]}"; do
    kind="${line%%$'\t'*}"
    value="${line#*$'\t'}"
    case "${kind}" in
        removed_arg) dynamic_removed_args+=("${value}") ;;
        excluded_file) excluded_files+=("${value}") ;;
    esac
done

removed_arg_flags=()
for arg in "${static_removed_args[@]}" "${dynamic_removed_args[@]}"; do
    removed_arg_flags+=("-removed-arg=${arg}")
done

source_filter='^.*$'
if ((${#excluded_files[@]} > 0)); then
    excluded_pattern="$(IFS='|'; echo "${excluded_files[*]}")"
    source_filter="^(?!(${excluded_pattern})$).*"
fi

exec run-clang-tidy \
    -p "${BUILD_DIR}" \
    -source-filter="${source_filter}" \
    "${removed_arg_flags[@]}" \
    "$@"
