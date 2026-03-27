#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

dir="${1:-.}"

python "${SCRIPT_DIR}/residual_curves.py" residuals*.txt
