#!/usr/bin/env bash
# Run after: module load intel-oneapi-mkl
set -euo pipefail

if [[ -z "${MKLROOT:-}" ]]; then
  echo "ERROR: MKLROOT not set. Did you load the MKL module?" >&2
  exit 1
fi

VSCODE_DIR="${1:-.vscode}"
mkdir -p "$VSCODE_DIR"

cat > "$VSCODE_DIR/c_cpp_properties.json" <<EOF
{
  "configurations": [
    {
      "name": "Linux-HPC",
      "includePath": [
        "\${workspaceFolder}/**",
        "${MKLROOT}/include",
        "\${userHome}/.local/include",
        "\${default}"
      ],
      "defines": [],
      "compilerPath": "$(which gcc)",
      "cStandard": "c17",
      "cppStandard": "c++20",
      "intelliSenseMode": "linux-gcc-x64"
    }
  ],
  "version": 4
}
EOF

echo "Written: $VSCODE_DIR/c_cpp_properties.json"
echo "  MKL:  ${MKLROOT}/include"
echo "  GCC:  $(which gcc)"