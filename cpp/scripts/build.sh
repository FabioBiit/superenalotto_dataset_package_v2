#!/usr/bin/env bash
# Build script for SuperEnalotto Engine (Linux/macOS bash).
#
# Prerequisites:
#   - CMake 3.24+, Ninja, vcpkg ($VCPKG_ROOT set)
#   - gcc 13+ or clang 16+ (C++20)
#   - Python 3.11+, pybind11 via vcpkg
#   - CUDA Toolkit (optional, only if --preset cuda)
#
# Usage:
#   ./scripts/build.sh                # CPU release
#   ./scripts/build.sh cuda           # CUDA release
#   ./scripts/build.sh cpu Debug      # CPU debug

set -euo pipefail

PRESET="${1:-cpu}"
CONFIG="${2:-Release}"

if [[ -z "${VCPKG_ROOT:-}" ]]; then
    echo "ERROR: VCPKG_ROOT is not set."
    exit 1
fi

cd "$(dirname "$0")/.."

echo "[1/3] Configuring (preset=$PRESET)..."
cmake --preset "$PRESET"

echo "[2/3] Building (config=$CONFIG)..."
cmake --build --preset "${PRESET}-$(echo "$CONFIG" | tr '[:upper:]' '[:lower:]')"

echo "[3/3] Running tests..."
ctest --preset "${PRESET}-tests"

EXE="build/${PRESET}/bin/${CONFIG}/se_cli"
[[ -f "$EXE" ]] && "$EXE" --version
