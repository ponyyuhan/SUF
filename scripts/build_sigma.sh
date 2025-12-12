#!/usr/bin/env bash
# Helper to build Sigma (EzPC GPU-MPC) CPU and GPU variants.
# Expects the Sigma checkout at external/sigma_ezpc/GPU-MPC relative to this repo.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SIGMA_ROOT="${ROOT}/external/sigma_ezpc/GPU-MPC"

if [[ ! -d "${SIGMA_ROOT}" ]]; then
  echo "Sigma root not found at ${SIGMA_ROOT}. Clone https://github.com/mpc-msri/EzPC.git sigma_ezpc under external/." >&2
  exit 1
fi

mkdir -p "${SIGMA_ROOT}/build_cpu" "${SIGMA_ROOT}/build_gpu"

echo "Building Sigma CPU..."
cmake -S "${SIGMA_ROOT}" -B "${SIGMA_ROOT}/build_cpu" -DCMAKE_BUILD_TYPE=Release -DSIGMA_BACKEND=CPU
cmake --build "${SIGMA_ROOT}/build_cpu" -j"$(nproc)"

echo "Building Sigma GPU..."
cmake -S "${SIGMA_ROOT}" -B "${SIGMA_ROOT}/build_gpu" -DCMAKE_BUILD_TYPE=Release -DSIGMA_BACKEND=GPU
cmake --build "${SIGMA_ROOT}/build_gpu" -j"$(nproc)"

echo "Done."
