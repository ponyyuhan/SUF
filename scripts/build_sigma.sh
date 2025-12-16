#!/usr/bin/env bash
# Helper to build Sigma (EzPC GPU-MPC SIGMA from ePrint 2023/1269).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EZPC_ROOT="${ROOT}/external/sigma_ezpc"
SIGMA_ROOT="${EZPC_ROOT}/GPU-MPC"

CUDA_BIN="${CUDA_BIN:-/usr/bin/nvcc}"
CUDA_VERSION="${CUDA_VERSION:-11.5}"
GPU_ARCH="${GPU_ARCH:-86}"
HOST_CXX="${HOST_CXX:-/usr/bin/g++-9}"
CMAKE_BIN="${CMAKE_BIN:-cmake}"

if [[ ! -d "${SIGMA_ROOT}" ]]; then
  mkdir -p "${ROOT}/external"
  echo "[sigma] cloning EzPC into ${EZPC_ROOT}"
  git clone --depth 1 https://github.com/mpc-msri/EzPC.git "${EZPC_ROOT}"
fi

echo "[sigma] updating submodules"
(cd "${SIGMA_ROOT}" && git submodule update --init --recursive)

PATCH="${ROOT}/scripts/patches/sigma_gpu_mpc.patch"
if [[ -f "${PATCH}" ]]; then
  echo "[sigma] applying local SIGMA patches"
  if git -C "${EZPC_ROOT}" apply --reverse --check "${PATCH}" >/dev/null 2>&1; then
    echo "[sigma] patch already applied"
  else
    git -C "${EZPC_ROOT}" apply "${PATCH}"
  fi
fi

echo "[sigma] building Sytorch (CUDA + g++ host)"
SYTORCH_BUILD="${SIGMA_ROOT}/ext/sytorch/build"
"${CMAKE_BIN}" -S "${SIGMA_ROOT}/ext/sytorch" -B "${SYTORCH_BUILD}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER="${CUDA_BIN}" \
  -DCMAKE_CUDA_HOST_COMPILER="${HOST_CXX}" \
  -DCUDAToolkit_ROOT=/usr
"${CMAKE_BIN}" --build "${SYTORCH_BUILD}" --target sytorch -j"$(nproc)"

echo "[sigma] building SIGMA binary"
# The Makefile assumes /usr/local/cuda-${CUDA_VERSION}/bin/nvcc by default; override it.
# NOTE: CUTLASS is vendored as a submodule and can take a long time to build if invoked.
make -C "${SIGMA_ROOT}" sigma \
  CXX="${CUDA_BIN}" \
  CUDA_VERSION="${CUDA_VERSION}" \
  GPU_ARCH="${GPU_ARCH}" \
  FLAGS="-O3 -gencode arch=compute_${GPU_ARCH},code=[sm_${GPU_ARCH},compute_${GPU_ARCH}] -std=c++17 -m64 -ccbin ${HOST_CXX} -Xcompiler=\"-O3,-w,-std=c++17,-fpermissive,-fpic,-pthread,-fopenmp,-march=native\""

if [[ ! -x "${SIGMA_ROOT}/experiments/sigma/sigma" ]]; then
  echo "[sigma] build finished but binary not found at ${SIGMA_ROOT}/experiments/sigma/sigma" >&2
  exit 1
fi

echo "[sigma] done: ${SIGMA_ROOT}/experiments/sigma/sigma"
