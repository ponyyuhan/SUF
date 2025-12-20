#!/usr/bin/env bash
# Helper to build Sigma (EzPC GPU-MPC SIGMA from ePrint 2023/1269).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EZPC_ROOT="${ROOT}/external/sigma_ezpc"
SIGMA_ROOT="${EZPC_ROOT}/GPU-MPC"
JOBS="${SIGMA_JOBS:-16}"

if [[ "${JOBS}" -le 0 ]]; then
  JOBS=16
fi

GEN=""
if command -v ninja >/dev/null 2>&1; then
  GEN="-GNinja"
fi

detect_cuda_bin() {
  if [[ -n "${CUDA_BIN:-}" ]]; then
    echo "${CUDA_BIN}"
    return
  fi
  if command -v nvcc >/dev/null 2>&1; then
    command -v nvcc
    return
  fi
  echo "/usr/local/cuda/bin/nvcc"
}

detect_cuda_version() {
  if [[ -n "${CUDA_VERSION:-}" ]]; then
    echo "${CUDA_VERSION}"
    return
  fi
  local nvcc
  nvcc="$(detect_cuda_bin)"
  local v
  v="$("${nvcc}" --version | sed -n 's/.*release \\([0-9]\\+\\.[0-9]\\+\\).*/\\1/p' | head -n1 || true)"
  if [[ -n "${v}" ]]; then
    echo "${v}"
  else
    echo "11.7"
  fi
}

detect_gpu_arch() {
  if [[ -n "${GPU_ARCH:-}" ]]; then
    echo "${GPU_ARCH}"
    return
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    local cap
    cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d ' ' || true)"
    if [[ "${cap}" =~ ^[0-9]+\.[0-9]+$ ]]; then
      local major="${cap%.*}"
      local minor="${cap#*.}"
      echo "${major}${minor}"
      return
    fi
  fi
  echo "86"
}

CUDA_BIN="$(detect_cuda_bin)"
CUDA_VERSION="$(detect_cuda_version)"
GPU_ARCH="$(detect_gpu_arch)"
HOST_CXX="${HOST_CXX:-$(command -v g++ || true)}"
HOST_CXX="${HOST_CXX:-/usr/bin/g++}"
CMAKE_BIN="${CMAKE_BIN:-cmake}"
MAKE_BIN="${MAKE_BIN:-make}"

CUDA_ROOT="/usr/local/cuda-${CUDA_VERSION}"
if [[ ! -d "${CUDA_ROOT}" ]]; then
  CUDA_ROOT="/usr/local/cuda"
fi

if [[ ! -d "${SIGMA_ROOT}" ]]; then
  mkdir -p "${ROOT}/external"
  echo "[sigma] cloning EzPC into ${EZPC_ROOT}"
  git clone --depth 1 https://github.com/mpc-msri/EzPC.git "${EZPC_ROOT}"
fi

echo "[sigma] updating submodules"
(cd "${SIGMA_ROOT}" && git submodule update --init --recursive --progress)

PATCH="${ROOT}/scripts/patches/sigma_gpu_mpc.patch"
if [[ -f "${PATCH}" ]]; then
  echo "[sigma] applying local SIGMA patches"
  if git -C "${EZPC_ROOT}" apply --reverse --check "${PATCH}" >/dev/null 2>&1; then
    echo "[sigma] patch already applied"
  else
    git -C "${EZPC_ROOT}" apply "${PATCH}"
  fi
fi

SCI_PATCH="${ROOT}/scripts/patches/sigma_sci_cleartext.patch"
if [[ -f "${SCI_PATCH}" ]]; then
  echo "[sigma] applying local SCI patches"
  if git -C "${EZPC_ROOT}" apply --reverse --check --ignore-whitespace "${SCI_PATCH}" >/dev/null 2>&1; then
    echo "[sigma] sci patch already applied"
  else
    git -C "${EZPC_ROOT}" apply --ignore-whitespace "${SCI_PATCH}"
  fi
fi

SEAL_PATCH="${ROOT}/scripts/patches/sigma_seal.patch"
SEAL_ROOT="${SIGMA_ROOT}/ext/sytorch/ext/sci/extern/SEAL"
if [[ -f "${SEAL_PATCH}" && -d "${SEAL_ROOT}" ]]; then
  echo "[sigma] applying local SEAL patch (mutex include)"
  if git -C "${SEAL_ROOT}" apply --reverse --check "${SEAL_PATCH}" >/dev/null 2>&1; then
    echo "[sigma] seal patch already applied"
  else
    git -C "${SEAL_ROOT}" apply "${SEAL_PATCH}"
  fi
fi

echo "[sigma] building Sytorch (CUDA + g++ host)"
SYTORCH_BUILD="${SIGMA_ROOT}/ext/sytorch/build"
mkdir -p "${SYTORCH_BUILD}"
# If the build directory already has a CMakeCache.txt, reuse its generator.
# (Passing a different -G causes a hard error and looks like a "deadlock".)
SYTORCH_GEN="${GEN}"
if [[ -f "${SYTORCH_BUILD}/CMakeCache.txt" ]]; then
  SYTORCH_GEN=""
fi
"${CMAKE_BIN}" ${SYTORCH_GEN} -S "${SIGMA_ROOT}/ext/sytorch" -B "${SYTORCH_BUILD}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER="${CUDA_BIN}" \
  -DCMAKE_CUDA_HOST_COMPILER="${HOST_CXX}" \
  -DCUDAToolkit_ROOT="${CUDA_ROOT}"
"${CMAKE_BIN}" --build "${SYTORCH_BUILD}" --target sytorch -j"${JOBS}"

echo "[sigma] building CUTLASS (arch=${GPU_ARCH})"
CUTLASS_BUILD="${SIGMA_ROOT}/ext/cutlass/build"
CUTLASS_LIB="${CUTLASS_BUILD}/tools/library/libcutlass.so"
if [[ "${SIGMA_BUILD_CUTLASS_LIB:-0}" == "1" ]]; then
  if [[ -f "${CUTLASS_LIB}" ]]; then
    echo "[sigma] CUTLASS already built: ${CUTLASS_LIB}"
  else
    mkdir -p "${CUTLASS_BUILD}"
    CUTLASS_GEN="${GEN}"
    if [[ -f "${CUTLASS_BUILD}/CMakeCache.txt" ]]; then
      CUTLASS_GEN=""
    fi
    "${CMAKE_BIN}" ${CUTLASS_GEN} -S "${SIGMA_ROOT}/ext/cutlass" -B "${CUTLASS_BUILD}" \
      -DCUTLASS_NVCC_ARCHS="${GPU_ARCH}" \
      -DCMAKE_CUDA_COMPILER_WORKS=1 \
      -DCMAKE_CUDA_COMPILER="${CUDA_BIN}"
    # Build the CUTLASS library artifacts (expensive; not required for SIGMA's binary).
    "${CMAKE_BIN}" --build "${CUTLASS_BUILD}" --target cutlass_lib -j"${JOBS}"
  fi
else
  echo "[sigma] skipping CUTLASS library build (set SIGMA_BUILD_CUTLASS_LIB=1 to enable)"
fi

echo "[sigma] building SIGMA binary"
# The Makefile assumes /usr/local/cuda-${CUDA_VERSION}/bin/nvcc by default; override it.
# NOTE: CUTLASS is vendored as a submodule and can take a long time to build if invoked.
"${MAKE_BIN}" -C "${SIGMA_ROOT}" sigma \
  CXX="${CUDA_BIN}" \
  CUDA_VERSION="${CUDA_VERSION}" \
  GPU_ARCH="${GPU_ARCH}" \
  FLAGS="-O3 -gencode arch=compute_${GPU_ARCH},code=[sm_${GPU_ARCH},compute_${GPU_ARCH}] -std=c++17 -m64 -ccbin ${HOST_CXX} -Xcompiler=\"-O3,-w,-std=c++17,-fpermissive,-fpic,-pthread,-fopenmp,-march=native\""

if [[ ! -x "${SIGMA_ROOT}/experiments/sigma/sigma" ]]; then
  echo "[sigma] build finished but binary not found at ${SIGMA_ROOT}/experiments/sigma/sigma" >&2
  exit 1
fi

echo "[sigma] done: ${SIGMA_ROOT}/experiments/sigma/sigma"
