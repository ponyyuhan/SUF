#pragma once

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

#include "proto/backend_clear.hpp"
#include "proto/backend_gpu.hpp"

namespace proto {

enum class PfssBackendKind { Auto, Cpu, Gpu };

struct PfssBackendOptions {
  PfssBackendKind kind = PfssBackendKind::Auto;
  bool allow_gpu_stub = true;  // if GPU is unavailable, fall back to CPU instead of throwing
};

inline PfssBackendKind parse_backend_kind(const char* env) {
  if (!env) return PfssBackendKind::Auto;
  std::string v(env);
  std::transform(v.begin(), v.end(), v.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  if (v == "gpu") return PfssBackendKind::Gpu;
  if (v == "cpu") return PfssBackendKind::Cpu;
  return PfssBackendKind::Auto;
}

inline std::unique_ptr<PfssBackendBatch> make_pfss_backend(const PfssBackendOptions& opts = {}) {
  auto kind = opts.kind;
  if (kind == PfssBackendKind::Auto) {
    const char* env = std::getenv("SUF_PFSS_BACKEND");
    kind = parse_backend_kind(env);
  }

  if (kind == PfssBackendKind::Gpu) {
#ifdef SUF_HAVE_CUDA
    auto gpu = make_real_gpu_backend();
    if (!gpu) throw std::runtime_error("GPU backend unavailable");
    return gpu;
#else
    if (!opts.allow_gpu_stub) {
      throw std::runtime_error("GPU backend requested but SUF_HAVE_CUDA is not enabled");
    }
    // Fall back to CPU backend while keeping the interface identical.
    return ClearBackend::make();
#endif
  }

  // CPU/default path.
  return ClearBackend::make();
}

inline std::unique_ptr<PfssBackendBatch> make_pfss_backend_from_env(const PfssBackendOptions& defaults = {}) {
  PfssBackendOptions opts = defaults;
  const char* env = std::getenv("SUF_PFSS_BACKEND");
  if (env) opts.kind = parse_backend_kind(env);
  return make_pfss_backend(opts);
}

}  // namespace proto
