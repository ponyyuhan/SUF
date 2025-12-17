#include "nn/model_specs.hpp"

#include <stdexcept>

namespace nn {
namespace {

// Starter table using common HF configs and the bitwidth guidance from bench.md.
static const std::vector<ModelSpec> kSpecs = {
    // Match SIGMA's bert-tiny bitwidth (see EzPC GPU-MPC experiments/sigma/sigma.cu).
    {"bert-tiny", "prajjwal1/bert-tiny", "gelu", /*causal=*/false, 2, 2, 128, 512, 512, 37, 12},
    {"bert-base", "bert-base-uncased", "gelu", /*causal=*/false, 12, 12, 768, 3072, 512, 50, 12},
    {"bert-large", "bert-large-uncased", "gelu", /*causal=*/false, 24, 16, 1024, 4096, 512, 51, 12},
    {"gpt2", "gpt2", "gelu", /*causal=*/true, 12, 12, 768, 3072, 1024, 50, 12},
    // Paper Table 1: n_head=16, d_model=2048.
    {"gpt-neo-1.3b", "EleutherAI/gpt-neo-1.3B", "gelu", /*causal=*/true, 24, 16, 2048, 8192, 2048, 51, 12},
    // Optional heavier models; keep bitwidths conservative.
    {"llama2-7b", "meta-llama/Llama-2-7b-hf", "silu", /*causal=*/true, 32, 32, 4096, 11008, 2048, 51, 12},
    {"llama2-13b", "meta-llama/Llama-2-13b-hf", "silu", /*causal=*/true, 40, 40, 5120, 13824, 2048, 51, 12},
};

}  // namespace

const std::vector<ModelSpec>& list_model_specs() { return kSpecs; }

const ModelSpec& get_model_spec(const std::string& name) {
  for (const auto& s : kSpecs) {
    if (s.name == name) return s;
  }
  throw std::runtime_error("unknown model spec: " + name);
}

}  // namespace nn
