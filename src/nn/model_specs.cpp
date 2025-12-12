#include "nn/model_specs.hpp"

#include <stdexcept>

namespace nn {
namespace {

// Starter table using common HF configs and the bitwidth guidance from bench.md.
static const std::vector<ModelSpec> kSpecs = {
    {"bert-base", "bert-base-uncased", 12, 12, 768, 3072, 512, 50, 12},
    {"bert-large", "bert-large-uncased", 24, 16, 1024, 4096, 512, 51, 12},
    {"gpt2", "gpt2", 12, 12, 768, 3072, 1024, 50, 12},
    {"gpt-neo-1.3b", "EleutherAI/gpt-neo-1.3B", 24, 20, 2048, 8192, 2048, 51, 12},
    // Optional heavier models; keep bitwidths conservative.
    {"llama2-7b", "meta-llama/Llama-2-7b-hf", 32, 32, 4096, 11008, 2048, 51, 12},
    {"llama2-13b", "meta-llama/Llama-2-13b-hf", 40, 40, 5120, 13824, 2048, 51, 12},
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
