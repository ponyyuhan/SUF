#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace nn {

struct ModelSpec {
  std::string name;
  std::string hf_id;
  std::size_t n_layers = 0;
  std::size_t n_heads = 0;
  std::size_t d_model = 0;
  std::size_t d_ff = 0;
  std::size_t max_seq_len = 0;
  unsigned n_bits = 0;
  unsigned frac_bits = 0;
};

// Returns all built-in model specs (BERT/GPT families used in bench.md).
const std::vector<ModelSpec>& list_model_specs();

// Throws std::runtime_error if the model is unknown.
const ModelSpec& get_model_spec(const std::string& name);

}  // namespace nn
