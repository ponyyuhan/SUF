#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace pfss {

// Opaque key blob (serialize-friendly)
struct Key {
  std::vector<uint8_t> bytes;
};

// Public parameters (if needed by backend)
struct PublicParams {
  std::string backend_name;
  int lambda_bits = 128;
};

// Generic “program description” (filled by compiler & read by dealer only)
struct ProgramDesc {
  std::string kind;                  // "predicates", "coeff_lut", ...
  std::vector<uint8_t> dealer_only_desc;  // backend-specific encoding
};

// Backend interface: ProgGen in dealer, Eval in online parties.
template<typename PayloadT>
struct Backend {
  virtual PublicParams setup(int lambda_bits) = 0;

  virtual std::pair<Key, Key> prog_gen(const PublicParams& pp,
                                       const ProgramDesc& desc) = 0;

  virtual PayloadT eval(int party,
                        const PublicParams& pp,
                        const Key& key,
                        uint64_t x_hat_public) const = 0;

  // Optional batched API (CPU multithread / GPU)
  virtual void eval_batch(int party,
                          const PublicParams& pp,
                          const Key& key,
                          const uint64_t* x_hat,
                          PayloadT* out,
                          size_t count) const {
    for (size_t i = 0; i < count; i++) out[i] = eval(party, pp, key, x_hat[i]);
  }

  virtual ~Backend() = default;
};

}  // namespace pfss
