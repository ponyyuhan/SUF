#pragma once

#include <memory>
#include <vector>
#include <cstring>

#include "compiler/truncation_lowering.hpp"
#include "gates/composite_fss.hpp"
#include "nn/tensor_view.hpp"
#include "proto/pfss_backend_batch.hpp"
#include "mpc/net.hpp"
#include "proto/channel.hpp"

namespace runtime {

// Adapter bridging net::Chan to proto::IChannel so callers can flush PFSS
// batches without rewriting networking code.
class ProtoChanFromNet final : public proto::IChannel {
 public:
  explicit ProtoChanFromNet(net::Chan& c) : ch_(c) {}
  void send_bytes(const void* data, size_t n) override {
    ch_.send_u64(static_cast<uint64_t>(n));
    const uint8_t* p = static_cast<const uint8_t*>(data);
    size_t off = 0;
    while (off < n) {
      uint64_t word = 0;
      size_t chunk = std::min<size_t>(8, n - off);
      std::memcpy(&word, p + off, chunk);
      ch_.send_u64(word);
      off += chunk;
    }
  }
  void recv_bytes(void* data, size_t n) override {
    uint64_t expect = ch_.recv_u64();
    if (expect != n) throw std::runtime_error("ProtoChanFromNet: length mismatch");
    uint8_t* p = static_cast<uint8_t*>(data);
    size_t off = 0;
    while (off < n) {
      uint64_t word = ch_.recv_u64();
      size_t chunk = std::min<size_t>(8, n - off);
      std::memcpy(p + off, &word, chunk);
      off += chunk;
    }
  }

 private:
  net::Chan& ch_;
};

// Handle to batched truncation work; intended to be extended to support
// predicate/coeff coalescing. Generic composite job so non-trunc gates can
// reuse the same batching surface.
struct PreparedCompositeJob {
  const suf::SUF<uint64_t>* suf = nullptr;
  const gates::CompositePartyKey* key = nullptr;
  gates::PostProcHook* hook = nullptr;
  std::vector<uint64_t> hatx_public;
  nn::TensorView<uint64_t> out;  // destination for masked output share
};

class PfssSuperBatch {
 public:
  void enqueue_truncation(const compiler::TruncationLoweringResult& bundle,
                          const gates::CompositePartyKey& key,
                          gates::PostProcHook& hook,
                          std::vector<uint64_t> hatx_public,
                          nn::TensorView<uint64_t> out) {
    PreparedCompositeJob job;
    job.suf = &bundle.suf;
    job.key = &key;
    job.hook = &hook;
    job.hatx_public = std::move(hatx_public);
    job.out = out;
    jobs_.push_back(std::move(job));
  }

  void enqueue_composite(PreparedCompositeJob job) { jobs_.push_back(std::move(job)); }

  bool empty() const { return jobs_.empty(); }

  // Evaluate all queued truncation jobs and write their outputs.
  void flush_and_finalize(int party, proto::PfssBackendBatch& backend, proto::IChannel& ch);

  void clear() { jobs_.clear(); }

 private:
  std::vector<PreparedCompositeJob> jobs_;
};

// Convenience: run a truncation bundle immediately on a flat vector of shares.
void run_truncation_now(int party,
                        proto::PfssBackendBatch& backend,
                        proto::IChannel& ch,
                        const compiler::TruncationLoweringResult& bundle,
                        const std::vector<uint64_t>& x_share,
                        std::vector<uint64_t>& y_share);

}  // namespace runtime
