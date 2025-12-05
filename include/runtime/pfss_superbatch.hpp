#pragma once

#include <cstddef>
#include <cstring>
#include <memory>
#include <type_traits>
#include <vector>
#if __has_include(<span>)
  #include <span>
#else
  #if !defined(SUF_SPAN_FALLBACK_DEFINED)
    #define SUF_SPAN_FALLBACK_DEFINED
    namespace std {
      template<typename T>
      class span {
       public:
        span() : data_(nullptr), size_(0) {}
        span(const T* ptr, std::size_t n) : data_(ptr), size_(n) {}
        template <typename U, typename = std::enable_if_t<std::is_same_v<std::remove_const_t<T>, U>>>
        span(const std::vector<U>& v) : data_(v.data()), size_(v.size()) {}
        std::size_t size() const { return size_; }
        const T* data() const { return data_; }
        const T& operator[](std::size_t i) const { return data_[i]; }
        span subspan(std::size_t off, std::size_t n) const {
          if (off > size_) return span();
          std::size_t len = (off + n > size_) ? (size_ - off) : n;
          return span(data_ + off, len);
        }
        const T* begin() const { return data_; }
        const T* end() const { return data_ + size_; }
      private:
        const T* data_;
        std::size_t size_;
      };
      template <typename T>
      const T* begin(span<T> s) { return s.data(); }
      template <typename T>
      const T* end(span<T> s) { return s.data() + s.size(); }
    }
  #endif
#endif

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

// Handle returned to callers so they can look up PFSS outputs after a flush.
struct PfssHandle {
  size_t token = static_cast<size_t>(-1);
};

// View into stored PFSS results (arith + bool shares). Pointers remain valid
// until clear() is called.
struct PfssResultView {
  const uint64_t* arith = nullptr;
  size_t arith_words = 0;
  const uint64_t* bools = nullptr;
  size_t bool_words = 0;
  size_t r = 0;
  size_t ell = 0;

  // Convenience: interpret arithmetic payload as SoA blocks. Caller ensures
  // arith_words == rows * cols.
  std::span<const uint64_t> soa_col(size_t col, size_t rows) const {
    if (arith == nullptr) return {};
    size_t off = col * rows;
    if (off >= arith_words) return {};
    size_t len = std::min(rows, arith_words - off);
    return std::span<const uint64_t>(arith + off, len);
  }
};

// Generic composite job so non-trunc gates can reuse the same batching surface.
struct PreparedCompositeJob {
  const suf::SUF<uint64_t>* suf = nullptr;
  const gates::CompositePartyKey* key = nullptr;
  gates::PostProcHook* hook = nullptr;
  std::vector<uint64_t> hatx_public;
  nn::TensorView<uint64_t> out;  // destination for masked output share
  size_t token = static_cast<size_t>(-1);  // filled by PfssSuperBatch
};

class PfssSuperBatch {
 public:
  // Enqueue a composite job; no implicit finalize is performed. Caller should
  // flush(), then read view() to consume outputs.
  PfssHandle enqueue_composite(PreparedCompositeJob job);
  // Legacy helper: enqueue truncation gate with postproc; kept for compatibility.
  PfssHandle enqueue_truncation(const compiler::TruncationLoweringResult& bundle,
                                const gates::CompositePartyKey& key,
                                gates::PostProcHook& hook,
                                std::vector<uint64_t> hatx_public,
                                nn::TensorView<uint64_t> out);

  bool empty() const { return jobs_.empty(); }
  bool has_pending() const { return !jobs_.empty() && !flushed_; }
  // Ready check for callers using multi-wave task scheduling.
  bool ready(const PfssHandle& h) const;

  // Evaluate all queued composite jobs and store PFSS outputs.
  void flush(int party, proto::PfssBackendBatch& backend, proto::IChannel& ch);

  // Legacy finalize that applies hooks and writes masked outputs.
  void finalize(int party, proto::IChannel& ch);

  // Convenience wrapper that performs flush() followed by finalize().
  void flush_and_finalize(int party, proto::PfssBackendBatch& backend, proto::IChannel& ch);

  // Drop all pending jobs and stored results.
  void clear();

  // Access raw PFSS outputs (arith/bool shares) for a completed handle.
  PfssResultView view(const PfssHandle& h) const;

 private:
  struct GroupResult {
    const suf::SUF<uint64_t>* suf = nullptr;
    const gates::CompositePartyKey* key = nullptr;
    size_t r = 0;
    size_t ell = 0;
    std::vector<uint64_t> arith;  // [N * r]
    std::vector<uint64_t> bools;  // [N * ell]
  };
  struct CompletedJob {
    size_t r = 0;
    size_t ell = 0;
    std::vector<uint64_t> arith;  // postproc + unmask
    std::vector<uint64_t> bools;  // raw PFSS bool slice for this job
  };
  struct JobSlice {
    size_t group_result = static_cast<size_t>(-1);
    size_t start = 0;
    size_t len = 0;
  };

  std::vector<PreparedCompositeJob> jobs_;
  std::vector<GroupResult> group_results_;
  std::vector<CompletedJob> completed_;
  std::vector<JobSlice> slices_;
  bool flushed_ = false;
};

// Convenience: run a truncation bundle immediately on a flat vector of shares.
void run_truncation_now(int party,
                        proto::PfssBackendBatch& backend,
                        proto::IChannel& ch,
                        const compiler::TruncationLoweringResult& bundle,
                        const std::vector<uint64_t>& x_share,
                        std::vector<uint64_t>& y_share);

}  // namespace runtime
