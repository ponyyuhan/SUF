#pragma once

#include <vector>
#include <span>
#include <stdexcept>

#include "runtime/phase_tasks.hpp"
#include "proto/common.hpp"
#include "proto/beaver.hpp"

namespace runtime {

// Secret matmul via Beaver triples: computes C = A * B where A is [M x K], B is [K x N]
// stored row-major as flat spans. Output C is written to out (size M*N).
// This is a basic two-phase task: one open of (D,E) followed by local reconstruction.
class MatmulTask final : public detail::PhaseTask {
 public:
  MatmulTask(int M,
             int K,
             int N,
             std::span<const uint64_t> A,
             std::span<const uint64_t> B,
             std::span<uint64_t> out,
             std::span<const proto::BeaverTriple64Share> triples)
      : M_(M), K_(K), N_(N), A_(A), B_(B), out_(out), triples_(triples) {
    if (A_.size() != static_cast<size_t>(M_) * static_cast<size_t>(K_)) {
      throw std::runtime_error("MatmulTask: A size mismatch (got " + std::to_string(A_.size()) +
                               ", expected " + std::to_string(static_cast<size_t>(M_) * static_cast<size_t>(K_)) +
                               ")");
    }
    if (B_.size() != static_cast<size_t>(K_) * static_cast<size_t>(N_)) {
      throw std::runtime_error("MatmulTask: B size mismatch (got " + std::to_string(B_.size()) +
                               ", expected " + std::to_string(static_cast<size_t>(K_) * static_cast<size_t>(N_)) +
                               ")");
    }
    if (out_.size() != static_cast<size_t>(M_) * static_cast<size_t>(N_)) {
      throw std::runtime_error("MatmulTask: out size mismatch (got " + std::to_string(out_.size()) +
                               ", expected " + std::to_string(static_cast<size_t>(M_) * static_cast<size_t>(N_)) +
                               ")");
    }
    size_t need = static_cast<size_t>(M_) * static_cast<size_t>(K_) * static_cast<size_t>(N_);
    if (triples_.size() < need) {
      throw std::runtime_error("MatmulTask: insufficient triples");
    }
  }

  bool done() const override { return st_ == St::Done; }

  detail::Need step(PhaseResources& R) override {
    switch (st_) {
      case St::Init: {
        if (!R.opens) throw std::runtime_error("MatmulTask: no OpenCollector");
        // Form D (A - a) and E (B - b) for all products needed.
        size_t total = static_cast<size_t>(M_) * static_cast<size_t>(N_) * static_cast<size_t>(K_);
        buf_de_.resize(2 * total);
        #pragma omp parallel for collapse(3) schedule(static)
        for (int m = 0; m < M_; ++m) {
          for (int n = 0; n < N_; ++n) {
            for (int k = 0; k < K_; ++k) {
              size_t idx = static_cast<size_t>(m) * static_cast<size_t>(N_) * static_cast<size_t>(K_) +
                           static_cast<size_t>(n) * static_cast<size_t>(K_) + static_cast<size_t>(k);
              size_t a_idx = static_cast<size_t>(m) * static_cast<size_t>(K_) + static_cast<size_t>(k);
              size_t b_idx = static_cast<size_t>(k) * static_cast<size_t>(N_) + static_cast<size_t>(n);
              buf_de_[idx] = proto::sub_mod(A_[a_idx], triples_[idx].a);
              buf_de_[total + idx] = proto::sub_mod(B_[b_idx], triples_[idx].b);
            }
          }
        }
        h_open_ = R.opens->enqueue(buf_de_, OpenKind::kBeaver);
        st_ = St::WaitOpen;
        return detail::Need::Open;
      }
      case St::WaitOpen: {
        if (!R.opens->ready(h_open_)) return detail::Need::Open;
        auto opened = R.opens->view(h_open_);
        size_t total = static_cast<size_t>(M_) * static_cast<size_t>(N_) * static_cast<size_t>(K_);
        if (opened.size() != 2 * total) {
          throw std::runtime_error("MatmulTask: opened size mismatch");
        }
        // Reconstruct all products and accumulate into out.
        #pragma omp parallel for collapse(2) schedule(static)
        for (int m = 0; m < M_; ++m) {
          for (int n = 0; n < N_; ++n) {
            uint64_t acc = 0;
            for (int k = 0; k < K_; ++k) {
              size_t idx = static_cast<size_t>(m) * static_cast<size_t>(N_) * static_cast<size_t>(K_) +
                           static_cast<size_t>(n) * static_cast<size_t>(K_) + static_cast<size_t>(k);
              uint64_t d = static_cast<uint64_t>(opened[idx]);
              uint64_t e = static_cast<uint64_t>(opened[total + idx]);
              uint64_t z = triples_[idx].c;
              z = proto::add_mod(z, proto::mul_mod(d, triples_[idx].b));
              z = proto::add_mod(z, proto::mul_mod(e, triples_[idx].a));
              if (R.party == 0) {
                z = proto::add_mod(z, proto::mul_mod(d, e));
              }
              acc = proto::add_mod(acc, z);
            }
            size_t out_idx = static_cast<size_t>(m) * static_cast<size_t>(N_) + static_cast<size_t>(n);
            out_[out_idx] = acc;
          }
        }
        st_ = St::Done;
        return detail::Need::None;
      }
      case St::Done:
        return detail::Need::None;
    }
    return detail::Need::None;
  }

 private:
  enum class St { Init, WaitOpen, Done } st_ = St::Init;
  int M_, K_, N_;
  std::span<const uint64_t> A_;
  std::span<const uint64_t> B_;
  std::span<uint64_t> out_;
  std::span<const proto::BeaverTriple64Share> triples_;
  std::vector<uint64_t> buf_de_;
  OpenHandle h_open_{};
};

}  // namespace runtime
