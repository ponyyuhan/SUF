#pragma once

#include <vector>
#include <cstddef>
#include <type_traits>
#include <random>
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
        T* data() { return const_cast<T*>(data_); }
        const T& operator[](std::size_t i) const { return data_[i]; }
        T& operator[](std::size_t i) { return const_cast<T&>(data_[i]); }
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
#include <stdexcept>
#include <iostream>

#include "compiler/truncation_lowering.hpp"
#include "compiler/range_analysis.hpp"
#include "gates/postproc_hooks.hpp"
#include "proto/beaver.hpp"
#include "proto/beaver_mul64.hpp"
#include "proto/common.hpp"
#include "runtime/open_collector.hpp"
#include "runtime/phase_executor.hpp"
#include "runtime/pfss_superbatch.hpp"

namespace runtime {

// Simple Beaver mul task (secret x secret). Uses OpenCollector if provided,
// otherwise falls back to direct channel opens.
class MulTask final : public detail::PhaseTask {
 public:
  MulTask(std::span<const uint64_t> x,
          std::span<const uint64_t> y,
          std::span<uint64_t> out,
          std::span<const proto::BeaverTriple64Share> triples)
      : x_(x), y_(y), out_(out), triples_(triples) {
    if (x_.size() != y_.size() || x_.size() != out_.size()) {
      throw std::runtime_error("MulTask: size mismatch");
    }
    if (triples_.size() < x_.size()) {
      throw std::runtime_error("MulTask: insufficient triples");
    }
  }

  bool done() const override { return st_ == St::Done; }

  detail::Need step(PhaseResources& R) override {
    switch (st_) {
      case St::Init: {
        if (!R.net_chan) throw std::runtime_error("MulTask: net channel missing");
        diff_.resize(2 * x_.size());
        for (size_t i = 0; i < x_.size(); ++i) {
          diff_[i] = proto::sub_mod(x_[i], triples_[i].a);
          diff_[x_.size() + i] = proto::sub_mod(y_[i], triples_[i].b);
        }
        if (R.opens) {
          h_ = R.opens->enqueue(diff_);
          st_ = St::WaitOpen;
          return detail::Need::Open;
        }
        opened_.assign(diff_.size(), 0);
        // Fallback direct open.
        for (size_t i = 0; i < diff_.size(); ++i) {
          if (R.party == 0) {
            R.net_chan->send_u64(diff_[i]);
            opened_[i] = static_cast<int64_t>(diff_[i] + R.net_chan->recv_u64());
          } else {
            opened_[i] = static_cast<int64_t>(diff_[i] + R.net_chan->recv_u64());
            R.net_chan->send_u64(diff_[i]);
          }
        }
        st_ = St::Finalize;
        [[fallthrough]];
      }
      case St::WaitOpen: {
        if (st_ == St::WaitOpen) {
          if (!R.opens) throw std::runtime_error("MulTask: no OpenCollector");
          if (!R.opens->ready(h_)) return detail::Need::Open;
          auto v = R.opens->view(h_);
          if (v.size() != diff_.size()) {
            throw std::runtime_error("MulTask: opened size mismatch");
          }
          opened_.assign(v.begin(), v.end());
          st_ = St::Finalize;
        }
        [[fallthrough]];
      }
      case St::Finalize: {
        for (size_t i = 0; i < x_.size(); ++i) {
          uint64_t d = static_cast<uint64_t>(opened_[i]);
          uint64_t e = static_cast<uint64_t>(opened_[x_.size() + i]);
          uint64_t z = triples_[i].c;
          z = proto::add_mod(z, proto::mul_mod(d, triples_[i].b));
          z = proto::add_mod(z, proto::mul_mod(e, triples_[i].a));
          if (R.party == 0) {
            z = proto::add_mod(z, proto::mul_mod(d, e));
          }
          out_[i] = z;
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
  enum class St { Init, WaitOpen, Finalize, Done } st_ = St::Init;
  std::span<const uint64_t> x_;
  std::span<const uint64_t> y_;
  std::span<uint64_t> out_;
  std::span<const proto::BeaverTriple64Share> triples_;

  std::vector<uint64_t> diff_;
  std::vector<int64_t> opened_;
  OpenHandle h_{};
};

// Faithful truncation task (TR/ARS/GapARS) using existing composite bundle.
class TruncTask final : public detail::PhaseTask {
 public:
  TruncTask(const compiler::TruncationLoweringResult* bundle,
            std::span<const uint64_t> in_share,
            std::span<uint64_t> out_share)
      : bundle_(bundle), in_(in_share), out_(out_share) {
    if (!bundle_) throw std::runtime_error("TruncTask: bundle null");
    if (in_.size() != out_.size()) {
      throw std::runtime_error("TruncTask: size mismatch");
    }
  }

  bool done() const override { return st_ == St::Done; }

  detail::Need step(PhaseResources& R) override {
    if (!R.pfss || !R.pfss_backend || !R.pfss_chan) {
      throw std::runtime_error("TruncTask: PFSS resources missing");
    }
    if (!R.net_chan) {
      throw std::runtime_error("TruncTask: net channel missing");
    }
    // Resolve party-specific view lazily.
    if (!key_) {
      key_ = (R.party == 0) ? &bundle_->keys.k0 : &bundle_->keys.k1;
      hook_ = (R.party == 0) ? bundle_->hook0.get() : bundle_->hook1.get();
      if (!hook_) throw std::runtime_error("TruncTask: hook missing");
      hook_->configure(key_->compiled.layout);
    }

    switch (st_) {
      case St::OpenXhat: {
        masked_.resize(in_.size());
        for (size_t i = 0; i < in_.size(); ++i) {
          masked_[i] = proto::add_mod(in_[i], key_->r_in_share);
        }
        if (R.opens) {
          h_open_ = R.opens->enqueue(masked_);
          st_ = St::WaitOpen;
          return detail::Need::Open;
        }
        opened_.assign(masked_.size(), 0);
        for (size_t i = 0; i < masked_.size(); ++i) {
          if (R.party == 0) {
            R.net_chan->send_u64(masked_[i]);
            opened_[i] = static_cast<int64_t>(masked_[i] + R.net_chan->recv_u64());
          } else {
            opened_[i] = static_cast<int64_t>(masked_[i] + R.net_chan->recv_u64());
            R.net_chan->send_u64(masked_[i]);
          }
        }
        st_ = St::EnqueuePfss;
        [[fallthrough]];
      }
      case St::WaitOpen: {
        if (st_ == St::WaitOpen) {
          if (!R.opens) throw std::runtime_error("TruncTask: no OpenCollector");
          if (!R.opens->ready(h_open_)) return detail::Need::Open;
          auto v = R.opens->view(h_open_);
          opened_.assign(v.begin(), v.end());
          st_ = St::EnqueuePfss;
        }
        [[fallthrough]];
      }
      case St::EnqueuePfss: {
        PreparedCompositeJob job;
        job.suf = &bundle_->suf;
        job.key = key_;
        job.hook = nullptr;
        job.hatx_public.resize(opened_.size());
        for (size_t i = 0; i < opened_.size(); ++i) {
          job.hatx_public[i] = static_cast<uint64_t>(opened_[i]);
        }
        token_ = R.pfss->enqueue_composite(std::move(job)).token;
        st_ = St::WaitPfss;
        return detail::Need::Pfss;
      }
      case St::WaitPfss: {
        if (!R.pfss->ready(PfssHandle{token_})) return detail::Need::Pfss;
        auto v = R.pfss->view(PfssHandle{token_});
        size_t elems = in_.size();
        if (v.arith_words < elems * v.r) {
          throw std::runtime_error("TruncTask: PFSS arith slice too small");
        }
        std::vector<uint64_t> hook_out(elems * v.r, 0);
        const std::vector<proto::BeaverTriple64Share>* triples = &key_->triples;
        size_t need_triples = std::max<size_t>(elems * v.ell, elems * v.r);
        if (triples->size() < need_triples) {
          fallback_triples_.clear();
          fallback_triples_.reserve(need_triples);
          std::mt19937_64 rng(key_->r_in_share + 12345);
          for (size_t i = 0; i < need_triples; ++i) {
            uint64_t a = rng();
            uint64_t b = rng();
            uint64_t c = proto::mul_mod(a, b);
            uint64_t a0 = rng();
            uint64_t a1 = a - a0;
            uint64_t b0 = rng();
            uint64_t b1 = b - b0;
            uint64_t c0 = rng();
            uint64_t c1 = c - c0;
            if (R.party == 0) {
              fallback_triples_.push_back({a0, b0, c0});
            } else {
              fallback_triples_.push_back({a1, b1, c1});
            }
          }
          triples = &fallback_triples_;
        }
        proto::BeaverMul64 mul{R.party, *R.pfss_chan, *triples};
        hook_->run_batch(R.party, *R.pfss_chan, mul,
                         job_hatx(),
                         v.arith, v.r,
                         v.bools, v.ell,
                         elems,
                         hook_out.data());
        for (size_t i = 0; i < elems; ++i) {
          out_[i] = hook_out[i * v.r];
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
  enum class St { OpenXhat, WaitOpen, EnqueuePfss, WaitPfss, Done } st_ = St::OpenXhat;
  const compiler::TruncationLoweringResult* bundle_ = nullptr;
  const gates::CompositePartyKey* key_ = nullptr;
  gates::PostProcHook* hook_ = nullptr;
  std::span<const uint64_t> in_;
  std::span<uint64_t> out_;
  std::vector<uint64_t> masked_;
  std::vector<int64_t> opened_;
  OpenHandle h_open_{};
  size_t token_ = static_cast<size_t>(-1);
  std::vector<proto::BeaverTriple64Share> fallback_triples_;

  const uint64_t* job_hatx() {
    if (hatx_public_.empty()) {
      hatx_public_.resize(opened_.size());
      for (size_t i = 0; i < opened_.size(); ++i) {
        hatx_public_[i] = static_cast<uint64_t>(opened_[i]);
      }
    }
    return hatx_public_.data();
  }

  std::vector<uint64_t> hatx_public_;
};

struct CubicPolyBundle {
  const suf::SUF<uint64_t>* suf = nullptr;
  const gates::CompositePartyKey* key0 = nullptr;
  const gates::CompositePartyKey* key1 = nullptr;
  const compiler::TruncationLoweringResult* trunc_f = nullptr;
  const compiler::TruncationLoweringResult* trunc_2f = nullptr;
  int frac_bits = 0;
};

struct RecipTaskBundle {
  const suf::SUF<uint64_t>* suf = nullptr;  // affine init coeff program
  const gates::CompositePartyKey* key0 = nullptr;
  const gates::CompositePartyKey* key1 = nullptr;
  const compiler::TruncationLoweringResult* trunc_fb = nullptr;
  int frac_bits = 0;
  int nr_iters = 1;
};

struct TruncChoice {
  const compiler::TruncationLoweringResult* gapars = nullptr;
  const compiler::TruncationLoweringResult* faithful = nullptr;
  int shift_bits = 0;
  bool signed_value = true;
};

inline const compiler::TruncationLoweringResult* select_trunc_bundle(
    const TruncChoice& choice,
    const compiler::RangeInterval& range,
    int frac_bits) {
  if (!choice.gapars || !choice.faithful) return choice.faithful;
  auto kind = compiler::select_trunc_kind(range, frac_bits);
  return (kind == compiler::GateKind::GapARS) ? choice.gapars : choice.faithful;
}

struct RowBroadcastTriple {
  std::span<const uint64_t> A;
  std::span<const uint64_t> B;
  std::span<const uint64_t> C;
};

class RowBroadcastTripleProvider {
 public:
  virtual ~RowBroadcastTripleProvider() = default;
  virtual RowBroadcastTriple reserve_mul(int rows, int cols) = 0;
};

// Mul exp[row,col] * inv[row] using row-broadcast triples to cut open size.
class MulRowBroadcastTask final : public detail::PhaseTask {
 public:
  MulRowBroadcastTask(std::span<const uint64_t> mat,
                      std::span<const uint64_t> vec,
                      int rows,
                      int cols,
                      std::span<const int> valid_lens,
                      std::span<uint64_t> out,
                      RowBroadcastTripleProvider* triples)
      : mat_(mat),
        vec_(vec),
        rows_(rows),
        cols_(cols),
        valid_lens_(valid_lens),
        out_(out),
        triple_provider_(triples) {
    if (mat_.size() != out_.size()) throw std::runtime_error("MulRowBroadcastTask: size mismatch");
    if (static_cast<int>(mat_.size()) != rows * cols) throw std::runtime_error("MulRowBroadcastTask: dims mismatch");
  }

  bool done() const override { return st_ == St::Done; }

  detail::Need step(PhaseResources& R) override {
    switch (st_) {
      case St::Init: {
        if (!triple_provider_) throw std::runtime_error("MulRowBroadcastTask: triple provider missing");
        triple_ = triple_provider_->reserve_mul(rows_, cols_);
        if (triple_.A.size() < mat_.size() || triple_.B.size() < static_cast<size_t>(rows_) ||
            triple_.C.size() < mat_.size()) {
          throw std::runtime_error("MulRowBroadcastTask: triple too small");
        }
        buf_de_.resize(mat_.size() + rows_);
        // D matrix
        for (int r = 0; r < rows_; ++r) {
          int L = (valid_lens_.size() == 0) ? cols_ : valid_lens_[r];
          for (int c = 0; c < cols_; ++c) {
            size_t idx = static_cast<size_t>(r * cols_ + c);
            if (c < L) {
              buf_de_[idx] = proto::sub_mod(mat_[idx], triple_.A[idx]);
            } else {
              buf_de_[idx] = 0;
            }
          }
        }
        // e vector
        size_t off_e = mat_.size();
        for (int r = 0; r < rows_; ++r) {
          buf_de_[off_e + r] = proto::sub_mod(vec_[r], triple_.B[r]);
        }
        if (!R.opens) throw std::runtime_error("MulRowBroadcastTask: no OpenCollector");
        h_open_ = R.opens->enqueue(buf_de_);
        st_ = St::WaitOpen;
        return detail::Need::Open;
      }
      case St::WaitOpen: {
        if (!R.opens->ready(h_open_)) return detail::Need::Open;
        auto opened = R.opens->view(h_open_);
        if (opened.size() != buf_de_.size()) throw std::runtime_error("MulRowBroadcastTask: opened mismatch");
        size_t off_e = mat_.size();
        // Compute Z shares.
        for (int r = 0; r < rows_; ++r) {
          int L = (valid_lens_.size() == 0) ? cols_ : valid_lens_[r];
          uint64_t e = static_cast<uint64_t>(opened[off_e + r]);
          for (int c = 0; c < cols_; ++c) {
            size_t idx = static_cast<size_t>(r * cols_ + c);
            if (c >= L) {
              out_[idx] = 0;
              continue;
            }
            uint64_t d = static_cast<uint64_t>(opened[idx]);
            uint64_t z = triple_.C[idx];
            z = proto::add_mod(z, proto::mul_mod(d, triple_.B[r]));
            z = proto::add_mod(z, proto::mul_mod(e, triple_.A[idx]));
            if (R.party == 0) {
              z = proto::add_mod(z, proto::mul_mod(d, e));
            }
            out_[idx] = z;
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
  std::span<const uint64_t> mat_;
  std::span<const uint64_t> vec_;
  int rows_;
  int cols_;
  std::span<const int> valid_lens_;
  std::span<uint64_t> out_;
  RowBroadcastTripleProvider* triple_provider_ = nullptr;

  RowBroadcastTriple triple_;
  std::vector<uint64_t> buf_de_;
  OpenHandle h_open_{};
};

// Two-trunc cubic evaluator for SiLU/nExp.
class CubicPolyTask final : public detail::PhaseTask {
 public:
  CubicPolyTask(const CubicPolyBundle& bundle,
                std::span<const uint64_t> x_qf,
                std::span<uint64_t> out_qf)
      : bundle_(bundle), x_(x_qf), out_(out_qf) {
    if (x_.size() != out_.size()) {
      throw std::runtime_error("CubicPolyTask: size mismatch");
    }
    if (!bundle_.suf) {
      throw std::runtime_error("CubicPolyTask: missing bundle pieces");
    }
  }

  bool done() const override { return st_ == St::Done; }

  detail::Need step(PhaseResources& R) override {
    if (!key_) {
      key_ = (R.party == 0) ? bundle_.key0 : bundle_.key1;
      if (!key_) throw std::runtime_error("CubicPolyTask: missing party key");
      triple_span_ = std::span<const proto::BeaverTriple64Share>(key_->triples);
    }
    switch (st_) {
      case St::OpenXhat: {
        if (!R.net_chan) throw std::runtime_error("CubicPolyTask: net channel missing");
        masked_.resize(x_.size());
        for (size_t i = 0; i < x_.size(); ++i) {
          masked_[i] = proto::add_mod(x_[i], key_->r_in_share);
        }
        if (R.opens) {
          h_open_ = R.opens->enqueue(masked_);
          st_ = St::WaitXhatOpen;
          return detail::Need::Open;
        }
        opened_.assign(masked_.size(), 0);
        for (size_t i = 0; i < masked_.size(); ++i) {
          if (R.party == 0) {
            R.net_chan->send_u64(masked_[i]);
            opened_[i] = static_cast<int64_t>(masked_[i] + R.net_chan->recv_u64());
          } else {
            opened_[i] = static_cast<int64_t>(masked_[i] + R.net_chan->recv_u64());
            R.net_chan->send_u64(masked_[i]);
          }
        }
        st_ = St::EnqueueCoeff;
        [[fallthrough]];
      }
      case St::WaitXhatOpen: {
        if (st_ == St::WaitXhatOpen) {
          if (!R.opens) throw std::runtime_error("CubicPolyTask: no OpenCollector");
          if (!R.opens->ready(h_open_)) return detail::Need::Open;
          auto v = R.opens->view(h_open_);
          opened_.assign(v.begin(), v.end());
          st_ = St::EnqueueCoeff;
        }
        [[fallthrough]];
      }
      case St::EnqueueCoeff: {
        PreparedCompositeJob job;
        job.suf = bundle_.suf;
        job.key = key_;
        job.hook = nullptr;
        job.hatx_public.resize(opened_.size());
        for (size_t i = 0; i < opened_.size(); ++i) {
          job.hatx_public[i] = static_cast<uint64_t>(opened_[i]);
        }
        coeff_token_ = R.pfss->enqueue_composite(std::move(job)).token;
        st_ = St::WaitCoeff;
        return detail::Need::Pfss;
      }
      case St::WaitCoeff: {
        if (!R.pfss->ready(PfssHandle{coeff_token_})) return detail::Need::Pfss;
        auto v = R.pfss->view(PfssHandle{coeff_token_});
        size_t elems = x_.size();
        // Some backends may already return the evaluated polynomial (r=1).
        if (v.r == 1 && v.arith_words == elems) {
          for (size_t i = 0; i < elems; ++i) out_[i] = v.arith[i];
          st_ = St::Done;
          return detail::Need::None;
        }
        if (v.r < 4) throw std::runtime_error("CubicPolyTask: coeff payload too small");
        coeff_buf_.assign(v.arith, v.arith + elems * v.r);  // AoS layout
        soa_buf_.assign(4 * elems, 0);
        for (size_t i = 0; i < elems; ++i) {
          soa_buf_[0 * elems + i] = coeff_buf_[i * v.r + 0];
          soa_buf_[1 * elems + i] = coeff_buf_[i * v.r + 1];
          soa_buf_[2 * elems + i] = coeff_buf_[i * v.r + 2];
          soa_buf_[3 * elems + i] = coeff_buf_[i * v.r + 3];
        }
        c0_ = std::span<const uint64_t>(soa_buf_.data() + 0 * elems, elems);
        c1_ = std::span<const uint64_t>(soa_buf_.data() + 1 * elems, elems);
        c2_ = std::span<const uint64_t>(soa_buf_.data() + 2 * elems, elems);
        c3_ = std::span<const uint64_t>(soa_buf_.data() + 3 * elems, elems);
        // Allocate temps.
        m1_.assign(elems, 0);
        m2_.assign(elems, 0);
        t2_.assign(elems, 0);
        m3_.assign(elems, 0);
        x_eff_.resize(elems);
        for (size_t i = 0; i < elems; ++i) {
          uint64_t hx = static_cast<uint64_t>(opened_[i]);
          x_eff_[i] = (R.party == 0) ? proto::sub_mod(hx, key_->r_in_share)
                                     : proto::sub_mod(0ull, key_->r_in_share);
        }

        // mul1: c3 * x
        auto triples1 = next_triples(x_.size());
        mul1_ = std::make_unique<MulTask>(c3_, std::span<const uint64_t>(x_eff_), std::span<uint64_t>(m1_.data(), m1_.size()), triples1);
        st_ = St::Mul1;
        return detail::Need::None;
      }
      case St::Mul1: {
        auto need = mul1_->step(R);
        if (!mul1_->done()) return need;
        for (size_t i = 0; i < x_.size(); ++i) {
          m2_[i] = proto::add_mod(m1_[i], c2_[i]);
        }
        auto triples2 = next_triples(x_.size());
        mul2_ = std::make_unique<MulTask>(std::span<const uint64_t>(m2_.data(), m2_.size()),
                                          std::span<const uint64_t>(x_eff_),
                                          std::span<uint64_t>(t2_.data(), t2_.size()),
                                          triples2);
        st_ = St::Mul2;
        return detail::Need::None;
      }
      case St::Mul2: {
        auto need = mul2_->step(R);
        if (!mul2_->done()) return need;
        for (size_t i = 0; i < x_.size(); ++i) {
          t2_[i] = proto::add_mod(t2_[i], c1_[i]);
        }
        auto triples3 = next_triples(x_.size());
        mul3_ = std::make_unique<MulTask>(std::span<const uint64_t>(t2_.data(), t2_.size()),
                                          std::span<const uint64_t>(x_eff_),
                                          std::span<uint64_t>(m3_.data(), m3_.size()),
                                          triples3);
        st_ = St::Mul3;
        return detail::Need::None;
      }
      case St::Mul3: {
        auto need = mul3_->step(R);
        if (!mul3_->done()) return need;
        for (size_t i = 0; i < x_.size(); ++i) {
          out_[i] = proto::add_mod(m3_[i], c0_[i]);
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
  enum class St {
    OpenXhat,
    WaitXhatOpen,
    EnqueueCoeff,
    WaitCoeff,
    Mul1,
    Mul2,
    Mul3,
    Done
  } st_ = St::OpenXhat;

  const CubicPolyBundle bundle_;
  const gates::CompositePartyKey* key_ = nullptr;
  std::span<const uint64_t> x_;
  std::span<uint64_t> out_;

  std::vector<uint64_t> masked_;
  std::vector<int64_t> opened_;
  OpenHandle h_open_{};
  size_t coeff_token_ = static_cast<size_t>(-1);
  std::span<const proto::BeaverTriple64Share> triple_span_;
  size_t triple_cursor_ = 0;

  // coeff storage
  std::vector<uint64_t> coeff_buf_;
  std::vector<uint64_t> soa_buf_;
  std::span<const uint64_t> c0_, c1_, c2_, c3_;

  // temps
  std::vector<uint64_t> m1_, m2_, t2_, m3_;
  std::vector<uint64_t> x_eff_;

  // sub tasks
  std::unique_ptr<MulTask> mul1_;
  std::unique_ptr<MulTask> mul2_;
  std::unique_ptr<MulTask> mul3_;

  std::span<const proto::BeaverTriple64Share> next_triples(size_t n) {
    if (triple_cursor_ + n > triple_span_.size()) {
      throw std::runtime_error("CubicPolyTask: not enough triples");
    }
    auto s = triple_span_.subspan(triple_cursor_, n);
    triple_cursor_ += n;
    return s;
  }
};

// Reciprocal via affine init + NR (1..k iters). Keeps everything on shares and
// uses PFSS/Open batching.
class RecipTask final : public detail::PhaseTask {
 public:
  RecipTask(const RecipTaskBundle& bundle,
            std::span<const uint64_t> x_qf,
            std::span<uint64_t> out_qf)
      : bundle_(bundle), x_(x_qf), out_(out_qf) {
    if (x_.size() != out_.size()) throw std::runtime_error("RecipTask: size mismatch");
    if (!bundle_.suf || !bundle_.trunc_fb) throw std::runtime_error("RecipTask: bundle missing parts");
  }

  bool done() const override { return st_ == St::Done; }

  detail::Need step(PhaseResources& R) override {
    if (!key_) {
      key_ = (R.party == 0) ? bundle_.key0 : bundle_.key1;
      if (!key_) throw std::runtime_error("RecipTask: missing party key");
      triple_span_ = std::span<const proto::BeaverTriple64Share>(key_->triples);
    }
    switch (st_) {
      case St::OpenXhat: {
        if (!R.net_chan) throw std::runtime_error("RecipTask: net channel missing");
        masked_.resize(x_.size());
        for (size_t i = 0; i < x_.size(); ++i) masked_[i] = proto::add_mod(x_[i], key_->r_in_share);
        if (R.opens) {
          h_open_ = R.opens->enqueue(masked_);
          st_ = St::WaitXhatOpen;
          return detail::Need::Open;
        }
        opened_.assign(masked_.begin(), masked_.end());
        st_ = St::EnqueueCoeff;
        [[fallthrough]];
      }
      case St::WaitXhatOpen: {
        if (st_ == St::WaitXhatOpen) {
          if (!R.opens) throw std::runtime_error("RecipTask: no OpenCollector");
          if (!R.opens->ready(h_open_)) return detail::Need::Open;
          auto v = R.opens->view(h_open_);
          opened_.assign(v.begin(), v.end());
          st_ = St::EnqueueCoeff;
        }
        [[fallthrough]];
      }
      case St::EnqueueCoeff: {
        PreparedCompositeJob job;
        job.suf = bundle_.suf;
        job.key = key_;
        job.hook = nullptr;
        job.hatx_public.resize(opened_.size());
        for (size_t i = 0; i < opened_.size(); ++i) job.hatx_public[i] = static_cast<uint64_t>(opened_[i]);
        coeff_token_ = R.pfss->enqueue_composite(std::move(job)).token;
        st_ = St::WaitCoeff;
        return detail::Need::Pfss;
      }
      case St::WaitCoeff: {
        if (!R.pfss->ready(PfssHandle{coeff_token_})) return detail::Need::Pfss;
        auto v = R.pfss->view(PfssHandle{coeff_token_});
        size_t elems = x_.size();
        // Some backends may already emit evaluated init (single arith word).
        if (v.r == 1 && v.arith_words >= elems) {
          y_.assign(v.arith, v.arith + elems);
          st_ = St::IterMul1;
          iter_ = 0;
          return detail::Need::None;
        }
        if (v.r < 2 || v.arith_words < elems * v.r) {
          throw std::runtime_error("RecipTask: coeff payload too small");
        }
        coeff_buf_.assign(v.arith, v.arith + elems * v.r);
        soa_buf_.assign(2 * elems, 0);
        for (size_t i = 0; i < elems; ++i) {
          soa_buf_[0 * elems + i] = coeff_buf_[i * v.r + 0];
          soa_buf_[1 * elems + i] = coeff_buf_[i * v.r + 1];
        }
        int shift_down = 32 - bundle_.frac_bits;
        if (shift_down < 0) shift_down = 0;
        for (size_t i = 0; i < elems; ++i) {
          uint32_t c0_q32 = static_cast<uint32_t>(soa_buf_[0 * elems + i] & 0xffffffffu);
          uint64_t half_inv_qf = (shift_down >= 32) ? 0ull
                                                    : (static_cast<uint64_t>(c0_q32) >> shift_down);
          uint64_t decoded = half_inv_qf;  // coeff already carries the needed offset in Qf
          soa_buf_[0 * elems + i] = decoded;
        }
        c0_ = std::span<const uint64_t>(soa_buf_.data() + 0 * elems, elems);
        c1_ = std::span<const uint64_t>(soa_buf_.data() + 1 * elems, elems);
        init_mul_out_.assign(elems, 0);
        auto triples = next_triples(elems);
        init_mul_ = std::make_unique<MulTask>(c1_, x_, std::span<uint64_t>(init_mul_out_.data(), init_mul_out_.size()), triples);
        st_ = St::InitMul;
        return detail::Need::None;
      }
      case St::InitMul: {
        auto need = init_mul_->step(R);
        if (!init_mul_->done()) return need;
        init_trunc_out_.assign(init_mul_out_.size(), 0);
        init_trunc_ = std::make_unique<TruncTask>(bundle_.trunc_fb,
                                                  std::span<const uint64_t>(init_mul_out_.data(), init_mul_out_.size()),
                                                  std::span<uint64_t>(init_trunc_out_.data(), init_trunc_out_.size()));
        st_ = St::InitTrunc;
        return detail::Need::None;
      }
      case St::InitTrunc: {
        auto need = init_trunc_->step(R);
        if (!init_trunc_->done()) return need;
        y_.assign(init_trunc_out_.begin(), init_trunc_out_.end());
        for (size_t i = 0; i < y_.size(); ++i) {
          y_[i] = proto::add_mod(y_[i], c0_[i]);
        }
        iter_ = 0;
        st_ = St::IterMul1;
        return detail::Need::None;
      }
      case St::IterMul1: {
        if (iter_ >= bundle_.nr_iters) {
          std::span<uint64_t> y_span(y_.data(), y_.size());
          for (size_t i = 0; i < y_span.size(); ++i) out_[i] = y_span[i];
          st_ = St::Done;
          return detail::Need::None;
        }
        auto triples = next_triples(x_.size());
        t_xy_.assign(x_.size(), 0);
        mul1_ = std::make_unique<MulTask>(
            std::span<const uint64_t>(y_.data(), y_.size()),
            x_,
            std::span<uint64_t>(t_xy_.data(), t_xy_.size()),
            triples);
        st_ = St::Mul1;
        return detail::Need::None;
      }
      case St::Mul1: {
        auto need = mul1_->step(R);
        if (!mul1_->done()) return need;
        t_xy_tr_.assign(t_xy_.size(), 0);
        trunc1_ = std::make_unique<TruncTask>(bundle_.trunc_fb,
                                              std::span<const uint64_t>(t_xy_.data(), t_xy_.size()),
                                              std::span<uint64_t>(t_xy_tr_.data(), t_xy_tr_.size()));
        st_ = St::Trunc1;
        return detail::Need::None;
      }
      case St::Trunc1: {
        auto need = trunc1_->step(R);
        if (!trunc1_->done()) return need;
        two_minus_.assign(x_.size(), 0);
        uint64_t two = (bundle_.frac_bits >= 64) ? 0ull : (uint64_t(2) << bundle_.frac_bits);
        for (size_t i = 0; i < x_.size(); ++i) {
          uint64_t const_share = (R.party == 0) ? two : 0ull;
          two_minus_[i] = proto::sub_mod(const_share, t_xy_tr_[i]);
        }
        auto triples = next_triples(x_.size());
        t_update_.assign(x_.size(), 0);
        mul2_ = std::make_unique<MulTask>(
            std::span<const uint64_t>(y_.data(), y_.size()),
            std::span<const uint64_t>(two_minus_.data(), two_minus_.size()),
            std::span<uint64_t>(t_update_.data(), t_update_.size()),
            triples);
        st_ = St::Mul2;
        return detail::Need::None;
      }
      case St::Mul2: {
        auto need = mul2_->step(R);
        if (!mul2_->done()) return need;
        t_update_tr_.assign(t_update_.size(), 0);
        trunc2_ = std::make_unique<TruncTask>(bundle_.trunc_fb,
                                              std::span<const uint64_t>(t_update_.data(), t_update_.size()),
                                              std::span<uint64_t>(t_update_tr_.data(), t_update_tr_.size()));
        st_ = St::Trunc2;
        return detail::Need::None;
      }
      case St::Trunc2: {
        auto need = trunc2_->step(R);
        if (!trunc2_->done()) return need;
        y_.assign(t_update_tr_.begin(), t_update_tr_.end());
        ++iter_;
        st_ = St::IterMul1;
        return detail::Need::None;
      }
      case St::Done:
        return detail::Need::None;
    }
    return detail::Need::None;
  }

 private:
  enum class St {
    OpenXhat,
    WaitXhatOpen,
    EnqueueCoeff,
    WaitCoeff,
    InitMul,
    InitTrunc,
    IterMul1,
    Mul1,
    Trunc1,
    Mul2,
    Trunc2,
    Done
  } st_ = St::OpenXhat;

  RecipTaskBundle bundle_;
  const gates::CompositePartyKey* key_ = nullptr;
  std::span<const uint64_t> x_;
  std::span<uint64_t> out_;

  std::vector<uint64_t> masked_;
  std::vector<int64_t> opened_;
  OpenHandle h_open_{};
  size_t coeff_token_ = static_cast<size_t>(-1);

  std::span<const proto::BeaverTriple64Share> triple_span_;
  size_t triple_cursor_ = 0;

  // coeff storage
  std::vector<uint64_t> coeff_buf_;
  std::vector<uint64_t> soa_buf_;
  std::span<const uint64_t> c0_;
  std::span<const uint64_t> c1_;

  std::vector<uint64_t> y_;
  int iter_ = 0;

  std::vector<uint64_t> init_mul_out_;
  std::vector<uint64_t> init_trunc_out_;
  std::unique_ptr<MulTask> init_mul_;
  std::unique_ptr<TruncTask> init_trunc_;

  std::vector<uint64_t> t_xy_;
  std::vector<uint64_t> t_xy_tr_;
  std::vector<uint64_t> two_minus_;
  std::vector<uint64_t> t_update_;
  std::vector<uint64_t> t_update_tr_;

  std::unique_ptr<MulTask> mul1_;
  std::unique_ptr<MulTask> mul2_;
  std::unique_ptr<TruncTask> trunc1_;
  std::unique_ptr<TruncTask> trunc2_;

  std::span<const proto::BeaverTriple64Share> next_triples(size_t n) {
    if (triple_cursor_ + n > triple_span_.size()) {
      throw std::runtime_error("RecipTask: not enough triples");
    }
    auto s = triple_span_.subspan(triple_cursor_, n);
    triple_cursor_ += n;
    return s;
  }
};

}  // namespace runtime
