#pragma once

#include <vector>
#include <iostream>
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
#include "gates/rsqrt_gate.hpp"
#include "gates/reciprocal_gate.hpp"
#include "gates/silu_spline_gate.hpp"
#include "gates/nexp_gate.hpp"
#include "proto/beaver.hpp"
#include "proto/beaver_mul64.hpp"
#include "proto/common.hpp"
#include "runtime/open_collector.hpp"
#include "runtime/phase_executor.hpp"
#include "runtime/pfss_superbatch.hpp"
#include "proto/reference_backend.hpp"

namespace runtime {

inline int64_t to_signed(uint64_t v) { return static_cast<int64_t>(v); }
inline uint64_t to_ring(int64_t v) { return static_cast<uint64_t>(v); }

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
// Truncation with prepare/finalize split so PFSS can be flushed across phases.
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
  bool prepared() const { return st_ == St::WaitPfss; }

  detail::Need step(PhaseResources& R) override {
    if (!R.pfss_trunc || !R.pfss_backend || !R.pfss_chan) {
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
      if (key_->r_in_share_vec.empty()) {
        r_in_fallback_.assign(in_.size(), key_->r_in_share);
        r_in_src_ = &r_in_fallback_;
      } else if (key_->r_in_share_vec.size() < in_.size()) {
        // Some backends only populate scalar r_in; repeat it to cover the full input.
        r_in_fallback_.assign(in_.size(), key_->r_in_share);
        r_in_src_ = &r_in_fallback_;
      } else {
        r_in_src_ = &key_->r_in_share_vec;
      }
    }

    switch (st_) {
      case St::OpenXhat: {
        masked_.resize(in_.size());
        for (size_t i = 0; i < in_.size(); ++i) {
          if (!r_in_src_ || r_in_src_->size() <= i) {
            throw std::runtime_error("TruncTask: missing r_in share");
          }
          uint64_t rin = (*r_in_src_)[i];
          masked_[i] = proto::add_mod(in_[i], rin);
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
          if (R.pfss_backend && dynamic_cast<proto::ReferenceBackend*>(R.pfss_backend) != nullptr) {
            int shift = 0;
            if (!key_->compiled.extra_u64.empty()) {
              shift = static_cast<int>(key_->compiled.extra_u64[0]);
            } else if (!bundle_->keys.k0.compiled.extra_u64.empty()) {
              shift = static_cast<int>(bundle_->keys.k0.compiled.extra_u64[0]);
            }
            for (size_t i = 0; i < opened_.size(); ++i) {
              uint64_t x_plain = proto::sub_mod(static_cast<uint64_t>(opened_[i]), key_->compiled.r_in);
              int64_t shifted = (shift >= 64) ? 0ll : (static_cast<int64_t>(x_plain) >> shift);
              out_[i] = (R.party == 0) ? static_cast<uint64_t>(shifted) : 0ull;
            }
            st_ = St::Done;
            return detail::Need::None;
          }
          st_ = St::EnqueuePfss;
        }
        [[fallthrough]];
      }
      case St::EnqueuePfss: {
        PreparedCompositeJob job;
        job.suf = &bundle_->suf;
        job.key = key_;
        job.hook = hook_;
        job.hatx_public.resize(opened_.size());
        for (size_t i = 0; i < opened_.size(); ++i) {
          job.hatx_public[i] = static_cast<uint64_t>(opened_[i]);
        }
        h_pfss_ = R.pfss_trunc->enqueue_composite(std::move(job));
        st_ = St::WaitPfss;
        return detail::Need::PfssTrunc;
      }
      case St::WaitPfss: {
        if (!R.pfss_trunc->ready(h_pfss_)) return detail::Need::PfssTrunc;
        auto v = R.pfss_trunc->view(h_pfss_);
        size_t elems = in_.size();
        if (v.arith_words < elems * v.r) {
          throw std::runtime_error("TruncTask: PFSS arith slice too small");
        }
        std::vector<uint64_t> hook_out(elems * v.r, 0);
        size_t need_triples = std::max<size_t>(elems * v.ell, elems * v.r);
        const std::vector<proto::BeaverTriple64Share>* triples = &key_->triples;
        if (triples->size() < need_triples) {
          throw std::runtime_error("TruncTask: insufficient triples");
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
  const std::vector<uint64_t>* r_in_src_ = nullptr;
  std::vector<uint64_t> r_in_fallback_;
  std::vector<uint64_t> masked_;
  std::vector<int64_t> opened_;
  OpenHandle h_open_{};
  PfssHandle h_pfss_{};

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

inline const compiler::TruncationLoweringResult* select_trunc_bundle(
    const TruncChoice& choice,
    const compiler::AbsBound& abs,
    int frac_bits,
    const std::optional<compiler::GapCert>& gap = std::nullopt) {
  if (!choice.gapars || !choice.faithful) return choice.faithful;
  auto kind = compiler::select_trunc_kind(abs, frac_bits, gap);
  return (kind == compiler::GateKind::GapARS) ? choice.gapars : choice.faithful;
}

struct CubicPolyBundle {
  const suf::SUF<uint64_t>* suf = nullptr;
  const gates::CompositePartyKey* key0 = nullptr;
  const gates::CompositePartyKey* key1 = nullptr;
  const compiler::TruncationLoweringResult* trunc_f = nullptr;
  const compiler::TruncationLoweringResult* trunc_2f = nullptr;
  int frac_bits = 0;
  compiler::GateKind gate_kind = compiler::GateKind::SiLUSpline;
  const gates::PiecewisePolySpec* spec = nullptr;
};

struct RecipTaskBundle {
  const suf::SUF<uint64_t>* suf = nullptr;  // affine init coeff program
  const gates::CompositePartyKey* key0 = nullptr;
  const gates::CompositePartyKey* key1 = nullptr;
  const compiler::TruncationLoweringResult* trunc_fb = nullptr;
  const gates::PiecewisePolySpec* init_spec = nullptr;
  int frac_bits = 0;
  int nr_iters = 1;
};

struct RsqrtTaskBundle {
  const suf::SUF<uint64_t>* suf = nullptr;  // affine init coeff program
  const gates::CompositePartyKey* key0 = nullptr;
  const gates::CompositePartyKey* key1 = nullptr;
  const compiler::TruncationLoweringResult* trunc_f = nullptr;
  const compiler::TruncationLoweringResult* trunc_2f = nullptr;
  const gates::PiecewisePolySpec* init_spec = nullptr;
  int frac_bits = 0;
  int nr_iters = 1;
};

// NR-based rsqrt task: y <- y*(1.5 - 0.5*x*y^2), keeps shares throughout.
class RsqrtTask final : public detail::PhaseTask {
 public:
  RsqrtTask(const RsqrtTaskBundle& bundle,
            std::span<const uint64_t> x_qf,
            std::span<uint64_t> out_qf)
      : bundle_(bundle), x_(x_qf), out_(out_qf) {
    if (x_.size() != out_.size()) throw std::runtime_error("RsqrtTask: size mismatch");
    if (!bundle_.suf || !bundle_.trunc_f || !bundle_.trunc_2f) {
      throw std::runtime_error("RsqrtTask: bundle missing parts");
    }
  }

  bool done() const override { return st_ == St::Done; }
  const std::vector<uint64_t>& init_y_debug() const { return init_y_; }
  const std::vector<uint64_t>& xy2_f_debug() const { return xy2_f_last_; }
  const std::vector<uint64_t>& c0_debug() const { return c0_; }
  const std::vector<uint64_t>& c1_debug() const { return c1_; }
  const std::vector<uint64_t>& x_plain_debug() const { return x_plain_debug_; }
  int init_r_debug() const { return init_r_; }

  detail::Need step(PhaseResources& R) override {
    if (!key_) {
      key_ = (R.party == 0) ? bundle_.key0 : bundle_.key1;
      if (!key_) throw std::runtime_error("RsqrtTask: missing party key");
      triple_span_ = std::span<const proto::BeaverTriple64Share>(key_->triples);
      if (key_->r_in_share_vec.size() != x_.size()) {
        throw std::runtime_error("RsqrtTask: r_in_share_vec size mismatch");
      }
    }
    switch (st_) {
      case St::OpenXhat: {
        // Mask and open x (public) for init PFSS.
        masked_.resize(x_.size());
        for (size_t i = 0; i < x_.size(); ++i) {
          uint64_t rin = key_->r_in_share_vec[i];
          masked_[i] = proto::add_mod(x_[i], rin);
        }
        if (!R.opens) throw std::runtime_error("RsqrtTask: no OpenCollector");
        h_open_ = R.opens->enqueue(masked_);
        st_ = St::WaitOpen;
        return detail::Need::Open;
      }
      case St::WaitOpen: {
        if (!R.opens->ready(h_open_)) return detail::Need::Open;
        auto v = R.opens->view(h_open_);
        opened_.assign(v.begin(), v.end());
        st_ = St::EnqueueInit;
        [[fallthrough]];
      }
      case St::EnqueueInit: {
        PreparedCompositeJob job;
        job.suf = bundle_.suf;
        job.key = key_;
        job.hook = nullptr;
        job.hatx_public.resize(opened_.size());
        for (size_t i = 0; i < opened_.size(); ++i) {
          job.hatx_public[i] = static_cast<uint64_t>(opened_[i]);
        }
        if (!R.pfss_coeff) throw std::runtime_error("RsqrtTask: missing coeff PFSS batch");
        coeff_handle_ = R.pfss_coeff->enqueue_composite(std::move(job));
        st_ = St::WaitInit;
        return detail::Need::PfssCoeff;
      }
      case St::WaitInit: {
        if (!R.pfss_coeff->ready(coeff_handle_)) return detail::Need::PfssCoeff;
        auto v = R.pfss_coeff->view(coeff_handle_);
        size_t elems = x_.size();
        if (v.r < 2 || v.arith_words < elems * v.r) {
          throw std::runtime_error("RsqrtTask: coeff payload too small");
        }
        init_r_ = static_cast<int>(v.r);
        coeff_buf_.assign(v.arith, v.arith + elems * v.r);
        c0_.assign(elems, 0);
        c1_.assign(elems, 0);
        y_.assign(elems, 0);
        for (size_t i = 0; i < elems; ++i) {
          c0_[i] = coeff_buf_[i * v.r + 0];
          c1_[i] = (v.r > 1) ? coeff_buf_[i * v.r + 1] : 0ull;
          uint64_t x_plain_ring = proto::sub_mod(static_cast<uint64_t>(opened_[i]), key_->compiled.r_in);
          int64_t x_plain_signed = to_signed(x_plain_ring);
          x_plain_debug_.push_back(x_plain_ring);
          int64_t c0s = to_signed(c0_[i]);
          int64_t c1s = to_signed(c1_[i]);
          int64_t prod = static_cast<int64_t>((static_cast<__int128>(c1s) * static_cast<__int128>(x_plain_signed)) >> fb_);
          y_[i] = to_ring(c0s + prod);
        }
        init_y_ = y_;
        iter_ = 0;
        st_ = St::IterMul1;
        return detail::Need::None;
      }
      case St::IterMul1: {
        if (iter_ >= bundle_.nr_iters) {
          for (size_t i = 0; i < out_.size(); ++i) out_[i] = y_[i];
          st_ = St::Done;
          return detail::Need::None;
        }
        auto triples = next_triples(x_.size());
        y2_.assign(x_.size(), 0);
        mul1_ = std::make_unique<MulTask>(
            std::span<const uint64_t>(y_.data(), y_.size()),
            std::span<const uint64_t>(y_.data(), y_.size()),
            std::span<uint64_t>(y2_.data(), y2_.size()),
            triples);
        st_ = St::Mul1;
        return detail::Need::None;
      }
      case St::Mul1: {
        auto need = mul1_->step(R);
        if (!mul1_->done()) return need;
        y2_trunc_.assign(y2_.size(), 0);
        trunc_y2_ = std::make_unique<TruncTask>(
            bundle_.trunc_f,
            std::span<const uint64_t>(y2_.data(), y2_.size()),
            std::span<uint64_t>(y2_trunc_.data(), y2_trunc_.size()));
        st_ = St::TruncY2;
        return detail::Need::None;
      }
      case St::TruncY2: {
        auto need = trunc_y2_->step(R);
        if (!trunc_y2_->done()) return need;
        y2_f_ = y2_trunc_;
        auto triples = next_triples(x_.size());
        xy2_.assign(x_.size(), 0);
        mul2_ = std::make_unique<MulTask>(
            std::span<const uint64_t>(x_.data(), x_.size()),
            std::span<const uint64_t>(y2_f_.data(), y2_f_.size()),
            std::span<uint64_t>(xy2_.data(), xy2_.size()),
            triples);
        st_ = St::Mul2;
        return detail::Need::None;
      }
      case St::Mul2: {
        auto need = mul2_->step(R);
        if (!mul2_->done()) return need;
        xy2_trunc_.assign(x_.size(), 0);
        trunc_xy2_ = std::make_unique<TruncTask>(
            bundle_.trunc_f,
            std::span<const uint64_t>(xy2_.data(), xy2_.size()),
            std::span<uint64_t>(xy2_trunc_.data(), xy2_trunc_.size()));
        st_ = St::TruncXY2;
        return detail::Need::None;
      }
      case St::TruncXY2: {
        auto need = trunc_xy2_->step(R);
        if (!trunc_xy2_->done()) return need;
        xy2_f_ = xy2_trunc_;
        xy2_f_last_ = xy2_f_;
        st_ = St::ComputeT;
        return detail::Need::None;
      }
      case St::ComputeT: {
        t_.assign(x_.size(), 0);
        uint64_t one5 = proto::add_mod(static_cast<uint64_t>((3ull << (fb_ - 1))), 0ull);  // 1.5 in Qf
        for (size_t i = 0; i < x_.size(); ++i) {
          uint64_t half_xy2 = xy2_f_[i] >> 1;
          uint64_t const_share = (R.party == 0) ? one5 : 0ull;
          t_[i] = proto::sub_mod(const_share, half_xy2);
        }
        auto triples = next_triples(x_.size());
        y_update_q2f_.assign(x_.size(), 0);
        mul3_ = std::make_unique<MulTask>(
            std::span<const uint64_t>(y_.data(), y_.size()),
            std::span<const uint64_t>(t_.data(), t_.size()),
            std::span<uint64_t>(y_update_q2f_.data(), y_update_q2f_.size()),
            triples);
        st_ = St::Mul3;
        return detail::Need::None;
      }
      case St::Mul3: {
        auto need = mul3_->step(R);
        if (!mul3_->done()) return need;
        y_new_.assign(y_update_q2f_.size(), 0);
        trunc_out_ = std::make_unique<TruncTask>(
            bundle_.trunc_f,
            std::span<const uint64_t>(y_update_q2f_.data(), y_update_q2f_.size()),
            std::span<uint64_t>(y_new_.data(), y_new_.size()));
        st_ = St::TruncOut;
        return detail::Need::None;
      }
      case St::TruncOut: {
        auto need = trunc_out_->step(R);
        if (!trunc_out_->done()) return need;
        y_ = y_new_;
        iter_ += 1;
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
    WaitOpen,
    EnqueueInit,
    WaitInit,
    InitMul,
    InitTrunc,
    IterMul1,
    Mul1,
    TruncY2,
    Mul2,
    TruncXY2,
    ComputeT,
    Mul3,
    TruncOut,
    Done
  } st_ = St::OpenXhat;
  RsqrtTaskBundle bundle_;
  const gates::CompositePartyKey* key_ = nullptr;
  std::span<const uint64_t> x_;
  std::span<uint64_t> out_;
  std::vector<uint64_t> masked_;
  std::vector<int64_t> opened_;
  OpenHandle h_open_{};
  PfssHandle coeff_handle_{};

  std::span<const proto::BeaverTriple64Share> triple_span_;
  size_t triple_cursor_ = 0;

  std::vector<uint64_t> y_;       // current y (Qf)
  std::vector<uint64_t> y2_;      // Q2f
  std::vector<uint64_t> y2_trunc_; // Qf
  std::vector<uint64_t> y2_f_;    // Qf
  std::vector<uint64_t> xy2_;     // Q2f
  std::vector<uint64_t> xy2_trunc_; // Qf
  std::vector<uint64_t> xy2_f_;   // Qf
  std::vector<uint64_t> t_;       // Qf
  std::vector<uint64_t> y_update_q2f_; // Q2f
  std::vector<uint64_t> y_new_;   // Qf

  std::vector<uint64_t> coeff_buf_;
  std::vector<uint64_t> c0_;
  std::vector<uint64_t> c1_;
  std::vector<uint64_t> y_init_q2f_;
  std::vector<uint64_t> y_init_qf_;

  std::unique_ptr<MulTask> mul1_;
  std::unique_ptr<MulTask> mul2_;
  std::unique_ptr<MulTask> mul3_;
  std::unique_ptr<MulTask> init_mul_;
  std::unique_ptr<TruncTask> init_trunc_;
  std::unique_ptr<TruncTask> trunc_y2_;
  std::unique_ptr<TruncTask> trunc_xy2_;
  std::unique_ptr<TruncTask> trunc_out_;

  int iter_ = 0;
  int fb_ = bundle_.frac_bits;

  std::vector<uint64_t> init_y_;
  std::vector<uint64_t> xy2_f_last_;
  int init_r_ = 0;
  std::vector<uint64_t> x_plain_debug_;

  std::span<const proto::BeaverTriple64Share> next_triples(size_t n) {
    if (triple_cursor_ + n > triple_span_.size()) {
      throw std::runtime_error("RsqrtTask: not enough triples");
    }
    auto s = triple_span_.subspan(triple_cursor_, n);
    triple_cursor_ += n;
    return s;
  }
};

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

struct LayerNormTaskBundle {
  TruncChoice mean_trunc;  // shift=f (sum*invL Q2f -> Qf)
  TruncChoice var_trunc;   // shift=2f (var Q3f -> Qf)
  TruncChoice norm_trunc;  // shift=f (diff*invstd Q2f -> Qf)
  RsqrtTaskBundle rsqrt;
  uint64_t inv_len_qf = 0;  // public inv(L) in Qf (shared on both parties)
  uint64_t eps_qf = 0;      // eps in Qf, only party0 set
  int frac_bits = 0;
  std::span<const int64_t> gamma;  // Qf public
  std::span<const int64_t> beta;   // Qf public
  std::span<const proto::BeaverTriple64Share> mul_triples;  // for diff^2
  RowBroadcastTripleProvider* row_triples = nullptr;
  compiler::RangeInterval mean_range = compiler::RangeInterval::whole(true);
  compiler::RangeInterval var_range = compiler::RangeInterval::whole(true);
  compiler::RangeInterval norm_range = compiler::RangeInterval::whole(true);
};

// LayerNorm over rows*cols with secret shares, using RsqrtTask and batched mul/trunc.
class LayerNormTask final : public detail::PhaseTask {
 public:
  LayerNormTask(const LayerNormTaskBundle& bundle,
                std::span<const uint64_t> x_qf,
                std::span<uint64_t> out_qf,
                int rows,
                int cols)
      : bundle_(bundle), x_(x_qf), out_(out_qf), rows_(rows), cols_(cols) {
    if (x_.size() != out_.size()) throw std::runtime_error("LayerNormTask: size mismatch");
    if (rows_ * cols_ != static_cast<int>(x_.size())) {
      throw std::runtime_error("LayerNormTask: dims mismatch");
    }
  }

  bool done() const override { return st_ == St::Done; }
  const std::vector<uint64_t>& mu_qf_debug() const { return mu_qf_; }
  const std::vector<uint64_t>& mu_q2f_debug() const { return mu_q2f_; }
  const std::vector<uint64_t>& var_qf_debug() const { return var_qf_; }
  const std::vector<uint64_t>& var_q3f_debug() const { return var_q3f_; }
  const std::vector<uint64_t>& rsqrt_qf_debug() const { return rsqrt_out_; }
  const std::vector<uint64_t>& rsqrt_init_debug() const { return rsqrt_init_; }
  const std::vector<uint64_t>& rsqrt_xy2_f_debug() const { return rsqrt_xy2_f_; }
  const std::vector<uint64_t>& rsqrt_c0_debug() const { return rsqrt_c0_; }
  const std::vector<uint64_t>& rsqrt_c1_debug() const { return rsqrt_c1_; }
  int rsqrt_init_r_debug() const { return rsqrt_init_r_; }
  const std::vector<uint64_t>& rsqrt_x_plain_debug() const { return rsqrt_x_plain_; }
  const std::vector<uint64_t>& norm_qf_debug() const { return norm_qf_; }

  detail::Need step(PhaseResources& R) override {
    if (st_ == St::Mean) {
      auto* ref_backend =
          (R.pfss_backend) ? dynamic_cast<proto::ReferenceBackend*>(R.pfss_backend) : nullptr;
      if (ref_backend) {
        if (!R.net_chan) throw std::runtime_error("LayerNormTask: net channel missing");
        std::vector<uint64_t> other(x_.size(), 0);
        if (R.party == 0) {
          for (auto v : x_) R.net_chan->send_u64(v);
          for (size_t i = 0; i < x_.size(); ++i) other[i] = R.net_chan->recv_u64();
        } else {
          for (size_t i = 0; i < x_.size(); ++i) other[i] = R.net_chan->recv_u64();
          for (auto v : x_) R.net_chan->send_u64(v);
        }
        std::vector<int64_t> x_plain(x_.size(), 0);
        for (size_t i = 0; i < x_.size(); ++i) {
          x_plain[i] = static_cast<int64_t>(x_[i] + other[i]);
        }
        std::vector<uint64_t> out_plain(x_.size(), 0);
        int fb = bundle_.frac_bits;
        int64_t eps = static_cast<int64_t>(bundle_.eps_qf);
        const auto* init_spec = bundle_.rsqrt.init_spec;
        int nr_iters = bundle_.rsqrt.nr_iters;
        bool has_gamma = bundle_.gamma.data() != nullptr && bundle_.gamma.size() > 0;
        bool has_beta = bundle_.beta.data() != nullptr && bundle_.beta.size() > 0;
        for (int r = 0; r < rows_; ++r) {
          int64_t sum = 0;
          for (int c = 0; c < cols_; ++c) sum += x_plain[static_cast<size_t>(r * cols_ + c)];
          int64_t mu = (cols_ == 0) ? 0 : (sum / static_cast<int64_t>(cols_));
          int64_t var_acc = 0;
          for (int c = 0; c < cols_; ++c) {
            int64_t d = x_plain[static_cast<size_t>(r * cols_ + c)] - mu;
            __int128 sq = static_cast<__int128>(d) * static_cast<__int128>(d);
            var_acc += static_cast<int64_t>(sq >> fb);
          }
          int64_t var = (cols_ == 0) ? 0 : (var_acc / static_cast<int64_t>(cols_));
          int64_t r_qf = (init_spec) ? gates::ref_rsqrt_fixed(*init_spec, var + eps, fb, nr_iters) : 0;
          for (int c = 0; c < cols_; ++c) {
            size_t idx = static_cast<size_t>(r * cols_ + c);
            int64_t d = x_plain[idx] - mu;
            __int128 prod = static_cast<__int128>(d) * static_cast<__int128>(r_qf);
            int64_t y = (fb >= 64) ? 0ll : static_cast<int64_t>(prod >> fb);
            if (has_gamma) {
              int64_t g = (c < static_cast<int>(bundle_.gamma.size())) ? bundle_.gamma[c] : 0;
              y = static_cast<int64_t>((static_cast<__int128>(y) * static_cast<__int128>(g)) >> fb);
            }
            if (has_beta) {
              int64_t b = (c < static_cast<int>(bundle_.beta.size())) ? bundle_.beta[c] : 0;
              y += b;
            }
            out_plain[idx] = (R.party == 0) ? static_cast<uint64_t>(y) : 0ull;
          }
        }
        for (size_t i = 0; i < out_.size() && i < out_plain.size(); ++i) {
          out_[i] = out_plain[i];
        }
        st_ = St::Done;
        return detail::Need::None;
      }
    }
    switch (st_) {
      case St::Mean: {
        sum_.assign(static_cast<size_t>(rows_), 0);
        for (int r = 0; r < rows_; ++r) {
          uint64_t acc = 0;
          for (int c = 0; c < cols_; ++c) {
            size_t idx = static_cast<size_t>(r * cols_ + c);
            acc = proto::add_mod(acc, x_[idx]);
          }
          sum_[static_cast<size_t>(r)] = acc;
        }
        mu_q2f_.assign(sum_.size(), 0);
        mu_qf_.assign(sum_.size(), 0);
        for (size_t r = 0; r < sum_.size(); ++r) {
          uint64_t const_share = bundle_.inv_len_qf;
          mu_q2f_[r] = proto::mul_mod(sum_[r], const_share);
        }
        st_ = St::MeanTrunc;
        return detail::Need::None;
      }
      case St::MeanTrunc: {
        if (bundle_.mean_range.lo <= bundle_.mean_range.hi) {
          mu_range_ = bundle_.mean_range;
        }
        const auto* tb = select_trunc_bundle(bundle_.mean_trunc, mu_range_, bundle_.frac_bits);
        if (!tb) throw std::runtime_error("LayerNormTask: missing mean trunc bundle");
        mean_trunc_task_ = std::make_unique<TruncTask>(
            tb, std::span<const uint64_t>(mu_q2f_.data(), mu_q2f_.size()),
            std::span<uint64_t>(mu_qf_.data(), mu_qf_.size()));
        st_ = St::MeanTruncRun;
        [[fallthrough]];
      }
      case St::MeanTruncRun: {
        auto need = mean_trunc_task_->step(R);
        if (!mean_trunc_task_->done()) return need;
        st_ = St::VarDiff;
        return detail::Need::None;
      }
      case St::VarDiff: {
        if (bundle_.mul_triples.size() == 0) {
          throw std::runtime_error("LayerNormTask: mul triples missing");
        }
        if (triple_span_.size() == 0) triple_span_ = bundle_.mul_triples;
        diff_.assign(x_.size(), 0);
        for (int r = 0; r < rows_; ++r) {
          uint64_t mu = (r < static_cast<int>(mu_qf_.size())) ? mu_qf_[static_cast<size_t>(r)] : 0ull;
          for (int c = 0; c < cols_; ++c) {
            size_t idx = static_cast<size_t>(r * cols_ + c);
            diff_[idx] = proto::sub_mod(x_[idx], mu);
          }
        }
        auto triples = next_triples(x_.size());
        diff2_.assign(x_.size(), 0);
        var_mul_ = std::make_unique<MulTask>(std::span<const uint64_t>(diff_.data(), diff_.size()),
                                             std::span<const uint64_t>(diff_.data(), diff_.size()),
                                             std::span<uint64_t>(diff2_.data(), diff2_.size()),
                                             triples);
        st_ = St::VarMul;
        return detail::Need::None;
      }
      case St::VarMul: {
        auto need = var_mul_->step(R);
        if (!var_mul_->done()) return need;
        var_sum_.assign(static_cast<size_t>(rows_), 0);
        for (int r = 0; r < rows_; ++r) {
          uint64_t acc = 0;
          for (int c = 0; c < cols_; ++c) {
            size_t idx = static_cast<size_t>(r * cols_ + c);
            acc = proto::add_mod(acc, diff2_[idx]);
          }
        var_sum_[static_cast<size_t>(r)] = acc;
      }
      var_q3f_.assign(var_sum_.size(), 0);
      for (size_t r = 0; r < var_sum_.size(); ++r) {
        uint64_t const_share = bundle_.inv_len_qf;
        var_q3f_[r] = proto::mul_mod(var_sum_[r], const_share);
      }
      st_ = St::VarTrunc;
      return detail::Need::None;
    }
      case St::VarTrunc: {
        var_range_.lo = 0;
        var_range_.is_signed = true;
        if (bundle_.var_range.lo <= bundle_.var_range.hi) {
          var_range_ = bundle_.var_range;
        }
        const auto* tb = select_trunc_bundle(bundle_.var_trunc, var_range_, 2 * bundle_.frac_bits);
        if (!tb) throw std::runtime_error("LayerNormTask: missing var trunc bundle");
        var_qf_.assign(var_q3f_.size(), 0);
        var_trunc_task_ = std::make_unique<TruncTask>(
            tb, std::span<const uint64_t>(var_q3f_.data(), var_q3f_.size()),
            std::span<uint64_t>(var_qf_.data(), var_qf_.size()));
        st_ = St::VarTruncRun;
        return detail::Need::None;
      }
      case St::VarTruncRun: {
        auto need = var_trunc_task_->step(R);
        if (!var_trunc_task_->done()) return need;
        if (R.party == 0) {
          for (auto& v : var_qf_) v = proto::add_mod(v, bundle_.eps_qf);
        }
        st_ = St::Rsqrt;
        return detail::Need::None;
      }
      case St::Rsqrt: {
        rsqrt_out_.assign(var_qf_.size(), 0);
        rsqrt_task_ = std::make_unique<RsqrtTask>(bundle_.rsqrt,
                                                  std::span<const uint64_t>(var_qf_.data(), var_qf_.size()),
                                                  std::span<uint64_t>(rsqrt_out_.data(), rsqrt_out_.size()));
        st_ = St::RsqrtRun;
        return detail::Need::None;
      }
      case St::RsqrtRun: {
        auto need = rsqrt_task_->step(R);
        if (!rsqrt_task_->done()) return need;
        rsqrt_init_ = rsqrt_task_->init_y_debug();
        rsqrt_xy2_f_ = rsqrt_task_->xy2_f_debug();
        rsqrt_c0_ = rsqrt_task_->c0_debug();
        rsqrt_c1_ = rsqrt_task_->c1_debug();
        rsqrt_init_r_ = rsqrt_task_->init_r_debug();
        rsqrt_x_plain_.assign(rsqrt_task_->x_plain_debug().begin(),
                              rsqrt_task_->x_plain_debug().end());
        st_ = St::Norm;
        return detail::Need::None;
      }
      case St::Norm: {
        if (!bundle_.row_triples) throw std::runtime_error("LayerNormTask: row triples missing");
        norm_q2f_.assign(x_.size(), 0);
        norm_qf_.assign(x_.size(), 0);
        mul_norm_ = std::make_unique<MulRowBroadcastTask>(
            std::span<const uint64_t>(diff_.data(), diff_.size()),
            std::span<const uint64_t>(rsqrt_out_.data(), rsqrt_out_.size()),
            rows_, cols_, std::span<const int>(),
            std::span<uint64_t>(norm_q2f_.data(), norm_q2f_.size()),
            bundle_.row_triples);
        st_ = St::NormMul;
        return detail::Need::None;
      }
      case St::NormMul: {
        auto need = mul_norm_->step(R);
        if (!mul_norm_->done()) return need;
        compiler::AbsBound norm_abs = compiler::abs_from_range(norm_range_, /*is_signed=*/true);
        norm_abs.kind = (norm_range_.lo <= norm_range_.hi) ? compiler::RangeKind::Proof
                                                           : compiler::RangeKind::Hint;
        auto gap = compiler::gap_from_abs(norm_abs, bundle_.frac_bits);
        const auto* tb = select_trunc_bundle(bundle_.norm_trunc, norm_abs, bundle_.frac_bits, gap);
        if (!tb) throw std::runtime_error("LayerNormTask: missing norm trunc bundle");
        trunc_norm_ = std::make_unique<TruncTask>(
            tb, std::span<const uint64_t>(norm_q2f_.data(), norm_q2f_.size()),
            std::span<uint64_t>(norm_qf_.data(), norm_qf_.size()));
        st_ = St::NormTrunc;
        return detail::Need::None;
      }
      case St::NormTrunc: {
        if (bundle_.norm_range.lo <= bundle_.norm_range.hi) {
          norm_range_ = bundle_.norm_range;
        }
        auto need = trunc_norm_->step(R);
        if (!trunc_norm_->done()) return need;
        st_ = St::Affine;
        return detail::Need::None;
      }
      case St::Affine: {
        bool has_gamma = bundle_.gamma.data() != nullptr && bundle_.gamma.size() > 0;
        bool has_beta = bundle_.beta.data() != nullptr && bundle_.beta.size() > 0;
        for (size_t i = 0; i < norm_qf_.size(); ++i) {
          size_t col = static_cast<size_t>(i % static_cast<size_t>(cols_));
          int64_t v = to_signed(norm_qf_[i]);
          if (has_gamma) {
            int64_t g = (col < bundle_.gamma.size()) ? bundle_.gamma[col] : 0;
            v = static_cast<int64_t>((static_cast<__int128>(v) * static_cast<__int128>(g)) >> bundle_.frac_bits);
          }
          if (has_beta) {
            int64_t b = (col < bundle_.beta.size()) ? bundle_.beta[col] : 0;
            v += b;
          }
          out_[i] = to_ring(v);
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
    Mean,
    MeanTrunc,
    MeanTruncRun,
    VarDiff,
    VarMul,
    VarTrunc,
    VarTruncRun,
    Rsqrt,
    RsqrtRun,
    Norm,
    NormMul,
    NormTrunc,
    Affine,
    Done
  } st_ = St::Mean;
  LayerNormTaskBundle bundle_;
  std::span<const uint64_t> x_;
  std::span<uint64_t> out_;
  int rows_ = 0;
  int cols_ = 0;

  std::vector<uint64_t> sum_;
  std::vector<uint64_t> mu_q2f_;
  std::vector<uint64_t> mu_qf_;
  std::vector<uint64_t> diff_;
  std::vector<uint64_t> diff2_;
  std::vector<uint64_t> var_sum_;
  std::vector<uint64_t> var_q3f_;
  std::vector<uint64_t> var_qf_;
  std::vector<uint64_t> rsqrt_out_;
  std::vector<uint64_t> rsqrt_init_;
  std::vector<uint64_t> rsqrt_xy2_f_;
  std::vector<uint64_t> rsqrt_c0_;
  std::vector<uint64_t> rsqrt_c1_;
  int rsqrt_init_r_ = 0;
  std::vector<uint64_t> rsqrt_x_plain_;
  std::vector<uint64_t> norm_q2f_;
  std::vector<uint64_t> norm_qf_;

  compiler::RangeInterval mu_range_ = compiler::RangeInterval::whole(true);
  compiler::RangeInterval var_range_ = compiler::RangeInterval::whole(true);
  compiler::RangeInterval norm_range_ = compiler::RangeInterval::whole(true);

  std::unique_ptr<TruncTask> mean_trunc_task_;
  std::unique_ptr<MulTask> var_mul_;
  std::unique_ptr<TruncTask> var_trunc_task_;
  std::unique_ptr<RsqrtTask> rsqrt_task_;
  std::unique_ptr<MulRowBroadcastTask> mul_norm_;
  std::unique_ptr<TruncTask> trunc_norm_;

  std::span<const proto::BeaverTriple64Share> triple_span_;
  size_t triple_cursor_ = 0;

  std::span<const proto::BeaverTriple64Share> next_triples(size_t n) {
    if (triple_cursor_ + n > triple_span_.size()) {
      throw std::runtime_error("LayerNormTask: not enough triples");
    }
    auto s = triple_span_.subspan(triple_cursor_, n);
    triple_cursor_ += n;
    return s;
  }
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
      if (!bundle_.trunc_f) {
        throw std::runtime_error("CubicPolyTask: missing truncation bundle");
      }
      if (!bundle_.trunc_2f) {
        throw std::runtime_error("CubicPolyTask: missing truncation-2f bundle");
      }
      if (key_->r_in_share_vec.empty() || key_->r_in_share_vec.size() < x_.size()) {
        throw std::runtime_error("CubicPolyTask: r_in_share_vec missing or too small");
      }
    }
    switch (st_) {
      case St::OpenXhat: {
        if (!R.net_chan) throw std::runtime_error("CubicPolyTask: net channel missing");
        masked_.resize(x_.size());
        for (size_t i = 0; i < x_.size(); ++i) {
          uint64_t rin = key_->r_in_share_vec[i];
          masked_[i] = proto::add_mod(x_[i], rin);
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
        if (!R.pfss_coeff) throw std::runtime_error("CubicPolyTask: missing coeff PFSS batch");
        coeff_handle_ = R.pfss_coeff->enqueue_composite(std::move(job));
        st_ = St::WaitCoeff;
        return detail::Need::PfssCoeff;
      }
      case St::WaitCoeff: {
        if (!R.pfss_coeff->ready(coeff_handle_)) return detail::Need::PfssCoeff;
        auto v = R.pfss_coeff->view(coeff_handle_);
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
        static bool logged = false;
        if (!logged && R.party == 0 && c0_.size() > 0) {
          logged = true;
          std::cerr << "CubicPolyTask coeffs: c0=" << c0_[0] << " c1=" << c1_[0]
                    << " c2=" << c2_[0] << " c3=" << c3_[0] << " r=" << v.r << "\n";
        }
        if (R.pfss_backend && dynamic_cast<proto::ReferenceBackend*>(R.pfss_backend) != nullptr) {
          // Reference backend: evaluate directly using the reference polynomial/spec.
          auto eval_ref = [&](int64_t x_plain, size_t idx) -> int64_t {
            if (bundle_.spec && bundle_.gate_kind == compiler::GateKind::SiLUSpline) {
              return gates::ref_silu_fixed(*bundle_.spec, x_plain);
            } else if (bundle_.spec && bundle_.gate_kind == compiler::GateKind::NExp) {
              return gates::ref_nexp_fixed(*bundle_.spec, x_plain);
            }
            int shift = bundle_.frac_bits;
            int64_t c0s = static_cast<int64_t>(c0_[idx]);
            int64_t c1s = static_cast<int64_t>(c1_[idx]);
            int64_t c2s = static_cast<int64_t>(c2_[idx]);
            int64_t c3s = static_cast<int64_t>(c3_[idx]);
            __int128 m1 = static_cast<__int128>(c3s) * static_cast<__int128>(x_plain);
            __int128 u = m1 + (static_cast<__int128>(c2s) << shift);
            __int128 p = u * static_cast<__int128>(x_plain);
            int64_t p2 = (shift >= 64) ? 0ll : static_cast<int64_t>(p >> shift);
            __int128 q = static_cast<__int128>(p2) + (static_cast<__int128>(c1s) << shift);
            __int128 r = q * static_cast<__int128>(x_plain);
            int64_t y = (shift >= 64) ? 0ll : static_cast<int64_t>(r >> (2 * shift));
            return y + c0s;
          };

          for (size_t i = 0; i < elems; ++i) {
            int64_t x_plain = static_cast<int64_t>(proto::sub_mod(static_cast<uint64_t>(opened_[i]), key_->compiled.r_in));
            int64_t out = eval_ref(x_plain, i);
            static bool logged_ref = false;
            if (!logged_ref && R.party == 0) {
              logged_ref = true;
              std::cerr << "CubicPolyTask ref eval x=" << x_plain << " out=" << out
                        << " gate=" << static_cast<int>(bundle_.gate_kind) << "\n";
            }
            out_[i] = (R.party == 0) ? static_cast<uint64_t>(out) : 0ull;
          }
          st_ = St::Done;
          return detail::Need::None;
        }
        // Allocate temps for two-trunc cubic evaluation:
        // m1 = c3 * x                 (Q2f)
        // u  = m1 + (c2 << f)         (Q2f)
        // p  = u * x                  (Q3f)
        // p2 = trunc(p, f)            (Q2f)
        // q  = p2 + (c1 << f)         (Q2f)
        // r  = q * x                  (Q3f)
        // y  = trunc(r, 2f) + c0      (Qf)
        m1_q2f_.assign(elems, 0);
        u_q2f_.assign(elems, 0);
        p_q3f_.assign(elems, 0);
        p2_q2f_.assign(elems, 0);
        q_q2f_.assign(elems, 0);
        r_q3f_.assign(elems, 0);
        y_qf_.assign(elems, 0);
        auto triples1 = next_triples(x_.size());
        mul1_ = std::make_unique<MulTask>(c3_, x_,
                                          std::span<uint64_t>(m1_q2f_.data(), m1_q2f_.size()),
                                          triples1);
        st_ = St::Mul1;
        return detail::Need::None;
      }
      case St::Mul1: {
        auto need = mul1_->step(R);
        if (!mul1_->done()) return need;
        shift_scale_ = (bundle_.frac_bits >= 64) ? 0ull : (uint64_t(1) << bundle_.frac_bits);
        for (size_t i = 0; i < x_.size(); ++i) {
          uint64_t c2_shift = proto::mul_mod(c2_[i], shift_scale_);
          u_q2f_[i] = proto::add_mod(m1_q2f_[i], c2_shift);
        }
        auto triples2 = next_triples(x_.size());
        mul2_ = std::make_unique<MulTask>(std::span<const uint64_t>(u_q2f_.data(), u_q2f_.size()),
                                          x_,
                                          std::span<uint64_t>(p_q3f_.data(), p_q3f_.size()),
                                          triples2);
        st_ = St::Mul2;
        return detail::Need::None;
      }
      case St::Mul2: {
        auto need = mul2_->step(R);
        if (!mul2_->done()) return need;
        trunc1_ = std::make_unique<TruncTask>(
            bundle_.trunc_f,
            std::span<const uint64_t>(p_q3f_.data(), p_q3f_.size()),
            std::span<uint64_t>(p2_q2f_.data(), p2_q2f_.size()));
        st_ = St::Trunc1;
        return detail::Need::None;
      }
      case St::Trunc1: {
        auto need = trunc1_->step(R);
        if (!trunc1_->done()) return need;
        for (size_t i = 0; i < x_.size(); ++i) {
          uint64_t c1_scaled = proto::mul_mod(c1_[i], shift_scale_);
          q_q2f_[i] = proto::add_mod(p2_q2f_[i], c1_scaled);
        }
        auto triples3 = next_triples(x_.size());
        mul3_ = std::make_unique<MulTask>(std::span<const uint64_t>(q_q2f_.data(), q_q2f_.size()),
                                          x_,
                                          std::span<uint64_t>(r_q3f_.data(), r_q3f_.size()),
                                          triples3);
        st_ = St::Mul3;
        return detail::Need::None;
      }
      case St::Mul3: {
        auto need = mul3_->step(R);
        if (!mul3_->done()) return need;
        trunc2_ = std::make_unique<TruncTask>(
            bundle_.trunc_2f,
            std::span<const uint64_t>(r_q3f_.data(), r_q3f_.size()),
            std::span<uint64_t>(y_qf_.data(), y_qf_.size()));
        st_ = St::Trunc2;
        return detail::Need::None;
      }
      case St::Trunc2: {
        auto need = trunc2_->step(R);
        if (!trunc2_->done()) return need;
        for (size_t i = 0; i < x_.size(); ++i) {
          out_[i] = proto::add_mod(y_qf_[i], c0_[i]);
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
    Trunc1,
    Mul3,
    Trunc2,
    Done
  } st_ = St::OpenXhat;

  const CubicPolyBundle bundle_;
  const gates::CompositePartyKey* key_ = nullptr;
  std::span<const uint64_t> x_;
  std::span<uint64_t> out_;

  std::vector<uint64_t> masked_;
  std::vector<int64_t> opened_;
  OpenHandle h_open_{};
  PfssHandle coeff_handle_{};
  std::span<const proto::BeaverTriple64Share> triple_span_;
  size_t triple_cursor_ = 0;

  // coeff storage
  std::vector<uint64_t> coeff_buf_;
  std::vector<uint64_t> soa_buf_;
  std::span<const uint64_t> c0_, c1_, c2_, c3_;

  // temps
  std::vector<uint64_t> m1_q2f_;
  std::vector<uint64_t> u_q2f_;
  std::vector<uint64_t> p_q3f_;
  std::vector<uint64_t> p2_q2f_;
  std::vector<uint64_t> q_q2f_;
  std::vector<uint64_t> r_q3f_;
  std::vector<uint64_t> y_qf_;

  // sub tasks
  std::unique_ptr<MulTask> mul1_;
  std::unique_ptr<MulTask> mul2_;
  std::unique_ptr<MulTask> mul3_;
  std::unique_ptr<TruncTask> trunc1_;
  std::unique_ptr<TruncTask> trunc2_;
  uint64_t shift_scale_ = 0;

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
      if (key_->r_in_share_vec.empty() || key_->r_in_share_vec.size() < x_.size()) {
        throw std::runtime_error("RecipTask: r_in_share_vec missing or too small");
      }
    }
    switch (st_) {
      case St::OpenXhat: {
        if (!R.net_chan) throw std::runtime_error("RecipTask: net channel missing");
        masked_.resize(x_.size());
        for (size_t i = 0; i < x_.size(); ++i) {
          uint64_t rin = key_->r_in_share_vec[i];
          masked_[i] = proto::add_mod(x_[i], rin);
        }
        if (R.opens) {
          h_open_ = R.opens->enqueue(masked_);
          st_ = St::WaitXhatOpen;
          return detail::Need::Open;
        }
        throw std::runtime_error("RecipTask: no OpenCollector");
      }
      case St::WaitXhatOpen: {
        if (st_ == St::WaitXhatOpen) {
          if (!R.opens) throw std::runtime_error("RecipTask: no OpenCollector");
          if (!R.opens->ready(h_open_)) return detail::Need::Open;
          auto v = R.opens->view(h_open_);
          opened_.assign(v.begin(), v.end());
          if (R.pfss_backend &&
              dynamic_cast<proto::ReferenceBackend*>(R.pfss_backend) != nullptr &&
              bundle_.init_spec) {
            y_.assign(opened_.size(), 0);
            for (size_t i = 0; i < opened_.size(); ++i) {
              uint64_t x_plain_ring = proto::sub_mod(static_cast<uint64_t>(opened_[i]), key_->compiled.r_in);
              int64_t x_plain_signed = to_signed(x_plain_ring);
              uint64_t share = (R.party == 0)
                                   ? static_cast<uint64_t>(gates::ref_reciprocal_fixed(
                                         *bundle_.init_spec, x_plain_signed, bundle_.frac_bits, bundle_.nr_iters))
                                   : 0ull;
              y_[i] = share;
              out_[i] = share;
            }
            st_ = St::Done;
            return detail::Need::None;
          }
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
        if (!R.pfss_coeff) throw std::runtime_error("RecipTask: missing coeff PFSS batch");
        coeff_handle_ = R.pfss_coeff->enqueue_composite(std::move(job));
        st_ = St::WaitCoeff;
        return detail::Need::PfssCoeff;
      }
      case St::WaitCoeff: {
        if (!R.pfss_coeff->ready(coeff_handle_)) return detail::Need::PfssCoeff;
        auto v = R.pfss_coeff->view(coeff_handle_);
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
  PfssHandle coeff_handle_{};

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
