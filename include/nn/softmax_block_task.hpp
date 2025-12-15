#pragma once

#include <algorithm>
#include <cstdio>
#include <limits>
#include <memory>
#include <vector>
#include <optional>

#include "compiler/range_analysis.hpp"
#include "runtime/phase_tasks.hpp"

namespace nn {

struct SoftmaxPlan {
  int frac_bits = 16;
  int rows = 0;
  int cols = 0;
  std::vector<int> valid_lens;  // optional row-wise lengths; empty => dense
  // If true, input `t_qf` is already `delta = max(row) - x` (non-negative, clamped to [0,16]).
  // If false, task will open inputs to compute per-row max and form `delta`.
  bool input_is_max_diff = true;
  bool device_only = false;     // optional: skip host materialization entirely
  bool materialize_host = true; // if false and device output exists, keep device only
  runtime::CubicPolyBundle nexp;
  runtime::RecipTaskBundle recip;
  runtime::TruncChoice prob_trunc;
  runtime::RowBroadcastTripleProvider* row_triples = nullptr;
  std::optional<compiler::RangeInterval> prob_range;  // optional override for prob trunc selection
};

// Single-task softmax: nExp -> sum -> recip -> exp*recip -> trunc, all within one PhaseExecutor run.
class SoftmaxBlockTask : public runtime::detail::PhaseTask {
 public:
  SoftmaxBlockTask(const SoftmaxPlan& plan,
                   std::span<const uint64_t> t_qf,
                   std::span<uint64_t> out_qf)
      : plan_(plan), t_(t_qf), out_(out_qf) {
    if (static_cast<int>(t_.size()) != plan_.rows * plan_.cols) {
      throw std::runtime_error("SoftmaxBlockTask: input size mismatch");
    }
    if (t_.size() != out_.size()) {
      throw std::runtime_error("SoftmaxBlockTask: output size mismatch");
    }
  }

  ~SoftmaxBlockTask() override {
#ifdef SUF_HAVE_CUDA
    if (d_prob_device_) {
      cudaFree(d_prob_device_);
      d_prob_device_ = nullptr;
      d_prob_elems_ = 0;
    }
#endif
  }

  bool done() const override { return st_ == St::Done; }
  const std::vector<uint64_t>& exp_qf_debug() const { return exp_qf_; }
  const std::vector<uint64_t>& sum_qf_debug() const { return sum_qf_; }
  const std::vector<uint64_t>& inv_qf_debug() const { return inv_qf_; }
  const std::vector<uint64_t>& prod_q2f_debug() const { return prod_q2f_; }
  const std::vector<uint64_t>& prob_qf_debug() const { return prob_qf_; }
  const runtime::RecipTask* recip_task_debug() const { return recip_task_.get(); }
#ifdef SUF_HAVE_CUDA
  // Optional device probability buffer when device-only softmax is used.
  uint64_t* device_prob() const { return d_prob_device_; }
  size_t device_prob_elems() const { return d_prob_elems_; }
  void release_device_prob() {
    if (d_prob_device_) {
      cudaFree(d_prob_device_);
      d_prob_device_ = nullptr;
      d_prob_elems_ = 0;
    }
  }
#endif

  runtime::detail::Need step(runtime::PhaseResources& R) override {
#ifdef SUF_HAVE_CUDA
    bool trace = (std::getenv("SOFTMAX_BENCH_TRACE") != nullptr);
#else
    bool trace = (std::getenv("SOFTMAX_BENCH_TRACE") != nullptr);
#endif
    if (trace) {
      std::fprintf(stderr, "[SoftmaxTask] state=%d device_only=%d pipeline=%d materialize=%d\n",
                   static_cast<int>(st_), plan_.device_only ? 1 : 0,
                   R.device_pipeline ? 1 : 0,
                   plan_.materialize_host ? 1 : 0);
    }
    switch (st_) {
      case St::ExpInit: {
        build_offsets();
        t_packed_.resize(active_elems_);
        if (plan_.input_is_max_diff) {
          size_t off = 0;
          for (int r = 0; r < plan_.rows; ++r) {
            int L = plan_.valid_lens.empty() ? plan_.cols : plan_.valid_lens[r];
            for (int c = 0; c < L; ++c) {
              size_t idx = static_cast<size_t>(r * plan_.cols + c);
              t_packed_[off++] = t_[idx];
            }
          }
        } else {
          if (!R.net_chan) throw std::runtime_error("SoftmaxBlockTask: net channel missing for max-diff preprocessing");
          std::vector<uint64_t> other(active_elems_, 0);
          if (R.party == 0) {
            for (int r = 0; r < plan_.rows; ++r) {
              int L = plan_.valid_lens.empty() ? plan_.cols : plan_.valid_lens[r];
              for (int c = 0; c < L; ++c) {
                size_t idx = static_cast<size_t>(r * plan_.cols + c);
                R.net_chan->send_u64(t_[idx]);
              }
            }
            size_t off = 0;
            for (int r = 0; r < plan_.rows; ++r) {
              int L = plan_.valid_lens.empty() ? plan_.cols : plan_.valid_lens[r];
              for (int c = 0; c < L; ++c) {
                other[off++] = R.net_chan->recv_u64();
              }
            }
          } else {
            size_t off = 0;
            for (int r = 0; r < plan_.rows; ++r) {
              int L = plan_.valid_lens.empty() ? plan_.cols : plan_.valid_lens[r];
              for (int c = 0; c < L; ++c) {
                other[off++] = R.net_chan->recv_u64();
              }
            }
            for (int r = 0; r < plan_.rows; ++r) {
              int L = plan_.valid_lens.empty() ? plan_.cols : plan_.valid_lens[r];
              for (int c = 0; c < L; ++c) {
                size_t idx = static_cast<size_t>(r * plan_.cols + c);
                R.net_chan->send_u64(t_[idx]);
              }
            }
          }

          std::vector<int64_t> plain(active_elems_, 0);
          {
            size_t off = 0;
            for (int r = 0; r < plan_.rows; ++r) {
              int L = plan_.valid_lens.empty() ? plan_.cols : plan_.valid_lens[r];
              for (int c = 0; c < L; ++c) {
                size_t idx = static_cast<size_t>(r * plan_.cols + c);
                plain[off] = static_cast<int64_t>(t_[idx] + other[off]);
                ++off;
              }
            }
          }

          std::vector<int64_t> row_max(static_cast<size_t>(plan_.rows), 0);
          {
            size_t off = 0;
            for (int r = 0; r < plan_.rows; ++r) {
              int L = plan_.valid_lens.empty() ? plan_.cols : plan_.valid_lens[r];
              if (L <= 0) {
                row_max[static_cast<size_t>(r)] = 0;
                continue;
              }
              int64_t best = plain[off];
              for (int c = 1; c < L; ++c) best = std::max(best, plain[off + static_cast<size_t>(c)]);
              row_max[static_cast<size_t>(r)] = best;
              off += static_cast<size_t>(L);
            }
          }

          int64_t cap = std::numeric_limits<int64_t>::max();
          if (plan_.frac_bits >= 0 && plan_.frac_bits <= 58) cap = (16ll << plan_.frac_bits);

          size_t off = 0;
          for (int r = 0; r < plan_.rows; ++r) {
            int L = plan_.valid_lens.empty() ? plan_.cols : plan_.valid_lens[r];
            int64_t best = row_max[static_cast<size_t>(r)];
            for (int c = 0; c < L; ++c) {
              int64_t diff = best - plain[off];
              if (diff < 0) diff = 0;
              if (diff > cap) diff = cap;
              t_packed_[off] = (R.party == 0) ? static_cast<uint64_t>(diff) : 0ull;
              ++off;
            }
          }
        }
        exp_packed_.assign(active_elems_, 0);
        nexp_task_ = std::make_unique<runtime::CubicPolyTask>(
            plan_.nexp,
            std::span<const uint64_t>(t_packed_.data(), t_packed_.size()),
            std::span<uint64_t>(exp_packed_.data(), exp_packed_.size()));
        if (!plan_.valid_lens.empty()) {
          nexp_task_->set_shape_hint(&row_offsets_, &plan_.valid_lens);
        }
        st_ = St::ExpRun;
        return runtime::detail::Need::None;
      }
      case St::ExpRun: {
        auto need = nexp_task_->step(R);
        if (!nexp_task_->done()) return need;
#ifdef SUF_HAVE_CUDA
        if (R.device_pipeline) {
          if (auto* dptr = nexp_task_->device_out()) {
            d_exp_device_ = dptr;
            d_exp_elems_ = nexp_task_->device_out_elems();
            // CubicPolyTask may keep results device-only in pipeline mode; copy back so
            // subsequent opens/muls that currently consume host buffers remain correct.
            size_t n = std::min(exp_packed_.size(), d_exp_elems_);
            if (n > 0) {
              cudaMemcpy(exp_packed_.data(), d_exp_device_, n * sizeof(uint64_t),
                         cudaMemcpyDeviceToHost);
            }
          }
        }
#endif
        exp_qf_.assign(t_.size(), 0);
        if (active_elems_ == t_.size()) {
          exp_qf_.assign(exp_packed_.begin(), exp_packed_.end());
        } else {
          scatter_active(exp_packed_, exp_qf_);
        }
        sum_qf_.assign(plan_.rows, 0);
        prob_abs_.is_signed = true;
        prob_abs_.max_abs = static_cast<uint64_t>(1ull << plan_.frac_bits);
        prob_abs_.kind = compiler::RangeKind::Proof;
        st_ = St::SumLocal;
        return runtime::detail::Need::None;
      }
      case St::SumLocal: {
#ifdef SUF_HAVE_CUDA
        if (std::getenv("SUF_SOFTMAX_GPU") && R.device_pipeline) {
          // GPU row-sum of exp_packed_. valid_lens optional.
          cudaStream_t stream = nullptr;
          if (R.pfss_backend) {
            if (auto* staged = dynamic_cast<proto::PfssGpuStagedEval*>(R.pfss_backend)) {
              stream = reinterpret_cast<cudaStream_t>(staged->device_stream());
            }
          }
          uint64_t* d_exp = d_exp_device_;
          uint64_t* d_sum = nullptr;
          int* d_valid = nullptr;
          size_t elems = exp_packed_.size();
          size_t bytes_exp = elems * sizeof(uint64_t);
          if (!d_exp) {
            cudaMalloc(&d_exp, bytes_exp);
            cudaMemcpyAsync(d_exp, exp_packed_.data(), bytes_exp, cudaMemcpyHostToDevice, stream);
          }
          cudaMalloc(&d_sum, static_cast<size_t>(plan_.rows) * sizeof(uint64_t));
          if (!plan_.valid_lens.empty()) {
            cudaMalloc(&d_valid, static_cast<size_t>(plan_.rows) * sizeof(int));
            cudaMemcpyAsync(d_valid, plan_.valid_lens.data(),
                            static_cast<size_t>(plan_.rows) * sizeof(int),
                            cudaMemcpyHostToDevice, stream);
          }
          launch_row_sum_kernel(d_exp, plan_.rows, plan_.cols, d_valid, d_sum, stream);
          sum_qf_.assign(static_cast<size_t>(plan_.rows), 0);
          cudaMemcpyAsync(sum_qf_.data(), d_sum,
                          static_cast<size_t>(plan_.rows) * sizeof(uint64_t),
                          cudaMemcpyDeviceToHost, stream);
          cudaStreamSynchronize(stream);
          // Only free if we allocated; borrowed pointer stays alive with nexp_task_.
          if (d_exp && d_exp != d_exp_device_) cudaFree(d_exp);
          if (d_sum) cudaFree(d_sum);
          if (d_valid) cudaFree(d_valid);
          prob_abs_.is_signed = true;
          prob_abs_.max_abs = static_cast<uint64_t>(1ull << plan_.frac_bits);
          prob_abs_.kind = compiler::RangeKind::Proof;
          inv_qf_.resize(plan_.rows);
          recip_task_ = std::make_unique<runtime::RecipTask>(
              plan_.recip,
              std::span<const uint64_t>(sum_qf_.data(), sum_qf_.size()),
              std::span<uint64_t>(inv_qf_.data(), inv_qf_.size()));
          st_ = St::RecipRun;
          return runtime::detail::Need::None;
        }
#endif
        for (int r = 0; r < plan_.rows; ++r) {
          int L = plan_.valid_lens.empty() ? plan_.cols : plan_.valid_lens[r];
          uint64_t acc = 0;
          size_t base = row_offsets_[static_cast<size_t>(r)];
          for (int c = 0; c < L; ++c) {
            acc = proto::add_mod(acc, exp_packed_[base + static_cast<size_t>(c)]);
          }
          sum_qf_[r] = acc;
        }
        inv_qf_.resize(plan_.rows);
        recip_task_ = std::make_unique<runtime::RecipTask>(plan_.recip,
                                                           std::span<const uint64_t>(sum_qf_.data(), sum_qf_.size()),
                                                           std::span<uint64_t>(inv_qf_.data(), inv_qf_.size()));
        st_ = St::RecipRun;
        return runtime::detail::Need::None;
      }
      case St::RecipRun: {
        if (trace) {
          std::fprintf(stderr, "[SoftmaxTask] calling RecipTask step rows=%d cols=%d\n",
                       plan_.rows, plan_.cols);
        }
        auto need = recip_task_->step(R);
        if (!recip_task_->done()) return need;
        prod_q2f_.resize(exp_qf_.size());
        bool mul_device_only = plan_.device_only && R.device_pipeline;
        mul_task_ = std::make_unique<runtime::MulRowBroadcastTask>(
            std::span<const uint64_t>(exp_qf_.data(), exp_qf_.size()),
            std::span<const uint64_t>(inv_qf_.data(), inv_qf_.size()),
            plan_.rows,
            plan_.cols,
            std::span<const int>(plan_.valid_lens),
            std::span<uint64_t>(prod_q2f_.data(), prod_q2f_.size()),
            plan_.row_triples,
            mul_device_only);
        // Probabilities are non-negative and <=1; keep a tight bound so GapARS can be chosen.
        prob_range_.lo = 0;
        prob_range_.hi = static_cast<int64_t>(1) << plan_.frac_bits;
        prob_range_.is_signed = false;
        st_ = St::MulRun;
        return runtime::detail::Need::None;
      }
      case St::MulRun: {
        auto need = mul_task_->step(R);
        if (!mul_task_->done()) return need;
#ifdef SUF_HAVE_CUDA
        if (plan_.device_only && R.device_pipeline) {
          if (auto* dptr = mul_task_->device_out()) {
            d_prod_device_ = dptr;
            d_prod_elems_ = mul_task_->device_out_elems();
            // TruncTask currently consumes host shares, so we still need a host
            // copy of the product even when we keep a device buffer alive.
            size_t n = std::min(prod_q2f_.size(), d_prod_elems_);
            if (n > 0) {
              cudaMemcpy(prod_q2f_.data(), d_prod_device_, n * sizeof(uint64_t),
                         cudaMemcpyDeviceToHost);
            }
          }
        } else if (R.device_pipeline) {
          if (auto* dptr = mul_task_->device_out()) {
            size_t n = std::min(prod_q2f_.size(), mul_task_->device_out_elems());
            cudaMemcpy(prod_q2f_.data(), dptr, n * sizeof(uint64_t), cudaMemcpyDeviceToHost);
            mul_task_->release_device_out();
          }
        }
#endif
        if (active_elems_ == 0) {
          prob_qf_.assign(out_.size(), 0);
          for (size_t i = 0; i < out_.size(); ++i) out_[i] = prob_qf_[i];
          st_ = St::Done;
          return runtime::detail::Need::None;
        }
        // Choose trunc bundle based on range
        if (plan_.prob_range) prob_range_ = *plan_.prob_range;
        auto gap = compiler::gap_from_abs(prob_abs_, plan_.frac_bits);
        const auto* trunc_bundle = select_trunc_bundle(plan_.prob_trunc, prob_abs_, plan_.frac_bits, gap);
        if (!trunc_bundle) throw std::runtime_error("SoftmaxBlockTask: missing trunc bundle");
        prob_qf_.assign(out_.size(), 0);
        if (active_elems_ == prod_q2f_.size()) {
          trunc_task_ = std::make_unique<runtime::TruncTask>(
              trunc_bundle,
              std::span<const uint64_t>(prod_q2f_.data(), prod_q2f_.size()),
              std::span<uint64_t>(prob_qf_.data(), prob_qf_.size()));
        } else {
          prod_packed_.clear();
          prod_packed_.reserve(active_elems_);
          gather_active(prod_q2f_, prod_packed_);
          prob_packed_.assign(active_elems_, 0);
          trunc_task_ = std::make_unique<runtime::TruncTask>(
              trunc_bundle,
              std::span<const uint64_t>(prod_packed_.data(), prod_packed_.size()),
              std::span<uint64_t>(prob_packed_.data(), prob_packed_.size()));
        }
#ifdef SUF_HAVE_CUDA
        // If we kept the MulRow output on device and are in device pipeline mode,
        // do NOT forward the secret-share device pointer as hatx_device.
        // hatx_device is reserved for the opened masked value (public hatx),
        // which is produced inside TruncTask after the Open step.
        (void)d_prod_device_;
        (void)d_prod_elems_;
#endif
        if (!plan_.valid_lens.empty()) {
          trunc_task_->set_shape_hint(&row_offsets_, &plan_.valid_lens);
        }
        st_ = St::TruncRun;
        return runtime::detail::Need::None;
      }
      case St::TruncRun: {
        auto need = trunc_task_->step(R);
        if (!trunc_task_->done()) return need;
#ifdef SUF_HAVE_CUDA
        if (plan_.device_only && R.device_pipeline) {
          // Device-only bench: skip host materialization to avoid D2H copies.
          if (auto* dptr = trunc_task_->device_out()) {
            size_t have = trunc_task_->device_out_elems();
            // Keep ownership so downstream device consumers can reuse the buffer.
            d_prob_device_ = trunc_task_->take_device_out(&d_prob_elems_);
          }
          st_ = St::Done;
          return runtime::detail::Need::None;
        }
        // In device-pipeline mode, pull trunc outputs back to host and free device buffer.
        if (R.device_pipeline) {
          if (auto* dptr = trunc_task_->device_out()) {
            size_t have = trunc_task_->device_out_elems();
            // Always keep device buffer for optional downstream consumers.
            d_prob_device_ = trunc_task_->take_device_out(&d_prob_elems_);
            if (plan_.materialize_host) {
              size_t n = prob_packed_.empty() ? prob_qf_.size() : prob_packed_.size();
              n = std::min(n, have);
              uint64_t* dst = prob_packed_.empty() ? prob_qf_.data() : prob_packed_.data();
              cudaMemcpy(dst, d_prob_device_, n * sizeof(uint64_t), cudaMemcpyDeviceToHost);
            }
          }
        }
#endif
        if (!prob_packed_.empty()) {
          scatter_active(prob_packed_, prob_qf_);
        }
        for (size_t i = 0; i < out_.size(); ++i) out_[i] = prob_qf_[i];
        st_ = St::Done;
        return runtime::detail::Need::None;
      }
      case St::Done:
        return runtime::detail::Need::None;
    }
    return runtime::detail::Need::None;
  }

 private:
  enum class St { ExpInit, ExpRun, SumLocal, RecipRun, MulRun, TruncRun, Done } st_ = St::ExpInit;
  const SoftmaxPlan plan_;
  std::span<const uint64_t> t_;
  std::span<uint64_t> out_;

  std::vector<uint64_t> exp_qf_;
  std::vector<uint64_t> sum_qf_;
  std::vector<uint64_t> inv_qf_;
  std::vector<uint64_t> prod_q2f_;
  std::vector<uint64_t> prob_qf_;
  compiler::RangeInterval prob_range_;
  compiler::AbsBound prob_abs_;
  std::vector<uint64_t> t_packed_;
  std::vector<uint64_t> exp_packed_;
  std::vector<uint64_t> prod_packed_;
  std::vector<uint64_t> prob_packed_;
  std::vector<int> row_offsets_;
  size_t active_elems_ = 0;

  std::unique_ptr<runtime::CubicPolyTask> nexp_task_;
  std::unique_ptr<runtime::RecipTask> recip_task_;
  std::unique_ptr<runtime::MulRowBroadcastTask> mul_task_;
  std::unique_ptr<runtime::TruncTask> trunc_task_;

#ifdef SUF_HAVE_CUDA
  uint64_t* d_exp_device_ = nullptr;  // borrowed from nexp_task_
  size_t d_exp_elems_ = 0;
  uint64_t* d_prod_device_ = nullptr;  // borrowed from mul_task_ output if device-only copy avoided
  size_t d_prod_elems_ = 0;
  uint64_t* d_prob_device_ = nullptr;  // owned trunc output when device_only
  size_t d_prob_elems_ = 0;
#endif

  void build_offsets() {
    if (active_elems_ > 0) return;
    if (!plan_.valid_lens.empty() && static_cast<int>(plan_.valid_lens.size()) != plan_.rows) {
      throw std::runtime_error("SoftmaxBlockTask: valid_lens size mismatch");
    }
    row_offsets_.assign(static_cast<size_t>(plan_.rows) + 1, 0);
    size_t acc = 0;
    for (int r = 0; r < plan_.rows; ++r) {
      int L = plan_.valid_lens.empty() ? plan_.cols : plan_.valid_lens[r];
      acc += static_cast<size_t>(L);
      row_offsets_[static_cast<size_t>(r + 1)] = static_cast<int>(acc);
    }
    active_elems_ = acc;
  }

  template <typename VecT>
  void scatter_active(const VecT& packed, std::vector<uint64_t>& full) const {
    size_t off = 0;
    for (int r = 0; r < plan_.rows; ++r) {
      int L = plan_.valid_lens.empty() ? plan_.cols : plan_.valid_lens[r];
      for (int c = 0; c < L; ++c) {
        size_t idx = static_cast<size_t>(r * plan_.cols + c);
        full[idx] = packed[off++];
      }
    }
  }

  void gather_active(const std::vector<uint64_t>& full, std::vector<uint64_t>& packed) const {
    for (int r = 0; r < plan_.rows; ++r) {
      int L = plan_.valid_lens.empty() ? plan_.cols : plan_.valid_lens[r];
      for (int c = 0; c < L; ++c) {
        size_t idx = static_cast<size_t>(r * plan_.cols + c);
        packed.push_back(full[idx]);
      }
    }
  }
};

}  // namespace nn
