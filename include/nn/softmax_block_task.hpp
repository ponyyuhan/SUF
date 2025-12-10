#pragma once

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

  bool done() const override { return st_ == St::Done; }
  const std::vector<uint64_t>& exp_qf_debug() const { return exp_qf_; }
  const std::vector<uint64_t>& sum_qf_debug() const { return sum_qf_; }
  const std::vector<uint64_t>& inv_qf_debug() const { return inv_qf_; }
  const std::vector<uint64_t>& prod_q2f_debug() const { return prod_q2f_; }
  const std::vector<uint64_t>& prob_qf_debug() const { return prob_qf_; }
  const runtime::RecipTask* recip_task_debug() const { return recip_task_.get(); }

  runtime::detail::Need step(runtime::PhaseResources& R) override {
    switch (st_) {
      case St::ExpInit: {
        build_offsets();
        t_packed_.resize(active_elems_);
        size_t off = 0;
        for (int r = 0; r < plan_.rows; ++r) {
          int L = plan_.valid_lens.empty() ? plan_.cols : plan_.valid_lens[r];
          for (int c = 0; c < L; ++c) {
            size_t idx = static_cast<size_t>(r * plan_.cols + c);
            t_packed_[off++] = t_[idx];
          }
        }
        exp_packed_.assign(active_elems_, 0);
        nexp_task_ = std::make_unique<runtime::CubicPolyTask>(
            plan_.nexp,
            std::span<const uint64_t>(t_packed_.data(), t_packed_.size()),
            std::span<uint64_t>(exp_packed_.data(), exp_packed_.size()));
        st_ = St::ExpRun;
        return runtime::detail::Need::None;
      }
      case St::ExpRun: {
        auto need = nexp_task_->step(R);
        if (!nexp_task_->done()) return need;
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
        auto need = recip_task_->step(R);
        if (!recip_task_->done()) return need;
        prod_q2f_.resize(exp_qf_.size());
        mul_task_ = std::make_unique<runtime::MulRowBroadcastTask>(
            std::span<const uint64_t>(exp_qf_.data(), exp_qf_.size()),
            std::span<const uint64_t>(inv_qf_.data(), inv_qf_.size()),
            plan_.rows,
            plan_.cols,
            std::span<const int>(plan_.valid_lens),
            std::span<uint64_t>(prod_q2f_.data(), prod_q2f_.size()),
            plan_.row_triples);
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
        st_ = St::TruncRun;
        return runtime::detail::Need::None;
      }
      case St::TruncRun: {
        auto need = trunc_task_->step(R);
        if (!trunc_task_->done()) return need;
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
