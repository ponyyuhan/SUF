#pragma once

#include <memory>
#include <vector>

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

  runtime::detail::Need step(runtime::PhaseResources& R) override {
    switch (st_) {
      case St::ExpInit: {
        exp_qf_.resize(t_.size());
        nexp_task_ = std::make_unique<runtime::CubicPolyTask>(plan_.nexp, t_, exp_qf_);
        st_ = St::ExpRun;
        return runtime::detail::Need::None;
      }
      case St::ExpRun: {
        auto need = nexp_task_->step(R);
        if (!nexp_task_->done()) return need;
        sum_qf_.assign(plan_.rows, 0);
        st_ = St::SumLocal;
        return runtime::detail::Need::None;
      }
      case St::SumLocal: {
        for (int r = 0; r < plan_.rows; ++r) {
          int L = plan_.valid_lens.empty() ? plan_.cols : plan_.valid_lens[r];
          uint64_t acc = 0;
          for (int c = 0; c < L; ++c) {
            size_t idx = static_cast<size_t>(r * plan_.cols + c);
            acc = proto::add_mod(acc, exp_qf_[idx]);
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
        // Set a conservative range for prob (Q2f): |exp|<=1, |inv| bounded by sum<=cols
        prob_range_ = compiler::RangeInterval::from_lo_hi(0, static_cast<int64_t>(plan_.cols) << plan_.frac_bits);
        st_ = St::MulRun;
        return runtime::detail::Need::None;
      }
      case St::MulRun: {
        auto need = mul_task_->step(R);
        if (!mul_task_->done()) return need;
        // Choose trunc bundle based on range
        const auto* trunc_bundle = select_trunc_bundle(plan_.prob_trunc, prob_range_, plan_.frac_bits);
        if (!trunc_bundle) throw std::runtime_error("SoftmaxBlockTask: missing trunc bundle");
        trunc_task_ = std::make_unique<runtime::TruncTask>(trunc_bundle,
                                                           std::span<const uint64_t>(prod_q2f_.data(), prod_q2f_.size()),
                                                           out_);
        st_ = St::TruncRun;
        return runtime::detail::Need::None;
      }
      case St::TruncRun: {
        auto need = trunc_task_->step(R);
        if (!trunc_task_->done()) return need;
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
  compiler::RangeInterval prob_range_;

  std::unique_ptr<runtime::CubicPolyTask> nexp_task_;
  std::unique_ptr<runtime::RecipTask> recip_task_;
  std::unique_ptr<runtime::MulRowBroadcastTask> mul_task_;
  std::unique_ptr<runtime::TruncTask> trunc_task_;
};

}  // namespace nn

