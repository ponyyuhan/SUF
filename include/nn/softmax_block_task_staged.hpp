#pragma once

#include <memory>
#include <vector>
#include <optional>

#include "compiler/range_analysis.hpp"
#include "runtime/phase_tasks.hpp"
#include "runtime/staged_executor.hpp"
#include "nn/softmax_block_task.hpp"

namespace nn {

// Staged softmax: prepare enqueues PFSS work (nExp, recip, trunc), finalize consumes results.
class StagedSoftmaxTask : public runtime::detail::PhaseTask {
 public:
  struct PrepResult {
    runtime::detail::Need need = runtime::detail::Need::None;
    bool done = false;
  };

  StagedSoftmaxTask(const SoftmaxPlan& plan,
                    std::span<const uint64_t> t_qf,
                    std::span<uint64_t> out_qf)
      : plan_(plan), t_(t_qf), out_(out_qf) {
    if (static_cast<int>(t_.size()) != plan_.rows * plan_.cols) {
      throw std::runtime_error("StagedSoftmaxTask: input size mismatch");
    }
    if (t_.size() != out_.size()) {
      throw std::runtime_error("StagedSoftmaxTask: output size mismatch");
    }
    // Reuse the packing logic from SoftmaxBlockTask.
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
  }

  bool done() const override { return st_ == St::Done; }

  // Prepare phase: run nexp, sum, recip, mul, enqueue trunc; return Need signals.
  PrepResult prepare(runtime::PhaseResources& R) {
    while (true) {
      switch (st_) {
        case St::ExpInit: {
          exp_packed_.assign(active_elems_, 0);
          nexp_task_ = std::make_unique<runtime::CubicPolyTask>(
              plan_.nexp,
              std::span<const uint64_t>(t_packed_.data(), t_packed_.size()),
              std::span<uint64_t>(exp_packed_.data(), exp_packed_.size()));
          if (!plan_.valid_lens.empty()) {
            nexp_task_->set_shape_hint(&row_offsets_, &plan_.valid_lens);
          }
          st_ = St::ExpRun;
          break;
        }
        case St::ExpRun: {
          auto need = nexp_task_->step(R);
          if (!nexp_task_->done()) return {need, false};
          sum_qf_.assign(plan_.rows, 0);
          st_ = St::SumLocal;
          break;
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
          recip_task_ = std::make_unique<runtime::RecipTask>(
              plan_.recip,
              std::span<const uint64_t>(sum_qf_.data(), sum_qf_.size()),
              std::span<uint64_t>(inv_qf_.data(), inv_qf_.size()));
          st_ = St::RecipRun;
          break;
        }
        case St::RecipRun: {
          auto need = recip_task_->step(R);
          if (!recip_task_->done()) return {need, false};
          prod_q2f_.resize(t_.size());
          // Materialize full exp_qf for MulRowBroadcastTask.
          exp_qf_.assign(t_.size(), 0);
          scatter_active(exp_packed_, exp_qf_);
          mul_task_ = std::make_unique<runtime::MulRowBroadcastTask>(
              std::span<const uint64_t>(exp_qf_.data(), exp_qf_.size()),
              std::span<const uint64_t>(inv_qf_.data(), inv_qf_.size()),
              plan_.rows,
              plan_.cols,
              std::span<const int>(plan_.valid_lens),
              std::span<uint64_t>(prod_q2f_.data(), prod_q2f_.size()),
              plan_.row_triples,
              /*device_only=*/false);
          prob_range_.lo = 0;
          prob_range_.hi = static_cast<int64_t>(1) << plan_.frac_bits;
          prob_range_.is_signed = false;
          prob_abs_.is_signed = true;
          prob_abs_.max_abs = static_cast<uint64_t>(1ull << plan_.frac_bits);
          prob_abs_.kind = compiler::RangeKind::Proof;
          st_ = St::MulRun;
          break;
        }
        case St::MulRun: {
          auto need = mul_task_->step(R);
          if (!mul_task_->done()) return {need, false};
          prob_qf_.assign(out_.size(), 0);
          if (plan_.prob_range) prob_range_ = *plan_.prob_range;
          auto gap = compiler::gap_from_abs(prob_abs_, plan_.frac_bits);
          const auto* trunc_bundle = runtime::select_trunc_bundle(plan_.prob_trunc, prob_abs_, plan_.frac_bits, gap);
          if (!trunc_bundle) throw std::runtime_error("StagedSoftmaxTask: missing trunc bundle");
          trunc_task_ = std::make_unique<runtime::TruncTask>(
              trunc_bundle,
              std::span<const uint64_t>(prod_q2f_.data(), prod_q2f_.size()),
              std::span<uint64_t>(prob_qf_.data(), prob_qf_.size()));
          if (!plan_.valid_lens.empty()) {
            trunc_task_->set_shape_hint(&row_offsets_, &plan_.valid_lens);
          }
          st_ = St::TruncRun;
          break;
        }
        case St::TruncRun: {
          auto need = trunc_task_->step(R);
          if (!trunc_task_->done()) return {need, false};
          st_ = St::Done;
          return {runtime::detail::Need::None, true};
        }
        case St::Done:
          return {runtime::detail::Need::None, true};
      }
    }
  }

  // Finalize phase: consume trunc output and write to out_.
  void finalize() {
    if (!trunc_task_ || !trunc_task_->done()) return;
    // prob_qf_ already filled by trunc_task_; write to out.
    for (size_t i = 0; i < out_.size(); ++i) out_[i] = prob_qf_[i];
    st_ = St::Done;
  }

  runtime::detail::Need step(runtime::PhaseResources& R) override {
    auto prep = prepare(R);
    if (prep.need != runtime::detail::Need::None) return prep.need;
    if (prep.done) {
      finalize();
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
  std::vector<int> row_offsets_;
  size_t active_elems_ = 0;

  std::unique_ptr<runtime::CubicPolyTask> nexp_task_;
  std::unique_ptr<runtime::RecipTask> recip_task_;
  std::unique_ptr<runtime::MulRowBroadcastTask> mul_task_;
  std::unique_ptr<runtime::TruncTask> trunc_task_;

  void build_offsets() {
    if (active_elems_ > 0) return;
    if (!plan_.valid_lens.empty() && static_cast<int>(plan_.valid_lens.size()) != plan_.rows) {
      throw std::runtime_error("StagedSoftmaxTask: valid_lens size mismatch");
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
};

}  // namespace nn
