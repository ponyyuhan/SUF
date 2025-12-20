#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

#include "runtime/phase_tasks.hpp"

namespace nn {

struct RowMaxDiffPlan {
  int rows = 0;
  int cols = 0;
  std::vector<int> valid_lens;  // optional row-wise lengths; empty => dense
  runtime::ReluBundle relu;
};

// Computes per-row `delta = max(row) - x` over secret-shared `x` using Composite-FSS ReLU:
//   max(a,b) = b + ReLU(a - b).
// The output `delta_qf` is written in the original [rows*cols] layout; entries beyond
// `valid_lens[r]` are set to 0 and are ignored by downstream ragged softmax code.
class RowMaxDiffTask final : public runtime::detail::PhaseTask {
 public:
  RowMaxDiffTask(const RowMaxDiffPlan& plan,
                 std::span<const uint64_t> x_qf,
                 std::span<uint64_t> delta_qf)
      : plan_(plan), x_(x_qf), delta_(delta_qf) {
    if (plan_.rows < 0 || plan_.cols < 0) throw std::runtime_error("RowMaxDiffTask: invalid shape");
    if (static_cast<size_t>(plan_.rows) * static_cast<size_t>(plan_.cols) != x_.size()) {
      throw std::runtime_error("RowMaxDiffTask: input size mismatch");
    }
    if (x_.size() != delta_.size()) throw std::runtime_error("RowMaxDiffTask: output size mismatch");
    if (!plan_.valid_lens.empty() && plan_.valid_lens.size() != static_cast<size_t>(plan_.rows)) {
      throw std::runtime_error("RowMaxDiffTask: valid_lens size mismatch");
    }
    if (!plan_.relu.suf) throw std::runtime_error("RowMaxDiffTask: missing relu SUF");
  }

  bool done() const override { return st_ == St::Done; }
  const std::vector<uint64_t>& row_max_debug() const { return row_max_; }

  runtime::detail::Need step(runtime::PhaseResources& R) override {
    switch (st_) {
      case St::Init: {
        build_active_();
        if (active_elems_ == 0 || plan_.rows == 0 || plan_.cols == 0) {
          std::fill(delta_.begin(), delta_.end(), 0ull);
          st_ = St::Done;
          return runtime::detail::Need::None;
        }
        row_max_.assign(static_cast<size_t>(plan_.rows), 0ull);
        cur_.resize(active_elems_);
        for (size_t i = 0; i < active_elems_; ++i) cur_[i] = x_[active_idx_[i]];
        // Snapshot original active values so the task remains correct even if
        // `x_` and `delta_` alias (in-place max-diff).
        orig_active_ = cur_;
        cur_row_offsets_ = active_row_offsets_;
        cur_row_lens_ = active_row_lens_;
        st_ = St::RoundPrep;
        return runtime::detail::Need::None;
      }
      case St::RoundPrep: {
        bool all_single = true;
        for (int L : cur_row_lens_) {
          if (L > 1) {
            all_single = false;
            break;
          }
        }
        if (all_single) {
          for (int r = 0; r < plan_.rows; ++r) {
            int L = cur_row_lens_[static_cast<size_t>(r)];
            if (L <= 0) {
              row_max_[static_cast<size_t>(r)] = 0ull;
              continue;
            }
            size_t off = static_cast<size_t>(cur_row_offsets_[static_cast<size_t>(r)]);
            row_max_[static_cast<size_t>(r)] = cur_[off];
          }
          st_ = St::Scatter;
          return runtime::detail::Need::None;
        }

        // Build one reduction layer: pairwise max.
        diffs_.clear();
        pair_b_.clear();
        pair_dst_.clear();

        next_row_lens_.assign(static_cast<size_t>(plan_.rows), 0);
        next_row_offsets_.assign(static_cast<size_t>(plan_.rows) + 1, 0);
        for (int r = 0; r < plan_.rows; ++r) {
          int L = cur_row_lens_[static_cast<size_t>(r)];
          int outL = (L <= 0) ? 0 : ((L + 1) / 2);
          next_row_lens_[static_cast<size_t>(r)] = outL;
          next_row_offsets_[static_cast<size_t>(r) + 1] =
              next_row_offsets_[static_cast<size_t>(r)] + outL;
          if (L > 1) diffs_.reserve(diffs_.size() + static_cast<size_t>(L / 2));
        }
        const size_t next_elems = static_cast<size_t>(next_row_offsets_.back());
        next_.assign(next_elems, 0ull);

        for (int r = 0; r < plan_.rows; ++r) {
          const int L = cur_row_lens_[static_cast<size_t>(r)];
          if (L <= 0) continue;
          const size_t in_off = static_cast<size_t>(cur_row_offsets_[static_cast<size_t>(r)]);
          const size_t out_off = static_cast<size_t>(next_row_offsets_[static_cast<size_t>(r)]);
          const int pairs = L / 2;
          for (int i = 0; i < pairs; ++i) {
            const uint64_t a = cur_[in_off + static_cast<size_t>(2 * i)];
            const uint64_t b = cur_[in_off + static_cast<size_t>(2 * i + 1)];
            diffs_.push_back(proto::sub_mod(a, b));  // a-b
            pair_b_.push_back(b);
            pair_dst_.push_back(out_off + static_cast<size_t>(i));
          }
          if (L & 1) {
            next_[out_off + static_cast<size_t>(pairs)] = cur_[in_off + static_cast<size_t>(L - 1)];
          }
        }

        relu_out_.assign(diffs_.size(), 0ull);
        relu_task_ = std::make_unique<runtime::ReluTask>(
            plan_.relu,
            std::span<const uint64_t>(diffs_.data(), diffs_.size()),
            std::span<uint64_t>(relu_out_.data(), relu_out_.size()));
        st_ = St::RoundRelu;
        return runtime::detail::Need::None;
      }
      case St::RoundRelu: {
        auto need = relu_task_->step(R);
        if (!relu_task_->done()) return need;
        relu_task_.reset();
        for (size_t i = 0; i < relu_out_.size(); ++i) {
          size_t dst = pair_dst_[i];
          next_[dst] = proto::add_mod(pair_b_[i], relu_out_[i]);
        }
        cur_.swap(next_);
        cur_row_lens_.swap(next_row_lens_);
        cur_row_offsets_.swap(next_row_offsets_);
        st_ = St::RoundPrep;
        return runtime::detail::Need::None;
      }
      case St::Scatter: {
        std::fill(delta_.begin(), delta_.end(), 0ull);
        for (size_t i = 0; i < active_elems_; ++i) {
          size_t idx = active_idx_[i];
          int row = active_row_[i];
          uint64_t mx = row_max_[static_cast<size_t>(row)];
          // Use the original input value captured at Init time to be robust to
          // in-place operation (`x_` and `delta_` may alias).
          delta_[idx] = proto::sub_mod(mx, orig_active_[i]);
        }
        st_ = St::Done;
        return runtime::detail::Need::None;
      }
      case St::Done:
        return runtime::detail::Need::None;
    }
    return runtime::detail::Need::None;
  }

 private:
  void build_active_() {
    active_row_lens_.assign(static_cast<size_t>(plan_.rows), 0);
    active_row_offsets_.assign(static_cast<size_t>(plan_.rows) + 1, 0);
    for (int r = 0; r < plan_.rows; ++r) {
      int L = plan_.valid_lens.empty() ? plan_.cols : plan_.valid_lens[static_cast<size_t>(r)];
      if (L < 0) L = 0;
      if (L > plan_.cols) L = plan_.cols;
      active_row_lens_[static_cast<size_t>(r)] = L;
      active_row_offsets_[static_cast<size_t>(r) + 1] = active_row_offsets_[static_cast<size_t>(r)] + L;
    }
    active_elems_ = static_cast<size_t>(active_row_offsets_.back());
    active_idx_.resize(active_elems_);
    active_row_.resize(active_elems_);
    size_t off = 0;
    for (int r = 0; r < plan_.rows; ++r) {
      int L = active_row_lens_[static_cast<size_t>(r)];
      for (int c = 0; c < L; ++c) {
        active_idx_[off] = static_cast<size_t>(r * plan_.cols + c);
        active_row_[off] = r;
        ++off;
      }
    }
  }

  enum class St { Init, RoundPrep, RoundRelu, Scatter, Done } st_ = St::Init;

  RowMaxDiffPlan plan_{};
  std::span<const uint64_t> x_;
  std::span<uint64_t> delta_;

  size_t active_elems_ = 0;
  std::vector<int> active_row_lens_;
  std::vector<int> active_row_offsets_;
  std::vector<size_t> active_idx_;
  std::vector<int> active_row_;

  std::vector<uint64_t> row_max_;

  std::vector<uint64_t> cur_;
  std::vector<uint64_t> orig_active_;
  std::vector<int> cur_row_lens_;
  std::vector<int> cur_row_offsets_;

  std::vector<uint64_t> next_;
  std::vector<int> next_row_lens_;
  std::vector<int> next_row_offsets_;

  std::vector<uint64_t> diffs_;
  std::vector<uint64_t> pair_b_;
  std::vector<size_t> pair_dst_;
  std::vector<uint64_t> relu_out_;
  std::unique_ptr<runtime::ReluTask> relu_task_;
};

}  // namespace nn
