#pragma once

#include <stdexcept>

#include "runtime/pfss_superbatch.hpp"
#include "runtime/pfss_async_runner.hpp"

namespace runtime {

// Minimal helper to force a single grouped flush/finalize for PFSS batches at
// phase boundaries. Useful when a phase already queued all jobs and we want to
// ensure trunc/coeff share one flush.
class PfssPhasePlanner {
 public:
  PfssPhasePlanner() = default;

  struct Stats {
    size_t coeff_jobs = 0;
    size_t trunc_jobs = 0;
    size_t coeff_flushes = 0;
    size_t trunc_flushes = 0;
    size_t coeff_hatx_words = 0;
    size_t trunc_hatx_words = 0;
  };

  void bind(PfssSuperBatch* coeff_batch, PfssSuperBatch* trunc_batch) {
    coeff_ = coeff_batch;
    trunc_ = trunc_batch;
  }

  template <typename PfssChanT>
  void finalize_phase(int party, proto::PfssBackendBatch& backend, PfssChanT& pfss_ch) {
    auto flush_once = [&](PfssSuperBatch* b) {
      if (!b) return;
      if (b->has_pending()) {
        b->flush_eval(party, backend, pfss_ch);
      }
      if (b->has_flushed()) {
        b->finalize_all(party, pfss_ch);
      }
    };
    // Flush coeff first, trunc second (common to have them aliased).
    flush_once(coeff_);
    if (trunc_ != coeff_) flush_once(trunc_);
    stats_.coeff_jobs = coeff_ ? coeff_->stats().jobs : 0;
    stats_.trunc_jobs = trunc_ ? trunc_->stats().jobs : 0;
    stats_.coeff_flushes = coeff_ ? coeff_->stats().flushes : 0;
    stats_.trunc_flushes = trunc_ ? trunc_->stats().flushes : 0;
    stats_.coeff_hatx_words = coeff_ ? coeff_->stats().max_bucket_hatx : 0;
    stats_.trunc_hatx_words = trunc_ ? trunc_->stats().max_bucket_hatx : 0;
  }

  const Stats& stats() const { return stats_; }

 private:
  PfssSuperBatch* coeff_ = nullptr;
  PfssSuperBatch* trunc_ = nullptr;
  Stats stats_{};
};

// Cross-phase planner: accumulates per-phase PFSS usage and enforces a
// fail-closed budget across an entire layer/sequence.
class PfssLayerPlanner {
 public:
  PfssLayerPlanner() = default;

  struct Limits {
    size_t max_coeff_jobs = 1ull << 16;
    size_t max_trunc_jobs = 1ull << 16;
    size_t max_coeff_hatx_words = 1ull << 23;
    size_t max_trunc_hatx_words = 1ull << 23;
    size_t max_coeff_flushes = 1ull << 10;
    size_t max_trunc_flushes = 1ull << 10;
    size_t max_phases = 1ull << 12;
  };

  struct Totals {
    size_t phases = 0;
    size_t coeff_jobs = 0;
    size_t trunc_jobs = 0;
    size_t coeff_hatx_words = 0;
    size_t trunc_hatx_words = 0;
    size_t coeff_flushes = 0;
    size_t trunc_flushes = 0;
  };

  void set_limits(const Limits& lim) { limits_ = lim; }
  const Totals& totals() const { return totals_; }
  void reset() { totals_ = Totals{}; }

  void record_phase(const PfssPhasePlanner& planner,
                    const PfssSuperBatch& coeff_batch,
                    const PfssSuperBatch& trunc_batch) {
    const auto& ps = planner.stats();
    totals_.phases++;
    totals_.coeff_jobs += ps.coeff_jobs;
    totals_.trunc_jobs += ps.trunc_jobs;
    totals_.coeff_hatx_words += coeff_batch.stats().max_bucket_hatx;
    totals_.trunc_hatx_words += trunc_batch.stats().max_bucket_hatx;
    totals_.coeff_flushes += ps.coeff_flushes;
    totals_.trunc_flushes += ps.trunc_flushes;
    enforce_limits();
  }

  // Optional safety flush at layer end; useful when phases forgot to clear.
  template <typename PfssChanT>
  void finalize_layer(int party,
                      proto::PfssBackendBatch& backend,
                      PfssSuperBatch& coeff_batch,
                      PfssSuperBatch& trunc_batch,
                      PfssChanT& pfss_ch,
                      PfssAsyncRunner* async_runner = nullptr,
                      bool async = false) {
    if (async_runner) {
      async_runner->start_flush(party, backend, coeff_batch, &trunc_batch, pfss_ch, async);
      if (async) async_runner->wait();  // caller can still choose to block
    } else {
      auto flush_once = [&](PfssSuperBatch& b, size_t& flush_counter) {
        if (b.has_pending()) {
          b.flush_eval(party, backend, pfss_ch);
          flush_counter += b.stats().flushes;
        }
        if (b.has_flushed()) {
          b.finalize_all(party, pfss_ch);
          flush_counter += b.stats().flushes;
        }
      };
      flush_once(coeff_batch, totals_.coeff_flushes);
      if (&trunc_batch != &coeff_batch) flush_once(trunc_batch, totals_.trunc_flushes);
    }
    totals_.coeff_jobs += coeff_batch.stats().jobs;
    totals_.trunc_jobs += trunc_batch.stats().jobs;
    totals_.coeff_hatx_words += coeff_batch.stats().max_bucket_hatx;
    totals_.trunc_hatx_words += trunc_batch.stats().max_bucket_hatx;
    enforce_limits();
  }

 private:
  void enforce_limits() const {
    if (totals_.phases > limits_.max_phases) {
      throw std::runtime_error("PfssLayerPlanner: phase budget exceeded");
    }
    if (totals_.coeff_jobs > limits_.max_coeff_jobs) {
      throw std::runtime_error("PfssLayerPlanner: coeff job budget exceeded");
    }
    if (totals_.trunc_jobs > limits_.max_trunc_jobs) {
      throw std::runtime_error("PfssLayerPlanner: trunc job budget exceeded");
    }
    if (totals_.coeff_hatx_words > limits_.max_coeff_hatx_words) {
      throw std::runtime_error("PfssLayerPlanner: coeff hatx budget exceeded");
    }
    if (totals_.trunc_hatx_words > limits_.max_trunc_hatx_words) {
      throw std::runtime_error("PfssLayerPlanner: trunc hatx budget exceeded");
    }
    if (totals_.coeff_flushes > limits_.max_coeff_flushes) {
      throw std::runtime_error("PfssLayerPlanner: coeff flush budget exceeded");
    }
    if (totals_.trunc_flushes > limits_.max_trunc_flushes) {
      throw std::runtime_error("PfssLayerPlanner: trunc flush budget exceeded");
    }
  }

  Limits limits_{};
  Totals totals_{};
};

}  // namespace runtime
