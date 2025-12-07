#pragma once

#include <stdexcept>
#include <algorithm>
#include <optional>

#include "runtime/pfss_superbatch.hpp"
#include "runtime/pfss_async_runner.hpp"
#include "runtime/open_collector.hpp"
#include "mpc/net.hpp"

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

  struct BarrierPolicy {
    bool drain_open = false;
    bool drain_pfss_coeff = false;
    bool drain_pfss_trunc = false;
    bool drain_all = false;
  };

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

  void begin_layer() { reset(); }
  void enter_phase() { totals_.phases++; enforce_limits(); }

  void record_phase(const PfssPhasePlanner& planner,
                    const PfssSuperBatch& coeff_batch,
                    const PfssSuperBatch& trunc_batch) {
    const auto& ps = planner.stats();
    totals_.coeff_jobs += ps.coeff_jobs;
    totals_.trunc_jobs += ps.trunc_jobs;
    totals_.coeff_hatx_words += coeff_batch.stats().max_bucket_hatx;
    totals_.trunc_hatx_words += trunc_batch.stats().max_bucket_hatx;
    totals_.coeff_flushes += ps.coeff_flushes;
    totals_.trunc_flushes += ps.trunc_flushes;
    enforce_limits();
  }

  template <typename PfssChanT>
  void barrier(int party,
               proto::PfssBackendBatch& backend,
               PfssSuperBatch& coeff_batch,
               PfssSuperBatch& trunc_batch,
               PfssChanT& pfss_ch,
               OpenCollector* opens,
               net::Chan* net_ch,
               const BarrierPolicy& policy) {
    auto flush_open = [&](OpenCollector* oc, net::Chan* nc) {
      if (!oc || !nc) return;
      if (oc->has_pending()) {
        oc->flush(party, *nc);
      }
    };
    auto flush_pfss = [&](PfssSuperBatch& b) {
      if (b.has_pending()) {
        b.flush_eval(party, backend, pfss_ch);
      }
      if (b.has_flushed()) {
        b.finalize_all(party, pfss_ch);
      }
    };
    if (policy.drain_all || policy.drain_open) {
      flush_open(opens, net_ch);
    }
    if (policy.drain_all || policy.drain_pfss_coeff) {
      flush_pfss(coeff_batch);
    }
    if (policy.drain_all || policy.drain_pfss_trunc) {
      if (&trunc_batch != &coeff_batch) flush_pfss(trunc_batch);
    }
    // We do not clear batches here; callers can keep slots alive across phases.
    // Totals will be updated when finalize_layer() is invoked.
  }

  // Optional safety flush at layer end; useful when phases forgot to clear.
  template <typename PfssChanT>
  void finalize_layer(int party,
                      proto::PfssBackendBatch& backend,
                      PfssSuperBatch& coeff_batch,
                      PfssSuperBatch& trunc_batch,
                      PfssChanT& pfss_ch,
                      PfssAsyncRunner* async_runner = nullptr,
                      bool wait = true,
                      std::mutex* chan_mu = nullptr) {
    auto pending_snapshot = [&](PfssSuperBatch& b) {
      struct Pending {
        size_t jobs = 0;
        size_t hatx_words = 0;
        size_t flushes = 0;
      };
      Pending p;
      const auto& st = b.stats();
      p.jobs = st.jobs + st.pending_jobs;
      p.hatx_words = std::max(st.max_bucket_hatx, st.pending_hatx);
      if (b.has_pending()) p.flushes += 1;
      p.flushes += st.flushes;
      return p;
    };

    if (async_runner) {
      auto coeff_pending = pending_snapshot(coeff_batch);
      std::optional<decltype(coeff_pending)> trunc_pending;
      if (&trunc_batch != &coeff_batch) trunc_pending = pending_snapshot(trunc_batch);
      bool spawn_async = !wait;
      async_runner->start_flush(party, backend, coeff_batch, &trunc_batch, pfss_ch, spawn_async, chan_mu);
      if (wait) {
        auto stats = async_runner->take_stats();
        if (stats) {
          totals_.coeff_jobs += stats->coeff.jobs;
          totals_.coeff_hatx_words += stats->coeff.max_bucket_hatx;
          totals_.coeff_flushes += stats->coeff.flushes;
          if (&trunc_batch != &coeff_batch) {
            totals_.trunc_jobs += stats->trunc.jobs;
            totals_.trunc_hatx_words += stats->trunc.max_bucket_hatx;
            totals_.trunc_flushes += stats->trunc.flushes;
          }
        }
      } else {
        totals_.coeff_jobs += coeff_pending.jobs;
        totals_.coeff_hatx_words += coeff_pending.hatx_words;
        totals_.coeff_flushes += coeff_pending.flushes;
        if (&trunc_batch != &coeff_batch && trunc_pending) {
          totals_.trunc_jobs += trunc_pending->jobs;
          totals_.trunc_hatx_words += trunc_pending->hatx_words;
          totals_.trunc_flushes += trunc_pending->flushes;
        }
      }
    } else {
      auto flush_once = [&](PfssSuperBatch& b, size_t& flush_counter, size_t& jobs_counter, size_t& hatx_counter) {
        if (b.has_pending()) {
          b.flush_eval(party, backend, pfss_ch);
        }
        if (b.has_flushed()) {
          b.finalize_all(party, pfss_ch);
        }
        jobs_counter += b.stats().jobs;
        hatx_counter += b.stats().max_bucket_hatx;
        flush_counter += b.stats().flushes;
        b.clear();
      };
      flush_once(coeff_batch, totals_.coeff_flushes, totals_.coeff_jobs, totals_.coeff_hatx_words);
      if (&trunc_batch != &coeff_batch) {
        flush_once(trunc_batch, totals_.trunc_flushes, totals_.trunc_jobs, totals_.trunc_hatx_words);
      }
    }
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
