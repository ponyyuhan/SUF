#pragma once

#include <stdexcept>

#include "runtime/pfss_superbatch.hpp"
#include "runtime/open_collector.hpp"
#include "runtime/pfss_phase_planner.hpp"

namespace runtime {

struct PhaseResources {
  int party = 0;
  proto::PfssBackendBatch* pfss_backend = nullptr;
  proto::IChannel* pfss_chan = nullptr;
  net::Chan* net_chan = nullptr;
  PfssSuperBatch* pfss_coeff = nullptr;
  PfssSuperBatch* pfss_trunc = nullptr;
  OpenCollector* opens = nullptr;
  PfssPhasePlanner* pfss_planner = nullptr;  // optional single-flush planner per phase
};

// Multi-wave phase executor: drives PhaseTasks that enqueue PFSS/open work.
namespace detail {
enum class Need : uint8_t { None, Open, PfssCoeff, PfssTrunc };
struct PhaseTask {
  virtual ~PhaseTask() = default;
  virtual bool done() const = 0;
  virtual Need step(PhaseResources& R) = 0;  // enqueue or consume; return requested flush kind
};
}  // namespace detail

class PhaseExecutor {
 public:
  PhaseExecutor() {
    PfssSuperBatch::Limits pfss_lim;
    pfss_lim.max_pending_jobs = 1ull << 12;
    pfss_lim.max_pending_hatx_words = 1ull << 20;
    pfss_lim.max_pending_hatx_bytes = pfss_lim.max_pending_hatx_words * sizeof(uint64_t);
    pfss_lim.max_flushes = 1ull << 9;
    pfss_coeff_.set_limits(pfss_lim);
    pfss_trunc_.set_limits(pfss_lim);
    OpenCollector::Limits open_lim;
    open_lim.max_pending_words = 1ull << 19;
    opens_.set_limits(open_lim);
    max_flushes_ = 1ull << 11;
  }

  enum class Phase : int {
    kLN1 = 0,
    kQKV_Score = 1,
    kSoftmax = 2,
    kOutProj = 3,
    kLN2_MLP = 4
  };

  template <typename TaskT>
  void add_task(std::unique_ptr<TaskT> t) {
    tasks_.push_back(std::move(t));
  }

  void begin_phase(Phase, bool clear_tasks = true) {
    if (clear_tasks) tasks_.clear();
  }

  struct Stats {
    size_t open_flushes = 0;
    size_t pfss_coeff_flushes = 0;
    size_t pfss_trunc_flushes = 0;
    size_t opened_words = 0;
    size_t pfss_coeff_jobs = 0;
    size_t pfss_coeff_arith_words = 0;
    size_t pfss_coeff_pred_bits = 0;
    size_t pfss_coeff_hatx_words = 0;
    size_t pfss_coeff_hatx_bytes = 0;
    size_t pfss_trunc_jobs = 0;
    size_t pfss_trunc_arith_words = 0;
    size_t pfss_trunc_pred_bits = 0;
    size_t pfss_trunc_hatx_words = 0;
    size_t pfss_trunc_hatx_bytes = 0;
  };

  // Legacy single-flush path (used by existing callers).
  template <typename PfssChanT>
  void flush_phase(int party, proto::PfssBackendBatch& backend, PfssChanT& pfss_ch, net::Chan& net_ch) {
    if (!pfss_coeff_.empty()) {
      pfss_coeff_.flush_and_finalize(party, backend, pfss_ch);
      if (!keep_batches_) pfss_coeff_.clear();
    }
    if (!pfss_trunc_.empty()) {
      pfss_trunc_.flush_and_finalize(party, backend, pfss_ch);
      if (!keep_batches_) pfss_trunc_.clear();
    }
    if (!opens_.empty()) {
      opens_.flush(party, net_ch);
      if (!keep_batches_) opens_.clear();
    }
  }

  PfssSuperBatch& pfss_coeff_batch() { return pfss_coeff_; }
  PfssSuperBatch& pfss_trunc_batch() { return pfss_trunc_; }
  OpenCollector& open_collector() { return opens_; }
  const Stats& stats() const { return stats_; }
  void set_max_flushes(size_t max_flushes) { max_flushes_ = max_flushes; }
  void clear_batches() {
    pfss_coeff_.clear();
    pfss_trunc_.clear();
    opens_.clear();
  }
  void reset_stats() {
    stats_ = Stats{};
    pfss_coeff_.reset_stats();
    pfss_trunc_.reset_stats();
    opens_.reset_stats();
  }

  struct LazyLimits {
    size_t open_pending_words = 1ull << 18;
    size_t coeff_pending_jobs = 1ull << 12;
    size_t trunc_pending_jobs = 1ull << 12;
    size_t hatx_pending_words = 1ull << 20;
  };

  void set_lazy_mode(bool enable) { lazy_mode_ = enable; }
  void set_lazy_limits(const LazyLimits& lim) { lazy_limits_ = lim; }
  void set_keep_batches(bool keep) { keep_batches_ = keep; }

  template <typename PfssChanT>
  void finalize_pfss_once(int party, proto::PfssBackendBatch& backend, PfssChanT& pfss_ch) {
    auto flush_once = [&](PfssSuperBatch& b) {
      if (b.has_pending()) b.flush_eval(party, backend, pfss_ch);
      if (b.has_flushed()) b.finalize_all(party, pfss_ch);
      b.clear();
    };
    flush_once(pfss_coeff_);
    if (&pfss_trunc_ != &pfss_coeff_) flush_once(pfss_trunc_);
  }

  void run(PhaseResources& R) {
    if (lazy_mode_) {
      run_lazy(R);
      return;
    }
    size_t flush_guard = 0;
    bool planner_flushed = false;
    for (;;) {
      bool any_not_done = false;
      bool want_open = false;
      bool want_pfss_coeff = false;
      bool want_pfss_trunc = false;
      bool progressed = false;
      for (auto& t : tasks_) {
        if (t->done()) continue;
        any_not_done = true;
        auto need = t->step(R);
        if (need == detail::Need::Open) want_open = true;
        if (need == detail::Need::PfssCoeff) want_pfss_coeff = true;
        if (need == detail::Need::PfssTrunc) want_pfss_trunc = true;
        if (need == detail::Need::None) progressed = true;
      }
      if (!any_not_done) break;

      auto do_flush_open = [&]() -> bool {
        if (!R.opens || !R.net_chan || !R.opens->has_pending()) return false;
        if (flush_guard + 1 > max_flushes_) {
          throw std::runtime_error("PhaseExecutor: open flush budget exceeded");
        }
        R.opens->flush(R.party, *R.net_chan);
        flush_guard++;
        return true;
      };
      auto do_flush_pfss = [&](PfssSuperBatch* b) -> bool {
        if (!b || !R.pfss_backend || !R.pfss_chan) return false;
        bool did = false;
        if (b->has_pending()) {
          if (flush_guard + 1 > max_flushes_) {
            throw std::runtime_error("PhaseExecutor: PFSS flush budget exceeded");
          }
          b->flush_eval(R.party, *R.pfss_backend, *R.pfss_chan);
          flush_guard++;
          did = true;
        }
        if (b->has_flushed()) {
          if (flush_guard + 1 > max_flushes_) {
            throw std::runtime_error("PhaseExecutor: PFSS finalize budget exceeded");
          }
          b->finalize_all(R.party, *R.pfss_chan);
          flush_guard++;
          did = true;
        }
        return did;
      };

      bool budget_open = R.opens &&
                         lazy_limits_.open_pending_words > 0 &&
                         R.opens->pending_words() >= lazy_limits_.open_pending_words;
      auto coeff_stats = pfss_coeff_.stats();
      auto trunc_stats = pfss_trunc_.stats();
      bool budget_coeff = lazy_limits_.coeff_pending_jobs > 0 &&
                          coeff_stats.pending_jobs >= lazy_limits_.coeff_pending_jobs;
      bool budget_trunc = lazy_limits_.trunc_pending_jobs > 0 &&
                          trunc_stats.pending_jobs >= lazy_limits_.trunc_pending_jobs;
      bool budget_hatx = lazy_limits_.hatx_pending_words > 0 &&
                         coeff_stats.pending_hatx >= lazy_limits_.hatx_pending_words;

      if (progressed) {
        if (budget_open) {
          do_flush_open();
        } else if (budget_coeff || budget_hatx) {
          do_flush_pfss(&pfss_coeff_);
        } else if (budget_trunc) {
          do_flush_pfss(&pfss_trunc_);
        }
        continue;
      }

      // Deadlock handling: nothing progressed; force flushes on demand.
      if (want_open && do_flush_open()) continue;
      if (want_pfss_coeff && do_flush_pfss(&pfss_coeff_)) continue;
      if (want_pfss_trunc && do_flush_pfss(&pfss_trunc_)) continue;
      if (R.pfss_planner && (want_pfss_coeff || want_pfss_trunc)) {
        if (!R.pfss_backend || !R.pfss_chan) {
          throw std::runtime_error("PhaseExecutor: PFSS planner missing backend/channel");
        }
        if (planner_flushed) {
          throw std::runtime_error("PhaseExecutor: planner single-flush budget exceeded");
        }
        if (flush_guard + 1 > max_flushes_) {
          throw std::runtime_error("PhaseExecutor: planner flush budget exceeded");
        }
        R.pfss_planner->finalize_phase(R.party, *R.pfss_backend, *R.pfss_chan);
        flush_guard++;
        planner_flushed = true;
        continue;
      }
      // Last chance: try flushing any pending batches even if tasks did not request it.
      if (do_flush_open()) continue;
      if (do_flush_pfss(&pfss_coeff_)) continue;
      if (do_flush_pfss(&pfss_trunc_)) continue;
      const auto pcs = pfss_coeff_.stats();
      const auto pts = pfss_trunc_.stats();
      std::string msg = "PhaseExecutor deadlock: tasks waiting but no pending flush";
      msg += " want_open=" + std::to_string(want_open);
      msg += " want_coeff=" + std::to_string(want_pfss_coeff);
      msg += " want_trunc=" + std::to_string(want_pfss_trunc);
      msg += " coeff_pending_jobs=" + std::to_string(pcs.pending_jobs);
      msg += " trunc_pending_jobs=" + std::to_string(pts.pending_jobs);
      msg += " coeff_pending_hatx=" + std::to_string(pcs.pending_hatx);
      msg += " trunc_pending_hatx=" + std::to_string(pts.pending_hatx);
      msg += " coeff_flushes=" + std::to_string(pcs.flushes);
      msg += " trunc_flushes=" + std::to_string(pts.flushes);
      throw std::runtime_error(msg);
    }
    const auto& os = opens_.stats();
    stats_.open_flushes = os.flushes;
    stats_.opened_words = os.opened_words;
    const auto& pcs = pfss_coeff_.stats();
    const auto& pts = pfss_trunc_.stats();
    stats_.pfss_coeff_flushes = pcs.flushes;
    stats_.pfss_coeff_jobs = pcs.jobs;
    stats_.pfss_coeff_arith_words = pcs.arith_words;
    stats_.pfss_coeff_pred_bits = pcs.pred_bits;
    stats_.pfss_coeff_hatx_words = pcs.hatx_words;
    stats_.pfss_coeff_hatx_bytes = pcs.hatx_bytes;
    stats_.pfss_trunc_flushes = pts.flushes;
    stats_.pfss_trunc_jobs = pts.jobs;
    stats_.pfss_trunc_arith_words = pts.arith_words;
    stats_.pfss_trunc_pred_bits = pts.pred_bits;
    stats_.pfss_trunc_hatx_words = pts.hatx_words;
    stats_.pfss_trunc_hatx_bytes = pts.hatx_bytes;

    // Clear batches/opens for next phase unless caller wants to keep them alive across phases.
    if (!keep_batches_) {
      pfss_coeff_.clear();
      pfss_trunc_.clear();
      opens_.clear();
    }
  }

 private:
  void run_lazy(PhaseResources& R) {
    size_t flush_guard = 0;
    for (;;) {
      bool any_not_done = false;
      bool want_open = false;
      bool want_pfss_coeff = false;
      bool want_pfss_trunc = false;
      bool progressed = false;
      for (auto& t : tasks_) {
        if (t->done()) continue;
        any_not_done = true;
        auto need = t->step(R);
        if (need == detail::Need::Open) want_open = true;
        if (need == detail::Need::PfssCoeff) want_pfss_coeff = true;
        if (need == detail::Need::PfssTrunc) want_pfss_trunc = true;
        if (need == detail::Need::None) progressed = true;
      }
      if (!any_not_done) break;

      auto do_flush_open = [&]() -> bool {
        if (!R.opens || !R.net_chan || !R.opens->has_pending()) return false;
        if (flush_guard + 1 > max_flushes_) {
          throw std::runtime_error("PhaseExecutor: open flush budget exceeded");
        }
        R.opens->flush(R.party, *R.net_chan);
        flush_guard++;
        return true;
      };
      auto do_flush_pfss = [&](PfssSuperBatch* b) -> bool {
        if (!b || !R.pfss_backend || !R.pfss_chan) return false;
        bool did = false;
        if (b->has_pending()) {
          if (flush_guard + 1 > max_flushes_) {
            throw std::runtime_error("PhaseExecutor: PFSS flush budget exceeded");
          }
          b->flush_eval(R.party, *R.pfss_backend, *R.pfss_chan);
          flush_guard++;
          did = true;
        }
        if (b->has_flushed()) {
          if (flush_guard + 1 > max_flushes_) {
            throw std::runtime_error("PhaseExecutor: PFSS finalize budget exceeded");
          }
          b->finalize_all(R.party, *R.pfss_chan);
          flush_guard++;
          did = true;
        }
        return did;
      };

      bool budget_open = R.opens &&
                         lazy_limits_.open_pending_words > 0 &&
                         R.opens->pending_words() >= lazy_limits_.open_pending_words;
      auto coeff_stats = pfss_coeff_.stats();
      auto trunc_stats = pfss_trunc_.stats();
      bool budget_coeff = lazy_limits_.coeff_pending_jobs > 0 &&
                          coeff_stats.pending_jobs >= lazy_limits_.coeff_pending_jobs;
      bool budget_trunc = lazy_limits_.trunc_pending_jobs > 0 &&
                          trunc_stats.pending_jobs >= lazy_limits_.trunc_pending_jobs;
      bool budget_hatx = lazy_limits_.hatx_pending_words > 0 &&
                         coeff_stats.pending_hatx >= lazy_limits_.hatx_pending_words;

      if (progressed) {
        if (budget_open) {
          do_flush_open();
        } else if (budget_coeff || budget_hatx) {
          do_flush_pfss(&pfss_coeff_);
        } else if (budget_trunc) {
          do_flush_pfss(&pfss_trunc_);
        }
        continue;
      }

      if (want_open && do_flush_open()) continue;
      if (want_pfss_coeff && do_flush_pfss(&pfss_coeff_)) continue;
      if (want_pfss_trunc && do_flush_pfss(&pfss_trunc_)) continue;
      if (do_flush_open()) continue;
      if (do_flush_pfss(&pfss_coeff_)) continue;
      if (do_flush_pfss(&pfss_trunc_)) continue;
      const auto pcs = pfss_coeff_.stats();
      const auto pts = pfss_trunc_.stats();
      std::string msg = "PhaseExecutor deadlock (lazy): tasks waiting but no pending flush";
      msg += " want_open=" + std::to_string(want_open);
      msg += " want_coeff=" + std::to_string(want_pfss_coeff);
      msg += " want_trunc=" + std::to_string(want_pfss_trunc);
      msg += " coeff_pending_jobs=" + std::to_string(pcs.pending_jobs);
      msg += " trunc_pending_jobs=" + std::to_string(pts.pending_jobs);
      msg += " coeff_pending_hatx=" + std::to_string(pcs.pending_hatx);
      msg += " trunc_pending_hatx=" + std::to_string(pts.pending_hatx);
      msg += " coeff_flushes=" + std::to_string(pcs.flushes);
      msg += " trunc_flushes=" + std::to_string(pts.flushes);
      throw std::runtime_error(msg);
    }

    const auto& os = opens_.stats();
    stats_.open_flushes = os.flushes;
    stats_.opened_words = os.opened_words;
    const auto& pcs = pfss_coeff_.stats();
    const auto& pts = pfss_trunc_.stats();
    stats_.pfss_coeff_flushes = pcs.flushes;
    stats_.pfss_coeff_jobs = pcs.jobs;
    stats_.pfss_coeff_arith_words = pcs.arith_words;
    stats_.pfss_coeff_pred_bits = pcs.pred_bits;
    stats_.pfss_coeff_hatx_words = pcs.hatx_words;
    stats_.pfss_coeff_hatx_bytes = pcs.hatx_bytes;
    stats_.pfss_trunc_flushes = pts.flushes;
    stats_.pfss_trunc_jobs = pts.jobs;
    stats_.pfss_trunc_arith_words = pts.arith_words;
    stats_.pfss_trunc_pred_bits = pts.pred_bits;
    stats_.pfss_trunc_hatx_words = pts.hatx_words;
    stats_.pfss_trunc_hatx_bytes = pts.hatx_bytes;

    if (!keep_batches_) {
      pfss_coeff_.clear();
      pfss_trunc_.clear();
      opens_.clear();
    }
  }

  std::vector<std::unique_ptr<detail::PhaseTask>> tasks_;
  PfssSuperBatch pfss_coeff_;
  PfssSuperBatch pfss_trunc_;
  OpenCollector opens_;
  Stats stats_;
  size_t max_flushes_ = 1ull << 16;  // safety guard
  bool lazy_mode_ = true;
  bool keep_batches_ = true;
  LazyLimits lazy_limits_{};
};

}  // namespace runtime
