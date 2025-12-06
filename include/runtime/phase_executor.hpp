#pragma once

#include <stdexcept>

#include "runtime/pfss_superbatch.hpp"
#include "runtime/open_collector.hpp"

namespace runtime {

struct PhaseResources {
  int party = 0;
  proto::PfssBackendBatch* pfss_backend = nullptr;
  proto::IChannel* pfss_chan = nullptr;
  net::Chan* net_chan = nullptr;
  PfssSuperBatch* pfss_coeff = nullptr;
  PfssSuperBatch* pfss_trunc = nullptr;
  OpenCollector* opens = nullptr;
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

  void begin_phase(Phase) {
    tasks_.clear();
  }

  struct Stats {
    size_t open_flushes = 0;
    size_t pfss_coeff_flushes = 0;
    size_t pfss_trunc_flushes = 0;
    size_t opened_words = 0;
    size_t pfss_coeff_jobs = 0;
    size_t pfss_coeff_arith_words = 0;
    size_t pfss_coeff_pred_bits = 0;
    size_t pfss_trunc_jobs = 0;
    size_t pfss_trunc_arith_words = 0;
    size_t pfss_trunc_pred_bits = 0;
  };

  // Legacy single-flush path (used by existing callers).
  template <typename PfssChanT>
  void flush_phase(int party, proto::PfssBackendBatch& backend, PfssChanT& pfss_ch, net::Chan& net_ch) {
    if (!pfss_coeff_.empty()) {
      pfss_coeff_.flush_and_finalize(party, backend, pfss_ch);
      pfss_coeff_.clear();
    }
    if (!pfss_trunc_.empty()) {
      pfss_trunc_.flush_and_finalize(party, backend, pfss_ch);
      pfss_trunc_.clear();
    }
    if (!opens_.empty()) {
      opens_.flush(party, net_ch);
      opens_.clear();
    }
  }

  PfssSuperBatch& pfss_coeff_batch() { return pfss_coeff_; }
  PfssSuperBatch& pfss_trunc_batch() { return pfss_trunc_; }
  OpenCollector& open_collector() { return opens_; }
  const Stats& stats() const { return stats_; }
  void reset_stats() {
    stats_ = Stats{};
    pfss_coeff_.reset_stats();
    pfss_trunc_.reset_stats();
    opens_.reset_stats();
  }

  void run(PhaseResources& R) {
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
      if (want_open && R.opens && R.net_chan && R.opens->has_pending()) {
        R.opens->flush(R.party, *R.net_chan);
        progressed = true;
      }
      if (want_pfss_coeff && R.pfss_coeff && R.pfss_backend && R.pfss_chan &&
          R.pfss_coeff->has_pending()) {
        R.pfss_coeff->flush_eval(R.party, *R.pfss_backend, *R.pfss_chan);
        progressed = true;
      }
      if (want_pfss_trunc && R.pfss_trunc && R.pfss_backend && R.pfss_chan &&
          R.pfss_trunc->has_pending()) {
        R.pfss_trunc->flush_eval(R.party, *R.pfss_backend, *R.pfss_chan);
        progressed = true;
      }
      if (!progressed) {
        throw std::runtime_error("PhaseExecutor deadlock: tasks waiting but no pending flush");
      }
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
    stats_.pfss_trunc_flushes = pts.flushes;
    stats_.pfss_trunc_jobs = pts.jobs;
    stats_.pfss_trunc_arith_words = pts.arith_words;
    stats_.pfss_trunc_pred_bits = pts.pred_bits;

    // Clear batches/opens for next phase; stats already recorded.
    pfss_coeff_.clear();
    pfss_trunc_.clear();
    opens_.clear();
  }

 private:
  std::vector<std::unique_ptr<detail::PhaseTask>> tasks_;
  PfssSuperBatch pfss_coeff_;
  PfssSuperBatch pfss_trunc_;
  OpenCollector opens_;
  Stats stats_;
};

}  // namespace runtime
