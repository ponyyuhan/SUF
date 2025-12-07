#pragma once

#include <vector>
#include <memory>
#include <stdexcept>

#include "runtime/phase_tasks.hpp"

namespace runtime {

// Simple two-phase executor: tasks call prepare() to enqueue PFSS/Open,
// executor flushes once, then tasks call finalize().
class StagedExecutor {
 public:
  void add_task(std::unique_ptr<detail::PhaseTask> t) { tasks_.push_back(std::move(t)); }

  template <typename PfssChanT>
  void run(PhaseResources& R, proto::PfssBackendBatch& backend, PfssChanT& pfss_ch) {
    // Drive tasks until all are done, flushing PFSS/Open when requested.
    while (true) {
      bool need_pfss = false;
      bool need_open = false;
      bool progressed = false;
      for (auto& t : tasks_) {
        if (t->done()) continue;
        auto need = t->step(R);
        switch (need) {
          case detail::Need::PfssCoeff:
          case detail::Need::PfssTrunc:
            need_pfss = true;
            break;
          case detail::Need::Open:
            need_open = true;
            break;
          case detail::Need::None:
            progressed = true;
            break;
        }
      }
      if (need_open) {
        if (!R.opens || !R.net_chan) throw std::runtime_error("StagedExecutor: missing open resources");
        if (R.opens->has_pending()) {
          R.opens->flush(R.party, *R.net_chan);
          progressed = true;
        }
      }
      if (need_pfss) {
        if (!R.pfss_coeff || !R.pfss_trunc) throw std::runtime_error("StagedExecutor: missing PFSS batches");
        if (R.pfss_coeff->has_pending()) {
          R.pfss_coeff->flush_eval(R.party, backend, pfss_ch);
          progressed = true;
        }
        if (R.pfss_trunc->has_pending()) {
          R.pfss_trunc->flush_eval(R.party, backend, pfss_ch);
          progressed = true;
        }
        if (R.pfss_coeff->has_flushed()) {
          R.pfss_coeff->finalize_all(R.party, pfss_ch);
          progressed = true;
        }
        if (R.pfss_trunc->has_flushed()) {
          R.pfss_trunc->finalize_all(R.party, pfss_ch);
          progressed = true;
        }
      }
      bool all_done = true;
      for (auto& t : tasks_) {
        if (!t->done()) {
          all_done = false;
          break;
        }
      }
      if (all_done) break;
      if (!progressed && !need_pfss && !need_open) {
        throw std::runtime_error("StagedExecutor: no progress; task may be stuck");
      }
    }
    tasks_.clear();
  }

 private:
  std::vector<std::unique_ptr<detail::PhaseTask>> tasks_;
};

}  // namespace runtime
