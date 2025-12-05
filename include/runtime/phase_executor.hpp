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
  PfssSuperBatch* pfss = nullptr;
  OpenCollector* opens = nullptr;
};

// Multi-wave phase executor: drives PhaseTasks that enqueue PFSS/open work.
namespace detail {
enum class Need : uint8_t { None, Open, Pfss };
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

  // Legacy single-flush path (used by existing callers).
  template <typename PfssChanT>
  void flush_phase(int party, proto::PfssBackendBatch& backend, PfssChanT& pfss_ch, net::Chan& net_ch) {
    if (!pfss_.empty()) {
      pfss_.flush_and_finalize(party, backend, pfss_ch);
      pfss_.clear();
    }
    if (!opens_.empty()) {
      opens_.flush(party, net_ch);
      opens_.clear();
    }
  }

  PfssSuperBatch& pfss_batch() { return pfss_; }
  OpenCollector& open_collector() { return opens_; }

  void run(PhaseResources& R) {
    for (;;) {
      bool any_not_done = false;
      bool want_open = false;
      bool want_pfss = false;
      bool progressed = false;
      for (auto& t : tasks_) {
        if (t->done()) continue;
        any_not_done = true;
        auto need = t->step(R);
        if (need == detail::Need::Open) want_open = true;
        if (need == detail::Need::Pfss) want_pfss = true;
      }
      if (!any_not_done) break;
      if (want_open && R.opens && R.net_chan && R.opens->has_pending()) {
        R.opens->flush(R.party, *R.net_chan);
        progressed = true;
      }
      if (want_pfss && R.pfss && R.pfss_backend && R.pfss_chan && R.pfss->has_pending()) {
        R.pfss->flush(R.party, *R.pfss_backend, *R.pfss_chan);
        progressed = true;
      }
      if (!progressed) {
        throw std::runtime_error("PhaseExecutor deadlock: tasks waiting but no pending flush");
      }
    }
    tasks_.clear();
  }

 private:
  std::vector<std::unique_ptr<detail::PhaseTask>> tasks_;
  PfssSuperBatch pfss_;
  OpenCollector opens_;
};

}  // namespace runtime
