#pragma once

#include "runtime/pfss_superbatch.hpp"
#include "runtime/open_collector.hpp"

namespace runtime {

// Minimal phase executor that batches PFSS composite jobs and Beaver opens and
// flushes at explicit phase boundaries.
class PhaseExecutor {
 public:
  enum class Phase : int {
    kLN1 = 0,
    kQKV_Score = 1,
    kSoftmax = 2,
    kOutProj = 3,
    kLN2_MLP = 4
  };

  void begin_phase(Phase) {}

  template <typename ChannelT>
  void flush_phase(int party, proto::PfssBackendBatch& backend, ChannelT& ch) {
    if (!pfss_.empty()) {
      pfss_.flush_and_finalize(party, backend, ch);
      pfss_.clear();
    }
    if (!opens_.empty()) {
      opens_.flush(party, ch);
      opens_.clear();
    }
  }

  PfssSuperBatch& pfss_batch() { return pfss_; }
  OpenCollector& open_collector() { return opens_; }

  void clear() {
    pfss_.clear();
    opens_.clear();
  }

 private:
  PfssSuperBatch pfss_;
  OpenCollector opens_;
};

}  // namespace runtime
