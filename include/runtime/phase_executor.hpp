#pragma once

#include "runtime/pfss_superbatch.hpp"

namespace runtime {

// Minimal phase executor that batches PFSS composite jobs and flushes at
// explicit phase boundaries. Open fusion can be added later.
class PhaseExecutor {
 public:
  void begin_phase() {}

  template <typename ChannelT>
  void flush_phase(int party, proto::PfssBackendBatch& backend, ChannelT& ch) {
    if (!pfss_.empty()) {
      pfss_.flush_and_finalize(party, backend, ch);
      pfss_.clear();
    }
  }

  PfssSuperBatch& pfss_batch() { return pfss_; }

 private:
  PfssSuperBatch pfss_;
};

}  // namespace runtime
