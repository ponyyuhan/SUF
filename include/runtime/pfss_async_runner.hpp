#pragma once

#include <thread>
#include <utility>
#include <stdexcept>

#include "runtime/pfss_superbatch.hpp"

namespace runtime {

// Minimal async helper for PFSS flush/finalize. By default you can use it
// synchronously (async=false) to keep semantics unchanged; async=true will
// spawn a worker thread and join on wait().
class PfssAsyncRunner {
 public:
  PfssAsyncRunner() = default;
  PfssAsyncRunner(const PfssAsyncRunner&) = delete;
  PfssAsyncRunner& operator=(const PfssAsyncRunner&) = delete;
  ~PfssAsyncRunner() { wait(); }

  template <typename PfssChanT>
  void start_flush(int party,
                   proto::PfssBackendBatch& backend,
                   PfssSuperBatch& coeff_batch,
                   PfssSuperBatch* trunc_batch,
                   PfssChanT& pfss_ch,
                   bool async = false) {
    auto runner = [=, &backend, &coeff_batch, trunc_batch, &pfss_ch]() {
      auto flush_once = [&](PfssSuperBatch& b) {
        if (b.has_pending()) b.flush_eval(party, backend, pfss_ch);
        if (b.has_flushed()) b.finalize_all(party, pfss_ch);
      };
      flush_once(coeff_batch);
      if (trunc_batch && trunc_batch != &coeff_batch) flush_once(*trunc_batch);
    };
    if (async) {
      wait();
      worker_ = std::thread(runner);
    } else {
      runner();
    }
  }

  void wait() {
    if (worker_.joinable()) worker_.join();
  }

 private:
  std::thread worker_;
};

}  // namespace runtime
