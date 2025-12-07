#pragma once

#include <thread>
#include <utility>
#include <stdexcept>
#include <mutex>
#include <optional>

#include "runtime/pfss_superbatch.hpp"

namespace runtime {

// Minimal async helper for PFSS flush/finalize. By default you can use it
// synchronously (async=false) to keep semantics unchanged; async=true will
// spawn a worker thread and join on wait().
class PfssAsyncRunner {
 public:
  struct ResultStats {
    PfssSuperBatch::Stats coeff;
    PfssSuperBatch::Stats trunc;
    bool has_trunc = false;
  };

  PfssAsyncRunner() = default;
  PfssAsyncRunner(const PfssAsyncRunner&) = delete;
  PfssAsyncRunner& operator=(const PfssAsyncRunner&) = delete;
  ~PfssAsyncRunner() { wait(); }

  template <typename PfssChanT>
  void start_flush(int party,
                   proto::PfssBackendBatch& backend,
                   PfssSuperBatch& coeff_batch,
                   PfssSuperBatch* trunc_batch,
                   PfssChanT pfss_ch,
                   bool async = false,
                   std::mutex* chan_mu = nullptr) {
    auto runner = [=, &backend, &coeff_batch, trunc_batch, pfss_ch, chan_mu, this]() mutable {
      std::unique_lock<std::mutex> lk;
      if (chan_mu) lk = std::unique_lock<std::mutex>(*chan_mu);
      ResultStats stats_out{};
      auto flush_once = [&](PfssSuperBatch& b, PfssSuperBatch::Stats& dst_stats) {
        if (b.has_pending()) b.flush_eval(party, backend, pfss_ch);
        if (b.has_flushed()) b.finalize_all(party, pfss_ch);
        dst_stats = b.stats();
        b.clear();
      };
      flush_once(coeff_batch, stats_out.coeff);
      if (trunc_batch && trunc_batch != &coeff_batch) {
        stats_out.has_trunc = true;
        flush_once(*trunc_batch, stats_out.trunc);
      }
      {
        std::lock_guard<std::mutex> g(stats_mu_);
        last_stats_ = stats_out;
      }
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

  std::optional<ResultStats> take_stats() {
    wait();
    std::lock_guard<std::mutex> g(stats_mu_);
    return last_stats_;
  }

 private:
  std::thread worker_;
  std::optional<ResultStats> last_stats_;
  std::mutex stats_mu_;
};

}  // namespace runtime
