#pragma once

#include <stdexcept>
#include <thread>
#include <exception>
#include <algorithm>
#include <cctype>
#include <string>

#include "runtime/pfss_superbatch.hpp"
#include "runtime/open_collector.hpp"
#include "runtime/pfss_phase_planner.hpp"
#include "proto/backend_gpu.hpp"

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
  bool device_pipeline = false;  // optional: keep PFSS outputs on device across phases
  // If true, PhaseExecutor may overlap PFSS flushing with open flushing in a
  // background thread. Safe only when PFSS uses a dedicated channel (i.e., not
  // a ProtoChanFromNet over the same net channel as OpenCollector).
  bool overlap_pfss_open = false;
  // Optional CUDA stream override for OpenCollector device packing/unpacking
  // even when the PFSS backend itself is CPU-based.
  void* cuda_stream = nullptr;  // cudaStream_t
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
    // Default to large batches: end-to-end transformer runs are dominated by
    // per-flush latency if we fragment opens/PFSS work too aggressively.
    pfss_lim.max_pending_jobs = 1ull << 18;
    pfss_lim.max_pending_hatx_words = 1ull << 24;
    pfss_lim.max_pending_hatx_bytes = pfss_lim.max_pending_hatx_words * sizeof(uint64_t);
    pfss_lim.max_flushes = 1ull << 16;
    pfss_coeff_.set_limits(pfss_lim);
    pfss_trunc_.set_limits(pfss_lim);
    OpenCollector::Limits open_lim;
    open_lim.max_pending_words = 1ull << 24;
    opens_.set_limits(open_lim);
    max_flushes_ = 1ull << 16;
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
    size_t opened_words_beaver = 0;
    size_t opened_words_mask = 0;
    size_t opened_words_other = 0;
    uint64_t open_wire_bytes_sent = 0;
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
      pfss_coeff_.flush_eval(party, backend, pfss_ch);
      pfss_coeff_.materialize_host(party, pfss_ch);
      if (!keep_batches_) pfss_coeff_.clear();
    }
    if (!pfss_trunc_.empty()) {
      pfss_trunc_.flush_eval(party, backend, pfss_ch);
      pfss_trunc_.materialize_host(party, pfss_ch);
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
    // Default batching thresholds. These are tuned for end-to-end transformer
    // runs where per-flush latency dominates if we flush too often.
    // (OpenCollector itself has a generous max_pending_words guardrail.)
    size_t open_pending_words = 1ull << 22;
    size_t coeff_pending_jobs = 1ull << 14;
    size_t trunc_pending_jobs = 1ull << 14;
    size_t hatx_pending_words = 1ull << 22;
  };

  void set_lazy_mode(bool enable) { lazy_mode_ = enable; }
  void set_lazy_limits(const LazyLimits& lim) { lazy_limits_ = lim; }
  void set_keep_batches(bool keep) { keep_batches_ = keep; }
  // Enable device pipeline: skip host materialization until caller explicitly
  // requests it (e.g., for device-only benches). Callers must later invoke
  // materialize_host() on the batches or manually clear them.
  void set_device_pipeline(bool enable) { device_pipeline_ = enable; }
  void set_device_pipeline_materialize(bool enable) { device_pipeline_materialize_ = enable; }

  template <typename PfssChanT>
  void finalize_pfss_once(int party, proto::PfssBackendBatch& backend, PfssChanT& pfss_ch) {
    auto flush_once = [&](PfssSuperBatch& b) {
      if (b.has_pending()) b.flush_eval(party, backend, pfss_ch);
      if (b.has_flushed() &&
          (!device_pipeline_ || device_pipeline_materialize_ ||
           (device_pipeline_ && b.needs_host_materialize()))) {
        b.materialize_host(party, pfss_ch);
      }
      if (b.has_flushed()) b.clear();
    };
    flush_once(pfss_coeff_);
    if (&pfss_trunc_ != &pfss_coeff_) flush_once(pfss_trunc_);
  }

  void run(PhaseResources& R) {
    R.device_pipeline = device_pipeline_;
    const bool allow_planner = (!device_pipeline_ || device_pipeline_materialize_);
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
#ifdef SUF_HAVE_CUDA
        if (R.pfss_backend) {
          if (auto* staged = dynamic_cast<proto::PfssGpuStagedEval*>(R.pfss_backend)) {
            R.opens->set_cuda_stream(staged->device_stream());
          }
        }
        if (R.cuda_stream) {
          R.opens->set_cuda_stream(R.cuda_stream);
        }
#endif
        // Opportunistically overlap a pending PFSS flush with the open flush when
        // the caller provided a dedicated PFSS channel (overlap_pfss_open=true).
        // This reduces effective round barriers in end-to-end transformer runs.
        bool open_uses_gpu = false;
#ifdef SUF_HAVE_CUDA
        // OpenCollector uses CUDA packing/scatter when a CUDA stream is present.
        // Overlapping PFSS (also GPU) with open flush tends to contend for the same
        // device and can *increase* end-to-end time, so only overlap when opens
        // are host-driven.
        const bool have_cuda_stream =
            (R.cuda_stream != nullptr) ||
            (R.pfss_backend && dynamic_cast<proto::PfssGpuStagedEval*>(R.pfss_backend) != nullptr);
        auto env_flag_enabled_default_local = [](const char* name, bool defv) -> bool {
          const char* env = std::getenv(name);
          if (!env) return defv;
          std::string v(env);
          std::transform(v.begin(), v.end(), v.begin(),
                         [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
          return !(v == "0" || v == "false" || v == "off" || v == "no");
        };
        const bool want_device_pack =
            env_flag_enabled_default_local("SUF_OPEN_PACK_DEVICE", have_cuda_stream);
        // OpenCollector's device pack path defaults to an internal non-blocking CUDA
        // stream (`cuda_pack_stream_`) unless `SUF_OPEN_PACK_USE_CALLER_STREAM=1`.
        // When it uses the internal stream, we can safely overlap PFSS flushing with
        // open packing/comm without serializing on the PFSS compute stream.
        const bool open_uses_caller_stream =
            env_flag_enabled_default_local("SUF_OPEN_PACK_USE_CALLER_STREAM", false);
        open_uses_gpu = have_cuda_stream && want_device_pack && open_uses_caller_stream;
#endif
        if (!open_uses_gpu &&
            R.overlap_pfss_open && R.pfss_backend && R.pfss_chan &&
            ((pfss_coeff_.has_pending() || pfss_coeff_.has_flushed()) ||
             (pfss_trunc_.has_pending() || pfss_trunc_.has_flushed()))) {
          std::exception_ptr pfss_exc;
          std::thread pfss_thread([&]() {
            try {
              auto flush_one = [&](PfssSuperBatch& b) {
                if (!R.pfss_backend || !R.pfss_chan) {
                  throw std::runtime_error("PhaseExecutor: PFSS flush missing backend/channel");
                }
                if (b.has_pending()) b.flush_eval(R.party, *R.pfss_backend, *R.pfss_chan);
                if (b.has_flushed()) {
                  if (!device_pipeline_ || device_pipeline_materialize_ ||
                      (device_pipeline_ && b.needs_host_materialize())) {
                    b.materialize_host(R.party, *R.pfss_chan);
                  }
                  b.clear();
                }
              };
              flush_one(pfss_coeff_);
              if (&pfss_trunc_ != &pfss_coeff_) flush_one(pfss_trunc_);
            } catch (...) {
              pfss_exc = std::current_exception();
            }
          });
          R.opens->flush(R.party, *R.net_chan);
          if (pfss_thread.joinable()) pfss_thread.join();
          if (pfss_exc) std::rethrow_exception(pfss_exc);
        } else {
          R.opens->flush(R.party, *R.net_chan);
        }
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
          if (!device_pipeline_ || device_pipeline_materialize_ ||
              (device_pipeline_ && b->needs_host_materialize())) {
            b->materialize_host(R.party, *R.pfss_chan);
          }
          b->clear();
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
      if (R.overlap_pfss_open && want_open && (want_pfss_coeff || want_pfss_trunc) &&
          R.opens && R.net_chan && R.opens->has_pending() &&
          ((pfss_coeff_.has_pending() || pfss_coeff_.has_flushed()) ||
           (pfss_trunc_.has_pending() || pfss_trunc_.has_flushed()))) {
        std::exception_ptr pfss_exc;
        std::thread pfss_thread([&]() {
          try {
            if (allow_planner && R.pfss_planner && (want_pfss_coeff || want_pfss_trunc) && !planner_flushed) {
              if (!R.pfss_backend || !R.pfss_chan) {
                throw std::runtime_error("PhaseExecutor: PFSS planner missing backend/channel");
              }
              R.pfss_planner->finalize_phase(R.party, *R.pfss_backend, *R.pfss_chan);
            } else {
              auto flush_one = [&](PfssSuperBatch& b) {
                if (!R.pfss_backend || !R.pfss_chan) {
                  throw std::runtime_error("PhaseExecutor: PFSS flush missing backend/channel");
                }
                if (b.has_pending()) b.flush_eval(R.party, *R.pfss_backend, *R.pfss_chan);
                if (b.has_flushed()) {
                  if (!device_pipeline_ || device_pipeline_materialize_ ||
                      (device_pipeline_ && b.needs_host_materialize())) {
                    b.materialize_host(R.party, *R.pfss_chan);
                  }
                  b.clear();
                }
              };
              if (want_pfss_coeff) flush_one(pfss_coeff_);
              if (want_pfss_trunc && &pfss_trunc_ != &pfss_coeff_) flush_one(pfss_trunc_);
            }
          } catch (...) {
            pfss_exc = std::current_exception();
          }
        });
        do_flush_open();
        if (pfss_thread.joinable()) pfss_thread.join();
        // Mark planner flushed so we don't run it twice in this phase wave.
        if (allow_planner && R.pfss_planner && (want_pfss_coeff || want_pfss_trunc) && !planner_flushed) {
          planner_flushed = true;
        }
        if (pfss_exc) std::rethrow_exception(pfss_exc);
        continue;
      }
      if (want_open && do_flush_open()) continue;
      if (allow_planner && R.pfss_planner && (want_pfss_coeff || want_pfss_trunc) && !planner_flushed) {
        if (!R.pfss_backend || !R.pfss_chan) {
          throw std::runtime_error("PhaseExecutor: PFSS planner missing backend/channel");
        }
        if (flush_guard + 1 > max_flushes_) {
          throw std::runtime_error("PhaseExecutor: planner flush budget exceeded");
        }
        // Prefer the planner when provided: it drains coeff+trunc in one grouped flush/finalize.
        R.pfss_planner->finalize_phase(R.party, *R.pfss_backend, *R.pfss_chan);
        flush_guard++;
        planner_flushed = true;
        continue;
      }
      if (want_pfss_coeff && do_flush_pfss(&pfss_coeff_)) continue;
      if (want_pfss_trunc && do_flush_pfss(&pfss_trunc_)) continue;
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
    stats_.opened_words_beaver =
        os.opened_words_by_kind[static_cast<size_t>(OpenKind::kBeaver)];
    stats_.opened_words_mask =
        os.opened_words_by_kind[static_cast<size_t>(OpenKind::kMask)];
    stats_.opened_words_other =
        os.opened_words_by_kind[static_cast<size_t>(OpenKind::kOther)];
    stats_.open_wire_bytes_sent = os.wire_bytes_sent;
    const auto& pcs = pfss_coeff_.total_stats();
    const auto& pts = pfss_trunc_.total_stats();
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
    bool planner_flushed = false;
    const bool allow_planner = (!device_pipeline_ || device_pipeline_materialize_);
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
#ifdef SUF_HAVE_CUDA
        if (R.pfss_backend) {
          if (auto* staged = dynamic_cast<proto::PfssGpuStagedEval*>(R.pfss_backend)) {
            R.opens->set_cuda_stream(staged->device_stream());
          }
        }
        if (R.cuda_stream) {
          R.opens->set_cuda_stream(R.cuda_stream);
        }
#endif
        // Opportunistically overlap a pending PFSS flush with the open flush when
        // the caller provided a dedicated PFSS channel (overlap_pfss_open=true).
        // This reduces effective round barriers in end-to-end transformer runs.
        bool allow_overlap = R.overlap_pfss_open;
        bool open_uses_gpu = false;
#ifdef SUF_HAVE_CUDA
        auto env_flag_enabled_default_local = [](const char* name, bool defv) -> bool {
          const char* env = std::getenv(name);
          if (!env) return defv;
          std::string v(env);
          std::transform(v.begin(), v.end(), v.begin(),
                         [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
          return !(v == "0" || v == "false" || v == "off" || v == "no");
        };
        allow_overlap = allow_overlap && env_flag_enabled_default_local("SUF_OVERLAP_PFSS_OPEN", true);
        const bool have_cuda_stream =
            (R.cuda_stream != nullptr) ||
            (R.pfss_backend && dynamic_cast<proto::PfssGpuStagedEval*>(R.pfss_backend) != nullptr);
        const bool want_device_pack =
            env_flag_enabled_default_local("SUF_OPEN_PACK_DEVICE", have_cuda_stream);
        const bool open_uses_caller_stream =
            env_flag_enabled_default_local("SUF_OPEN_PACK_USE_CALLER_STREAM", false);
        open_uses_gpu = have_cuda_stream && want_device_pack && open_uses_caller_stream;
        if (open_uses_gpu && !env_flag_enabled_default_local("SUF_OVERLAP_PFSS_OPEN_GPU", true)) {
          allow_overlap = false;
        }
#else
        (void)open_uses_gpu;
#endif
        if (allow_overlap &&
            R.pfss_backend && R.pfss_chan &&
            ((pfss_coeff_.has_pending() || pfss_coeff_.has_flushed()) ||
             (pfss_trunc_.has_pending() || pfss_trunc_.has_flushed()))) {
          std::exception_ptr pfss_exc;
          std::thread pfss_thread([&]() {
            try {
              auto flush_one = [&](PfssSuperBatch& b) {
                if (!R.pfss_backend || !R.pfss_chan) {
                  throw std::runtime_error("PhaseExecutor: PFSS flush missing backend/channel");
                }
                if (b.has_pending()) b.flush_eval(R.party, *R.pfss_backend, *R.pfss_chan);
                if (b.has_flushed()) {
                  if (!device_pipeline_ || device_pipeline_materialize_ ||
                      (device_pipeline_ && b.needs_host_materialize())) {
                    b.materialize_host(R.party, *R.pfss_chan);
                  }
                  b.clear();
                }
              };
              flush_one(pfss_coeff_);
              if (&pfss_trunc_ != &pfss_coeff_) flush_one(pfss_trunc_);
            } catch (...) {
              pfss_exc = std::current_exception();
            }
          });
          R.opens->flush(R.party, *R.net_chan);
          if (pfss_thread.joinable()) pfss_thread.join();
          if (pfss_exc) std::rethrow_exception(pfss_exc);
        } else {
          R.opens->flush(R.party, *R.net_chan);
        }
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
          if (!device_pipeline_ || device_pipeline_materialize_ ||
              (device_pipeline_ && b->needs_host_materialize())) {
            b->materialize_host(R.party, *R.pfss_chan);
          }
          b->clear();
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

      if (R.overlap_pfss_open && want_open && (want_pfss_coeff || want_pfss_trunc) &&
          R.opens && R.net_chan && R.opens->has_pending() &&
          ((pfss_coeff_.has_pending() || pfss_coeff_.has_flushed()) ||
           (pfss_trunc_.has_pending() || pfss_trunc_.has_flushed()))) {
        std::exception_ptr pfss_exc;
        std::thread pfss_thread([&]() {
          try {
            if (allow_planner && R.pfss_planner && (want_pfss_coeff || want_pfss_trunc) && !planner_flushed) {
              if (!R.pfss_backend || !R.pfss_chan) {
                throw std::runtime_error("PhaseExecutor: PFSS planner missing backend/channel");
              }
              R.pfss_planner->finalize_phase(R.party, *R.pfss_backend, *R.pfss_chan);
            } else {
              auto flush_one = [&](PfssSuperBatch& b) {
                if (!R.pfss_backend || !R.pfss_chan) {
                  throw std::runtime_error("PhaseExecutor: PFSS flush missing backend/channel");
                }
                if (b.has_pending()) b.flush_eval(R.party, *R.pfss_backend, *R.pfss_chan);
                if (b.has_flushed()) {
                  if (!device_pipeline_ || device_pipeline_materialize_ ||
                      (device_pipeline_ && b.needs_host_materialize())) {
                    b.materialize_host(R.party, *R.pfss_chan);
                  }
                  b.clear();
                }
              };
              if (want_pfss_coeff) flush_one(pfss_coeff_);
              if (want_pfss_trunc && &pfss_trunc_ != &pfss_coeff_) flush_one(pfss_trunc_);
            }
          } catch (...) {
            pfss_exc = std::current_exception();
          }
        });
        do_flush_open();
        if (pfss_thread.joinable()) pfss_thread.join();
        if (allow_planner && R.pfss_planner && (want_pfss_coeff || want_pfss_trunc) && !planner_flushed) {
          planner_flushed = true;
        }
        if (pfss_exc) std::rethrow_exception(pfss_exc);
        continue;
      }
      if (want_open && do_flush_open()) continue;
      if (allow_planner && R.pfss_planner && (want_pfss_coeff || want_pfss_trunc) && !planner_flushed) {
        if (!R.pfss_backend || !R.pfss_chan) {
          throw std::runtime_error("PhaseExecutor: PFSS planner missing backend/channel");
        }
        if (flush_guard + 1 > max_flushes_) {
          throw std::runtime_error("PhaseExecutor: planner flush budget exceeded");
        }
        R.pfss_planner->finalize_phase(R.party, *R.pfss_backend, *R.pfss_chan);
        flush_guard++;
        planner_flushed = true;
        continue;
      }
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
    stats_.opened_words_beaver =
        os.opened_words_by_kind[static_cast<size_t>(OpenKind::kBeaver)];
    stats_.opened_words_mask =
        os.opened_words_by_kind[static_cast<size_t>(OpenKind::kMask)];
    stats_.opened_words_other =
        os.opened_words_by_kind[static_cast<size_t>(OpenKind::kOther)];
    stats_.open_wire_bytes_sent = os.wire_bytes_sent;
    const auto& pcs = pfss_coeff_.total_stats();
    const auto& pts = pfss_trunc_.total_stats();
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

    if (device_pipeline_ && R.pfss_chan) {
      if (pfss_coeff_.has_flushed() &&
          (device_pipeline_materialize_ || pfss_coeff_.needs_host_materialize())) {
        pfss_coeff_.materialize_host(R.party, *R.pfss_chan);
      }
      if (pfss_trunc_.has_flushed() &&
          (device_pipeline_materialize_ || pfss_trunc_.needs_host_materialize())) {
        pfss_trunc_.materialize_host(R.party, *R.pfss_chan);
      }
    }
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
  bool device_pipeline_ = false;
  bool device_pipeline_materialize_ = true;
  LazyLimits lazy_limits_{};
};

}  // namespace runtime
