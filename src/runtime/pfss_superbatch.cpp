#include "runtime/pfss_superbatch.hpp"

#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <random>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <limits>
#include <mutex>

#ifdef SUF_HAVE_CUDA
#include <cuda_runtime.h>
#include "runtime/cuda_primitives.hpp"
#include "proto/backend_gpu.hpp"
#endif
#include "runtime/bench_online_profile.hpp"

namespace runtime {

namespace {

static const std::vector<proto::BeaverTriple64Share> k_empty_triples;

inline size_t hatx_words_for_job(const PreparedCompositeJob& job) {
  if (!job.hatx_public.empty()) return job.hatx_public.size();
  if (job.hatx_device && job.hatx_device_words) return job.hatx_device_words;
  if (job.shape.total_elems) return static_cast<size_t>(job.shape.total_elems);
  return 0;
}

inline uint64_t mask_bits_host(int bits) {
  if (bits <= 0) return 0;
  if (bits >= 64) return ~uint64_t(0);
  return (uint64_t(1) << bits) - 1ull;
}

inline size_t packed_words_host(size_t elems, int eff_bits) {
  if (eff_bits <= 0 || eff_bits > 64) return elems;
  if (eff_bits == 64) return elems;
  uint64_t bits = static_cast<uint64_t>(elems) * static_cast<uint64_t>(eff_bits);
  return static_cast<size_t>((bits + 63) >> 6);
}

inline bool env_flag_enabled_default_local(const char* name, bool defv) {
  const char* env = std::getenv(name);
  if (!env) return defv;
  std::string v(env);
  std::transform(v.begin(), v.end(), v.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return !(v == "0" || v == "false" || v == "off" || v == "no");
}

inline size_t beaver_u64_mul_per_elem(const ::gates::CompositePartyKey& k) {
  const auto& comp = k.compiled;
  // If predicate shares are additive-u64, composite_eval_batch_backend uses a
  // non-interactive path (no Beaver muls) for degree-0 gates.
  if (k.pred_meta.sem == ::proto::ShareSemantics::AddU64 &&
      comp.degree == 0 &&
      comp.coeff.cutpoints_ge.empty()) {
    return 0;
  }
  // Truncation gates can be evaluated without Beaver muls when predicate shares
  // are emitted directly in the additive ring (0/1 as u64). This avoids the
  // XOR->add conversion and selector-weighted blending that the generic path
  // would otherwise perform.
  if ((comp.gate_kind == ::compiler::GateKind::FaithfulTR ||
       comp.gate_kind == ::compiler::GateKind::FaithfulARS ||
       comp.gate_kind == ::compiler::GateKind::GapARS) &&
      k.pred_meta.sem == ::proto::ShareSemantics::AddU64 &&
      comp.degree == 0 &&
      comp.coeff.cutpoints_ge.empty()) {
    return 0;
  }
  // Interval-LUT coeff mode can make coefficient selection Beaver-free. For the
  // current usage in this repo (payload-only gates: degree=0, ell=0), the PFSS
  // evaluation does not consume any Beaver 64-bit triples.
  if (comp.coeff.mode == ::compiler::CoeffMode::kIntervalLut &&
      comp.degree == 0 &&
      comp.ell == 0) {
    return 0;
  }
  // Current Composite-FSS evaluation is Beaver-free for arithmetic outputs: the
  // compiler shifts piecewise polynomials into the public `hatx` domain, and
  // Horner is done locally. Remaining Beaver usage (u64 triples) comes from:
  //  - selector network (XOR->add for cut bits, plus chain multiplications),
  //  - boolean output blending (XOR->add for per-piece bools, plus selector-weighted muls).
  if (comp.ell <= 0) return 0;
  const size_t cut_count = comp.coeff.cutpoints_ge.size();
  const size_t piece_count = cut_count + 1;
  const size_t selector_chain_mul = (cut_count > 0) ? (cut_count - 1) : 0;
  const size_t ell = static_cast<size_t>(comp.ell);
  const size_t bool_b2a_mul = piece_count * ell;
  const size_t bool_select_mul = piece_count * ell;
  return cut_count + selector_chain_mul + bool_b2a_mul + bool_select_mul;
}

inline size_t beaver_u64_capacity_elems(const ::gates::CompositePartyKey& k) {
  const size_t per_elem = beaver_u64_mul_per_elem(k);
  if (per_elem == 0) return std::numeric_limits<size_t>::max();
  return k.triples.size() / per_elem;
}

std::vector<uint64_t> pack_eff_bits_host(const std::vector<uint64_t>& xs, int eff_bits) {
  if (eff_bits <= 0 || eff_bits > 64) throw std::runtime_error("pack_eff_bits_host: eff_bits out of range");
  if (eff_bits == 64) return xs;
  size_t words = packed_words_host(xs.size(), eff_bits);
  std::vector<uint64_t> packed(words, 0);
  uint64_t mask = mask_bits_host(eff_bits);
  for (size_t i = 0; i < xs.size(); i++) {
    uint64_t v = xs[i] & mask;
    size_t bit_idx = i * static_cast<size_t>(eff_bits);
    size_t w = bit_idx >> 6;
    int off = static_cast<int>(bit_idx & 63);
    packed[w] |= (v << off);
    int spill = off + eff_bits - 64;
    if (spill > 0 && w + 1 < packed.size()) {
      packed[w + 1] |= (v >> (eff_bits - spill));
    }
  }
  return packed;
}

}  // namespace

PfssHandle PfssSuperBatch::enqueue_composite(PreparedCompositeJob job) {
  if (job.token == static_cast<size_t>(-1)) {
    job.token = completed_.size();
  }
  if (job.key) {
    const auto& comp = job.key->compiled;
    uint16_t pred_eff = (comp.pred.eff_bits > 0 && comp.pred.eff_bits <= comp.pred.n)
                            ? static_cast<uint16_t>(comp.pred.eff_bits)
                            : 0;
    uint16_t coeff_eff = (comp.coeff.eff_bits > 0 && comp.coeff.eff_bits <= comp.coeff.n)
                             ? static_cast<uint16_t>(comp.coeff.eff_bits)
                             : 0;
    uint16_t eff_bits = job.shape.eff_bits;
    if (eff_bits == 0 || eff_bits == 64) {
      eff_bits = pred_eff ? pred_eff : (coeff_eff ? coeff_eff : 64);
    } else if (pred_eff && pred_eff < eff_bits) {
      eff_bits = pred_eff;
    }
    if (eff_bits == 0 || eff_bits > 64) eff_bits = 64;
    job.shape.eff_bits = eff_bits;
  }
  if (job.shape.total_elems == 0) {
    job.shape.total_elems = static_cast<uint32_t>(hatx_words_for_job(job));
  }
  if (slots_.size() <= job.token) {
    slots_.resize(job.token + 1);
  }
  if (!slots_[job.token]) {
    slots_[job.token] = std::make_shared<PfssResultSlot>();
  }
  size_t hatx_words = hatx_words_for_job(job);
  size_t hatx_bytes = hatx_words * sizeof(uint64_t);
  bool ragged = !job.row_offsets.empty() && !job.row_lengths.empty();
  if (ragged) {
    if (job.row_offsets.size() != job.row_lengths.size() + 1) {
      throw std::runtime_error("PfssSuperBatch: ragged offsets must be len+1");
    }
    size_t total = job.row_offsets.back();
    if (total != hatx_words) {
      throw std::runtime_error("PfssSuperBatch: ragged offsets mismatch hatx size");
    }
    job.shape.ragged = true;
    if (job.shape.num_rows == 0) {
      job.shape.num_rows = static_cast<uint16_t>(job.row_lengths.size());
    }
    if (job.shape.total_elems == 0) {
      job.shape.total_elems = static_cast<uint32_t>(total);
    }
    if (job.shape.max_row_len == 0) {
      for (int L : job.row_lengths) {
        job.shape.max_row_len =
            std::max<uint16_t>(job.shape.max_row_len, static_cast<uint16_t>(L));
      }
    }
  } else {
    if (job.shape.num_rows == 0 && job.shape.total_elems > 0) {
      job.shape.num_rows = static_cast<uint16_t>(job.shape.total_elems);
    }
    if (job.shape.max_row_len == 0 && job.shape.total_elems > 0) {
      job.shape.max_row_len = static_cast<uint16_t>(job.shape.total_elems);
    }
  }
  size_t new_pending_jobs = pending_jobs_ + 1;
  size_t new_pending_hatx = pending_hatx_words_ + hatx_words;
  if (limits_.max_pending_jobs > 0 && new_pending_jobs > limits_.max_pending_jobs) {
    throw std::runtime_error("PfssSuperBatch: pending job limit exceeded");
  }
  if (limits_.max_pending_hatx_words > 0 && new_pending_hatx > limits_.max_pending_hatx_words) {
    throw std::runtime_error("PfssSuperBatch: pending hatx packing limit exceeded");
  }
  if (limits_.max_pending_hatx_bytes > 0 &&
      new_pending_hatx * sizeof(uint64_t) > limits_.max_pending_hatx_bytes) {
    throw std::runtime_error("PfssSuperBatch: pending hatx byte limit exceeded");
  }
  if (gpu_stager_ && limits_.max_pending_device_bytes > 0) {
    size_t new_pending_dev = pending_dev_bytes_ + hatx_bytes;
    if (new_pending_dev > limits_.max_pending_device_bytes) {
      throw std::runtime_error("PfssSuperBatch: pending device staging budget exceeded");
    }
    pending_dev_bytes_ = new_pending_dev;
  }
  pending_jobs_ = new_pending_jobs;
  pending_hatx_words_ = new_pending_hatx;
  stats_.active_elems += job.shape.total_elems;
  stats_.cost_effbits += static_cast<size_t>(job.shape.total_elems) * static_cast<size_t>(job.shape.eff_bits);
  total_stats_.active_elems += job.shape.total_elems;
  total_stats_.cost_effbits += static_cast<size_t>(job.shape.total_elems) * static_cast<size_t>(job.shape.eff_bits);
  stats_.pending_jobs = pending_jobs_;
  stats_.pending_hatx = pending_hatx_words_;
  stats_.pending_device_bytes = pending_dev_bytes_;
  if (completed_.size() <= job.token) {
    completed_.resize(job.token + 1);
  }
  jobs_.push_back(std::move(job));
  flushed_ = false;
  return PfssHandle{jobs_.back().token, slots_[jobs_.back().token]};
}

PfssHandle PfssSuperBatch::enqueue_truncation(const compiler::TruncationLoweringResult& bundle,
                                              const gates::CompositePartyKey& key,
                                              gates::PostProcHook& hook,
                                              std::vector<uint64_t> hatx_public,
                                              nn::TensorView<uint64_t> out) {
  PreparedCompositeJob job;
  job.suf = &bundle.suf;
  job.key = &key;
  job.hook = &hook;
  job.hatx_public = std::move(hatx_public);
  job.out = out;
  return enqueue_composite(std::move(job));
}

bool PfssSuperBatch::ready(const PfssHandle& h) const {
  if (h.slot && h.slot->ready.load()) return true;
  return h.token < completed_.size() && !completed_[h.token].arith.empty();
}

bool PfssSuperBatch::needs_host_materialize() const {
  for (const auto& job : jobs_) {
    if (job.out.data && job.out.numel() > 0) return true;
  }
  return false;
}

void PfssSuperBatch::flush_eval(int party, proto::PfssBackendBatch& backend, proto::IChannel& ch) {
  const bool prof = runtime::bench::online_profiling_enabled();
  const auto prof_now = [] { return std::chrono::steady_clock::now(); };
  const auto prof_ns = [](const std::chrono::steady_clock::time_point& a,
                          const std::chrono::steady_clock::time_point& b) -> uint64_t {
    if (b <= a) return 0;
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count());
  };
  const auto t_total0 = prof ? prof_now() : std::chrono::steady_clock::time_point{};
  uint64_t ns_stage_hatx = 0;
  uint64_t ns_eval = 0;
  uint64_t ns_stage_out = 0;

  if (limits_.max_flushes > 0 && stats_.flushes + 1 > limits_.max_flushes) {
    throw std::runtime_error("PfssSuperBatch: flush budget exceeded");
  }
  stats_.flushes += 1;
  stats_.jobs += jobs_.size();
  total_stats_.flushes += 1;
  total_stats_.jobs += jobs_.size();

  // Fast path: a single job is extremely common in end-to-end transformer runs
  // (each trunc/coeff stage drives one PFSS eval). Avoid the heavyweight
  // grouping/bucketing path and, critically, avoid copying `hatx_public` into
  // an intermediate bucket buffer.
  if (jobs_.size() == 1) {
    if (gpu_stager_) {
      // Drop any group-local device buffers from a prior flush.
      for (auto& gr : group_results_) {
        if (gr.dev_arith.ptr && !gr.dev_arith_owner) gpu_stager_->free_bytes(gr.dev_arith);
        if (gr.dev_bools.ptr && !gr.dev_bools_owner) gpu_stager_->free_bytes(gr.dev_bools);
        gr.dev_arith_owner.reset();
        gr.dev_bools_owner.reset();
        gr.dev_arith = DeviceBufferRef{};
        gr.dev_bools = DeviceBufferRef{};
      }
    }
    group_results_.clear();
    slices_.assign(jobs_.size(), JobSlice{});

    auto& job = jobs_[0];
    if (!job.suf || !job.key) {
      throw std::runtime_error("PfssSuperBatch: incomplete composite job");
    }
    const auto& comp = job.key->compiled;
    const size_t r = static_cast<size_t>(comp.r);
    const size_t ell = static_cast<size_t>(comp.ell);
    const size_t N = hatx_words_for_job(job);
    size_t total_hatx_words = N;

    stats_.max_bucket_hatx = std::max(stats_.max_bucket_hatx, N);
    stats_.max_bucket_jobs = std::max(stats_.max_bucket_jobs, size_t{1});
    total_stats_.max_bucket_hatx = std::max(total_stats_.max_bucket_hatx, N);
    total_stats_.max_bucket_jobs = std::max(total_stats_.max_bucket_jobs, size_t{1});

    const bool dev_capable =
        gpu_stager_ && (dynamic_cast<runtime::CpuPassthroughStager*>(gpu_stager_) == nullptr);
    // Prefer device-resident `hatx` when possible: Composite-FSS has fast GPU
    // broadcast-key paths that avoid per-element key replication on host.
    const bool use_device_hatx_input = [&]() -> bool {
      bool gpu_backend = false;
#ifdef SUF_HAVE_CUDA
      gpu_backend = (dynamic_cast<proto::PfssGpuStagedEval*>(&backend) != nullptr);
#endif
      const bool caller_provided = (job.hatx_device != nullptr);
      // Default-on for the GPU backend because it removes a major host-side
      // bottleneck (key replication + H2D copies). Can be disabled via env.
      const bool want =
          env_flag_enabled_default_local("SUF_PFSS_USE_DEVICE_HATX", gpu_backend || caller_provided);
      return want;
    }();
    DeviceBufferRef dev_hatx;
    bool own_dev_hatx = false;
    auto free_dev_hatx = [&]() {
      if (!gpu_stager_ || !dev_capable) return;
      if (own_dev_hatx && dev_hatx.ptr) {
        gpu_stager_->free_bytes(dev_hatx);
        dev_hatx = DeviceBufferRef{};
        own_dev_hatx = false;
      }
    };

    try {
      if (use_device_hatx_input && gpu_stager_ && dev_capable && !job.hatx_device && !job.hatx_public.empty()) {
        const auto t_stage0 = prof ? prof_now() : std::chrono::steady_clock::time_point{};
        HostBufferRef host{job.hatx_public.data(), job.hatx_public.size() * sizeof(uint64_t)};
        dev_hatx = gpu_stager_->stage_to_device(host);
        own_dev_hatx = true;
        if (prof) ns_stage_hatx += prof_ns(t_stage0, prof_now());
      }

      gates::CompositeBatchInput in{
          job.hatx_public.empty() ? nullptr : job.hatx_public.data(),
          N,
          use_device_hatx_input
              ? (job.hatx_device
                     ? job.hatx_device
                     : (dev_capable && dev_hatx.ptr ? reinterpret_cast<const uint64_t*>(dev_hatx.ptr) : nullptr))
              : nullptr,
          device_outputs_};
      // We stage raw PFSS outputs ourselves (via PfssGpuStager) so device pointers
      // remain stable across flushes/finalize boundaries.
      in.device_outputs = false;
      const auto t_eval0 = prof ? prof_now() : std::chrono::steady_clock::time_point{};
      auto out = gates::composite_eval_batch_backend(party, backend, ch, *job.key, *job.suf, in);
      if (prof) ns_eval += prof_ns(t_eval0, prof_now());

      GroupResult gr;
      gr.suf = job.suf;
      gr.key = job.key;
      gr.r = r;
      gr.ell = ell;
      gr.arith = std::move(out.haty_share);
      gr.bools = std::move(out.bool_share);
      gr.dev_arith_ptr = out.haty_device;
      gr.dev_arith_words = out.haty_device_words;
      gr.dev_bools_ptr = out.bool_device;
      gr.dev_bools_words = out.bool_device_words;

      if (device_outputs_ && dev_capable && gpu_stager_) {
        if (!gr.arith.empty()) {
          const auto t_stage0 = prof ? prof_now() : std::chrono::steady_clock::time_point{};
          HostBufferRef host_arith{gr.arith.data(), gr.arith.size() * sizeof(uint64_t)};
          gr.dev_arith = gpu_stager_->stage_to_device(host_arith);
          if (prof) ns_stage_out += prof_ns(t_stage0, prof_now());
          if (gr.dev_arith.ptr) {
            void* ptr = gr.dev_arith.ptr;
            size_t bytes = gr.dev_arith.bytes;
#ifdef SUF_HAVE_CUDA
            gr.dev_arith_owner = std::shared_ptr<void>(ptr, [bytes](void* p) {
              (void)bytes;
              if (!p) return;
              cudaFree(p);
            });
#else
            PfssGpuStager* stager = gpu_stager_;
            gr.dev_arith_owner = std::shared_ptr<void>(ptr, [stager, bytes](void* p) {
              if (!p) return;
              if (!stager) return;
              stager->free_bytes(DeviceBufferRef{p, bytes});
            });
#endif
          }
        }
        if (!gr.bools.empty()) {
          const auto t_stage0 = prof ? prof_now() : std::chrono::steady_clock::time_point{};
          HostBufferRef host_bools{gr.bools.data(), gr.bools.size() * sizeof(uint64_t)};
          gr.dev_bools = gpu_stager_->stage_to_device(host_bools);
          if (prof) ns_stage_out += prof_ns(t_stage0, prof_now());
          if (gr.dev_bools.ptr) {
            void* ptr = gr.dev_bools.ptr;
            size_t bytes = gr.dev_bools.bytes;
#ifdef SUF_HAVE_CUDA
            gr.dev_bools_owner = std::shared_ptr<void>(ptr, [bytes](void* p) {
              (void)bytes;
              if (!p) return;
              cudaFree(p);
            });
#else
            PfssGpuStager* stager = gpu_stager_;
            gr.dev_bools_owner = std::shared_ptr<void>(ptr, [stager, bytes](void* p) {
              if (!p) return;
              if (!stager) return;
              stager->free_bytes(DeviceBufferRef{p, bytes});
            });
#endif
          }
        }
      }

      const size_t total_arith_words = gr.arith.size();
      const size_t total_bool_words = gr.bools.size();
      slices_[0].group_result = 0;
      slices_[0].start = 0;
      slices_[0].len = N;
      group_results_.push_back(std::move(gr));

      flushed_ = true;
      stats_.arith_words += total_arith_words;
      stats_.pred_bits += total_bool_words * 64;
      stats_.hatx_words += total_hatx_words;
      stats_.hatx_bytes += total_hatx_words * sizeof(uint64_t);
      total_stats_.arith_words += total_arith_words;
      total_stats_.pred_bits += total_bool_words * 64;
      total_stats_.hatx_words += total_hatx_words;
      total_stats_.hatx_bytes += total_hatx_words * sizeof(uint64_t);
      pending_jobs_ = 0;
      pending_hatx_words_ = 0;
      pending_dev_bytes_ = 0;
      stats_.pending_jobs = 0;
      stats_.pending_hatx = 0;
      stats_.pending_device_bytes = 0;
      free_dev_hatx();
      if (prof) {
        runtime::bench::add_online_ns(runtime::bench::OnlineTimeKind::PfssFlushEvalStageHatx, ns_stage_hatx);
        runtime::bench::add_online_ns(runtime::bench::OnlineTimeKind::PfssFlushEvalEval, ns_eval);
        runtime::bench::add_online_ns(runtime::bench::OnlineTimeKind::PfssFlushEvalStageOut, ns_stage_out);
        runtime::bench::add_online_ns(runtime::bench::OnlineTimeKind::PfssFlushEvalTotal,
                                      prof_ns(t_total0, prof_now()));
      }

      // Publish per-handle views for consumers waiting on `ready()` / `view()`.
      // The general multi-job path does this at the end of flush_eval; the
      // single-job fast path must mirror it to avoid deadlocks.
      if (store_results_) {
        populate_completed_();
      } else {
        populate_device_views_();
        for (const auto& j : jobs_) {
          if (j.token < slots_.size() && slots_[j.token]) {
            slots_[j.token]->ready.store(true);
          }
        }
      }
      return;
    } catch (...) {
      free_dev_hatx();
      throw;
    }
  }

  size_t total_hatx_words = 0;
  for (const auto& job : jobs_) {
    if (job.hatx_public.empty() && job.hatx_device && hatx_words_for_job(job) > 0) {
      throw std::runtime_error(
          "PfssSuperBatch: device-only hatx requires single-job flush (batch grouping needs host hatx)");
    }
  }
  struct GroupKey {
    int r = 0;
    int ell = 0;
    int degree = 0;
    int pred_n = 0;
    int pred_eff_bits = 64;
    int pred_out_mode = 0;
    int coeff_n = 0;
    int coeff_eff_bits = 64;
    int coeff_mode = 0;
    int coeff_words = 0;
    bool use_packed_pred = false;
    bool use_packed_cut = false;
    int packed_pred_words = 0;
    int packed_cut_words = 0;
    bool operator==(const GroupKey& o) const {
      return r == o.r && ell == o.ell && degree == o.degree &&
             pred_n == o.pred_n && pred_eff_bits == o.pred_eff_bits && pred_out_mode == o.pred_out_mode &&
             coeff_n == o.coeff_n && coeff_eff_bits == o.coeff_eff_bits &&
             coeff_mode == o.coeff_mode && coeff_words == o.coeff_words &&
             use_packed_pred == o.use_packed_pred && use_packed_cut == o.use_packed_cut &&
             packed_pred_words == o.packed_pred_words && packed_cut_words == o.packed_cut_words;
    }
  };
  struct GroupKeyHash {
    size_t operator()(const GroupKey& k) const {
      size_t h = std::hash<int>{}(k.r + (k.ell << 4) + (k.degree << 8));
      h ^= std::hash<int>{}(k.pred_n << 12) ^ std::hash<int>{}(k.pred_eff_bits << 16) ^ std::hash<int>{}(k.pred_out_mode << 20);
      h ^= std::hash<int>{}(k.coeff_n << 24) ^ std::hash<int>{}(k.coeff_eff_bits << 28) ^ std::hash<int>{}(k.coeff_mode << 12);
      h ^= std::hash<int>{}(k.coeff_words << 4);
      h ^= std::hash<int>{}(k.packed_pred_words) ^ (std::hash<int>{}(k.packed_cut_words) << 1);
      h ^= std::hash<bool>{}(k.use_packed_pred) ^ (std::hash<bool>{}(k.use_packed_cut) << 2);
      return h;
    }
  };
  struct GroupData {
    std::vector<size_t> jobs;
  };
  struct BucketJob {
    size_t job_idx = 0;
    size_t start = 0;
    size_t len = 0;
  };
  struct Bucket {
    const gates::CompositePartyKey* key = nullptr;
    const suf::SUF<uint64_t>* suf = nullptr;
    std::vector<BucketJob> jobs;
    std::vector<uint64_t> hatx;
    std::vector<int> row_offsets;
    std::vector<int> row_lengths;
    DeviceBufferRef dev_hatx;
    bool own_dev_hatx = false;
    DeviceBufferRef dev_hatx_packed;
    uint16_t eff_bits = 64;
    size_t r = 0;
    size_t ell = 0;
  };

  std::unordered_map<GroupKey, size_t, GroupKeyHash> group_index;
  std::vector<GroupData> groups;

  if (gpu_stager_) {
    // Drop any group-local device buffers from a prior flush. If device outputs were
    // published to per-handle slots via shared owners, those owners keep the buffers
    // alive; we must not free them here.
    for (auto& gr : group_results_) {
      if (gr.dev_arith.ptr && !gr.dev_arith_owner) gpu_stager_->free_bytes(gr.dev_arith);
      if (gr.dev_bools.ptr && !gr.dev_bools_owner) gpu_stager_->free_bytes(gr.dev_bools);
      gr.dev_arith_owner.reset();
      gr.dev_bools_owner.reset();
      gr.dev_arith = DeviceBufferRef{};
      gr.dev_bools = DeviceBufferRef{};
    }
  }
  group_results_.clear();
  slices_.assign(jobs_.size(), JobSlice{});
  size_t total_arith_words = 0;
  size_t total_bool_words = 0;

  for (size_t idx = 0; idx < jobs_.size(); ++idx) {
    auto& job = jobs_[idx];
    if (!job.suf || !job.key) {
      throw std::runtime_error("PfssSuperBatch: incomplete composite job");
    }
    const auto& comp = job.key->compiled;
    GroupKey gk;
    gk.r = comp.r;
    gk.ell = comp.ell;
    gk.degree = comp.degree;
    gk.pred_n = comp.pred.n;
    gk.pred_eff_bits = comp.pred.eff_bits;
    gk.pred_out_mode = static_cast<int>(comp.pred.out_mode);
    gk.coeff_n = comp.coeff.n;
    gk.coeff_eff_bits = comp.coeff.eff_bits;
    gk.coeff_mode = static_cast<int>(comp.coeff.mode);
    gk.coeff_words = comp.coeff.out_words;
    gk.use_packed_pred = job.key->use_packed_pred;
    gk.use_packed_cut = job.key->use_packed_cut;
    gk.packed_pred_words = job.key->packed_pred_words;
    gk.packed_cut_words = job.key->packed_cut_words;
    size_t g_idx;
    auto it = group_index.find(gk);
    if (it == group_index.end()) {
      g_idx = groups.size();
      group_index.emplace(gk, g_idx);
      groups.emplace_back();
    } else {
      g_idx = it->second;
    }
    groups[g_idx].jobs.push_back(idx);
  }

  for (const auto& gd : groups) {
    std::unordered_map<const gates::CompositePartyKey*, size_t> bucket_index;
    std::vector<Bucket> buckets;
    for (auto job_idx : gd.jobs) {
      auto& job = jobs_[job_idx];
      if (std::getenv("SUF_PFSS_BEAVER_TRACE")) {
        static std::mutex mu;
        static std::unordered_set<uint64_t> seen;
        const auto& ck = job.key->compiled;
        const size_t per_elem = beaver_u64_mul_per_elem(*job.key);
        if (per_elem > 0) {
          // Deduplicate logs by gate metadata; many jobs share the same key shape.
          uint64_t tag = 0;
          tag ^= static_cast<uint64_t>(static_cast<int>(ck.gate_kind)) & 0xFFull;
          tag ^= (static_cast<uint64_t>(static_cast<int>(ck.coeff.mode)) & 0xFFull) << 8;
          tag ^= (static_cast<uint64_t>(static_cast<int>(job.key->pred_meta.sem)) & 0xFFull) << 16;
          tag ^= (static_cast<uint64_t>(ck.r) & 0xFFull) << 24;
          tag ^= (static_cast<uint64_t>(ck.ell) & 0xFFull) << 32;
          tag ^= (static_cast<uint64_t>(ck.degree) & 0xFFull) << 40;
          tag ^= (static_cast<uint64_t>(ck.coeff.cutpoints_ge.size()) & 0xFFFFull) << 48;
          std::lock_guard<std::mutex> lg(mu);
          if (seen.insert(tag).second) {
            std::fprintf(stderr,
                         "[pfss_beaver] gate_kind=%d coeff_mode=%d pred_sem=%d r=%d ell=%d degree=%d "
                         "cuts=%zu per_elem_mul=%zu hatx=%zu triples=%zu\n",
                         static_cast<int>(ck.gate_kind),
                         static_cast<int>(ck.coeff.mode),
                         static_cast<int>(job.key->pred_meta.sem),
                         ck.r,
                         ck.ell,
                         ck.degree,
                         ck.coeff.cutpoints_ge.size(),
                         per_elem,
                         job.hatx_public.size(),
                         job.key->triples.size());
          }
        }
      }
      const size_t cap_elems = beaver_u64_capacity_elems(*job.key);
      if (cap_elems != std::numeric_limits<size_t>::max() && job.hatx_public.size() > cap_elems) {
        const auto& ck = job.key->compiled;
        throw std::runtime_error("PfssSuperBatch: job exceeds provisioned Beaver triple capacity (hatx=" +
                                 std::to_string(job.hatx_public.size()) + " cap=" + std::to_string(cap_elems) +
                                 " triples=" + std::to_string(job.key->triples.size()) +
                                 " gate_kind=" + std::to_string(static_cast<int>(ck.gate_kind)) +
                                 " pred_sem=" + std::to_string(static_cast<int>(job.key->pred_meta.sem)) +
                                 " r=" + std::to_string(ck.r) +
                                 " ell=" + std::to_string(ck.ell) +
                                 " degree=" + std::to_string(ck.degree) +
                                 " cutpoints=" + std::to_string(ck.coeff.cutpoints_ge.size()) +
                                 " coeff_mode=" + std::to_string(static_cast<int>(ck.coeff.mode)) + ")");
      }
      size_t bidx;
      auto it = bucket_index.find(job.key);
      if (it == bucket_index.end()) {
        bidx = buckets.size();
        bucket_index.emplace(job.key, bidx);
        Bucket b;
        b.key = job.key;
        b.suf = job.suf;
        b.r = static_cast<size_t>(job.key->compiled.r);
        b.ell = static_cast<size_t>(job.key->compiled.ell);
        buckets.push_back(std::move(b));
      } else {
        bidx = it->second;
        if (cap_elems != std::numeric_limits<size_t>::max() &&
            buckets[bidx].hatx.size() + job.hatx_public.size() > cap_elems) {
          bidx = buckets.size();
          bucket_index[job.key] = bidx;
          Bucket b;
          b.key = job.key;
          b.suf = job.suf;
          b.r = static_cast<size_t>(job.key->compiled.r);
          b.ell = static_cast<size_t>(job.key->compiled.ell);
          buckets.push_back(std::move(b));
        }
      }
      Bucket& b = buckets[bidx];
      if (job.shape.eff_bits > 0 && job.shape.eff_bits <= 64) {
        b.eff_bits = std::max<uint16_t>(b.eff_bits, job.shape.eff_bits);
      }
      BucketJob bj;
      bj.job_idx = job_idx;
      bj.start = b.hatx.size();
      bj.len = job.hatx_public.size();
      b.hatx.insert(b.hatx.end(), job.hatx_public.begin(), job.hatx_public.end());
      total_hatx_words += job.hatx_public.size();
      b.jobs.push_back(bj);
      // If caller provided a device hatx buffer and this bucket currently
      // only holds this job, reuse the device pointer to skip staging.
      if (job.hatx_device && job.hatx_device_words == job.hatx_public.size() &&
          b.jobs.size() == 1) {
        b.dev_hatx.ptr = const_cast<uint64_t*>(job.hatx_device);
        b.dev_hatx.bytes = job.hatx_device_words * sizeof(uint64_t);
        b.own_dev_hatx = false;
      }
      if (!job.row_offsets.empty()) {
        int base = b.row_offsets.empty() ? 0 : b.row_offsets.back();
        for (size_t k = 0; k + 1 < job.row_offsets.size(); ++k) {
          b.row_offsets.push_back(base + job.row_offsets[k]);
        }
        b.row_lengths.insert(b.row_lengths.end(), job.row_lengths.begin(), job.row_lengths.end());
        if (b.row_offsets.empty()) b.row_offsets.push_back(0);
        b.row_offsets.push_back(base + static_cast<int>(job.hatx_public.size()));
      }
    }

    for (auto& b : buckets) {
      stats_.max_bucket_hatx = std::max(stats_.max_bucket_hatx, b.hatx.size());
      stats_.max_bucket_jobs = std::max(stats_.max_bucket_jobs, b.jobs.size());
      total_stats_.max_bucket_hatx = std::max(total_stats_.max_bucket_hatx, b.hatx.size());
      total_stats_.max_bucket_jobs = std::max(total_stats_.max_bucket_jobs, b.jobs.size());
      bool dev_capable = gpu_stager_ && (dynamic_cast<runtime::CpuPassthroughStager*>(gpu_stager_) == nullptr);
      auto free_bucket_hatx = [&]() {
        if (!gpu_stager_ || !dev_capable) return;
        if (b.dev_hatx_packed.ptr) {
          gpu_stager_->free_bytes(b.dev_hatx_packed);
          b.dev_hatx_packed = DeviceBufferRef{};
        }
        if (b.own_dev_hatx && b.dev_hatx.ptr) {
          gpu_stager_->free_bytes(b.dev_hatx);
          b.dev_hatx = DeviceBufferRef{};
          b.own_dev_hatx = false;
        }
      };

      try {
        const bool use_device_hatx_input = [&]() -> bool {
          bool gpu_backend = false;
#ifdef SUF_HAVE_CUDA
          gpu_backend = (dynamic_cast<proto::PfssGpuStagedEval*>(&backend) != nullptr);
#endif
          return env_flag_enabled_default_local("SUF_PFSS_USE_DEVICE_HATX", gpu_backend);
        }();
        if (use_device_hatx_input && gpu_stager_ && dev_capable && !b.dev_hatx.ptr) {
          const auto t_stage0 = prof ? prof_now() : std::chrono::steady_clock::time_point{};
#ifdef SUF_HAVE_CUDA
          int eff_bits = static_cast<int>(b.eff_bits);
          void* st_stream = gpu_stager_->stream();
          // Only pack when eff_bits is meaningfully smaller than 64 to avoid
          // doubling device memory for near-64-bit values.
          const int max_pack_bits = []() {
            const char* env = std::getenv("SUF_PFSS_STAGE_PACK_MAX_BITS");
            if (!env) return 48;
            int v = std::atoi(env);
            if (v <= 0 || v > 64) return 48;
            return v;
          }();
          if (st_stream && !b.hatx.empty() && eff_bits > 0 && eff_bits < 64 && eff_bits <= max_pack_bits) {
            if (std::getenv("SUF_VALIDATE_EFFBITS")) {
              uint64_t mask = mask_bits_host(eff_bits);
              for (size_t i = 0; i < b.hatx.size(); i++) {
                if ((b.hatx[i] & ~mask) != 0) {
                  throw std::runtime_error("PfssSuperBatch: eff_bits pack overflow (hatx has high bits)");
                }
              }
            }
            auto packed = pack_eff_bits_host(b.hatx, eff_bits);
            HostBufferRef host_packed{packed.data(), packed.size() * sizeof(uint64_t)};
            b.dev_hatx_packed = gpu_stager_->stage_to_device(host_packed);
            b.dev_hatx = gpu_stager_->alloc_bytes(b.hatx.size() * sizeof(uint64_t));
            b.own_dev_hatx = true;
            launch_unpack_eff_bits_kernel(reinterpret_cast<const uint64_t*>(b.dev_hatx_packed.ptr),
                                          eff_bits,
                                          reinterpret_cast<uint64_t*>(b.dev_hatx.ptr),
                                          b.hatx.size(),
                                          st_stream);
            cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(st_stream));
            gpu_stager_->free_bytes(b.dev_hatx_packed);
            b.dev_hatx_packed = DeviceBufferRef{};
          } else
#endif
          {
            HostBufferRef host{b.hatx.data(), b.hatx.size() * sizeof(uint64_t)};
            b.dev_hatx = gpu_stager_->stage_to_device(host);
            b.own_dev_hatx = true;
          }
          if (prof) ns_stage_hatx += prof_ns(t_stage0, prof_now());
        }

        gates::CompositeBatchInput in{b.hatx.data(), b.hatx.size(),
                                      (use_device_hatx_input && dev_capable && b.dev_hatx.ptr)
                                          ? reinterpret_cast<const uint64_t*>(b.dev_hatx.ptr)
                                          : nullptr};
        // We stage raw PFSS outputs ourselves (via PfssGpuStager) so device pointers
        // remain stable across flushes/finalize boundaries.
        in.device_outputs = false;
        const auto t_eval0 = prof ? prof_now() : std::chrono::steady_clock::time_point{};
        auto out = gates::composite_eval_batch_backend(party, backend, ch, *b.key, *b.suf, in);
        if (prof) ns_eval += prof_ns(t_eval0, prof_now());
      size_t gr_idx = group_results_.size();
      GroupResult gr;
      gr.suf = b.suf;
      gr.key = b.key;
      gr.r = b.r;
      gr.ell = b.ell;
      gr.arith = std::move(out.haty_share);
      gr.bools = std::move(out.bool_share);
      if (device_outputs_ && dev_capable && gpu_stager_) {
        if (!gr.arith.empty()) {
          const auto t_stage0 = prof ? prof_now() : std::chrono::steady_clock::time_point{};
          HostBufferRef host{gr.arith.data(), gr.arith.size() * sizeof(uint64_t)};
          gr.dev_arith = gpu_stager_->stage_to_device(host);
          gr.dev_arith_words = gr.arith.size();
          if (prof) ns_stage_out += prof_ns(t_stage0, prof_now());
          if (gr.dev_arith.ptr) {
            void* ptr = gr.dev_arith.ptr;
            size_t bytes = gr.dev_arith.bytes;
#ifdef SUF_HAVE_CUDA
            gr.dev_arith_owner = std::shared_ptr<void>(ptr, [bytes](void* p) {
              (void)bytes;
              if (!p) return;
              cudaFree(p);
            });
#else
            PfssGpuStager* stager = gpu_stager_;
            gr.dev_arith_owner = std::shared_ptr<void>(ptr, [stager, bytes](void* p) {
              if (!p) return;
              if (!stager) return;
              stager->free_bytes(DeviceBufferRef{p, bytes});
            });
#endif
          }
        }
        if (!gr.bools.empty()) {
          const auto t_stage0 = prof ? prof_now() : std::chrono::steady_clock::time_point{};
          HostBufferRef host_b{gr.bools.data(), gr.bools.size() * sizeof(uint64_t)};
          gr.dev_bools = gpu_stager_->stage_to_device(host_b);
          gr.dev_bools_words = gr.bools.size();
          if (prof) ns_stage_out += prof_ns(t_stage0, prof_now());
          if (gr.dev_bools.ptr) {
            void* ptr = gr.dev_bools.ptr;
            size_t bytes = gr.dev_bools.bytes;
#ifdef SUF_HAVE_CUDA
            gr.dev_bools_owner = std::shared_ptr<void>(ptr, [bytes](void* p) {
              (void)bytes;
              if (!p) return;
              cudaFree(p);
            });
#else
            PfssGpuStager* stager = gpu_stager_;
            gr.dev_bools_owner = std::shared_ptr<void>(ptr, [stager, bytes](void* p) {
              if (!p) return;
              if (!stager) return;
              stager->free_bytes(DeviceBufferRef{p, bytes});
            });
#endif
          }
        }
      }
      total_arith_words += gr.arith.size();
      total_bool_words += gr.bools.size();
      for (const auto& bj : b.jobs) {
        if (bj.job_idx >= slices_.size()) continue;
        slices_[bj.job_idx].group_result = gr_idx;
        slices_[bj.job_idx].start = bj.start;
        slices_[bj.job_idx].len = bj.len;
      }
      group_results_.push_back(std::move(gr));
        free_bucket_hatx();
      } catch (...) {
        free_bucket_hatx();
        throw;
      }
    }
  }
  flushed_ = true;
  stats_.arith_words += total_arith_words;
  stats_.pred_bits += total_bool_words * 64;
  stats_.hatx_words += total_hatx_words;
  stats_.hatx_bytes += total_hatx_words * sizeof(uint64_t);
  total_stats_.arith_words += total_arith_words;
  total_stats_.pred_bits += total_bool_words * 64;
  total_stats_.hatx_words += total_hatx_words;
  total_stats_.hatx_bytes += total_hatx_words * sizeof(uint64_t);
  pending_jobs_ = 0;
  pending_hatx_words_ = 0;
  pending_dev_bytes_ = 0;
  stats_.pending_jobs = 0;
  stats_.pending_hatx = 0;
  stats_.pending_device_bytes = 0;
  if (store_results_) {
    populate_completed_();
  } else {
    populate_device_views_();
    for (const auto& job : jobs_) {
      if (job.token < slots_.size() && slots_[job.token]) {
        slots_[job.token]->ready.store(true);
      }
    }
  }
  if (prof) {
    runtime::bench::add_online_ns(runtime::bench::OnlineTimeKind::PfssFlushEvalStageHatx, ns_stage_hatx);
    runtime::bench::add_online_ns(runtime::bench::OnlineTimeKind::PfssFlushEvalEval, ns_eval);
    runtime::bench::add_online_ns(runtime::bench::OnlineTimeKind::PfssFlushEvalStageOut, ns_stage_out);
    runtime::bench::add_online_ns(runtime::bench::OnlineTimeKind::PfssFlushEvalTotal,
                                  prof_ns(t_total0, prof_now()));
  }
}

PfssResultView PfssSuperBatch::view(const PfssHandle& h) const {
  PfssResultView v;
  if (h.slot && h.slot->ready.load()) {
    if (h.slot->arith_storage) {
      v.arith = h.slot->arith_storage->data();
      v.arith_words = h.slot->arith_storage->size();
    }
    if (h.slot->bool_storage) {
      v.bools = h.slot->bool_storage->data();
      v.bool_words = h.slot->bool_storage->size();
    }
    v.arith_device = h.slot->arith_device;
    v.arith_device_words = h.slot->arith_device_words;
    v.bools_device = h.slot->bools_device;
    v.bools_device_words = h.slot->bools_device_words;
    v.r = h.slot->r;
    v.ell = h.slot->ell;
    return v;
  }
  if (h.token >= completed_.size()) return v;
  const auto& cj = completed_[h.token];
  v.arith = cj.arith.data();
  v.arith_words = cj.arith.size();
  v.bools = cj.bools.data();
  v.bool_words = cj.bools.size();
  if (h.token < slices_.size()) {
    const auto& sl = slices_[h.token];
    if (sl.group_result < group_results_.size()) {
      const auto& gr = group_results_[sl.group_result];
      const uint64_t* dev_arith_base = nullptr;
      if (gr.dev_arith_owner) {
        dev_arith_base = reinterpret_cast<const uint64_t*>(gr.dev_arith_owner.get());
      } else if (gr.dev_arith.ptr) {
        dev_arith_base = reinterpret_cast<const uint64_t*>(gr.dev_arith.ptr);
      } else if (gr.dev_arith_ptr) {
        dev_arith_base = gr.dev_arith_ptr;
      }
      if (dev_arith_base) {
        v.arith_device = dev_arith_base + sl.start * cj.r;
        v.arith_device_words = sl.len * cj.r;
      }

      const uint64_t* dev_bools_base = nullptr;
      if (gr.dev_bools_owner) {
        dev_bools_base = reinterpret_cast<const uint64_t*>(gr.dev_bools_owner.get());
      } else if (gr.dev_bools.ptr) {
        dev_bools_base = reinterpret_cast<const uint64_t*>(gr.dev_bools.ptr);
      } else if (gr.dev_bools_ptr) {
        dev_bools_base = gr.dev_bools_ptr;
      }
      if (dev_bools_base) {
        v.bools_device = dev_bools_base + sl.start * cj.ell;
        v.bools_device_words = sl.len * cj.ell;
      }
    }
  }
  v.r = cj.r;
  v.ell = cj.ell;
  return v;
}

PfssSharedResult PfssSuperBatch::view_shared(const PfssHandle& h) const {
  PfssSharedResult out;
  if (h.slot && h.slot->ready.load()) {
    out.r = h.slot->r;
    out.ell = h.slot->ell;
    out.arith = h.slot->arith_storage;
    out.bools = h.slot->bool_storage;
    return out;
  }
  if (h.token >= completed_.size()) return out;
  const auto& cj = completed_[h.token];
  out.r = cj.r;
  out.ell = cj.ell;
  out.arith = std::make_shared<std::vector<uint64_t>>(cj.arith);
  out.bools = std::make_shared<std::vector<uint64_t>>(cj.bools);
  return out;
}

void PfssSuperBatch::populate_completed_() {
  // Populate per-job completed slices (without running hooks).
  for (size_t idx = 0; idx < jobs_.size(); ++idx) {
    if (idx >= slices_.size()) continue;
    const auto& sl = slices_[idx];
    if (sl.group_result == static_cast<size_t>(-1)) continue;
    if (sl.group_result >= group_results_.size()) continue;
    if (idx >= completed_.size()) continue;
    const auto& job = jobs_[idx];
    const auto& gr = group_results_[sl.group_result];
    size_t r = gr.r;
    size_t ell = gr.ell;
    size_t arith_words = sl.len * r;
    size_t bool_words = sl.len * ell;
    const uint64_t* arith_base = gr.arith.data() + sl.start * r;
    const uint64_t* bool_base =
        (ell > 0 && !gr.bools.empty()) ? (gr.bools.data() + sl.start * ell) : nullptr;

    CompletedJob& cj = completed_[job.token];
    cj.r = r;
    cj.ell = ell;
    cj.arith.assign(arith_base, arith_base + arith_words);
    if (bool_base && bool_words > 0) {
      cj.bools.assign(bool_base, bool_base + bool_words);
    } else {
      cj.bools.clear();
    }
    if (job.token < slots_.size() && slots_[job.token]) {
      auto& slot = slots_[job.token];
      slot->r = r;
      slot->ell = ell;
      if (!slot->arith_storage) slot->arith_storage = std::make_shared<std::vector<uint64_t>>();
      if (!slot->bool_storage) slot->bool_storage = std::make_shared<std::vector<uint64_t>>();
      slot->arith_storage->assign(arith_base, arith_base + arith_words);
      if (bool_base && bool_words > 0) {
        slot->bool_storage->assign(bool_base, bool_base + bool_words);
      } else {
        slot->bool_storage->clear();
      }

      // Optional device views for this slice (if staged by flush_eval()).
      slot->arith_device_owner.reset();
      slot->arith_device = nullptr;
      slot->arith_device_words = 0;
      slot->bools_device_owner.reset();
      slot->bools_device = nullptr;
      slot->bools_device_words = 0;

      const uint64_t* dev_arith_base = nullptr;
      if (gr.dev_arith_owner) {
        dev_arith_base = reinterpret_cast<const uint64_t*>(gr.dev_arith_owner.get());
        slot->arith_device_owner = gr.dev_arith_owner;
      } else if (gr.dev_arith.ptr) {
        dev_arith_base = reinterpret_cast<const uint64_t*>(gr.dev_arith.ptr);
      } else if (gr.dev_arith_ptr) {
        dev_arith_base = gr.dev_arith_ptr;
      }
      if (dev_arith_base) {
        slot->arith_device = dev_arith_base + sl.start * r;
        slot->arith_device_words = arith_words;
      }

      const uint64_t* dev_bools_base = nullptr;
      if (gr.dev_bools_owner) {
        dev_bools_base = reinterpret_cast<const uint64_t*>(gr.dev_bools_owner.get());
        slot->bools_device_owner = gr.dev_bools_owner;
      } else if (gr.dev_bools.ptr) {
        dev_bools_base = reinterpret_cast<const uint64_t*>(gr.dev_bools.ptr);
      } else if (gr.dev_bools_ptr) {
        dev_bools_base = gr.dev_bools_ptr;
      }
      if (dev_bools_base) {
        slot->bools_device = dev_bools_base + sl.start * ell;
        slot->bools_device_words = bool_words;
      }

      slot->ready.store(true);
    }
  }
}

void PfssSuperBatch::populate_device_views_() {
  // Populate per-handle device slice pointers/owners from group results (no host copies).
  for (size_t idx = 0; idx < jobs_.size(); ++idx) {
    if (idx >= slices_.size()) continue;
    const auto& sl = slices_[idx];
    if (sl.group_result == static_cast<size_t>(-1)) continue;
    if (sl.group_result >= group_results_.size()) continue;
    const auto& job = jobs_[idx];
    if (job.token >= slots_.size()) continue;
    auto& slot = slots_[job.token];
    if (!slot) continue;
    const auto& gr = group_results_[sl.group_result];
    const size_t r = gr.r;
    const size_t ell = gr.ell;

    slot->r = r;
    slot->ell = ell;
    slot->arith_device_owner.reset();
    slot->arith_device = nullptr;
    slot->arith_device_words = 0;
    slot->bools_device_owner.reset();
    slot->bools_device = nullptr;
    slot->bools_device_words = 0;

    const uint64_t* dev_arith_base = nullptr;
    if (gr.dev_arith_owner) {
      dev_arith_base = reinterpret_cast<const uint64_t*>(gr.dev_arith_owner.get());
      slot->arith_device_owner = gr.dev_arith_owner;
    } else if (gr.dev_arith.ptr) {
      dev_arith_base = reinterpret_cast<const uint64_t*>(gr.dev_arith.ptr);
    } else if (gr.dev_arith_ptr) {
      dev_arith_base = gr.dev_arith_ptr;
    }
    if (dev_arith_base) {
      slot->arith_device = dev_arith_base + sl.start * r;
      slot->arith_device_words = sl.len * r;
    }

    const uint64_t* dev_bools_base = nullptr;
    if (gr.dev_bools_owner) {
      dev_bools_base = reinterpret_cast<const uint64_t*>(gr.dev_bools_owner.get());
      slot->bools_device_owner = gr.dev_bools_owner;
    } else if (gr.dev_bools.ptr) {
      dev_bools_base = reinterpret_cast<const uint64_t*>(gr.dev_bools.ptr);
    } else if (gr.dev_bools_ptr) {
      dev_bools_base = gr.dev_bools_ptr;
    }
    if (dev_bools_base) {
      slot->bools_device = dev_bools_base + sl.start * ell;
      slot->bools_device_words = sl.len * ell;
    }
  }
}

void PfssSuperBatch::clear() {
  if (gpu_stager_) {
    for (auto& gr : group_results_) {
      if (gr.dev_arith.ptr && !gr.dev_arith_owner) {
        gpu_stager_->free_bytes(gr.dev_arith);
        gr.dev_arith = DeviceBufferRef{};
      }
      if (gr.dev_bools.ptr && !gr.dev_bools_owner) {
        gpu_stager_->free_bytes(gr.dev_bools);
        gr.dev_bools = DeviceBufferRef{};
      }
      gr.dev_arith_owner.reset();
      gr.dev_bools_owner.reset();
    }
  }
  jobs_.clear();
  group_results_.clear();
  completed_.clear();
  slices_.clear();
  slots_.clear();
  flushed_ = false;
  pending_jobs_ = 0;
  pending_hatx_words_ = 0;
  stats_.pending_jobs = 0;
  stats_.pending_hatx = 0;
  stats_.pending_device_bytes = 0;
}

void PfssSuperBatch::finalize_all(int party, proto::IChannel& ch) {
  if (!flushed_) return;
  const bool prof = runtime::bench::online_profiling_enabled();
  const auto t0 = prof ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
  populate_device_views_();
  if (!store_results_) {
    // Scratch buffer for hook outputs to avoid per-job allocations/zeroing.
    std::vector<uint64_t> arith_hooked;
    for (size_t idx = 0; idx < jobs_.size(); ++idx) {
      const auto& job = jobs_[idx];
      if (idx >= slices_.size()) continue;
      const auto& sl = slices_[idx];
      if (sl.group_result == static_cast<size_t>(-1)) continue;
      if (sl.group_result >= group_results_.size()) continue;
      const auto& gr = group_results_[sl.group_result];

      const size_t r = gr.r;
      const size_t ell = gr.ell;
      const size_t arith_words = sl.len * r;
      const size_t bool_words = sl.len * ell;
      const uint64_t* arith_base = gr.arith.data() + sl.start * r;
      const uint64_t* bool_base =
          (ell > 0 && !gr.bools.empty()) ? (gr.bools.data() + sl.start * ell) : nullptr;

      const uint64_t* arith_in = arith_base;
      if (job.hook && job.out.data) {
        job.hook->configure(job.key->compiled.layout);
        // Hooks write all arithmetic outputs; no need to memset the buffer.
        arith_hooked.resize(arith_words);

        const bool hook_needs_mul =
            !(dynamic_cast<const gates::FaithfulTruncPostProc*>(job.hook) ||
              dynamic_cast<const gates::FaithfulArsPostProc*>(job.hook) ||
              dynamic_cast<const gates::GapArsPostProc*>(job.hook));
        const size_t need_triples = hook_needs_mul ? std::max(arith_words, bool_words) : 0;

        const std::vector<proto::BeaverTriple64Share>* triples =
            (job.key != nullptr) ? &job.key->triples : nullptr;
        if (need_triples > 0) {
          if (!triples || triples->empty()) {
            throw std::runtime_error("PfssSuperBatch::finalize_all: missing Beaver triples for hook");
          }
          if (triples->size() < need_triples) {
            throw std::runtime_error("PfssSuperBatch::finalize_all: insufficient Beaver triples for hook");
          }
        }
        proto::BeaverMul64 mul{party, ch, hook_needs_mul ? *triples : k_empty_triples, 0};
        job.hook->run_batch(party,
                            ch,
                            mul,
                            job.hatx_public.data(),
                            arith_base,
                            r,
                            bool_base,
                            ell,
                            sl.len,
                            arith_hooked.data());
        arith_in = arith_hooked.data();
      }

      if (job.out.data && job.key) {
        const size_t out_cap = job.out.numel();
        const size_t nwrite = std::min(arith_words, out_cap);
#ifdef _OPENMP
#pragma omp parallel for if (nwrite >= (1ull << 15)) schedule(static)
#endif
        for (size_t out_idx = 0; out_idx < nwrite; ++out_idx) {
          const size_t rr = (r > 0) ? (out_idx % r) : 0;
          uint64_t rout = (rr < job.key->r_out_share.size()) ? job.key->r_out_share[rr] : 0ull;
          job.out.data[out_idx] = proto::sub_mod(arith_in[out_idx], rout);
        }
      }
      if (job.token < slots_.size() && slots_[job.token]) {
        auto& slot = slots_[job.token];
        slot->r = r;
        slot->ell = ell;
        slot->ready.store(true);
      }
    }
    jobs_.clear();
    slices_.clear();
    group_results_.clear();
    flushed_ = false;
    if (prof) {
      const auto t1 = std::chrono::steady_clock::now();
      runtime::bench::add_online_ns(
          runtime::bench::OnlineTimeKind::PfssFinalizeTotal,
          static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()));
    }
    return;
  }

  if (completed_.empty()) populate_completed_();
  // Scratch buffer for hook outputs to avoid per-job allocations/zeroing.
  std::vector<uint64_t> arith_hooked;
  for (size_t idx = 0; idx < jobs_.size(); ++idx) {
    const auto& job = jobs_[idx];
    if (job.token >= completed_.size()) continue;
    if (idx >= slices_.size()) continue;
    const auto& sl = slices_[idx];
    if (sl.group_result == static_cast<size_t>(-1)) continue;
    if (sl.group_result >= group_results_.size()) continue;
    const auto& gr = group_results_[sl.group_result];

    size_t r = gr.r;
    size_t ell = gr.ell;
    size_t arith_words = sl.len * r;
    size_t bool_words = sl.len * ell;
    const uint64_t* arith_base = gr.arith.data() + sl.start * r;
    const uint64_t* bool_base =
        (ell > 0 && !gr.bools.empty()) ? (gr.bools.data() + sl.start * ell) : nullptr;

    const uint64_t* arith_in = arith_base;
    if (job.hook) {
      job.hook->configure(job.key->compiled.layout);
      // Hooks write all arithmetic outputs; no need to memset the buffer.
      arith_hooked.resize(arith_words);

      const bool hook_needs_mul =
          !(dynamic_cast<const gates::FaithfulTruncPostProc*>(job.hook) ||
            dynamic_cast<const gates::FaithfulArsPostProc*>(job.hook) ||
            dynamic_cast<const gates::GapArsPostProc*>(job.hook));
      const size_t need_triples = hook_needs_mul ? std::max(arith_words, bool_words) : 0;

      const std::vector<proto::BeaverTriple64Share>* triples =
          (job.key != nullptr) ? &job.key->triples : nullptr;
      if (need_triples > 0) {
        if (!triples || triples->empty()) {
          throw std::runtime_error("PfssSuperBatch::finalize_all: missing Beaver triples for hook");
        }
        if (triples->size() < need_triples) {
          throw std::runtime_error("PfssSuperBatch::finalize_all: insufficient Beaver triples for hook");
        }
      }
      proto::BeaverMul64 mul{party, ch, hook_needs_mul ? *triples : k_empty_triples, 0};
      job.hook->run_batch(party,
                          ch,
                          mul,
                          job.hatx_public.data(),
                          arith_base,
                          r,
                          bool_base,
                          ell,
                          sl.len,
                          arith_hooked.data());
      arith_in = arith_hooked.data();
    }

    CompletedJob& cj = completed_[job.token];
    cj.r = r;
    cj.ell = ell;
    cj.arith.resize(arith_words);
    for (size_t i = 0; i < sl.len; ++i) {
      for (size_t rr = 0; rr < r; ++rr) {
        size_t out_idx = i * r + rr;
        uint64_t val = arith_in[out_idx];
        uint64_t rout = (rr < job.key->r_out_share.size()) ? job.key->r_out_share[rr] : 0ull;
        val = proto::sub_mod(val, rout);
        cj.arith[out_idx] = val;
        if (out_idx < job.out.numel() && job.out.data) {
          job.out.data[out_idx] = val;
        }
      }
    }
    if (bool_base && bool_words > 0) {
      cj.bools.assign(bool_base, bool_base + bool_words);
    } else {
      cj.bools.clear();
    }
    if (job.token < slots_.size() && slots_[job.token]) {
      auto& slot = slots_[job.token];
      slot->r = r;
      slot->ell = ell;
      if (!slot->arith_storage) slot->arith_storage = std::make_shared<std::vector<uint64_t>>();
      if (!slot->bool_storage) slot->bool_storage = std::make_shared<std::vector<uint64_t>>();
      slot->arith_storage->assign(cj.arith.begin(), cj.arith.end());
      slot->bool_storage->assign(cj.bools.begin(), cj.bools.end());
      slot->ready.store(true);
    }
  }
  jobs_.clear();
  slices_.clear();
  group_results_.clear();
  flushed_ = false;
  if (prof) {
    const auto t1 = std::chrono::steady_clock::now();
    runtime::bench::add_online_ns(
        runtime::bench::OnlineTimeKind::PfssFinalizeTotal,
        static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()));
  }
}

void PfssSuperBatch::materialize_host(int party, proto::IChannel& ch) {
  if (!flushed_) return;
  const bool prof = runtime::bench::online_profiling_enabled();
  const auto t0 = prof ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
  populate_device_views_();
  if (!store_results_) {
    for (size_t idx = 0; idx < jobs_.size(); ++idx) {
      const auto& job = jobs_[idx];
      if (idx >= slices_.size()) continue;
      const auto& sl = slices_[idx];
      if (sl.group_result == static_cast<size_t>(-1)) continue;
      if (sl.group_result >= group_results_.size()) continue;
      const auto& gr = group_results_[sl.group_result];

      const size_t r = gr.r;
      const size_t ell = gr.ell;
      const size_t arith_words = sl.len * r;
      const size_t bool_words = sl.len * ell;
      const uint64_t* arith_base = gr.arith.data() + sl.start * r;
      const uint64_t* bool_base =
          (ell > 0 && !gr.bools.empty()) ? (gr.bools.data() + sl.start * ell) : nullptr;

      const uint64_t* arith_in = arith_base;
      std::vector<uint64_t> arith_hooked;
      if (job.hook && job.out.data) {
        job.hook->configure(job.key->compiled.layout);
        arith_hooked.assign(arith_words, 0);

        const bool hook_needs_mul =
            !(dynamic_cast<const gates::FaithfulTruncPostProc*>(job.hook) ||
              dynamic_cast<const gates::FaithfulArsPostProc*>(job.hook) ||
              dynamic_cast<const gates::GapArsPostProc*>(job.hook));
        const size_t need_triples = hook_needs_mul ? std::max(arith_words, bool_words) : 0;

        const std::vector<proto::BeaverTriple64Share>* triples =
            (job.key != nullptr) ? &job.key->triples : nullptr;
        if (need_triples > 0) {
          if (!triples || triples->empty()) {
            throw std::runtime_error("PfssSuperBatch::materialize_host: missing Beaver triples for hook");
          }
          if (triples->size() < need_triples) {
            throw std::runtime_error("PfssSuperBatch::materialize_host: insufficient Beaver triples for hook");
          }
        }
        proto::BeaverMul64 mul{party, ch, hook_needs_mul ? *triples : k_empty_triples, 0};
        job.hook->run_batch(party,
                            ch,
                            mul,
                            job.hatx_public.data(),
                            arith_base,
                            r,
                            bool_base,
                            ell,
                            sl.len,
                            arith_hooked.data());
        arith_in = arith_hooked.data();
      }

      if (job.out.data && job.key) {
        const size_t out_cap = job.out.numel();
        const size_t nwrite = std::min(arith_words, out_cap);
#ifdef _OPENMP
#pragma omp parallel for if (nwrite >= (1ull << 15)) schedule(static)
#endif
        for (size_t out_idx = 0; out_idx < nwrite; ++out_idx) {
          const size_t rr = (r > 0) ? (out_idx % r) : 0;
          uint64_t rout = (rr < job.key->r_out_share.size()) ? job.key->r_out_share[rr] : 0ull;
          job.out.data[out_idx] = proto::sub_mod(arith_in[out_idx], rout);
        }
      }
      if (job.token < slots_.size() && slots_[job.token]) {
        auto& slot = slots_[job.token];
        slot->r = r;
        slot->ell = ell;
        slot->ready.store(true);
      }
    }
    if (prof) {
      const auto t1 = std::chrono::steady_clock::now();
      runtime::bench::add_online_ns(
          runtime::bench::OnlineTimeKind::PfssMaterializeHost,
          static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()));
    }
    return;
  }
  if (completed_.empty()) populate_completed_();
  for (size_t idx = 0; idx < jobs_.size(); ++idx) {
    const auto& job = jobs_[idx];
    if (job.token >= completed_.size()) continue;
    if (idx >= slices_.size()) continue;
    const auto& sl = slices_[idx];
    if (sl.group_result == static_cast<size_t>(-1)) continue;
    if (sl.group_result >= group_results_.size()) continue;
    const auto& gr = group_results_[sl.group_result];

    size_t r = gr.r;
    size_t ell = gr.ell;
    size_t arith_words = sl.len * r;
    size_t bool_words = sl.len * ell;
    const uint64_t* arith_base = gr.arith.data() + sl.start * r;
    const uint64_t* bool_base =
        (ell > 0 && !gr.bools.empty()) ? (gr.bools.data() + sl.start * ell) : nullptr;

    const uint64_t* arith_in = arith_base;
    std::vector<uint64_t> arith_hooked;
    if (job.hook) {
      job.hook->configure(job.key->compiled.layout);
      arith_hooked.assign(arith_words, 0);

      const bool hook_needs_mul =
          !(dynamic_cast<const gates::FaithfulTruncPostProc*>(job.hook) ||
            dynamic_cast<const gates::FaithfulArsPostProc*>(job.hook) ||
            dynamic_cast<const gates::GapArsPostProc*>(job.hook));
      const size_t need_triples = hook_needs_mul ? std::max(arith_words, bool_words) : 0;

      const std::vector<proto::BeaverTriple64Share>* triples =
          (job.key != nullptr) ? &job.key->triples : nullptr;
      if (need_triples > 0) {
        if (!triples || triples->empty()) {
          throw std::runtime_error("PfssSuperBatch::materialize_host: missing Beaver triples for hook");
        }
        if (triples->size() < need_triples) {
          throw std::runtime_error("PfssSuperBatch::materialize_host: insufficient Beaver triples for hook");
        }
      }
      proto::BeaverMul64 mul{party, ch, hook_needs_mul ? *triples : k_empty_triples, 0};
      job.hook->run_batch(party,
                          ch,
                          mul,
                          job.hatx_public.data(),
                          arith_base,
                          r,
                          bool_base,
                          ell,
                          sl.len,
                          arith_hooked.data());
      arith_in = arith_hooked.data();
    }

    CompletedJob& cj = completed_[job.token];
    cj.r = r;
    cj.ell = ell;
    cj.arith.resize(arith_words);
    for (size_t i = 0; i < sl.len; ++i) {
      for (size_t rr = 0; rr < r; ++rr) {
        size_t out_idx = i * r + rr;
        uint64_t val = arith_in[out_idx];
        uint64_t rout = (rr < job.key->r_out_share.size()) ? job.key->r_out_share[rr] : 0ull;
        val = proto::sub_mod(val, rout);
        cj.arith[out_idx] = val;
        if (out_idx < job.out.numel() && job.out.data) {
          job.out.data[out_idx] = val;
        }
      }
    }
    if (bool_base && bool_words > 0) {
      cj.bools.assign(bool_base, bool_base + bool_words);
    } else {
      cj.bools.clear();
    }
    if (job.token < slots_.size() && slots_[job.token]) {
      auto& slot = slots_[job.token];
      slot->r = r;
      slot->ell = ell;
      if (!slot->arith_storage) slot->arith_storage = std::make_shared<std::vector<uint64_t>>();
      if (!slot->bool_storage) slot->bool_storage = std::make_shared<std::vector<uint64_t>>();
      slot->arith_storage->assign(cj.arith.begin(), cj.arith.end());
      slot->bool_storage->assign(cj.bools.begin(), cj.bools.end());
      slot->ready.store(true);
    }
  }
  if (prof) {
    const auto t1 = std::chrono::steady_clock::now();
    runtime::bench::add_online_ns(
        runtime::bench::OnlineTimeKind::PfssMaterializeHost,
        static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()));
  }
}

void PfssSuperBatch::flush_and_finalize(int party,
                                        proto::PfssBackendBatch& backend,
                                        proto::IChannel& ch) {
  flush_eval(party, backend, ch);
  populate_completed_();
  finalize_all(party, ch);
  clear();
}

void run_truncation_now(int party,
                        proto::PfssBackendBatch& backend,
                        proto::IChannel& ch,
                        const compiler::TruncationLoweringResult& bundle,
                        const std::vector<uint64_t>& x_share,
                        std::vector<uint64_t>& y_share) {
  const auto& key = (party == 0) ? bundle.keys.k0 : bundle.keys.k1;
  const gates::PostProcHook* hook_ptr = (party == 0) ? bundle.hook0.get() : bundle.hook1.get();
  if (!hook_ptr) {
    throw std::runtime_error("run_truncation_now: missing truncation hook");
  }
  gates::PostProcHook* hook = const_cast<gates::PostProcHook*>(hook_ptr);
  hook->configure(key.compiled.layout);
  size_t N = x_share.size();
  std::vector<uint64_t> hatx_share(N);
  for (size_t i = 0; i < N; ++i) {
    uint64_t rin = (!key.r_in_share_vec.empty() && key.r_in_share_vec.size() > i)
                       ? key.r_in_share_vec[i]
                       : key.r_in_share;
    hatx_share[i] = proto::add_mod(x_share[i], rin);
  }
  std::vector<uint64_t> other(N, 0);
  if (party == 0) {
    ch.send_bytes(hatx_share.data(), N * sizeof(uint64_t));
    ch.recv_bytes(other.data(), N * sizeof(uint64_t));
  } else {
    ch.recv_bytes(other.data(), N * sizeof(uint64_t));
    ch.send_bytes(hatx_share.data(), N * sizeof(uint64_t));
  }
  std::vector<uint64_t> hatx_public(N);
  for (size_t i = 0; i < N; ++i) {
    hatx_public[i] = proto::add_mod(hatx_share[i], other[i]);
  }
  gates::CompositeBatchInput in{hatx_public.data(), N, nullptr};
  auto out = gates::composite_eval_batch_with_postproc(
      party, backend, ch, key, bundle.suf, in, *hook);
  uint64_t r_out_share = key.r_out_share.empty() ? 0ull : key.r_out_share[0];
  y_share.resize(N);
  for (size_t i = 0; i < N; ++i) {
    y_share[i] = proto::sub_mod(out.haty_share[i], r_out_share);
  }
}

}  // namespace runtime
