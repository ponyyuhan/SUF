#pragma once

#include <vector>
#include <cstdio>
#include <iostream>
#include <cstddef>
#include <string>
#include <type_traits>
#include <random>
#include <cstring>
#include <cstdlib>
#ifdef SUF_HAVE_CUDA
#include <cuda_runtime.h>
#endif
#if __has_include(<span>)
  #include <span>
#else
  #if !defined(SUF_SPAN_FALLBACK_DEFINED)
    #define SUF_SPAN_FALLBACK_DEFINED
    namespace std {
      template<typename T>
      class span {
       public:
        span() : data_(nullptr), size_(0) {}
        span(const T* ptr, std::size_t n) : data_(ptr), size_(n) {}
        template <typename U, typename = std::enable_if_t<std::is_same_v<std::remove_const_t<T>, U>>>
        span(const std::vector<U>& v) : data_(v.data()), size_(v.size()) {}
        std::size_t size() const { return size_; }
        const T* data() const { return data_; }
        T* data() { return const_cast<T*>(data_); }
        const T& operator[](std::size_t i) const { return data_[i]; }
        T& operator[](std::size_t i) { return const_cast<T&>(data_[i]); }
        span subspan(std::size_t off, std::size_t n) const {
          if (off > size_) return span();
          std::size_t len = (off + n > size_) ? (size_ - off) : n;
          return span(data_ + off, len);
        }
        const T* begin() const { return data_; }
        const T* end() const { return data_ + size_; }
      private:
        const T* data_;
        std::size_t size_;
      };
      template <typename T>
      const T* begin(span<T> s) { return s.data(); }
      template <typename T>
      const T* end(span<T> s) { return s.data() + s.size(); }
    }
  #endif
#endif
#include <stdexcept>
#include <iostream>

#include "compiler/truncation_lowering.hpp"
#include "compiler/range_analysis.hpp"
#include "gates/postproc_hooks.hpp"
#include "gates/rsqrt_gate.hpp"
#include "gates/reciprocal_gate.hpp"
#include "gates/silu_spline_gate.hpp"
#include "gates/nexp_gate.hpp"
#include "proto/beaver.hpp"
#include "proto/beaver_mul64.hpp"
#include "proto/common.hpp"
#include "proto/backend_gpu.hpp"
#include "runtime/open_collector.hpp"
#include "runtime/phase_executor.hpp"
#include "runtime/pfss_superbatch.hpp"
#include "runtime/cuda_primitives.hpp"
#include "proto/reference_backend.hpp"

namespace runtime {

inline int64_t to_signed(uint64_t v) { return proto::to_signed(v); }
inline uint64_t to_ring(int64_t v) { return proto::from_signed(v); }

inline bool env_flag_enabled_default(const char* name, bool defv) {
  const char* env = std::getenv(name);
  if (!env) return defv;
  std::string v(env);
  std::transform(v.begin(), v.end(), v.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return !(v == "0" || v == "false" || v == "off" || v == "no");
}

inline size_t env_size_t_default(const char* name, size_t defv) {
  const char* env = std::getenv(name);
  if (!env) return defv;
  try {
    long long v = std::stoll(std::string(env));
    if (v <= 0) return defv;
    return static_cast<size_t>(v);
  } catch (...) {
    return defv;
  }
}

// Simple Beaver mul task (secret x secret). Uses OpenCollector if provided,
// otherwise falls back to direct channel opens.
class MulTask final : public detail::PhaseTask {
 public:
  MulTask(std::span<const uint64_t> x,
          std::span<const uint64_t> y,
          std::span<uint64_t> out,
          std::span<const proto::BeaverTriple64Share> triples)
      : x_(x), y_(y), out_(out), triples_(triples) {
    if (x_.size() != y_.size() || x_.size() != out_.size()) {
      throw std::runtime_error("MulTask: size mismatch");
    }
    if (triples_.size() < x_.size()) {
      throw std::runtime_error("MulTask: insufficient triples");
    }
  }

  bool done() const override { return st_ == St::Done; }

  detail::Need step(PhaseResources& R) override {
    switch (st_) {
      case St::Init: {
        if (!R.net_chan) throw std::runtime_error("MulTask: net channel missing");
        if (R.opens) {
          const size_t n = x_.size();
          auto res = R.opens->reserve(2 * n, OpenKind::kBeaver);
          for (size_t i = 0; i < n; ++i) {
            res.diff[i] = proto::sub_mod(x_[i], triples_[i].a);
            res.diff[n + i] = proto::sub_mod(y_[i], triples_[i].b);
          }
          h_ = res.handle;
          st_ = St::WaitOpen;
          // Treat enqueue as progress so PhaseExecutor can batch multiple opens
          // before forcing a flush.
          return detail::Need::None;
        }
        diff_.resize(2 * x_.size());
        for (size_t i = 0; i < x_.size(); ++i) {
          diff_[i] = proto::sub_mod(x_[i], triples_[i].a);
          diff_[x_.size() + i] = proto::sub_mod(y_[i], triples_[i].b);
        }
        opened_.assign(diff_.size(), 0);
        // Fallback direct open.
        for (size_t i = 0; i < diff_.size(); ++i) {
          if (R.party == 0) {
            R.net_chan->send_u64(diff_[i]);
            opened_[i] = proto::add_mod(diff_[i], R.net_chan->recv_u64());
          } else {
            opened_[i] = proto::add_mod(diff_[i], R.net_chan->recv_u64());
            R.net_chan->send_u64(diff_[i]);
          }
        }
        st_ = St::Finalize;
        [[fallthrough]];
      }
      case St::WaitOpen: {
        if (st_ == St::WaitOpen) {
          if (!R.opens) throw std::runtime_error("MulTask: no OpenCollector");
          if (!R.opens->ready(h_)) return detail::Need::Open;
#ifdef SUF_HAVE_CUDA
          // Prefer device opens when available (avoids a full D2H materialization).
          d_opened_ = R.opens->view_device_u64(h_);
          d_opened_words_ = d_opened_ ? (2 * x_.size()) : 0;
          if (!d_opened_) {
            auto v = R.opens->view_u64(h_);
            if (v.size() != 2 * x_.size()) {
              throw std::runtime_error("MulTask: opened size mismatch");
            }
            opened_.assign(v.begin(), v.end());
          }
#else
          auto v = R.opens->view_u64(h_);
          if (v.size() != 2 * x_.size()) {
            throw std::runtime_error("MulTask: opened size mismatch");
          }
          opened_.assign(v.begin(), v.end());
#endif
          st_ = St::Finalize;
        }
        [[fallthrough]];
      }
      case St::Finalize: {
        bool use_gpu = false;
#ifdef SUF_HAVE_CUDA
        const bool gpu_backend =
            (R.pfss_backend && dynamic_cast<proto::PfssGpuStagedEval*>(R.pfss_backend) != nullptr) ||
            (R.cuda_stream != nullptr);
        const bool want_gpu = env_flag_enabled_default("SUF_MUL_GPU", gpu_backend);
        const size_t min_elems =
            env_size_t_default("SUF_MUL_GPU_MIN_ELEMS", gpu_backend ? (1ull << 12) : (1ull << 20));
        if (!force_cpu_mul_ && want_gpu && x_.size() >= min_elems) use_gpu = true;
#endif
        if (use_gpu) {
#ifdef SUF_HAVE_CUDA
          cudaStream_t stream = nullptr;
          if (R.pfss_backend) {
            if (auto* staged = dynamic_cast<proto::PfssGpuStagedEval*>(R.pfss_backend)) {
              stream = reinterpret_cast<cudaStream_t>(staged->device_stream());
            }
          }
          size_t n = x_.size();
          size_t bytes = n * sizeof(uint64_t);
          uint64_t* d_out = nullptr;
          uint64_t* d_open_tmp = nullptr;
          void* d_tri = nullptr;
          auto do_malloc = [&](uint64_t** p, size_t sz) {
            cudaError_t st = cudaSuccess;
#if defined(CUDART_VERSION) && (CUDART_VERSION >= 11020)
            st = cudaMallocAsync(reinterpret_cast<void**>(p), sz, stream ? stream : 0);
            if (st == cudaErrorNotSupported) {
              // Some runtimes (or vGPU configurations) may not support async alloc.
              (void)cudaGetLastError();  // clear sticky "not supported" for later cudaGetLastError() checks
              st = cudaMalloc(reinterpret_cast<void**>(p), sz);
            }
#else
            (void)stream;
            st = cudaMalloc(reinterpret_cast<void**>(p), sz);
#endif
            if (st != cudaSuccess) {
              throw std::runtime_error(std::string("MulTask cudaMalloc failed: ") +
                                       cudaGetErrorString(st));
            }
          };
          auto do_free = [&](uint64_t* p) {
            if (!p) return;
            cudaError_t st = cudaSuccess;
#if defined(CUDART_VERSION) && (CUDART_VERSION >= 11020)
            st = cudaFreeAsync(p, stream ? stream : 0);
            if (st == cudaErrorNotSupported) {
              (void)cudaGetLastError();  // clear sticky "not supported"
              st = cudaFree(p);
            }
#else
            (void)stream;
            st = cudaFree(p);
#endif
            if (st != cudaSuccess) {
              throw std::runtime_error(std::string("MulTask cudaFree failed: ") +
                                       cudaGetErrorString(st));
            }
          };
          do_malloc(&d_out, bytes);
          const uint64_t* d_open = d_opened_;
          if (!d_open || d_opened_words_ < 2 * n) {
            if (opened_.empty()) {
              if (!R.opens) throw std::runtime_error("MulTask: no OpenCollector");
              auto v = R.opens->view_u64(h_);
              if (v.size() != 2 * n) throw std::runtime_error("MulTask: opened size mismatch");
              opened_.assign(v.begin(), v.end());
            }
            do_malloc(&d_open_tmp, 2 * bytes);
            cudaMemcpyAsync(d_open_tmp, opened_.data(), 2 * bytes, cudaMemcpyHostToDevice, stream);
            d_open = d_open_tmp;
          }
          {
            cudaError_t st = cudaMalloc(&d_tri, n * sizeof(proto::BeaverTriple64Share));
            if (st != cudaSuccess) {
              throw std::runtime_error(std::string("MulTask cudaMalloc(d_tri) failed: ") +
                                       cudaGetErrorString(st));
            }
          }
          {
            cudaError_t st = cudaMemcpyAsync(d_tri,
                                             triples_.data(),
                                             n * sizeof(proto::BeaverTriple64Share),
                                             cudaMemcpyHostToDevice,
                                             stream);
            if (st != cudaSuccess) {
              throw std::runtime_error(std::string("MulTask cudaMemcpyAsync(triples) failed: ") +
                                       cudaGetErrorString(st));
            }
          }
          launch_beaver_mul_aos_kernel(R.party,
                                       d_tri,
                                       d_open,
                                       d_open + n,
                                       d_out,
                                       n,
                                       stream);
          cudaMemcpyAsync(out_.data(), d_out, bytes, cudaMemcpyDeviceToHost, stream);
          cudaStreamSynchronize(stream);
          if (d_open_tmp) do_free(d_open_tmp);
          if (d_tri) cudaFree(d_tri);
          do_free(d_out);
#else
          (void)use_gpu;
#endif
        } else {
          if (opened_.empty() && R.opens) {
            auto v = R.opens->view_u64(h_);
            if (v.size() != 2 * x_.size()) {
              throw std::runtime_error("MulTask: opened size mismatch");
            }
            opened_.assign(v.begin(), v.end());
          }
          #pragma omp parallel for schedule(static)
          for (size_t i = 0; i < x_.size(); ++i) {
            uint64_t d = opened_[i];
            uint64_t e = opened_[x_.size() + i];
            uint64_t z = triples_[i].c;
            z = proto::add_mod(z, proto::mul_mod(d, triples_[i].b));
            z = proto::add_mod(z, proto::mul_mod(e, triples_[i].a));
            if (R.party == 0) {
              z = proto::add_mod(z, proto::mul_mod(d, e));
            }
            out_[i] = z;
          }
        }
        st_ = St::Done;
        return detail::Need::None;
      }
      case St::Done:
        return detail::Need::None;
    }
    return detail::Need::None;
  }

 private:
  enum class St { Init, WaitOpen, Finalize, Done } st_ = St::Init;
  std::span<const uint64_t> x_;
  std::span<const uint64_t> y_;
  std::span<uint64_t> out_;
	  std::span<const proto::BeaverTriple64Share> triples_;

	  std::vector<uint64_t> diff_;
	  std::vector<uint64_t> opened_;
	  OpenHandle h_{};
#ifdef SUF_HAVE_CUDA
  const uint64_t* d_opened_ = nullptr;  // points into OpenCollector device opened buffer
  size_t d_opened_words_ = 0;
#endif
	  bool force_cpu_mul_ = false; // preserve for future tuning
};

// Faithful truncation task (TR/ARS/GapARS) using existing composite bundle.
// Truncation with prepare/finalize split so PFSS can be flushed across phases.
class TruncTask final : public detail::PhaseTask {
 public:
  TruncTask(const compiler::TruncationLoweringResult* bundle,
            std::span<const uint64_t> in_share,
            std::span<uint64_t> out_share)
      : bundle_(bundle), in_(in_share), out_(out_share) {
    if (!bundle_) throw std::runtime_error("TruncTask: bundle null");
    if (in_.size() != out_.size()) {
      throw std::runtime_error("TruncTask: size mismatch");
    }
  }

  ~TruncTask() override {
#ifdef SUF_HAVE_CUDA
    if (d_hatx_public_) {
      cudaFree(d_hatx_public_);
      d_hatx_public_ = nullptr;
      d_hatx_words_ = 0;
    }
    if (d_out_device_) {
      cudaFree(d_out_device_);
      d_out_device_ = nullptr;
      d_out_elems_ = 0;
    }
#endif
  }

  bool done() const override { return st_ == St::Done; }
  bool prepared() const { return st_ == St::WaitPfss; }
  // Optional device output when trunc postproc ran on GPU in device-pipeline mode.
#ifdef SUF_HAVE_CUDA
  uint64_t* device_out() const { return d_out_device_; }
  size_t device_out_elems() const { return d_out_elems_; }
  void release_device_out() {
    if (d_out_device_) {
      cudaFree(d_out_device_);
      d_out_device_ = nullptr;
      d_out_elems_ = 0;
    }
  }
  // Transfer ownership of the device buffer to the caller without freeing.
  // Caller becomes responsible for cudaFree.
  uint64_t* take_device_out(size_t* elems_out = nullptr) {
    if (elems_out) *elems_out = d_out_elems_;
    uint64_t* p = d_out_device_;
    d_out_device_ = nullptr;
    d_out_elems_ = 0;
    return p;
  }
#endif
  void set_shape_hint(const std::vector<int>* row_offsets,
                      const std::vector<int>* row_lengths,
                      uint16_t eff_bits_hint = 0) {
    row_offsets_hint_ = row_offsets;
    row_lengths_hint_ = row_lengths;
    eff_bits_hint_ = eff_bits_hint;
  }
  // Optional device input for device-pipeline callers (non-owning).
  void set_device_input(const uint64_t* d_ptr, size_t elems) {
#ifdef SUF_HAVE_CUDA
    d_in_device_ = d_ptr;
    d_in_elems_ = elems;
#else
    (void)d_ptr; (void)elems;
#endif
  }

  detail::Need step(PhaseResources& R) override {
    if (!R.pfss_trunc || !R.pfss_backend || !R.pfss_chan) {
      throw std::runtime_error("TruncTask: PFSS resources missing");
    }
    if (!R.net_chan) {
      throw std::runtime_error("TruncTask: net channel missing");
    }
    // Resolve party-specific view lazily.
    if (!key_ && !per_element_) {
      if (!bundle_->per_elems.empty() && bundle_->per_elems.size() == in_.size()) {
        per_element_ = true;
        per_keys_.resize(in_.size());
        per_hooks_.resize(in_.size());
        for (size_t i = 0; i < in_.size(); ++i) {
          const auto& pe = bundle_->per_elems[i];
          per_keys_[i] = (R.party == 0) ? &pe.keys.k0 : &pe.keys.k1;
          auto* hk = (R.party == 0) ? pe.hook0.get() : pe.hook1.get();
          if (!hk) throw std::runtime_error("TruncTask: per-element hook missing");
          hk->configure(per_keys_[i]->compiled.layout);
          per_hooks_[i] = hk;
        }
      } else {
        key_ = (R.party == 0) ? &bundle_->keys.k0 : &bundle_->keys.k1;
        hook_ = (R.party == 0) ? bundle_->hook0.get() : bundle_->hook1.get();
        if (!hook_) throw std::runtime_error("TruncTask: hook missing");
        hook_->configure(key_->compiled.layout);
        if (std::getenv("SUF_TRUNC_KEY_TRACE")) {
          static std::atomic<bool> logged{false};
          bool expect = false;
          if (logged.compare_exchange_strong(expect, true)) {
            std::fprintf(stderr,
                         "[TruncTask] gate_kind=%d pred_sem=%d pred_out_bytes=%d triples=%zu ell=%d degree=%d\n",
                         static_cast<int>(key_->compiled.gate_kind),
                         static_cast<int>(key_->pred_meta.sem),
                         key_->pred_meta.out_bytes,
                         key_->triples.size(),
                         key_->compiled.ell,
                         key_->compiled.degree);
          }
        }
      }
    }

    switch (st_) {
      case St::OpenXhat: {
        if (R.opens) {
          auto res = R.opens->reserve(in_.size(), OpenKind::kMask);
          if (per_element_) {
            for (size_t i = 0; i < in_.size(); ++i) {
              if (!per_keys_[i]) throw std::runtime_error("TruncTask: missing per-element key");
              uint64_t rin = per_keys_[i]->r_in_share;
              res.diff[i] = proto::add_mod(in_[i], rin);
            }
          } else {
            for (size_t i = 0; i < in_.size(); ++i) {
              uint64_t rin = (key_->r_in_share_vec.size() > i) ? key_->r_in_share_vec[i] : key_->r_in_share;
              res.diff[i] = proto::add_mod(in_[i], rin);
            }
          }
          h_open_ = res.handle;
          st_ = St::WaitOpen;
          // Allow other tasks to enqueue opens before forcing a flush.
          return detail::Need::None;
        }
        masked_.resize(in_.size());
        if (per_element_) {
          for (size_t i = 0; i < in_.size(); ++i) {
            if (!per_keys_[i]) throw std::runtime_error("TruncTask: missing per-element key");
            uint64_t rin = per_keys_[i]->r_in_share;
            masked_[i] = proto::add_mod(in_[i], rin);
          }
        } else {
          for (size_t i = 0; i < in_.size(); ++i) {
            uint64_t rin = (key_->r_in_share_vec.size() > i) ? key_->r_in_share_vec[i] : key_->r_in_share;
            masked_[i] = proto::add_mod(in_[i], rin);
          }
        }
        opened_.assign(masked_.size(), 0);
        for (size_t i = 0; i < masked_.size(); ++i) {
          if (R.party == 0) {
            R.net_chan->send_u64(masked_[i]);
            opened_[i] = proto::add_mod(masked_[i], R.net_chan->recv_u64());
          } else {
            opened_[i] = proto::add_mod(masked_[i], R.net_chan->recv_u64());
            R.net_chan->send_u64(masked_[i]);
          }
        }
        st_ = St::EnqueuePfss;
        [[fallthrough]];
      }
      case St::WaitOpen: {
        if (st_ == St::WaitOpen) {
          if (!R.opens) throw std::runtime_error("TruncTask: no OpenCollector");
          if (!R.opens->ready(h_open_)) return detail::Need::Open;
          const size_t elems = in_.size();
#ifdef SUF_HAVE_CUDA
          // Device-only hatx is only safe/beneficial in device-pipeline mode, where
          // downstream consumers can keep data on GPU and avoid host staging.
          const bool want_device_hatx =
              env_flag_enabled_default("SUF_TRUNC_DEVICE_HATX", false) &&
              R.device_pipeline && R.pfss_backend &&
              dynamic_cast<proto::PfssGpuStagedEval*>(R.pfss_backend) != nullptr;
          if (want_device_hatx) {
            const uint64_t* d_hatx_src = R.opens->view_device_u64(h_open_);
            if (d_hatx_src) {
              // Ensure we own a stable device buffer across subsequent OpenCollector flushes.
              if (!d_hatx_public_ || d_hatx_words_ < elems) {
                if (d_hatx_public_) cudaFree(d_hatx_public_);
                d_hatx_public_ = nullptr;
                d_hatx_words_ = 0;
                cudaError_t st = cudaMalloc(reinterpret_cast<void**>(&d_hatx_public_), elems * sizeof(uint64_t));
                if (st != cudaSuccess) {
                  throw std::runtime_error(std::string("TruncTask cudaMalloc(d_hatx_public_) failed: ") +
                                           cudaGetErrorString(st));
                }
                d_hatx_words_ = elems;
              }
              cudaStream_t stream = nullptr;
              if (auto* staged = dynamic_cast<proto::PfssGpuStagedEval*>(R.pfss_backend)) {
                stream = reinterpret_cast<cudaStream_t>(staged->device_stream());
              }
              cudaError_t st = cudaMemcpyAsync(d_hatx_public_, d_hatx_src,
                                               elems * sizeof(uint64_t),
                                               cudaMemcpyDeviceToDevice,
                                               stream);
              if (st != cudaSuccess) {
                throw std::runtime_error(std::string("TruncTask cudaMemcpyAsync hatx D2D failed: ") +
                                         cudaGetErrorString(st));
              }
              // Skip host materialization; EnqueuePfss will use device-only hatx.
              opened_.clear();
              device_hatx_only_ = true;
              st_ = St::EnqueuePfss;
              break;
            }
          }
#endif
          auto v = R.opens->view_u64(h_open_);
          opened_.assign(v.begin(), v.end());
          device_hatx_only_ = false;
          if (R.pfss_backend &&
              dynamic_cast<proto::ReferenceBackend*>(R.pfss_backend) != nullptr &&
              !std::getenv("SUF_FORCE_PFSS")) {
            if (per_element_) {
              for (size_t i = 0; i < opened_.size(); ++i) {
                int shift = 0;
                const auto* k_i = per_keys_[i];
                if (!k_i) throw std::runtime_error("TruncTask: missing per-element key (ref backend)");
                if (!k_i->compiled.extra_u64.empty()) {
                  shift = static_cast<int>(k_i->compiled.extra_u64[0]);
                }
                uint64_t x_plain = proto::sub_mod(opened_[i], k_i->compiled.r_in);
                int64_t shifted = (shift >= 64) ? 0ll : (static_cast<int64_t>(x_plain) >> shift);
                out_[i] = (R.party == 0) ? static_cast<uint64_t>(shifted) : 0ull;
              }
            } else {
              int shift = 0;
              if (!key_->compiled.extra_u64.empty()) {
                shift = static_cast<int>(key_->compiled.extra_u64[0]);
              } else if (!bundle_->keys.k0.compiled.extra_u64.empty()) {
                shift = static_cast<int>(bundle_->keys.k0.compiled.extra_u64[0]);
              }
              for (size_t i = 0; i < opened_.size(); ++i) {
                uint64_t x_plain = proto::sub_mod(opened_[i], key_->compiled.r_in);
                int64_t shifted = (shift >= 64) ? 0ll : (static_cast<int64_t>(x_plain) >> shift);
                out_[i] = (R.party == 0) ? static_cast<uint64_t>(shifted) : 0ull;
              }
            }
            st_ = St::Done;
            return detail::Need::None;
          }
          st_ = St::EnqueuePfss;
        }
        [[fallthrough]];
      }
      case St::EnqueuePfss: {
        if (per_element_) {
          constexpr size_t kMaxEnqueuePerStep = 1024;
          pfss_handles_.clear();
          pfss_chunk_begin_ = pfss_next_elem_;
          size_t enqueued = 0;
          while (pfss_next_elem_ < opened_.size() && enqueued < kMaxEnqueuePerStep) {
            size_t elem = pfss_next_elem_;
            PreparedCompositeJob job;
            job.suf = &bundle_->per_elems[elem].suf;
            job.key = per_keys_[elem];
            // Per-element masks are primarily a correctness mode; always materialize via PfssSuperBatch
            // so the task doesn't need to `view()` and copy raw PFSS payloads.
            job.hook = per_hooks_[elem];
            job.hatx_public.resize(1);
            job.hatx_public[0] = opened_[elem];
            job.out = nn::TensorView<uint64_t>(&out_[elem], {1});
            try {
              pfss_handles_.push_back(R.pfss_trunc->enqueue_composite(std::move(job)));
              pfss_next_elem_++;
              enqueued++;
            } catch (const std::exception& e) {
              std::string msg = e.what();
              bool budget = (msg.find("PfssSuperBatch: pending") != std::string::npos);
              if (!budget || pfss_handles_.empty()) throw;
              break;
            }
          }
        } else {
          PreparedCompositeJob job;
          job.suf = &bundle_->suf;
          job.key = key_;
          job.hook = R.device_pipeline ? nullptr : hook_;
          const size_t elems = in_.size();
#ifdef SUF_HAVE_CUDA
          if (R.device_pipeline &&
              R.pfss_backend && dynamic_cast<proto::PfssGpuStagedEval*>(R.pfss_backend) != nullptr &&
              d_hatx_public_ && d_hatx_words_ >= elems) {
            job.hatx_public.clear();  // device-only hatx (GPU backend will use hatx_device)
            job.hatx_device = d_hatx_public_;
            job.hatx_device_words = elems;
            job.shape.total_elems = static_cast<uint32_t>(elems);
          } else
#endif
          {
            job.hatx_public.resize(opened_.size());
            if (!opened_.empty()) {
              std::memcpy(job.hatx_public.data(), opened_.data(), opened_.size() * sizeof(uint64_t));
            }
            if (R.device_pipeline && d_in_device_) {
              job.hatx_device = d_in_device_;
              job.hatx_device_words = std::min(d_in_elems_, opened_.size());
            }
          }
          if (!R.device_pipeline) {
            job.out = nn::TensorView<uint64_t>(out_.data(), {out_.size()});
          }
          if (row_offsets_hint_ && row_lengths_hint_ &&
              row_offsets_hint_->size() == row_lengths_hint_->size() + 1 &&
              !row_offsets_hint_->empty()) {
            job.row_offsets = *row_offsets_hint_;
            job.row_lengths = *row_lengths_hint_;
            job.shape.ragged = true;
            job.shape.num_rows = static_cast<uint16_t>(row_lengths_hint_->size());
            for (int L : *row_lengths_hint_) {
              job.shape.max_row_len =
                  std::max<uint16_t>(job.shape.max_row_len, static_cast<uint16_t>(L));
            }
            job.shape.total_elems = static_cast<uint32_t>(row_offsets_hint_->back());
          }
          if (eff_bits_hint_ > 0 && eff_bits_hint_ <= 64) {
            job.shape.eff_bits = eff_bits_hint_;
          }
          h_pfss_ = R.pfss_trunc->enqueue_composite(std::move(job));
        }
        st_ = St::WaitPfss;
        // Treat enqueue as progress so PhaseExecutor can batch multiple PFSS jobs
        // before forcing a flush/finalize cycle.
        return detail::Need::None;
      }
      case St::WaitPfss: {
        if (per_element_) {
          for (size_t i = 0; i < pfss_handles_.size(); ++i) {
            if (!R.pfss_trunc->ready(pfss_handles_[i])) return detail::Need::PfssTrunc;
          }
          pfss_handles_.clear();
          if (pfss_next_elem_ < opened_.size()) {
            st_ = St::EnqueuePfss;
            return detail::Need::None;
          }
          st_ = St::Done;
          return detail::Need::None;
        }
        if (!R.pfss_trunc->ready(h_pfss_)) return detail::Need::PfssTrunc;
        if (!R.device_pipeline) {
          st_ = St::Done;
          return detail::Need::None;
        }
        auto v = R.pfss_trunc->view(h_pfss_);
        size_t elems = in_.size();
        if (v.arith_words < elems * v.r) {
          throw std::runtime_error("TruncTask: PFSS arith slice too small");
        }
        bool gpu_direct = false;
        uint64_t* d_tmp_out = nullptr;
        cudaStream_t trunc_stream = nullptr;
#ifdef SUF_HAVE_CUDA
        if (R.device_pipeline && R.pfss_backend &&
            env_flag_enabled_default(
                "SUF_TRUNC_GPU",
                dynamic_cast<proto::PfssGpuStagedEval*>(R.pfss_backend) != nullptr) &&
            v.arith_device) {
            if (auto* staged = dynamic_cast<proto::PfssGpuStagedEval*>(R.pfss_backend)) {
              trunc_stream = reinterpret_cast<cudaStream_t>(staged->device_stream());
              gpu_direct = true;
	              auto do_malloc = [&](uint64_t** p, size_t sz) {
	                cudaError_t st = cudaSuccess;
	#if defined(CUDART_VERSION) && (CUDART_VERSION >= 11020)
	                st = cudaMallocAsync(reinterpret_cast<void**>(p), sz, trunc_stream ? trunc_stream : 0);
	                if (st == cudaErrorNotSupported) {
	                  (void)cudaGetLastError();  // clear sticky "not supported"
	                  st = cudaMalloc(reinterpret_cast<void**>(p), sz);
	                }
	#else
	                (void)trunc_stream;
	                st = cudaMalloc(reinterpret_cast<void**>(p), sz);
	#endif
                if (st != cudaSuccess) {
                  throw std::runtime_error(std::string("TruncTask cudaMalloc failed: ") +
                                           cudaGetErrorString(st));
                }
              };
	              auto do_free = [&](uint64_t* p) {
	                if (!p) return;
	                cudaError_t st = cudaSuccess;
	#if defined(CUDART_VERSION) && (CUDART_VERSION >= 11020)
	                st = cudaFreeAsync(p, trunc_stream ? trunc_stream : 0);
	                if (st == cudaErrorNotSupported) {
	                  (void)cudaGetLastError();  // clear sticky "not supported"
	                  st = cudaFree(p);
	                }
	#else
	                (void)trunc_stream;
	                st = cudaFree(p);
	#endif
                if (st != cudaSuccess) {
                  throw std::runtime_error(std::string("TruncTask cudaFree failed: ") +
                                           cudaGetErrorString(st));
                }
              };
              // Prefer the stable device hatx buffer when available (avoids host materialization).
              const uint64_t* d_hatx = nullptr;
              uint64_t* d_hatx_tmp = nullptr;
              if (d_hatx_public_ && d_hatx_words_ >= elems) {
                d_hatx = d_hatx_public_;
              } else {
                const uint64_t* hatx_host = job_hatx();
                do_malloc(&d_hatx_tmp, elems * sizeof(uint64_t));
                cudaMemcpyAsync(d_hatx_tmp, hatx_host, elems * sizeof(uint64_t),
                                cudaMemcpyHostToDevice, trunc_stream);
                d_hatx = d_hatx_tmp;
              }
              // Stage bools to device if needed.
              uint64_t* d_bools = nullptr;
              if (v.bools_device) {
                d_bools = const_cast<uint64_t*>(v.bools_device);
              } else if (v.bool_words >= elems * v.ell && v.ell > 0) {
                do_malloc(&d_bools, v.bool_words * sizeof(uint64_t));
                cudaMemcpyAsync(d_bools, v.bools, v.bool_words * sizeof(uint64_t),
                                cudaMemcpyHostToDevice, trunc_stream);
              }
              // IMPORTANT: d_tmp_out may escape the task in device-pipeline mode (via
              // d_out_device_/take_device_out). Allocate it with cudaMalloc so it can be
              // safely freed with cudaFree by downstream owners/destructors.
              {
                cudaError_t st = cudaMalloc(reinterpret_cast<void**>(&d_tmp_out),
                                            elems * sizeof(uint64_t));
                if (st != cudaSuccess) {
                  throw std::runtime_error(std::string("TruncTask cudaMalloc(d_tmp_out) failed: ") +
                                           cudaGetErrorString(st));
                }
              }
              int f_bits = 0;
              if (key_ && !key_->compiled.extra_u64.empty()) {
                f_bits = static_cast<int>(key_->compiled.extra_u64[0] & 0xFFFFu);
              } else if (bundle_ && !bundle_->keys.k0.compiled.extra_u64.empty()) {
                f_bits = static_cast<int>(bundle_->keys.k0.compiled.extra_u64[0] & 0xFFFFu);
              }
              int carry_idx = -1;
              int sign_idx = -1;
              int wrap_idx = -1;
              if (key_) {
                auto findb = [&](const std::string& name)->int {
                  for (size_t bi = 0; bi < key_->compiled.layout.bool_ports.size(); ++bi) {
                    if (key_->compiled.layout.bool_ports[bi] == name) return static_cast<int>(bi);
                  }
                  return -1;
                };
                carry_idx = findb("carry");
                sign_idx = findb("sign");
                wrap_idx = findb("wrap");
              }
              int kind_gapars = (key_ && key_->compiled.gate_kind == compiler::GateKind::GapARS) ? 1 : 0;
              uint64_t r_hi_share = key_ ? key_->r_hi_share : 0ull;
              uint64_t m_share = (kind_gapars && key_) ? key_->wrap_sign_share : 0ull;
              launch_trunc_postproc_kernel(R.party,
                                           kind_gapars,
                                           f_bits,
                                           r_hi_share,
                                           m_share,
                                           d_hatx,
                                           v.arith_device,
                                           v.r,
                                           /*arith_idx=*/0,
                                           d_bools,
                                           v.ell,
                                           carry_idx,
                                           sign_idx,
                                           wrap_idx,
                                           d_tmp_out,
                                           elems,
                                           trunc_stream);
              if (!R.device_pipeline) {
                cudaMemcpyAsync(out_.data(), d_tmp_out, elems * sizeof(uint64_t),
                                cudaMemcpyDeviceToHost, trunc_stream);
                cudaStreamSynchronize(trunc_stream);
                cudaFree(d_tmp_out);
                d_tmp_out = nullptr;
                d_out_elems_ = 0;
              } else {
                // Device-pipeline mode: keep the device buffer for downstream kernels.
                d_out_device_ = d_tmp_out;
                d_out_elems_ = elems;
                d_tmp_out = nullptr;  // keep ownership; freed after downstream use.
              }
              if (std::getenv("SOFTMAX_TRUNC_VALIDATE") && !R.device_pipeline) {
                // Run the host hook on the CPU view for comparison (does not overwrite unless mismatch).
                std::vector<uint64_t> hook_out(elems * v.r, 0);
                if (!hook_) throw std::runtime_error("TruncTask: hook missing for validate");
                // Use generous synthetic triples (same pattern as finalize_all).
                size_t generous_need = std::max<size_t>(512, elems * std::max<size_t>(v.ell, v.r));
                std::vector<proto::BeaverTriple64Share> synth_triples;
                synth_triples.reserve(generous_need);
                std::mt19937_64 rng(key_ ? key_->compiled.r_in ^ 0x7472756e63u : 0x7472756e63u);
                for (size_t t = 0; t < generous_need; ++t) {
                  uint64_t a = rng(), b = rng(), c = proto::mul_mod(a, b);
                  uint64_t a0 = rng(), b0 = rng(), c0 = rng();
                  synth_triples.push_back((R.party == 0)
                                              ? proto::BeaverTriple64Share{a0, b0, c0}
                                              : proto::BeaverTriple64Share{a - a0, b - b0, c - c0});
                }
                proto::BeaverMul64 mul{R.party, *R.pfss_chan, synth_triples, 0};
                hook_->run_batch(R.party,
                                 *R.pfss_chan,
                                 mul,
                                 hatx_public_.data(),
                                 v.arith,
                                 v.r,
                                 v.bools,
                                 v.ell,
                                 elems,
                                 hook_out.data());
                bool mismatch = false;
                size_t dump_n = std::min<size_t>(2, elems);
                for (size_t i = 0; i < elems; ++i) {
                  uint64_t gpu = out_[i];
                  uint64_t host_v = hook_out[i * v.r];
                  if (gpu != host_v) {
                    if (!mismatch) {
                      std::cerr << "[TruncTask p" << R.party << "] validate gpu vs hook f_bits=" << f_bits
                                << " kind=" << (kind_gapars ? "GapARS" : "faithful")
                                << " r=" << v.r << " ell=" << v.ell << "\n";
                    }
	                    mismatch = true;
	                    if (i < dump_n) {
	                      uint64_t base = v.arith[i * v.r];
	                      uint64_t carry = (carry_idx >= 0 && v.bools)
	                                           ? v.bools[i * v.ell + static_cast<size_t>(carry_idx)]
	                                           : 0ull;
	                      uint64_t sign = (sign_idx >= 0 && v.bools)
	                                          ? v.bools[i * v.ell + static_cast<size_t>(sign_idx)]
	                                          : 0ull;
	                      uint64_t wrap = (wrap_idx >= 0 && v.bools)
	                                          ? v.bools[i * v.ell + static_cast<size_t>(wrap_idx)]
	                                          : 0ull;
	                      uint64_t hx = hatx_public_[i];
	                      std::cerr << "  i=" << i
	                                << " hx=" << hx
	                                << " base=" << base
	                                << " carry=" << carry
	                                << " sign=" << sign
	                                << " wrap=" << wrap
	                                << " gpu=" << gpu
	                                << " hook=" << host_v
	                                << "\n";
	                    }
                    out_[i] = host_v;  // heal for downstream correctness.
                  }
                }
              }
              if (d_hatx_tmp) do_free(d_hatx_tmp);
              if (d_bools && d_bools != v.bools_device) do_free(d_bools);
            }
        }
#endif
        if (gpu_direct || std::getenv("SOFTMAX_TRUNC_DIRECT")) {
          if (!gpu_direct) {
            for (size_t i = 0; i < elems; ++i) {
              out_[i] = v.arith[i * v.r];
            }
          }
#ifdef SUF_HAVE_CUDA
          if (d_tmp_out) {
#if defined(CUDART_VERSION) && (CUDART_VERSION >= 11020)
            if (cudaFreeAsync(d_tmp_out, trunc_stream ? trunc_stream : 0) == cudaErrorNotSupported) {
              cudaFree(d_tmp_out);
            }
#else
            cudaFree(d_tmp_out);
#endif
          }
#endif
          st_ = St::Done;
          return detail::Need::None;
        }
        if (std::getenv("SOFTMAX_TRUNC_DUMP")) {
          std::cerr << "[TruncTask p" << R.party << "] r=" << v.r
                    << " ell=" << v.ell
                    << " arith0=" << (v.arith_words > 0 ? v.arith[0] : 0)
                    << " bool0=" << (v.bool_words > 0 ? v.bools[0] : 0)
                    << "\n";
        }
        std::vector<uint64_t> hook_out(elems * v.r, 0);
        const bool hook_needs_mul =
            !(dynamic_cast<const gates::FaithfulTruncPostProc*>(hook_) ||
              dynamic_cast<const gates::FaithfulArsPostProc*>(hook_) ||
              dynamic_cast<const gates::GapArsPostProc*>(hook_));
        size_t need_triples = hook_needs_mul ? std::max<size_t>(elems * v.ell, elems * v.r) : 0;
        const std::vector<proto::BeaverTriple64Share>* triples = &key_->triples;
        std::vector<proto::BeaverTriple64Share> tmp_triples;
        if (triples->size() < need_triples && !triples->empty()) {
          tmp_triples.reserve(need_triples);
          for (size_t t = 0; t < need_triples; ++t) {
            tmp_triples.push_back((*triples)[t % triples->size()]);
          }
          triples = &tmp_triples;
        }
        if (triples->size() < need_triples) {
          throw std::runtime_error("TruncTask: insufficient triples");
        }
        static const std::vector<proto::BeaverTriple64Share> k_empty_triples;
        proto::BeaverMul64 mul{R.party, *R.pfss_chan, hook_needs_mul ? *triples : k_empty_triples};
        hook_->run_batch(R.party, *R.pfss_chan, mul,
                         job_hatx(),
                         v.arith, v.r,
                         v.bools, v.ell,
                         elems,
                         hook_out.data());
        for (size_t i = 0; i < elems; ++i) {
          out_[i] = hook_out[i * v.r];
        }
        if (std::getenv("SOFTMAX_TRUNC_VALIDATE")) {
          size_t dump_n = std::min<size_t>(2, elems);
          int carry_idx = 0;
          int sign_idx = 1;
          if (key_) {
            auto findb = [&](const std::string& name)->int {
              for (size_t bi = 0; bi < key_->compiled.layout.bool_ports.size(); ++bi) {
                if (key_->compiled.layout.bool_ports[bi] == name) return static_cast<int>(bi);
              }
              return -1;
            };
            int c = findb("carry");
            int s = findb("sign");
            if (c >= 0) carry_idx = c;
            if (s >= 0) sign_idx = s;
          }
          int f_bits = 0;
          if (key_ && !key_->compiled.extra_u64.empty()) {
            f_bits = static_cast<int>(key_->compiled.extra_u64[0] & 0xFFFFu);
          } else if (bundle_ && !bundle_->keys.k0.compiled.extra_u64.empty()) {
            f_bits = static_cast<int>(bundle_->keys.k0.compiled.extra_u64[0] & 0xFFFFu);
          }
          int kind_gapars = (key_ && key_->compiled.gate_kind == compiler::GateKind::GapARS) ? 1 : 0;
          std::cerr << "[TruncTask p" << R.party << "] validate hook-only f_bits=" << f_bits
                    << " kind=" << (kind_gapars ? "GapARS" : "faithful")
                    << " r_hi_share=" << (key_ ? key_->r_hi_share : 0ull)
                    << " r=" << v.r << " ell=" << v.ell << "\n";
          for (size_t i = 0; i < dump_n; ++i) {
            uint64_t base = v.arith[i * v.r];
            uint64_t carry = (v.ell > 0 && v.bools) ? v.bools[i * v.ell + static_cast<size_t>(carry_idx)] : 0ull;
            uint64_t sign = (v.ell > static_cast<size_t>(sign_idx) && v.bools) ? v.bools[i * v.ell + static_cast<size_t>(sign_idx)] : 0ull;
            uint64_t hx = job_hatx()[i];
            uint64_t hookv = hook_out[i * v.r];
            std::cerr << "  i=" << i
                      << " hx=" << hx
                      << " base=" << base
                      << " carry=" << carry
                      << " sign=" << sign
                      << " hook_out=" << hookv
                      << "\n";
          }
        }
        st_ = St::Done;
        return detail::Need::None;
      }
      case St::Done:
        return detail::Need::None;
    }
    return detail::Need::None;
  }

 private:
  enum class St { OpenXhat, WaitOpen, EnqueuePfss, WaitPfss, Done } st_ = St::OpenXhat;
  const compiler::TruncationLoweringResult* bundle_ = nullptr;
  const gates::CompositePartyKey* key_ = nullptr;
  gates::PostProcHook* hook_ = nullptr;
  bool per_element_ = false;
  std::vector<const gates::CompositePartyKey*> per_keys_;
  std::vector<gates::PostProcHook*> per_hooks_;
  std::vector<PfssHandle> pfss_handles_;
  size_t pfss_next_elem_ = 0;
	  size_t pfss_chunk_begin_ = 0;
	  std::span<const uint64_t> in_;
	  std::span<uint64_t> out_;
	  std::vector<uint64_t> masked_;
	  std::vector<uint64_t> opened_;
	  OpenHandle h_open_{};
  PfssHandle h_pfss_{};
  const std::vector<int>* row_offsets_hint_ = nullptr;
  const std::vector<int>* row_lengths_hint_ = nullptr;
  uint16_t eff_bits_hint_ = 0;
  uint64_t* d_out_device_ = nullptr;  // optional device output when device_pipeline is enabled
  size_t d_out_elems_ = 0;
  const uint64_t* d_in_device_ = nullptr;  // optional device hatx input (non-owning)
  size_t d_in_elems_ = 0;
  bool device_hatx_only_ = false;
#ifdef SUF_HAVE_CUDA
  uint64_t* d_hatx_public_ = nullptr;  // stable device hatx (owned)
  size_t d_hatx_words_ = 0;
#endif

	  const uint64_t* job_hatx() {
	    if (hatx_public_.empty()) {
	      hatx_public_.resize(opened_.size());
	      for (size_t i = 0; i < opened_.size(); ++i) hatx_public_[i] = opened_[i];
	    }
	    return hatx_public_.data();
	  }

  std::vector<uint64_t> hatx_public_;
};

struct TruncChoice {
  const compiler::TruncationLoweringResult* gapars = nullptr;
  const compiler::TruncationLoweringResult* faithful = nullptr;
  int shift_bits = 0;
  bool signed_value = true;
};

inline const compiler::TruncationLoweringResult* select_trunc_bundle(
    const TruncChoice& choice,
    const compiler::RangeInterval& range,
    int frac_bits) {
  if (!choice.gapars || !choice.faithful) return choice.faithful;
  auto kind = compiler::select_trunc_kind(range, frac_bits);
  return (kind == compiler::GateKind::GapARS) ? choice.gapars : choice.faithful;
}

inline const compiler::TruncationLoweringResult* select_trunc_bundle(
    const TruncChoice& choice,
    const compiler::AbsBound& abs,
    int frac_bits,
    const std::optional<compiler::GapCert>& gap = std::nullopt) {
  if (!choice.gapars || !choice.faithful) return choice.faithful;
  auto kind = compiler::select_trunc_kind(abs, frac_bits, gap);
  return (kind == compiler::GateKind::GapARS) ? choice.gapars : choice.faithful;
}

struct CubicPolyBundle {
  const suf::SUF<uint64_t>* suf = nullptr;
  const gates::CompositePartyKey* key0 = nullptr;
  const gates::CompositePartyKey* key1 = nullptr;
  const compiler::TruncationLoweringResult* trunc_f = nullptr;
  const compiler::TruncationLoweringResult* trunc_2f = nullptr;
  int frac_bits = 0;
  compiler::GateKind gate_kind = compiler::GateKind::SiLUSpline;
  const gates::PiecewisePolySpec* spec = nullptr;
};

struct RecipTaskBundle {
  const suf::SUF<uint64_t>* suf = nullptr;  // affine init coeff program
  const gates::CompositePartyKey* key0 = nullptr;
  const gates::CompositePartyKey* key1 = nullptr;
  const compiler::TruncationLoweringResult* trunc_fb = nullptr;
  const gates::PiecewisePolySpec* init_spec = nullptr;
  int frac_bits = 0;
  int nr_iters = 1;
};

struct ReluBundle {
  const suf::SUF<uint64_t>* suf = nullptr;
  const gates::CompositePartyKey* key0 = nullptr;
  const gates::CompositePartyKey* key1 = nullptr;
};

// Generic composite-eval wrapper for ReLU (degree-1 piecewise). Runs the full
// Composite-FSS pipeline and writes unmasked arithmetic output shares.
class ReluTask final : public detail::PhaseTask {
 public:
  ReluTask(const ReluBundle& bundle,
           std::span<const uint64_t> in_share,
           std::span<uint64_t> out_share)
      : bundle_(bundle), in_(in_share), out_(out_share) {
    if (in_.size() != out_.size()) throw std::runtime_error("ReluTask: size mismatch");
    if (!bundle_.suf) throw std::runtime_error("ReluTask: missing SUF");
  }

  bool done() const override { return st_ == St::Done; }

  detail::Need step(PhaseResources& R) override {
    if (!R.net_chan) throw std::runtime_error("ReluTask: net channel missing");
    if (!R.pfss_backend || !R.pfss_chan) throw std::runtime_error("ReluTask: PFSS backend/channel missing");
    if (!R.pfss_trunc && !R.pfss_coeff) throw std::runtime_error("ReluTask: PFSS batches missing");
    if (!key_) {
      key_ = (R.party == 0) ? bundle_.key0 : bundle_.key1;
      if (!key_) throw std::runtime_error("ReluTask: missing party key");
    }
    auto* batch = R.pfss_trunc ? R.pfss_trunc : R.pfss_coeff;
    switch (st_) {
      case St::OpenXhat: {
        if (R.opens) {
          auto res = R.opens->reserve(in_.size(), OpenKind::kMask);
          for (size_t i = 0; i < in_.size(); ++i) {
            uint64_t rin = (key_->r_in_share_vec.size() > i) ? key_->r_in_share_vec[i] : key_->r_in_share;
            res.diff[i] = proto::add_mod(in_[i], rin);
          }
          h_open_ = res.handle;
          st_ = St::WaitOpen;
          return detail::Need::None;
        }
        masked_.resize(in_.size());
        for (size_t i = 0; i < in_.size(); ++i) {
          uint64_t rin = (key_->r_in_share_vec.size() > i) ? key_->r_in_share_vec[i] : key_->r_in_share;
          masked_[i] = proto::add_mod(in_[i], rin);
        }
        opened_.assign(masked_.size(), 0);
        for (size_t i = 0; i < masked_.size(); ++i) {
          if (R.party == 0) {
            R.net_chan->send_u64(masked_[i]);
            opened_[i] = proto::add_mod(masked_[i], R.net_chan->recv_u64());
          } else {
            opened_[i] = proto::add_mod(masked_[i], R.net_chan->recv_u64());
            R.net_chan->send_u64(masked_[i]);
          }
        }
        st_ = St::EnqueuePfss;
        [[fallthrough]];
      }
      case St::WaitOpen: {
        if (st_ == St::WaitOpen) {
          if (!R.opens) throw std::runtime_error("ReluTask: no OpenCollector");
          if (!R.opens->ready(h_open_)) return detail::Need::Open;
          auto v = R.opens->view_u64(h_open_);
          opened_.assign(v.begin(), v.end());
          st_ = St::EnqueuePfss;
        }
        [[fallthrough]];
      }
      case St::EnqueuePfss: {
        PreparedCompositeJob job;
        job.suf = bundle_.suf;
        job.key = key_;
        job.hook = nullptr;
        job.hatx_public.resize(opened_.size());
        if (!opened_.empty()) {
          std::memcpy(job.hatx_public.data(), opened_.data(), opened_.size() * sizeof(uint64_t));
        }
        job.out = nn::TensorView<uint64_t>(out_.data(), {out_.size()});
        h_pfss_ = batch->enqueue_composite(std::move(job));
        st_ = St::WaitPfss;
        return detail::Need::None;
      }
      case St::WaitPfss: {
        if (!batch->ready(h_pfss_)) return R.pfss_trunc ? detail::Need::PfssTrunc : detail::Need::PfssCoeff;
        st_ = St::Done;
        return detail::Need::None;
      }
      case St::Done:
        return detail::Need::None;
    }
    return detail::Need::None;
  }

 private:
  enum class St { OpenXhat, WaitOpen, EnqueuePfss, WaitPfss, Done } st_ = St::OpenXhat;
  ReluBundle bundle_{};
  std::span<const uint64_t> in_;
  std::span<uint64_t> out_;
  const gates::CompositePartyKey* key_ = nullptr;

  std::vector<uint64_t> masked_;
  std::vector<uint64_t> opened_;
  OpenHandle h_open_{};
  PfssHandle h_pfss_{};
};

struct RsqrtTaskBundle {
  const suf::SUF<uint64_t>* suf = nullptr;  // affine init coeff program
  const gates::CompositePartyKey* key0 = nullptr;
  const gates::CompositePartyKey* key1 = nullptr;
  const compiler::TruncationLoweringResult* trunc_f = nullptr;
  const compiler::TruncationLoweringResult* trunc_2f = nullptr;
  const gates::PiecewisePolySpec* init_spec = nullptr;
  int frac_bits = 0;
  int nr_iters = 1;
};

// NR-based rsqrt task: y <- y*(1.5 - 0.5*x*y^2), keeps shares throughout.
class RsqrtTask final : public detail::PhaseTask {
 public:
  RsqrtTask(const RsqrtTaskBundle& bundle,
            std::span<const uint64_t> x_qf,
            std::span<uint64_t> out_qf)
      : bundle_(bundle), x_(x_qf), out_(out_qf) {
    if (x_.size() != out_.size()) throw std::runtime_error("RsqrtTask: size mismatch");
    if (!bundle_.suf || !bundle_.trunc_f || !bundle_.trunc_2f) {
      throw std::runtime_error("RsqrtTask: bundle missing parts");
    }
  }

  bool done() const override { return st_ == St::Done; }
  const std::vector<uint64_t>& init_y_debug() const { return init_y_; }
  const std::vector<uint64_t>& xy2_f_debug() const { return xy2_f_last_; }
  const std::vector<uint64_t>& c0_debug() const { return c0_; }
  const std::vector<uint64_t>& c1_debug() const { return c1_; }
  const std::vector<uint64_t>& x_plain_debug() const { return x_plain_debug_; }
  int init_r_debug() const { return init_r_; }

  detail::Need step(PhaseResources& R) override {
    if (!key_) {
      key_ = (R.party == 0) ? bundle_.key0 : bundle_.key1;
      if (!key_) throw std::runtime_error("RsqrtTask: missing party key");
      triple_span_ = std::span<const proto::BeaverTriple64Share>(key_->triples);
      if (std::getenv("SOFTMAX_BENCH_TRACE")) {
        const bool have_vec = (key_->r_in_share_vec.size() == x_.size());
        std::fprintf(stderr, "[RsqrtTask p%d] r_in_share_vec size=%zu expected=%zu (using %s)\n",
                     R.party, key_->r_in_share_vec.size(), x_.size(), have_vec ? "vec" : "scalar");
      }
    }
    switch (st_) {
      case St::OpenXhat: {
        // Mask and open x (public) for init PFSS.
        if (!R.opens) throw std::runtime_error("RsqrtTask: no OpenCollector");
        auto res = R.opens->reserve(x_.size(), OpenKind::kMask);
        for (size_t i = 0; i < x_.size(); ++i) {
          uint64_t rin = (key_->r_in_share_vec.size() > i) ? key_->r_in_share_vec[i] : key_->r_in_share;
          res.diff[i] = proto::add_mod(x_[i], rin);
        }
        h_open_ = res.handle;
        st_ = St::WaitOpen;
        // Allow batching of opens across tasks.
        return detail::Need::None;
      }
      case St::WaitOpen: {
        if (!R.opens->ready(h_open_)) return detail::Need::Open;
        auto v = R.opens->view_u64(h_open_);
        opened_.assign(v.begin(), v.end());
        st_ = St::EnqueueInit;
        [[fallthrough]];
      }
      case St::EnqueueInit: {
        PreparedCompositeJob job;
        job.suf = bundle_.suf;
        job.key = key_;
        job.hook = nullptr;
        job.hatx_public.resize(opened_.size());
        if (!opened_.empty()) {
          std::memcpy(job.hatx_public.data(), opened_.data(), opened_.size() * sizeof(uint64_t));
        }
        const size_t elems = x_.size();
        const size_t r = static_cast<size_t>(std::max(0, key_->compiled.r));
        if (r == 0) throw std::runtime_error("RsqrtTask: compiled coeff gate has r=0");
        coeff_buf_.assign(elems * r, 0ull);
        job.out = nn::TensorView<uint64_t>(coeff_buf_.data(), {coeff_buf_.size()});
        if (!R.pfss_coeff) throw std::runtime_error("RsqrtTask: missing coeff PFSS batch");
        coeff_handle_ = R.pfss_coeff->enqueue_composite(std::move(job));
        st_ = St::WaitInit;
        // Allow batching of PFSS enqueues across tasks.
        return detail::Need::None;
      }
      case St::WaitInit: {
        if (!R.pfss_coeff->ready(coeff_handle_)) return detail::Need::PfssCoeff;
        size_t elems = x_.size();
        const size_t r = static_cast<size_t>(std::max(0, key_->compiled.r));
        if (r < 2 || coeff_buf_.size() < elems * r) {
          throw std::runtime_error("RsqrtTask: coeff payload too small");
        }
        init_r_ = static_cast<int>(r);
        c0_.assign(elems, 0);
        c1_.assign(elems, 0);
        y_.assign(elems, 0);
        for (size_t i = 0; i < elems; ++i) {
          c0_[i] = coeff_buf_[i * r + 0];
          c1_[i] = (r > 1) ? coeff_buf_[i * r + 1] : 0ull;
          uint64_t x_plain_ring = proto::sub_mod(opened_[i], key_->compiled.r_in);
          int64_t x_plain_signed = to_signed(x_plain_ring);
          x_plain_debug_.push_back(x_plain_ring);
          int64_t c0s = to_signed(c0_[i]);
          int64_t c1s = to_signed(c1_[i]);
          int64_t prod = static_cast<int64_t>((static_cast<__int128>(c1s) * static_cast<__int128>(x_plain_signed)) >> fb_);
          y_[i] = to_ring(c0s + prod);
        }
        init_y_ = y_;
        iter_ = 0;
        st_ = St::IterMul1;
        return detail::Need::None;
      }
      case St::IterMul1: {
        if (iter_ >= bundle_.nr_iters) {
          for (size_t i = 0; i < out_.size(); ++i) out_[i] = y_[i];
          st_ = St::Done;
          return detail::Need::None;
        }
        auto triples = next_triples(x_.size());
        y2_.assign(x_.size(), 0);
        mul1_ = std::make_unique<MulTask>(
            std::span<const uint64_t>(y_.data(), y_.size()),
            std::span<const uint64_t>(y_.data(), y_.size()),
            std::span<uint64_t>(y2_.data(), y2_.size()),
            triples);
        st_ = St::Mul1;
        return detail::Need::None;
      }
      case St::Mul1: {
        auto need = mul1_->step(R);
        if (!mul1_->done()) return need;
        y2_trunc_.assign(y2_.size(), 0);
        trunc_y2_ = std::make_unique<TruncTask>(
            bundle_.trunc_f,
            std::span<const uint64_t>(y2_.data(), y2_.size()),
            std::span<uint64_t>(y2_trunc_.data(), y2_trunc_.size()));
        st_ = St::TruncY2;
        return detail::Need::None;
      }
      case St::TruncY2: {
        auto need = trunc_y2_->step(R);
        if (!trunc_y2_->done()) return need;
#ifdef SUF_HAVE_CUDA
        if (R.device_pipeline) {
          if (auto* dptr = trunc_y2_->device_out()) {
            size_t n = std::min(y2_trunc_.size(), trunc_y2_->device_out_elems());
            if (n > 0) cudaMemcpy(y2_trunc_.data(), dptr, n * sizeof(uint64_t), cudaMemcpyDeviceToHost);
            trunc_y2_->release_device_out();
          }
        }
#endif
        y2_f_ = y2_trunc_;
        auto triples = next_triples(x_.size());
        xy2_.assign(x_.size(), 0);
        mul2_ = std::make_unique<MulTask>(
            std::span<const uint64_t>(x_.data(), x_.size()),
            std::span<const uint64_t>(y2_f_.data(), y2_f_.size()),
            std::span<uint64_t>(xy2_.data(), xy2_.size()),
            triples);
        st_ = St::Mul2;
        return detail::Need::None;
      }
      case St::Mul2: {
        auto need = mul2_->step(R);
        if (!mul2_->done()) return need;
        xy2_trunc_.assign(x_.size(), 0);
        trunc_xy2_ = std::make_unique<TruncTask>(
            bundle_.trunc_f,
            std::span<const uint64_t>(xy2_.data(), xy2_.size()),
            std::span<uint64_t>(xy2_trunc_.data(), xy2_trunc_.size()));
        st_ = St::TruncXY2;
        return detail::Need::None;
      }
      case St::TruncXY2: {
        auto need = trunc_xy2_->step(R);
        if (!trunc_xy2_->done()) return need;
#ifdef SUF_HAVE_CUDA
        if (R.device_pipeline) {
          if (auto* dptr = trunc_xy2_->device_out()) {
            size_t n = std::min(xy2_trunc_.size(), trunc_xy2_->device_out_elems());
            if (n > 0) cudaMemcpy(xy2_trunc_.data(), dptr, n * sizeof(uint64_t), cudaMemcpyDeviceToHost);
            trunc_xy2_->release_device_out();
          }
        }
#endif
        xy2_f_ = xy2_trunc_;
        xy2_f_last_ = xy2_f_;
        st_ = St::ComputeT;
        return detail::Need::None;
      }
      case St::ComputeT: {
        t_.assign(x_.size(), 0);
        uint64_t one5 = proto::add_mod(static_cast<uint64_t>((3ull << (fb_ - 1))), 0ull);  // 1.5 in Qf
        for (size_t i = 0; i < x_.size(); ++i) {
          uint64_t half_xy2 = xy2_f_[i] >> 1;
          uint64_t const_share = (R.party == 0) ? one5 : 0ull;
          t_[i] = proto::sub_mod(const_share, half_xy2);
        }
        auto triples = next_triples(x_.size());
        y_update_q2f_.assign(x_.size(), 0);
        mul3_ = std::make_unique<MulTask>(
            std::span<const uint64_t>(y_.data(), y_.size()),
            std::span<const uint64_t>(t_.data(), t_.size()),
            std::span<uint64_t>(y_update_q2f_.data(), y_update_q2f_.size()),
            triples);
        st_ = St::Mul3;
        return detail::Need::None;
      }
      case St::Mul3: {
        auto need = mul3_->step(R);
        if (!mul3_->done()) return need;
        y_new_.assign(y_update_q2f_.size(), 0);
        trunc_out_ = std::make_unique<TruncTask>(
            bundle_.trunc_f,
            std::span<const uint64_t>(y_update_q2f_.data(), y_update_q2f_.size()),
            std::span<uint64_t>(y_new_.data(), y_new_.size()));
        st_ = St::TruncOut;
        return detail::Need::None;
      }
      case St::TruncOut: {
        auto need = trunc_out_->step(R);
        if (!trunc_out_->done()) return need;
#ifdef SUF_HAVE_CUDA
        if (R.device_pipeline) {
          if (auto* dptr = trunc_out_->device_out()) {
            size_t n = std::min(y_new_.size(), trunc_out_->device_out_elems());
            if (n > 0) cudaMemcpy(y_new_.data(), dptr, n * sizeof(uint64_t), cudaMemcpyDeviceToHost);
            trunc_out_->release_device_out();
          }
        }
#endif
        y_ = y_new_;
        iter_ += 1;
        st_ = St::IterMul1;
        return detail::Need::None;
      }
      case St::Done:
        return detail::Need::None;
    }
    return detail::Need::None;
  }

 private:
  enum class St {
    OpenXhat,
    WaitOpen,
    EnqueueInit,
    WaitInit,
    InitMul,
    InitTrunc,
    IterMul1,
    Mul1,
    TruncY2,
    Mul2,
    TruncXY2,
    ComputeT,
    Mul3,
    TruncOut,
    Done
  } st_ = St::OpenXhat;
  RsqrtTaskBundle bundle_;
  const gates::CompositePartyKey* key_ = nullptr;
  std::span<const uint64_t> x_;
  std::span<uint64_t> out_;
  std::vector<uint64_t> masked_;
  std::vector<uint64_t> opened_;
  OpenHandle h_open_{};
  PfssHandle coeff_handle_{};

  std::span<const proto::BeaverTriple64Share> triple_span_;
  size_t triple_cursor_ = 0;

  std::vector<uint64_t> y_;       // current y (Qf)
  std::vector<uint64_t> y2_;      // Q2f
  std::vector<uint64_t> y2_trunc_; // Qf
  std::vector<uint64_t> y2_f_;    // Qf
  std::vector<uint64_t> xy2_;     // Q2f
  std::vector<uint64_t> xy2_trunc_; // Qf
  std::vector<uint64_t> xy2_f_;   // Qf
  std::vector<uint64_t> t_;       // Qf
  std::vector<uint64_t> y_update_q2f_; // Q2f
  std::vector<uint64_t> y_new_;   // Qf

  std::vector<uint64_t> coeff_buf_;
  std::vector<uint64_t> c0_;
  std::vector<uint64_t> c1_;
  std::vector<uint64_t> y_init_q2f_;
  std::vector<uint64_t> y_init_qf_;

  std::unique_ptr<MulTask> mul1_;
  std::unique_ptr<MulTask> mul2_;
  std::unique_ptr<MulTask> mul3_;
  std::unique_ptr<MulTask> init_mul_;
  std::unique_ptr<TruncTask> init_trunc_;
  std::unique_ptr<TruncTask> trunc_y2_;
  std::unique_ptr<TruncTask> trunc_xy2_;
  std::unique_ptr<TruncTask> trunc_out_;

  int iter_ = 0;
  int fb_ = bundle_.frac_bits;

  std::vector<uint64_t> init_y_;
  std::vector<uint64_t> xy2_f_last_;
  int init_r_ = 0;
  std::vector<uint64_t> x_plain_debug_;

#ifdef SUF_HAVE_CUDA
  uint64_t* d_out_device_ = nullptr;
  size_t d_out_elems_ = 0;
#endif

  std::span<const proto::BeaverTriple64Share> next_triples(size_t n) {
    if (triple_cursor_ + n > triple_span_.size()) {
      throw std::runtime_error("RsqrtTask: not enough triples");
    }
    auto s = triple_span_.subspan(triple_cursor_, n);
    triple_cursor_ += n;
    return s;
  }
};

struct RowBroadcastTriple {
  std::span<const uint64_t> A;
  std::span<const uint64_t> B;
  std::span<const uint64_t> C;
};

class RowBroadcastTripleProvider {
 public:
  virtual ~RowBroadcastTripleProvider() = default;
  virtual RowBroadcastTriple reserve_mul(int rows, int cols) = 0;
};

// Mul exp[row,col] * inv[row] using row-broadcast triples to cut open size.
class MulRowBroadcastTask final : public detail::PhaseTask {
 public:
  MulRowBroadcastTask(std::span<const uint64_t> mat,
                      std::span<const uint64_t> vec,
                      int rows,
                      int cols,
                      std::span<const int> valid_lens,
                      std::span<uint64_t> out,
                      RowBroadcastTripleProvider* triples,
                      bool device_only = false)
      : mat_(mat),
        vec_(vec),
        rows_(rows),
        cols_(cols),
        valid_lens_(valid_lens),
        out_(out),
        triple_provider_(triples),
        device_only_(device_only) {
    if (mat_.size() != out_.size()) throw std::runtime_error("MulRowBroadcastTask: size mismatch");
    if (static_cast<int>(mat_.size()) != rows * cols) throw std::runtime_error("MulRowBroadcastTask: dims mismatch");
  }

  ~MulRowBroadcastTask() override {
#ifdef SUF_HAVE_CUDA
    if (d_out_device_) {
      cudaFree(d_out_device_);
      d_out_device_ = nullptr;
      d_out_elems_ = 0;
    }
#endif
  }

  bool done() const override { return st_ == St::Done; }
#ifdef SUF_HAVE_CUDA
  uint64_t* device_out() const { return d_out_device_; }
  size_t device_out_elems() const { return d_out_elems_; }
  void release_device_out() {
    if (d_out_device_) {
      cudaFree(d_out_device_);
      d_out_device_ = nullptr;
      d_out_elems_ = 0;
    }
  }
#endif

  detail::Need step(PhaseResources& R) override {
    switch (st_) {
      case St::Init: {
        if (!triple_provider_) throw std::runtime_error("MulRowBroadcastTask: triple provider missing");
        triple_ = triple_provider_->reserve_mul(rows_, cols_);
        if (triple_.A.size() < mat_.size() || triple_.B.size() < static_cast<size_t>(rows_) ||
            triple_.C.size() < mat_.size()) {
          throw std::runtime_error("MulRowBroadcastTask: triple too small");
        }
        if (!R.opens) throw std::runtime_error("MulRowBroadcastTask: no OpenCollector");
        const size_t total_words = mat_.size() + static_cast<size_t>(rows_);
        auto res = R.opens->reserve(total_words, OpenKind::kBeaver);
        // D matrix
        for (int r = 0; r < rows_; ++r) {
          int L = (valid_lens_.size() == 0) ? cols_ : valid_lens_[r];
          for (int c = 0; c < cols_; ++c) {
            size_t idx = static_cast<size_t>(r * cols_ + c);
            if (c < L) {
              res.diff[idx] = proto::sub_mod(mat_[idx], triple_.A[idx]);
            } else {
              res.diff[idx] = 0;
            }
          }
        }
        // e vector
        size_t off_e = mat_.size();
        for (int r = 0; r < rows_; ++r) {
          res.diff[off_e + static_cast<size_t>(r)] = proto::sub_mod(vec_[r], triple_.B[r]);
        }
        h_open_ = res.handle;
        st_ = St::WaitOpen;
        // Allow batching of Beaver opens across tasks.
        return detail::Need::None;
      }
      case St::WaitOpen: {
        if (!R.opens->ready(h_open_)) return detail::Need::Open;
        size_t off_e = mat_.size();
        if (h_open_.len != mat_.size() + static_cast<size_t>(rows_)) {
          throw std::runtime_error("MulRowBroadcastTask: opened mismatch");
        }
        bool gpu_mul = false;
#ifdef SUF_HAVE_CUDA
        const bool gpu_backend =
            (R.pfss_backend && dynamic_cast<proto::PfssGpuStagedEval*>(R.pfss_backend) != nullptr) ||
            (R.cuda_stream != nullptr);
        const bool want_gpu = env_flag_enabled_default("SUF_MUL_GPU", gpu_backend);
        const size_t min_elems =
            env_size_t_default("SUF_MUL_GPU_MIN_ELEMS", gpu_backend ? (1ull << 12) : (1ull << 20));
        if (want_gpu && mat_.size() >= min_elems) gpu_mul = true;
#endif
        if (gpu_mul) {
#ifdef SUF_HAVE_CUDA
          cudaStream_t stream = nullptr;
          if (R.pfss_backend) {
            if (auto* staged = dynamic_cast<proto::PfssGpuStagedEval*>(R.pfss_backend)) {
              stream = reinterpret_cast<cudaStream_t>(staged->device_stream());
            }
          }
          size_t elems = mat_.size();
          size_t bytes_mat = elems * sizeof(uint64_t);
          const size_t bytes_rows = static_cast<size_t>(rows_) * sizeof(uint64_t);
          const uint64_t* d_opened = R.opens->view_device_u64(h_open_);
          const uint64_t* d_d_open = nullptr;
          const uint64_t* d_e_open_rows = nullptr;
          if (d_opened) {
            d_d_open = d_opened;
            d_e_open_rows = d_opened + off_e;
          }
          uint64_t *d_d = nullptr, *d_a = nullptr, *d_b_rows = nullptr, *d_c = nullptr, *d_e_rows = nullptr, *d_out = nullptr;
          int* d_valid = nullptr;
          bool own_d_d = false;
          bool own_d_e = false;
          if (!d_d_open) {
            cudaMalloc(&d_d, bytes_mat);
            own_d_d = true;
            d_d_open = d_d;
          }
          cudaMalloc(&d_a, bytes_mat);
          cudaMalloc(&d_b_rows, bytes_rows);
          cudaMalloc(&d_c, bytes_mat);
          if (!d_e_open_rows) {
            cudaMalloc(&d_e_rows, bytes_rows);
            own_d_e = true;
            d_e_open_rows = d_e_rows;
          }
          cudaMalloc(&d_out, bytes_mat);
          if (valid_lens_.size() != 0) {
            cudaMalloc(&d_valid, static_cast<size_t>(rows_) * sizeof(int));
            cudaMemcpyAsync(d_valid, valid_lens_.data(),
                            static_cast<size_t>(rows_) * sizeof(int),
                            cudaMemcpyHostToDevice, stream);
          }
          if (!d_opened) {
            auto opened = R.opens->view_u64(h_open_);
            if (opened.size() != mat_.size() + static_cast<size_t>(rows_)) {
              throw std::runtime_error("MulRowBroadcastTask: opened mismatch");
            }
            cudaMemcpyAsync(d_d, opened.data(), bytes_mat, cudaMemcpyHostToDevice, stream);
            std::vector<uint64_t> e_rows(static_cast<size_t>(rows_), 0);
            for (int r = 0; r < rows_; ++r) {
              e_rows[static_cast<size_t>(r)] = opened[off_e + static_cast<size_t>(r)];
            }
            cudaMemcpyAsync(d_e_rows, e_rows.data(), bytes_rows, cudaMemcpyHostToDevice, stream);
          }
          cudaMemcpyAsync(d_a, triple_.A.data(), bytes_mat, cudaMemcpyHostToDevice, stream);
          cudaMemcpyAsync(d_c, triple_.C.data(), bytes_mat, cudaMemcpyHostToDevice, stream);
          cudaMemcpyAsync(d_b_rows, triple_.B.data(), bytes_rows, cudaMemcpyHostToDevice, stream);
          launch_row_broadcast_mul_kernel(R.party,
                                          d_a,
                                          d_b_rows,
                                          d_c,
                                          d_d_open,
                                          d_e_open_rows,
                                          rows_,
                                          cols_,
                                          d_valid,
                                          d_out,
                                          elems,
                                          stream);
          const bool keep_device_out = (R.device_pipeline && device_only_);
          if (!keep_device_out) {
            cudaMemcpyAsync(out_.data(), d_out, bytes_mat, cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
          } else {
            d_out_device_ = d_out;
            d_out_elems_ = elems;
            d_out = nullptr;
          }
          if (own_d_d && d_d) cudaFree(d_d);
          if (d_a) cudaFree(d_a);
          if (d_b_rows) cudaFree(d_b_rows);
          if (d_c) cudaFree(d_c);
          if (own_d_e && d_e_rows) cudaFree(d_e_rows);
          if (d_valid) cudaFree(d_valid);
          if (d_out) cudaFree(d_out);
          st_ = St::Done;
          return detail::Need::None;
#endif
        }
        // CPU fallback: compute Z shares.
        auto opened = R.opens->view_u64(h_open_);
        if (opened.size() != mat_.size() + static_cast<size_t>(rows_)) {
          throw std::runtime_error("MulRowBroadcastTask: opened mismatch");
        }
        for (int r = 0; r < rows_; ++r) {
          int L = (valid_lens_.size() == 0) ? cols_ : valid_lens_[r];
          uint64_t e = opened[off_e + static_cast<size_t>(r)];
          for (int c = 0; c < cols_; ++c) {
            size_t idx = static_cast<size_t>(r * cols_ + c);
            if (c >= L) {
              out_[idx] = 0;
              continue;
            }
            uint64_t d = opened[idx];
            uint64_t z = triple_.C[idx];
            z = proto::add_mod(z, proto::mul_mod(d, triple_.B[r]));
            z = proto::add_mod(z, proto::mul_mod(e, triple_.A[idx]));
            if (R.party == 0) {
              z = proto::add_mod(z, proto::mul_mod(d, e));
            }
            out_[idx] = z;
          }
        }
        st_ = St::Done;
        return detail::Need::None;
      }
      case St::Done:
        return detail::Need::None;
    }
      return detail::Need::None;
  }

 private:
  enum class St { Init, WaitOpen, Done } st_ = St::Init;
  std::span<const uint64_t> mat_;
  std::span<const uint64_t> vec_;
  int rows_;
  int cols_;
  std::span<const int> valid_lens_;
  std::span<uint64_t> out_;
  RowBroadcastTripleProvider* triple_provider_ = nullptr;

  RowBroadcastTriple triple_;
  std::vector<uint64_t> buf_de_;
  OpenHandle h_open_{};
  uint64_t* d_out_device_ = nullptr;
  size_t d_out_elems_ = 0;
  bool device_only_ = false;
};

struct LayerNormTaskBundle {
  TruncChoice mean_trunc;  // shift=f (sum*invL Q2f -> Qf)
  TruncChoice var_trunc;   // shift=2f (var Q3f -> Qf)
  TruncChoice norm_trunc;  // shift=f (diff*invstd Q2f -> Qf)
  RsqrtTaskBundle rsqrt;
  uint64_t inv_len_qf = 0;  // public inv(L) in Qf (shared on both parties)
  uint64_t eps_qf = 0;      // eps in Qf, only party0 set
  int frac_bits = 0;
  std::span<const int64_t> gamma;  // Qf public
  std::span<const int64_t> beta;   // Qf public
  std::span<const proto::BeaverTriple64Share> mul_triples;  // for diff^2
  RowBroadcastTripleProvider* row_triples = nullptr;
  compiler::RangeInterval mean_range = compiler::RangeInterval::whole(true);
  compiler::RangeInterval var_range = compiler::RangeInterval::whole(true);
  compiler::RangeInterval norm_range = compiler::RangeInterval::whole(true);
};

// LayerNorm over rows*cols with secret shares, using RsqrtTask and batched mul/trunc.
class LayerNormTask final : public detail::PhaseTask {
 public:
  LayerNormTask(const LayerNormTaskBundle& bundle,
                std::span<const uint64_t> x_qf,
                std::span<uint64_t> out_qf,
                int rows,
                int cols)
      : bundle_(bundle), x_(x_qf), out_(out_qf), rows_(rows), cols_(cols) {
    if (x_.size() != out_.size()) throw std::runtime_error("LayerNormTask: size mismatch");
    if (rows_ * cols_ != static_cast<int>(x_.size())) {
      throw std::runtime_error("LayerNormTask: dims mismatch");
    }
  }

  bool done() const override { return st_ == St::Done; }
  const std::vector<uint64_t>& mu_qf_debug() const { return mu_qf_; }
  const std::vector<uint64_t>& mu_q2f_debug() const { return mu_q2f_; }
  const std::vector<uint64_t>& var_qf_debug() const { return var_qf_; }
  const std::vector<uint64_t>& var_q3f_debug() const { return var_q3f_; }
  const std::vector<uint64_t>& rsqrt_qf_debug() const { return rsqrt_out_; }
  const std::vector<uint64_t>& rsqrt_init_debug() const { return rsqrt_init_; }
  const std::vector<uint64_t>& rsqrt_xy2_f_debug() const { return rsqrt_xy2_f_; }
  const std::vector<uint64_t>& rsqrt_c0_debug() const { return rsqrt_c0_; }
  const std::vector<uint64_t>& rsqrt_c1_debug() const { return rsqrt_c1_; }
  int rsqrt_init_r_debug() const { return rsqrt_init_r_; }
  const std::vector<uint64_t>& rsqrt_x_plain_debug() const { return rsqrt_x_plain_; }
  const std::vector<uint64_t>& norm_qf_debug() const { return norm_qf_; }

  detail::Need step(PhaseResources& R) override {
    if (st_ == St::Mean) {
      auto* ref_backend =
          (R.pfss_backend) ? dynamic_cast<proto::ReferenceBackend*>(R.pfss_backend) : nullptr;
      if (ref_backend && !std::getenv("SUF_FORCE_PFSS")) {
        if (!R.net_chan) throw std::runtime_error("LayerNormTask: net channel missing");
        std::vector<uint64_t> other(x_.size(), 0);
        if (R.party == 0) {
          for (auto v : x_) R.net_chan->send_u64(v);
          for (size_t i = 0; i < x_.size(); ++i) other[i] = R.net_chan->recv_u64();
        } else {
          for (size_t i = 0; i < x_.size(); ++i) other[i] = R.net_chan->recv_u64();
          for (auto v : x_) R.net_chan->send_u64(v);
        }
        std::vector<int64_t> x_plain(x_.size(), 0);
        for (size_t i = 0; i < x_.size(); ++i) {
          x_plain[i] = static_cast<int64_t>(x_[i] + other[i]);
        }
        std::vector<uint64_t> out_plain(x_.size(), 0);
        int fb = bundle_.frac_bits;
        int64_t eps = static_cast<int64_t>(bundle_.eps_qf);
        const auto* init_spec = bundle_.rsqrt.init_spec;
        int nr_iters = bundle_.rsqrt.nr_iters;
        bool has_gamma = bundle_.gamma.data() != nullptr && bundle_.gamma.size() > 0;
        bool has_beta = bundle_.beta.data() != nullptr && bundle_.beta.size() > 0;
        for (int r = 0; r < rows_; ++r) {
          int64_t sum = 0;
          for (int c = 0; c < cols_; ++c) sum += x_plain[static_cast<size_t>(r * cols_ + c)];
          int64_t mu = (cols_ == 0) ? 0 : (sum / static_cast<int64_t>(cols_));
          int64_t var_acc = 0;
          for (int c = 0; c < cols_; ++c) {
            int64_t d = x_plain[static_cast<size_t>(r * cols_ + c)] - mu;
            __int128 sq = static_cast<__int128>(d) * static_cast<__int128>(d);
            var_acc += static_cast<int64_t>(sq >> fb);
          }
          int64_t var = (cols_ == 0) ? 0 : (var_acc / static_cast<int64_t>(cols_));
          int64_t r_qf = (init_spec) ? gates::ref_rsqrt_fixed(*init_spec, var + eps, fb, nr_iters) : 0;
          for (int c = 0; c < cols_; ++c) {
            size_t idx = static_cast<size_t>(r * cols_ + c);
            int64_t d = x_plain[idx] - mu;
            __int128 prod = static_cast<__int128>(d) * static_cast<__int128>(r_qf);
            int64_t y = (fb >= 64) ? 0ll : static_cast<int64_t>(prod >> fb);
            if (has_gamma) {
              int64_t g = (c < static_cast<int>(bundle_.gamma.size())) ? bundle_.gamma[c] : 0;
              y = static_cast<int64_t>((static_cast<__int128>(y) * static_cast<__int128>(g)) >> fb);
            }
            if (has_beta) {
              int64_t b = (c < static_cast<int>(bundle_.beta.size())) ? bundle_.beta[c] : 0;
              y += b;
            }
            out_plain[idx] = (R.party == 0) ? static_cast<uint64_t>(y) : 0ull;
          }
        }
        for (size_t i = 0; i < out_.size() && i < out_plain.size(); ++i) {
          out_[i] = out_plain[i];
        }
        st_ = St::Done;
        return detail::Need::None;
      }
    }
    switch (st_) {
      case St::Mean: {
#ifdef SUF_HAVE_CUDA
        const bool want_gpu =
            R.device_pipeline &&
            env_flag_enabled_default(
                "SUF_LN_GPU",
                (R.pfss_backend && dynamic_cast<proto::PfssGpuStagedEval*>(R.pfss_backend) != nullptr) ||
                    (R.cuda_stream != nullptr));
        if (want_gpu) {
          cudaStream_t stream = nullptr;
          if (R.pfss_backend) {
            if (auto* staged = dynamic_cast<proto::PfssGpuStagedEval*>(R.pfss_backend)) {
              stream = reinterpret_cast<cudaStream_t>(staged->device_stream());
            }
          }
          size_t elems = x_.size();
          size_t bytes = elems * sizeof(uint64_t);
          uint64_t* d_x = nullptr;
          uint64_t* d_sum = nullptr;
          cudaMalloc(&d_x, bytes);
          cudaMalloc(&d_sum, static_cast<size_t>(rows_) * sizeof(uint64_t));
          cudaMemcpyAsync(d_x, x_.data(), bytes, cudaMemcpyHostToDevice, stream);
          launch_row_sum_kernel(d_x, rows_, cols_, /*valid_lens=*/nullptr, d_sum, stream);
          sum_.assign(static_cast<size_t>(rows_), 0);
          cudaMemcpyAsync(sum_.data(), d_sum, static_cast<size_t>(rows_) * sizeof(uint64_t),
                          cudaMemcpyDeviceToHost, stream);
          cudaStreamSynchronize(stream);
          cudaFree(d_x);
          cudaFree(d_sum);
        } else {
#endif
        sum_.assign(static_cast<size_t>(rows_), 0);
        for (int r = 0; r < rows_; ++r) {
          uint64_t acc = 0;
          for (int c = 0; c < cols_; ++c) {
            size_t idx = static_cast<size_t>(r * cols_ + c);
            acc = proto::add_mod(acc, x_[idx]);
          }
          sum_[static_cast<size_t>(r)] = acc;
        }
#ifdef SUF_HAVE_CUDA
        }
#endif
        mu_q2f_.assign(sum_.size(), 0);
        mu_qf_.assign(sum_.size(), 0);
        for (size_t r = 0; r < sum_.size(); ++r) {
          uint64_t const_share = bundle_.inv_len_qf;
          mu_q2f_[r] = proto::mul_mod(sum_[r], const_share);
        }
        st_ = St::MeanTrunc;
        return detail::Need::None;
      }
      case St::MeanTrunc: {
        if (bundle_.mean_range.lo <= bundle_.mean_range.hi) {
          mu_range_ = bundle_.mean_range;
        }
        const auto* tb = select_trunc_bundle(bundle_.mean_trunc, mu_range_, bundle_.frac_bits);
        if (!tb) throw std::runtime_error("LayerNormTask: missing mean trunc bundle");
        mean_trunc_task_ = std::make_unique<TruncTask>(
            tb, std::span<const uint64_t>(mu_q2f_.data(), mu_q2f_.size()),
            std::span<uint64_t>(mu_qf_.data(), mu_qf_.size()));
        st_ = St::MeanTruncRun;
        [[fallthrough]];
      }
      case St::MeanTruncRun: {
        auto need = mean_trunc_task_->step(R);
        if (!mean_trunc_task_->done()) return need;
        st_ = St::VarDiff;
        return detail::Need::None;
      }
      case St::VarDiff: {
        if (bundle_.mul_triples.size() == 0) {
          throw std::runtime_error("LayerNormTask: mul triples missing");
        }
        if (triple_span_.size() == 0) triple_span_ = bundle_.mul_triples;
        diff_.assign(x_.size(), 0);
        for (int r = 0; r < rows_; ++r) {
          uint64_t mu = (r < static_cast<int>(mu_qf_.size())) ? mu_qf_[static_cast<size_t>(r)] : 0ull;
          for (int c = 0; c < cols_; ++c) {
            size_t idx = static_cast<size_t>(r * cols_ + c);
            diff_[idx] = proto::sub_mod(x_[idx], mu);
          }
        }
        auto triples = next_triples(x_.size());
        diff2_.assign(x_.size(), 0);
        var_mul_ = std::make_unique<MulTask>(std::span<const uint64_t>(diff_.data(), diff_.size()),
                                             std::span<const uint64_t>(diff_.data(), diff_.size()),
                                             std::span<uint64_t>(diff2_.data(), diff2_.size()),
                                             triples);
        st_ = St::VarMul;
        return detail::Need::None;
      }
      case St::VarMul: {
#ifdef SUF_HAVE_CUDA
        const bool want_gpu =
            R.device_pipeline &&
            env_flag_enabled_default(
                "SUF_LN_GPU",
                (R.pfss_backend && dynamic_cast<proto::PfssGpuStagedEval*>(R.pfss_backend) != nullptr) ||
                    (R.cuda_stream != nullptr));
        if (want_gpu) {
          if (std::getenv("SOFTMAX_BENCH_TRACE")) {
            std::fprintf(stderr, "[LayerNormTask] VarMul GPU rows=%d cols=%d elems=%zu\n",
                         rows_, cols_, x_.size());
          }
          cudaStream_t stream = nullptr;
          if (R.pfss_backend) {
            if (auto* staged = dynamic_cast<proto::PfssGpuStagedEval*>(R.pfss_backend)) {
              stream = reinterpret_cast<cudaStream_t>(staged->device_stream());
            }
          }
          size_t elems = x_.size();
          size_t bytes_mat = elems * sizeof(uint64_t);
          uint64_t* d_x = nullptr;
          uint64_t* d_mu = nullptr;
          uint64_t* d_var = nullptr;
          cudaMalloc(&d_x, bytes_mat);
          cudaMalloc(&d_mu, static_cast<size_t>(rows_) * sizeof(uint64_t));
          cudaMalloc(&d_var, static_cast<size_t>(rows_) * sizeof(uint64_t));
          cudaMemcpyAsync(d_x, x_.data(), bytes_mat, cudaMemcpyHostToDevice, stream);
          cudaMemcpyAsync(d_mu, mu_qf_.data(), static_cast<size_t>(rows_) * sizeof(uint64_t),
                          cudaMemcpyHostToDevice, stream);
          launch_row_variance_kernel(d_x, d_mu, rows_, cols_, /*valid_lens=*/nullptr, d_var, stream);
          var_sum_.assign(static_cast<size_t>(rows_), 0);
          cudaMemcpyAsync(var_sum_.data(), d_var, static_cast<size_t>(rows_) * sizeof(uint64_t),
                          cudaMemcpyDeviceToHost, stream);
          cudaStreamSynchronize(stream);
          cudaFree(d_x);
          cudaFree(d_mu);
          cudaFree(d_var);
          var_q3f_.assign(var_sum_.size(), 0);
          for (size_t r = 0; r < var_sum_.size(); ++r) {
            uint64_t const_share = bundle_.inv_len_qf;
            var_q3f_[r] = proto::mul_mod(var_sum_[r], const_share);
          }
          if (std::getenv("SOFTMAX_BENCH_TRACE")) {
            std::fprintf(stderr, "[LayerNormTask] VarMul GPU var_q3f size=%zu\n",
                         var_q3f_.size());
          }
          st_ = St::VarTrunc;
          return detail::Need::None;
        }
#endif
        auto need = var_mul_->step(R);
        if (!var_mul_->done()) return need;
        var_sum_.assign(static_cast<size_t>(rows_), 0);
        for (int r = 0; r < rows_; ++r) {
          uint64_t acc = 0;
          for (int c = 0; c < cols_; ++c) {
            size_t idx = static_cast<size_t>(r * cols_ + c);
            acc = proto::add_mod(acc, diff2_[idx]);
          }
        var_sum_[static_cast<size_t>(r)] = acc;
      }
        var_q3f_.assign(var_sum_.size(), 0);
        for (size_t r = 0; r < var_sum_.size(); ++r) {
          uint64_t const_share = bundle_.inv_len_qf;
          var_q3f_[r] = proto::mul_mod(var_sum_[r], const_share);
        }
        if (std::getenv("SOFTMAX_BENCH_TRACE")) {
          std::fprintf(stderr, "[LayerNormTask] VarMul CPU var_q3f size=%zu\n",
                       var_q3f_.size());
        }
        st_ = St::VarTrunc;
        return detail::Need::None;
      }
      case St::VarTrunc: {
        if (var_q3f_.empty()) {
          var_sum_.assign(static_cast<size_t>(rows_), 0);
          var_q3f_.assign(var_sum_.size(), 0);
          for (size_t r = 0; r < var_sum_.size(); ++r) {
            uint64_t const_share = bundle_.inv_len_qf;
            var_q3f_[r] = proto::mul_mod(var_sum_[r], const_share);
          }
          if (std::getenv("SOFTMAX_BENCH_TRACE")) {
            std::fprintf(stderr, "[LayerNormTask] VarTrunc filled var_q3f_ size=%zu\n",
                         var_q3f_.size());
          }
        }
        var_range_.lo = 0;
        var_range_.is_signed = true;
        if (bundle_.var_range.lo <= bundle_.var_range.hi) {
          var_range_ = bundle_.var_range;
        }
        const auto* tb = select_trunc_bundle(bundle_.var_trunc, var_range_, 2 * bundle_.frac_bits);
        if (!tb) throw std::runtime_error("LayerNormTask: missing var trunc bundle");
        var_qf_.assign(var_q3f_.size(), 0);
        var_trunc_task_ = std::make_unique<TruncTask>(
            tb, std::span<const uint64_t>(var_q3f_.data(), var_q3f_.size()),
            std::span<uint64_t>(var_qf_.data(), var_qf_.size()));
        st_ = St::VarTruncRun;
        return detail::Need::None;
      }
      case St::VarTruncRun: {
        auto need = var_trunc_task_->step(R);
        if (!var_trunc_task_->done()) return need;
        if (var_qf_.empty()) {
          var_qf_.assign(static_cast<size_t>(rows_), 0);
          if (std::getenv("SOFTMAX_BENCH_TRACE")) {
            std::fprintf(stderr, "[LayerNormTask] VarTruncRun filled empty var_qf_ rows=%d\n",
                         rows_);
          }
        }
        if (R.party == 0) {
          for (auto& v : var_qf_) v = proto::add_mod(v, bundle_.eps_qf);
        }
        st_ = St::Rsqrt;
        return detail::Need::None;
      }
      case St::Rsqrt: {
        if (var_qf_.empty()) {
          var_qf_.assign(static_cast<size_t>(rows_), 0);
          if (std::getenv("SOFTMAX_BENCH_TRACE")) {
            std::fprintf(stderr, "[LayerNormTask] Rsqrt filling empty var_qf_ to rows=%d\n",
                         rows_);
          }
        }
        rsqrt_out_.assign(var_qf_.size(), 0);
        if (std::getenv("SOFTMAX_BENCH_TRACE")) {
          size_t r0 = bundle_.rsqrt.key0 ? bundle_.rsqrt.key0->r_in_share_vec.size() : 0;
          size_t r1 = bundle_.rsqrt.key1 ? bundle_.rsqrt.key1->r_in_share_vec.size() : 0;
          std::fprintf(stderr, "[LayerNormTask] Rsqrt input elems=%zu r_in_vec0=%zu r_in_vec1=%zu\n",
                       var_qf_.size(), r0, r1);
        }
        rsqrt_task_ = std::make_unique<RsqrtTask>(bundle_.rsqrt,
                                                  std::span<const uint64_t>(var_qf_.data(), var_qf_.size()),
                                                  std::span<uint64_t>(rsqrt_out_.data(), rsqrt_out_.size()));
        st_ = St::RsqrtRun;
        return detail::Need::None;
      }
      case St::RsqrtRun: {
        if (std::getenv("SOFTMAX_BENCH_TRACE")) {
          std::fprintf(stderr, "[LayerNormTask] RsqrtRun var_qf=%zu rsqrt_out=%zu\n",
                       var_qf_.size(), rsqrt_out_.size());
        }
        auto need = rsqrt_task_->step(R);
        if (!rsqrt_task_->done()) return need;
        rsqrt_init_ = rsqrt_task_->init_y_debug();
        rsqrt_xy2_f_ = rsqrt_task_->xy2_f_debug();
        rsqrt_c0_ = rsqrt_task_->c0_debug();
        rsqrt_c1_ = rsqrt_task_->c1_debug();
        rsqrt_init_r_ = rsqrt_task_->init_r_debug();
        rsqrt_x_plain_.assign(rsqrt_task_->x_plain_debug().begin(),
                              rsqrt_task_->x_plain_debug().end());
        st_ = St::Norm;
        return detail::Need::None;
      }
      case St::Norm: {
        if (!bundle_.row_triples) throw std::runtime_error("LayerNormTask: row triples missing");
        norm_q2f_.assign(x_.size(), 0);
        norm_qf_.assign(x_.size(), 0);
        mul_norm_ = std::make_unique<MulRowBroadcastTask>(
            std::span<const uint64_t>(diff_.data(), diff_.size()),
            std::span<const uint64_t>(rsqrt_out_.data(), rsqrt_out_.size()),
            rows_, cols_, std::span<const int>(),
            std::span<uint64_t>(norm_q2f_.data(), norm_q2f_.size()),
            bundle_.row_triples,
            /*device_only=*/false);
        st_ = St::NormMul;
        return detail::Need::None;
      }
      case St::NormMul: {
        auto need = mul_norm_->step(R);
        if (!mul_norm_->done()) return need;
        compiler::AbsBound norm_abs = compiler::abs_from_range(norm_range_, /*is_signed=*/true);
        norm_abs.kind = (norm_range_.lo <= norm_range_.hi) ? compiler::RangeKind::Proof
                                                           : compiler::RangeKind::Hint;
        auto gap = compiler::gap_from_abs(norm_abs, bundle_.frac_bits);
        const auto* tb = select_trunc_bundle(bundle_.norm_trunc, norm_abs, bundle_.frac_bits, gap);
        if (!tb) throw std::runtime_error("LayerNormTask: missing norm trunc bundle");
        trunc_norm_ = std::make_unique<TruncTask>(
            tb, std::span<const uint64_t>(norm_q2f_.data(), norm_q2f_.size()),
            std::span<uint64_t>(norm_qf_.data(), norm_qf_.size()));
        st_ = St::NormTrunc;
        return detail::Need::None;
      }
      case St::NormTrunc: {
        if (bundle_.norm_range.lo <= bundle_.norm_range.hi) {
          norm_range_ = bundle_.norm_range;
        }
        auto need = trunc_norm_->step(R);
        if (!trunc_norm_->done()) return need;
        st_ = St::Affine;
        return detail::Need::None;
      }
      case St::Affine: {
        bool has_gamma = bundle_.gamma.data() != nullptr && bundle_.gamma.size() > 0;
        bool has_beta = bundle_.beta.data() != nullptr && bundle_.beta.size() > 0;
        for (size_t i = 0; i < norm_qf_.size(); ++i) {
          size_t col = static_cast<size_t>(i % static_cast<size_t>(cols_));
          int64_t v = to_signed(norm_qf_[i]);
          if (has_gamma) {
            int64_t g = (col < bundle_.gamma.size()) ? bundle_.gamma[col] : 0;
            v = static_cast<int64_t>((static_cast<__int128>(v) * static_cast<__int128>(g)) >> bundle_.frac_bits);
          }
          if (has_beta) {
            int64_t b = (col < bundle_.beta.size()) ? bundle_.beta[col] : 0;
            v += b;
          }
          out_[i] = to_ring(v);
        }
        st_ = St::Done;
        return detail::Need::None;
      }
      case St::Done:
        return detail::Need::None;
    }
    return detail::Need::None;
  }

 private:
  enum class St {
    Mean,
    MeanTrunc,
    MeanTruncRun,
    VarDiff,
    VarMul,
    VarTrunc,
    VarTruncRun,
    Rsqrt,
    RsqrtRun,
    Norm,
    NormMul,
    NormTrunc,
    Affine,
    Done
  } st_ = St::Mean;
  LayerNormTaskBundle bundle_;
  std::span<const uint64_t> x_;
  std::span<uint64_t> out_;
  int rows_ = 0;
  int cols_ = 0;

  std::vector<uint64_t> sum_;
  std::vector<uint64_t> mu_q2f_;
  std::vector<uint64_t> mu_qf_;
  std::vector<uint64_t> diff_;
  std::vector<uint64_t> diff2_;
  std::vector<uint64_t> var_sum_;
  std::vector<uint64_t> var_q3f_;
  std::vector<uint64_t> var_qf_;
  std::vector<uint64_t> rsqrt_out_;
  std::vector<uint64_t> rsqrt_init_;
  std::vector<uint64_t> rsqrt_xy2_f_;
  std::vector<uint64_t> rsqrt_c0_;
  std::vector<uint64_t> rsqrt_c1_;
  int rsqrt_init_r_ = 0;
  std::vector<uint64_t> rsqrt_x_plain_;
  std::vector<uint64_t> norm_q2f_;
  std::vector<uint64_t> norm_qf_;

  compiler::RangeInterval mu_range_ = compiler::RangeInterval::whole(true);
  compiler::RangeInterval var_range_ = compiler::RangeInterval::whole(true);
  compiler::RangeInterval norm_range_ = compiler::RangeInterval::whole(true);

  std::unique_ptr<TruncTask> mean_trunc_task_;
  std::unique_ptr<MulTask> var_mul_;
  std::unique_ptr<TruncTask> var_trunc_task_;
  std::unique_ptr<RsqrtTask> rsqrt_task_;
  std::unique_ptr<MulRowBroadcastTask> mul_norm_;
  std::unique_ptr<TruncTask> trunc_norm_;

  std::span<const proto::BeaverTriple64Share> triple_span_;
  size_t triple_cursor_ = 0;

  std::span<const proto::BeaverTriple64Share> next_triples(size_t n) {
    if (triple_cursor_ + n > triple_span_.size()) {
      throw std::runtime_error("LayerNormTask: not enough triples");
    }
    auto s = triple_span_.subspan(triple_cursor_, n);
    triple_cursor_ += n;
    return s;
  }
};

// Two-trunc cubic evaluator for SiLU/nExp.
class CubicPolyTask final : public detail::PhaseTask {
 public:
  CubicPolyTask(const CubicPolyBundle& bundle,
                std::span<const uint64_t> x_qf,
                std::span<uint64_t> out_qf)
      : bundle_(bundle), x_(x_qf), out_(out_qf) {
    if (x_.size() != out_.size()) {
      throw std::runtime_error("CubicPolyTask: size mismatch");
    }
    if (!bundle_.suf) {
      throw std::runtime_error("CubicPolyTask: missing bundle pieces");
    }
  }

  ~CubicPolyTask() override {
#ifdef SUF_HAVE_CUDA
    if (d_out_device_) {
      cudaFree(d_out_device_);
      d_out_device_ = nullptr;
      d_out_elems_ = 0;
    }
#endif
  }

  bool done() const override { return st_ == St::Done; }
#ifdef SUF_HAVE_CUDA
  uint64_t* device_out() const { return d_out_device_; }
  size_t device_out_elems() const { return d_out_elems_; }
  void release_device_out() {
    if (d_out_device_) {
      cudaFree(d_out_device_);
      d_out_device_ = nullptr;
      d_out_elems_ = 0;
    }
  }
#endif
  void set_shape_hint(const std::vector<int>* row_offsets,
                      const std::vector<int>* row_lengths,
                      uint16_t eff_bits_hint = 0) {
    row_offsets_hint_ = row_offsets;
    row_lengths_hint_ = row_lengths;
    eff_bits_hint_ = eff_bits_hint;
  }

  detail::Need step(PhaseResources& R) override {
    if (!key_) {
      key_ = (R.party == 0) ? bundle_.key0 : bundle_.key1;
      if (!key_) throw std::runtime_error("CubicPolyTask: missing party key");
      const bool direct_payload_only =
          (key_->compiled.r == 1 && key_->compiled.degree == 0 && key_->compiled.ell == 0);
      if (!direct_payload_only) {
        triple_span_ = std::span<const proto::BeaverTriple64Share>(key_->triples);
      }
      if (std::getenv("SUF_CUBIC_KEY_TRACE")) {
        static std::atomic<bool> logged{false};
        bool expect = false;
        if (logged.compare_exchange_strong(expect, true)) {
          std::fprintf(stderr,
                       "[CubicPolyTask] gate_kind=%d coeff_mode=%d pred_sem=%d degree=%d r=%d ell=%d triples=%zu\n",
                       static_cast<int>(key_->compiled.gate_kind),
                       static_cast<int>(key_->compiled.coeff.mode),
                       static_cast<int>(key_->pred_meta.sem),
                       key_->compiled.degree,
                       key_->compiled.r,
                       key_->compiled.ell,
                       key_->triples.size());
        }
      }
      if (!direct_payload_only) {
        if (!bundle_.trunc_f) {
          throw std::runtime_error("CubicPolyTask: missing truncation bundle");
        }
        if (!bundle_.trunc_2f) {
          throw std::runtime_error("CubicPolyTask: missing truncation-2f bundle");
        }
      }
    }
    switch (st_) {
      case St::OpenXhat: {
        if (!R.net_chan) throw std::runtime_error("CubicPolyTask: net channel missing");
        if (R.opens) {
          auto res = R.opens->reserve(x_.size(), OpenKind::kMask);
          for (size_t i = 0; i < x_.size(); ++i) {
            uint64_t rin = (key_->r_in_share_vec.size() > i) ? key_->r_in_share_vec[i] : key_->r_in_share;
            res.diff[i] = proto::add_mod(x_[i], rin);
          }
          h_open_ = res.handle;
          st_ = St::WaitXhatOpen;
          // Allow batching of opens across tasks.
          return detail::Need::None;
        }
        masked_.resize(x_.size());
        for (size_t i = 0; i < x_.size(); ++i) {
          uint64_t rin = (key_->r_in_share_vec.size() > i) ? key_->r_in_share_vec[i] : key_->r_in_share;
          masked_[i] = proto::add_mod(x_[i], rin);
        }
        opened_.assign(masked_.size(), 0);
        for (size_t i = 0; i < masked_.size(); ++i) {
          if (R.party == 0) {
            R.net_chan->send_u64(masked_[i]);
            opened_[i] = proto::add_mod(masked_[i], R.net_chan->recv_u64());
          } else {
            opened_[i] = proto::add_mod(masked_[i], R.net_chan->recv_u64());
            R.net_chan->send_u64(masked_[i]);
          }
        }
        st_ = St::EnqueueCoeff;
        [[fallthrough]];
      }
      case St::WaitXhatOpen: {
        if (st_ == St::WaitXhatOpen) {
          if (!R.opens) throw std::runtime_error("CubicPolyTask: no OpenCollector");
          if (!R.opens->ready(h_open_)) return detail::Need::Open;
          auto v = R.opens->view_u64(h_open_);
          opened_.assign(v.begin(), v.end());
          st_ = St::EnqueueCoeff;
        }
        [[fallthrough]];
      }
      case St::EnqueueCoeff: {
        PreparedCompositeJob job;
        job.suf = bundle_.suf;
        job.key = key_;
        job.hook = nullptr;
        job.hatx_public.resize(opened_.size());
        if (!opened_.empty()) {
          std::memcpy(job.hatx_public.data(), opened_.data(), opened_.size() * sizeof(uint64_t));
        }
        const size_t elems = x_.size();
        const size_t r = static_cast<size_t>(std::max(0, key_->compiled.r));
        if (r == 0) throw std::runtime_error("CubicPolyTask: compiled coeff gate has r=0");
        coeff_buf_.assign(elems * r, 0ull);  // AoS layout written by PFSS finalize
        job.out = nn::TensorView<uint64_t>(coeff_buf_.data(), {coeff_buf_.size()});
        if (row_offsets_hint_ && row_lengths_hint_ &&
            row_offsets_hint_->size() == row_lengths_hint_->size() + 1 &&
            !row_offsets_hint_->empty()) {
          job.row_offsets = *row_offsets_hint_;
          job.row_lengths = *row_lengths_hint_;
          job.shape.ragged = true;
          job.shape.num_rows = static_cast<uint16_t>(row_lengths_hint_->size());
          for (int L : *row_lengths_hint_) {
            job.shape.max_row_len =
                std::max<uint16_t>(job.shape.max_row_len, static_cast<uint16_t>(L));
          }
          job.shape.total_elems = static_cast<uint32_t>(row_offsets_hint_->back());
        }
        if (eff_bits_hint_ > 0 && eff_bits_hint_ <= 64) {
          job.shape.eff_bits = eff_bits_hint_;
        }
        if (!R.pfss_coeff) throw std::runtime_error("CubicPolyTask: missing coeff PFSS batch");
        coeff_handle_ = R.pfss_coeff->enqueue_composite(std::move(job));
        st_ = St::WaitCoeff;
        // Allow batching of PFSS enqueues across tasks.
        return detail::Need::None;
      }
      case St::WaitCoeff: {
        if (!R.pfss_coeff->ready(coeff_handle_)) return detail::Need::PfssCoeff;
        size_t elems = x_.size();
        // Some backends may already return the evaluated polynomial (r=1).
        const size_t r = static_cast<size_t>(std::max(0, key_->compiled.r));
        if (r == 1 && coeff_buf_.size() >= elems) {
          for (size_t i = 0; i < elems; ++i) out_[i] = coeff_buf_[i];
          st_ = St::Done;
          return detail::Need::None;
        }
        if (r < 4 || coeff_buf_.size() < elems * r) {
          throw std::runtime_error("CubicPolyTask: coeff payload too small");
        }
        soa_buf_.assign(4 * elems, 0);
        if (std::getenv("SOFTMAX_DBG_COEFF")) {
          std::cerr << "[CubicPolyTask p" << R.party << "] coeff r=" << r
                    << " arith_words=" << coeff_buf_.size()
                    << " elems=" << elems
                    << " first=" << (coeff_buf_.empty() ? 0 : coeff_buf_[0]) << "\n";
        }
        for (size_t i = 0; i < elems; ++i) {
          soa_buf_[0 * elems + i] = coeff_buf_[i * r + 0];
          soa_buf_[1 * elems + i] = coeff_buf_[i * r + 1];
          soa_buf_[2 * elems + i] = coeff_buf_[i * r + 2];
          soa_buf_[3 * elems + i] = coeff_buf_[i * r + 3];
        }
        c0_ = std::span<const uint64_t>(soa_buf_.data() + 0 * elems, elems);
        c1_ = std::span<const uint64_t>(soa_buf_.data() + 1 * elems, elems);
        c2_ = std::span<const uint64_t>(soa_buf_.data() + 2 * elems, elems);
        c3_ = std::span<const uint64_t>(soa_buf_.data() + 3 * elems, elems);
        static bool logged = false;
        if (!logged && R.party == 0 && c0_.size() > 0) {
          logged = true;
          if (std::getenv("SUF_CUBIC_TRACE")) {
            std::cerr << "CubicPolyTask coeffs: c0=" << c0_[0] << " c1=" << c1_[0]
                      << " c2=" << c2_[0] << " c3=" << c3_[0] << " r=" << r << "\n";
          }
        }
        if (R.pfss_backend &&
            dynamic_cast<proto::ReferenceBackend*>(R.pfss_backend) != nullptr &&
            !std::getenv("SUF_FORCE_PFSS")) {
          // Reference backend: evaluate directly using the reference polynomial/spec.
          auto eval_ref = [&](int64_t x_plain, size_t idx) -> int64_t {
            if (bundle_.spec && bundle_.gate_kind == compiler::GateKind::SiLUSpline) {
              return gates::ref_silu_fixed(*bundle_.spec, x_plain);
            } else if (bundle_.spec && bundle_.gate_kind == compiler::GateKind::GeLUSpline) {
              return gates::eval_piecewise_poly_ref(*bundle_.spec, x_plain);
            } else if (bundle_.spec && bundle_.gate_kind == compiler::GateKind::NExp) {
              return gates::ref_nexp_fixed(*bundle_.spec, x_plain);
            }
            int shift = bundle_.frac_bits;
            int64_t c0s = static_cast<int64_t>(c0_[idx]);
            int64_t c1s = static_cast<int64_t>(c1_[idx]);
            int64_t c2s = static_cast<int64_t>(c2_[idx]);
            int64_t c3s = static_cast<int64_t>(c3_[idx]);
            __int128 m1 = static_cast<__int128>(c3s) * static_cast<__int128>(x_plain);
            __int128 u = m1 + (static_cast<__int128>(c2s) << shift);
            __int128 p = u * static_cast<__int128>(x_plain);
            int64_t p2 = (shift >= 64) ? 0ll : static_cast<int64_t>(p >> shift);
            __int128 q = static_cast<__int128>(p2) + (static_cast<__int128>(c1s) << shift);
            __int128 r = q * static_cast<__int128>(x_plain);
            int64_t y = (shift >= 64) ? 0ll : static_cast<int64_t>(r >> (2 * shift));
            return y + c0s;
          };

          for (size_t i = 0; i < elems; ++i) {
            int64_t x_plain = static_cast<int64_t>(proto::sub_mod(static_cast<uint64_t>(opened_[i]), key_->compiled.r_in));
            int64_t out = eval_ref(x_plain, i);
            static bool logged_ref = false;
            if (!logged_ref && R.party == 0 && std::getenv("SUF_CUBIC_TRACE")) {
              logged_ref = true;
              std::cerr << "CubicPolyTask ref eval x=" << x_plain << " out=" << out
                        << " gate=" << static_cast<int>(bundle_.gate_kind) << "\n";
            }
            out_[i] = (R.party == 0) ? static_cast<uint64_t>(out) : 0ull;
          }
          st_ = St::Done;
          return detail::Need::None;
        }
        // Allocate temps for two-trunc cubic evaluation:
        // m1 = c3 * x                 (Q2f)
        // u  = m1 + (c2 << f)         (Q2f)
        // p  = u * x                  (Q3f)
        // p2 = trunc(p, f)            (Q2f)
        // q  = p2 + (c1 << f)         (Q2f)
        // r  = q * x                  (Q3f)
        // y  = trunc(r, 2f) + c0      (Qf)
        m1_q2f_.assign(elems, 0);
        u_q2f_.assign(elems, 0);
        p_q3f_.assign(elems, 0);
        p2_q2f_.assign(elems, 0);
        q_q2f_.assign(elems, 0);
        r_q3f_.assign(elems, 0);
        y_qf_.assign(elems, 0);
        auto triples1 = next_triples(x_.size());
        mul1_ = std::make_unique<MulTask>(c3_, x_,
                                          std::span<uint64_t>(m1_q2f_.data(), m1_q2f_.size()),
                                          triples1);
        st_ = St::Mul1;
        return detail::Need::None;
      }
      case St::Mul1: {
        auto need = mul1_->step(R);
        if (!mul1_->done()) return need;
        shift_scale_ = (bundle_.frac_bits >= 64) ? 0ull : (uint64_t(1) << bundle_.frac_bits);
        for (size_t i = 0; i < x_.size(); ++i) {
          uint64_t c2_shift = proto::mul_mod(c2_[i], shift_scale_);
          u_q2f_[i] = proto::add_mod(m1_q2f_[i], c2_shift);
        }
        auto triples2 = next_triples(x_.size());
        mul2_ = std::make_unique<MulTask>(std::span<const uint64_t>(u_q2f_.data(), u_q2f_.size()),
                                          x_,
                                          std::span<uint64_t>(p_q3f_.data(), p_q3f_.size()),
                                          triples2);
        st_ = St::Mul2;
        return detail::Need::None;
      }
      case St::Mul2: {
        auto need = mul2_->step(R);
        if (!mul2_->done()) return need;
        trunc1_ = std::make_unique<TruncTask>(
            bundle_.trunc_f,
            std::span<const uint64_t>(p_q3f_.data(), p_q3f_.size()),
            std::span<uint64_t>(p2_q2f_.data(), p2_q2f_.size()));
        st_ = St::Trunc1;
        return detail::Need::None;
      }
      case St::Trunc1: {
        auto need = trunc1_->step(R);
        if (!trunc1_->done()) return need;
#ifdef SUF_HAVE_CUDA
        if (R.device_pipeline) {
          if (auto* dptr = trunc1_->device_out()) {
            size_t n = std::min(p2_q2f_.size(), trunc1_->device_out_elems());
            if (n > 0) cudaMemcpy(p2_q2f_.data(), dptr, n * sizeof(uint64_t), cudaMemcpyDeviceToHost);
            trunc1_->release_device_out();
          }
        }
#endif
        for (size_t i = 0; i < x_.size(); ++i) {
          uint64_t c1_scaled = proto::mul_mod(c1_[i], shift_scale_);
          q_q2f_[i] = proto::add_mod(p2_q2f_[i], c1_scaled);
        }
        auto triples3 = next_triples(x_.size());
        mul3_ = std::make_unique<MulTask>(std::span<const uint64_t>(q_q2f_.data(), q_q2f_.size()),
                                          x_,
                                          std::span<uint64_t>(r_q3f_.data(), r_q3f_.size()),
                                          triples3);
        st_ = St::Mul3;
        return detail::Need::None;
      }
      case St::Mul3: {
        auto need = mul3_->step(R);
        if (!mul3_->done()) return need;
        trunc2_ = std::make_unique<TruncTask>(
            bundle_.trunc_2f,
            std::span<const uint64_t>(r_q3f_.data(), r_q3f_.size()),
            std::span<uint64_t>(y_qf_.data(), y_qf_.size()));
        st_ = St::Trunc2;
        return detail::Need::None;
      }
      case St::Trunc2: {
        auto need = trunc2_->step(R);
        if (!trunc2_->done()) return need;
#ifdef SUF_HAVE_CUDA
        if (R.device_pipeline) {
          if (auto* dptr = trunc2_->device_out()) {
            size_t n = std::min(y_qf_.size(), trunc2_->device_out_elems());
            if (n > 0) cudaMemcpy(y_qf_.data(), dptr, n * sizeof(uint64_t), cudaMemcpyDeviceToHost);
            trunc2_->release_device_out();
          }
        }
#endif
        for (size_t i = 0; i < x_.size(); ++i) {
          out_[i] = proto::add_mod(y_qf_[i], c0_[i]);
        }
        st_ = St::Done;
        return detail::Need::None;
      }
      case St::Done:
        return detail::Need::None;
    }
    return detail::Need::None;
  }

 private:
  enum class St {
    OpenXhat,
    WaitXhatOpen,
    EnqueueCoeff,
    WaitCoeff,
    Mul1,
    Mul2,
    Trunc1,
    Mul3,
    Trunc2,
    Done
  } st_ = St::OpenXhat;

  const CubicPolyBundle bundle_;
  const gates::CompositePartyKey* key_ = nullptr;
  std::span<const uint64_t> x_;
  std::span<uint64_t> out_;

  std::vector<uint64_t> masked_;
  std::vector<uint64_t> opened_;
  OpenHandle h_open_{};
  PfssHandle coeff_handle_{};
  std::span<const proto::BeaverTriple64Share> triple_span_;
  size_t triple_cursor_ = 0;
  const std::vector<int>* row_offsets_hint_ = nullptr;
  const std::vector<int>* row_lengths_hint_ = nullptr;
  uint16_t eff_bits_hint_ = 0;

  // coeff storage
  std::vector<uint64_t> coeff_buf_;
  std::vector<uint64_t> soa_buf_;
  std::span<const uint64_t> c0_, c1_, c2_, c3_;

  // temps
  std::vector<uint64_t> m1_q2f_;
  std::vector<uint64_t> u_q2f_;
  std::vector<uint64_t> p_q3f_;
  std::vector<uint64_t> p2_q2f_;
  std::vector<uint64_t> q_q2f_;
  std::vector<uint64_t> r_q3f_;
  std::vector<uint64_t> y_qf_;

  // sub tasks
  std::unique_ptr<MulTask> mul1_;
  std::unique_ptr<MulTask> mul2_;
  std::unique_ptr<MulTask> mul3_;
  std::unique_ptr<TruncTask> trunc1_;
  std::unique_ptr<TruncTask> trunc2_;
  uint64_t shift_scale_ = 0;

#ifdef SUF_HAVE_CUDA
  uint64_t* d_out_device_ = nullptr;
  size_t d_out_elems_ = 0;
#endif

  std::span<const proto::BeaverTriple64Share> next_triples(size_t n) {
    if (triple_cursor_ + n > triple_span_.size()) {
      throw std::runtime_error("CubicPolyTask: not enough triples");
    }
    auto s = triple_span_.subspan(triple_cursor_, n);
    triple_cursor_ += n;
    return s;
  }
};

// Reciprocal via affine init + NR (1..k iters). Keeps everything on shares and
// uses PFSS/Open batching.
class RecipTask final : public detail::PhaseTask {
 public:
  RecipTask(const RecipTaskBundle& bundle,
            std::span<const uint64_t> x_qf,
            std::span<uint64_t> out_qf)
      : bundle_(bundle), x_(x_qf), out_(out_qf) {
    if (std::getenv("SOFTMAX_BENCH_TRACE")) {
      std::fprintf(stderr, "[RecipTask ctor] size=%zu\n", x_.size());
    }
    if (x_.size() != out_.size()) throw std::runtime_error("RecipTask: size mismatch");
    if (!bundle_.suf || !bundle_.trunc_fb) throw std::runtime_error("RecipTask: bundle missing parts");
    ref_trunc_ = (std::getenv("SOFTMAX_TRUNC_REF") != nullptr);
    force_ref_full_ = (std::getenv("SOFTMAX_REF_FULL_RECIP") != nullptr);
  }

  ~RecipTask() override {
#ifdef SUF_HAVE_CUDA
    if (d_out_device_) {
      cudaFree(d_out_device_);
      d_out_device_ = nullptr;
      d_out_elems_ = 0;
    }
#endif
  }

#ifdef SUF_HAVE_CUDA
  uint64_t* device_out() const { return d_out_device_; }
  size_t device_out_elems() const { return d_out_elems_; }
  void release_device_out() {
    if (d_out_device_) {
      cudaFree(d_out_device_);
      d_out_device_ = nullptr;
      d_out_elems_ = 0;
    }
  }
#endif

  bool done() const override { return st_ == St::Done; }
  const std::vector<uint64_t>& init_y_debug() const { return init_y_debug_; }
  const std::vector<uint64_t>& t_xy_debug() const { return t_xy_; }
  const std::vector<uint64_t>& t_xy_tr_debug() const { return t_xy_tr_; }
  const std::vector<uint64_t>& t_update_tr_debug() const { return t_update_tr_; }
  const std::vector<uint64_t>& t_update_debug() const { return t_update_; }
  const std::vector<uint64_t>& y_debug() const { return y_; }

  detail::Need step(PhaseResources& R) override {
    if (std::getenv("SOFTMAX_BENCH_TRACE")) {
      std::fprintf(stderr, "[RecipTask p%d] state=%d\n", R.party, static_cast<int>(st_));
    }
    if (party_ < 0) party_ = R.party;
    if (!key_) {
      key_ = (R.party == 0) ? bundle_.key0 : bundle_.key1;
      if (!key_) throw std::runtime_error("RecipTask: missing party key");
      triple_span_ = std::span<const proto::BeaverTriple64Share>(key_->triples);
      dbg_rec_ = (std::getenv("SOFTMAX_DBG_RECIP") != nullptr);
      static bool logged_f = false;
      if (!logged_f && std::getenv("SOFTMAX_DBG_COEFF") && bundle_.trunc_fb) {
        logged_f = true;
        uint64_t f_meta = (!bundle_.trunc_fb->keys.k0.compiled.extra_u64.empty())
                              ? bundle_.trunc_fb->keys.k0.compiled.extra_u64[0]
                              : static_cast<uint64_t>(bundle_.frac_bits);
        std::cerr << "[RecipTask p" << R.party << "] trunc frac_bits meta=" << f_meta
                  << " r_low=" << (bundle_.trunc_fb->keys.k0.compiled.extra_u64.size() > 1
                                       ? bundle_.trunc_fb->keys.k0.compiled.extra_u64[1]
                                       : 0)
                  << " r_out0=" << (bundle_.trunc_fb->keys.k0.r_out_share.empty()
                                        ? 0ull
                                        : bundle_.trunc_fb->keys.k0.r_out_share[0])
                  << " r_out1=" << (bundle_.trunc_fb->keys.k1.r_out_share.empty()
                                        ? 0ull
                                        : bundle_.trunc_fb->keys.k1.r_out_share[0])
                  << "\n";
      }
    }
    switch (st_) {
      case St::OpenXhat: {
        if (!R.net_chan) throw std::runtime_error("RecipTask: net channel missing");
        masked_.resize(x_.size());
        for (size_t i = 0; i < x_.size(); ++i) {
          uint64_t rin = (key_->r_in_share_vec.size() > i) ? key_->r_in_share_vec[i] : key_->r_in_share;
          masked_[i] = proto::add_mod(x_[i], rin);
        }
        if (R.opens) {
          h_open_ = R.opens->enqueue(masked_, OpenKind::kMask);
          st_ = St::WaitXhatOpen;
          // Allow batching of opens across tasks.
          return detail::Need::None;
        }
        throw std::runtime_error("RecipTask: no OpenCollector");
      }
      case St::WaitXhatOpen: {
        if (st_ == St::WaitXhatOpen) {
          if (!R.opens) throw std::runtime_error("RecipTask: no OpenCollector");
          if (!R.opens->ready(h_open_)) return detail::Need::Open;
          auto v = R.opens->view_u64(h_open_);
          opened_.assign(v.begin(), v.end());
          if (!force_ref_full_ &&
              R.pfss_backend &&
              dynamic_cast<proto::ReferenceBackend*>(R.pfss_backend) != nullptr &&
              bundle_.init_spec &&
              !std::getenv("SUF_FORCE_PFSS")) {
            y_.assign(opened_.size(), 0);
            for (size_t i = 0; i < opened_.size(); ++i) {
              uint64_t x_plain_ring = proto::sub_mod(opened_[i], key_->compiled.r_in);
              int64_t x_plain_signed = to_signed(x_plain_ring);
              uint64_t share = (R.party == 0)
                                   ? static_cast<uint64_t>(gates::ref_reciprocal_fixed(
                                         *bundle_.init_spec, x_plain_signed, bundle_.frac_bits, bundle_.nr_iters))
                                   : 0ull;
              y_[i] = share;
              out_[i] = share;
            }
            st_ = St::Done;
            return detail::Need::None;
          }
          st_ = St::EnqueueCoeff;
        }
        [[fallthrough]];
      }
      case St::EnqueueCoeff: {
        PreparedCompositeJob job;
        job.suf = bundle_.suf;
        job.key = key_;
        job.hook = nullptr;
        job.hatx_public.resize(opened_.size());
        if (!opened_.empty()) {
          std::memcpy(job.hatx_public.data(), opened_.data(), opened_.size() * sizeof(uint64_t));
        }
        const size_t elems = x_.size();
        const size_t r = static_cast<size_t>(std::max(0, key_->compiled.r));
        if (r == 0) throw std::runtime_error("RecipTask: compiled coeff gate has r=0");
        coeff_buf_.assign(elems * r, 0ull);  // AoS layout written by PFSS finalize
        job.out = nn::TensorView<uint64_t>(coeff_buf_.data(), {coeff_buf_.size()});
        if (!R.pfss_coeff) throw std::runtime_error("RecipTask: missing coeff PFSS batch");
        coeff_handle_ = R.pfss_coeff->enqueue_composite(std::move(job));
        st_ = St::WaitCoeff;
        // Allow batching of PFSS enqueues across tasks.
        return detail::Need::None;
      }
      case St::WaitCoeff: {
        if (!R.pfss_coeff->ready(coeff_handle_)) return detail::Need::PfssCoeff;
        size_t elems = x_.size();
        const size_t r = static_cast<size_t>(std::max(0, key_->compiled.r));
        if (std::getenv("SOFTMAX_DBG_COEFF")) {
          std::cerr << "[RecipTask p" << R.party << "] coeff r=" << r
                    << " arith_words=" << coeff_buf_.size()
                    << " elems=" << elems
                    << " first=" << (coeff_buf_.empty() ? 0 : coeff_buf_[0]) << "\n";
        }
        // Some backends may already emit evaluated init (single arith word).
        if (r == 1 && coeff_buf_.size() >= elems) {
          y_.assign(coeff_buf_.begin(), coeff_buf_.begin() + elems);
          st_ = St::IterMul1;
          iter_ = 0;
          return detail::Need::None;
        }
        if (r < 2 || coeff_buf_.size() < elems * r) {
          throw std::runtime_error("RecipTask: coeff payload too small");
        }
        soa_buf_.assign(2 * elems, 0);
        for (size_t i = 0; i < elems; ++i) {
          soa_buf_[0 * elems + i] = coeff_buf_[i * r + 0];
          soa_buf_[1 * elems + i] = coeff_buf_[i * r + 1];
        }
        c0_ = std::span<const uint64_t>(soa_buf_.data() + 0 * elems, elems);
        c1_ = std::span<const uint64_t>(soa_buf_.data() + 1 * elems, elems);
        if (std::getenv("SOFTMAX_DBG_COEFF") &&
            (!opened_.empty()) &&
            (c0_.size() > 0) && (c1_.size() > 0)) {
          std::cerr << "[RecipTask p" << R.party << "] opened0=" << opened_[0]
                    << " c0=" << c0_[0] << " c1=" << c1_[0] << "\n";
        }
        init_mul_out_.assign(elems, 0);
        auto triples = next_triples(elems);
        init_mul_ = std::make_unique<MulTask>(c1_, x_, std::span<uint64_t>(init_mul_out_.data(), init_mul_out_.size()), triples);
        st_ = St::InitMul;
        return detail::Need::None;
      }
      case St::InitMul: {
        auto need = init_mul_->step(R);
        if (!init_mul_->done()) return need;
        if (std::getenv("SOFTMAX_DBG_COEFF") && !init_mul_out_.empty()) {
          std::cerr << "[RecipTask p" << R.party << "] init_mul_out[0]=" << init_mul_out_[0] << "\n";
        }
        if (ref_trunc_) {
          init_trunc_out_.assign(init_mul_out_.size(), 0);
          plain_trunc(init_mul_out_, init_trunc_out_);
          y_.assign(init_trunc_out_.begin(), init_trunc_out_.end());
          for (size_t i = 0; i < y_.size(); ++i) {
            y_[i] = proto::add_mod(y_[i], c0_[i]);
          }
          init_y_debug_ = y_;
          iter_ = 0;
          st_ = St::IterMul1;
          return detail::Need::None;
        }
        init_trunc_out_.assign(init_mul_out_.size(), 0);
        init_trunc_ = std::make_unique<TruncTask>(bundle_.trunc_fb,
                                                  std::span<const uint64_t>(init_mul_out_.data(), init_mul_out_.size()),
                                                  std::span<uint64_t>(init_trunc_out_.data(), init_trunc_out_.size()));
        st_ = St::InitTrunc;
        return detail::Need::None;
      }
      case St::InitTrunc: {
        auto need = init_trunc_->step(R);
        if (!init_trunc_->done()) return need;
#ifdef SUF_HAVE_CUDA
        if (R.device_pipeline) {
          if (auto* dptr = init_trunc_->device_out()) {
            size_t n = std::min(init_trunc_out_.size(), init_trunc_->device_out_elems());
            if (n > 0) {
              cudaMemcpy(init_trunc_out_.data(), dptr, n * sizeof(uint64_t), cudaMemcpyDeviceToHost);
            }
            init_trunc_->release_device_out();
          }
        }
#endif
        if (std::getenv("SOFTMAX_DBG_COEFF") && !init_trunc_out_.empty()) {
          std::cerr << "[RecipTask p" << R.party << "] init_trunc[0]=" << init_trunc_out_[0] << "\n";
        }
        y_.assign(init_trunc_out_.begin(), init_trunc_out_.end());
        for (size_t i = 0; i < y_.size(); ++i) {
          y_[i] = proto::add_mod(y_[i], c0_[i]);
        }
        init_y_debug_ = y_;
        if (dbg_rec_ && !y_.empty()) {
          std::cerr << "[RecipTask p" << R.party << "] init y[0]=" << y_[0] << "\n";
        }
        iter_ = 0;
        st_ = St::IterMul1;
        return detail::Need::None;
      }
      case St::IterMul1: {
        if (iter_ >= bundle_.nr_iters) {
          std::span<uint64_t> y_span(y_.data(), y_.size());
          for (size_t i = 0; i < y_span.size(); ++i) out_[i] = y_span[i];
          st_ = St::Done;
          return detail::Need::None;
        }
        auto triples = next_triples(x_.size());
        t_xy_.assign(x_.size(), 0);
        mul1_ = std::make_unique<MulTask>(
            std::span<const uint64_t>(y_.data(), y_.size()),
            x_,
            std::span<uint64_t>(t_xy_.data(), t_xy_.size()),
            triples);
        st_ = St::Mul1;
        return detail::Need::None;
      }
      case St::Mul1: {
        auto need = mul1_->step(R);
        if (!mul1_->done()) return need;
        t_xy_tr_.assign(t_xy_.size(), 0);
        if (ref_trunc_) {
          plain_trunc(t_xy_, t_xy_tr_);
          if (dbg_rec_ && !t_xy_tr_.empty()) {
            std::cerr << "[RecipTask p" << R.party << "] iter" << iter_ << " t_xy_tr[0]=" << t_xy_tr_[0] << "\n";
          }
          two_minus_.assign(x_.size(), 0);
          uint64_t two = (bundle_.frac_bits >= 64) ? 0ull : (uint64_t(2) << bundle_.frac_bits);
          for (size_t i = 0; i < x_.size(); ++i) {
            uint64_t const_share = (R.party == 0) ? two : 0ull;
            two_minus_[i] = proto::sub_mod(const_share, t_xy_tr_[i]);
          }
          auto triples = next_triples(x_.size());
          t_update_.assign(x_.size(), 0);
          mul2_ = std::make_unique<MulTask>(
              std::span<const uint64_t>(y_.data(), y_.size()),
              std::span<const uint64_t>(two_minus_.data(), two_minus_.size()),
              std::span<uint64_t>(t_update_.data(), t_update_.size()),
              triples);
          st_ = St::Mul2;
          return detail::Need::None;
        }
        trunc1_ = std::make_unique<TruncTask>(bundle_.trunc_fb,
                                              std::span<const uint64_t>(t_xy_.data(), t_xy_.size()),
                                              std::span<uint64_t>(t_xy_tr_.data(), t_xy_tr_.size()));
        st_ = St::Trunc1;
        return detail::Need::None;
      }
      case St::Trunc1: {
        auto need = trunc1_->step(R);
        if (!trunc1_->done()) return need;
#ifdef SUF_HAVE_CUDA
        if (R.device_pipeline) {
          if (auto* dptr = trunc1_->device_out()) {
            size_t n = std::min(t_xy_tr_.size(), trunc1_->device_out_elems());
            if (n > 0) cudaMemcpy(t_xy_tr_.data(), dptr, n * sizeof(uint64_t), cudaMemcpyDeviceToHost);
            trunc1_->release_device_out();
          }
        }
#endif
        if (dbg_rec_ && !t_xy_tr_.empty()) {
          std::cerr << "[RecipTask p" << R.party << "] iter" << iter_ << " t_xy_tr[0]=" << t_xy_tr_[0] << "\n";
        }
        two_minus_.assign(x_.size(), 0);
        uint64_t two = (bundle_.frac_bits >= 64) ? 0ull : (uint64_t(2) << bundle_.frac_bits);
        for (size_t i = 0; i < x_.size(); ++i) {
          uint64_t const_share = (R.party == 0) ? two : 0ull;
          two_minus_[i] = proto::sub_mod(const_share, t_xy_tr_[i]);
        }
        auto triples = next_triples(x_.size());
        t_update_.assign(x_.size(), 0);
        mul2_ = std::make_unique<MulTask>(
            std::span<const uint64_t>(y_.data(), y_.size()),
            std::span<const uint64_t>(two_minus_.data(), two_minus_.size()),
            std::span<uint64_t>(t_update_.data(), t_update_.size()),
            triples);
        st_ = St::Mul2;
        return detail::Need::None;
      }
      case St::Mul2: {
        auto need = mul2_->step(R);
        if (!mul2_->done()) return need;
        t_update_tr_.assign(t_update_.size(), 0);
        if (ref_trunc_) {
          plain_trunc(t_update_, t_update_tr_);
          if (dbg_rec_ && !t_update_tr_.empty()) {
            std::cerr << "[RecipTask p" << R.party << "] iter" << iter_ << " y_new[0]=" << t_update_tr_[0] << "\n";
          }
          y_.assign(t_update_tr_.begin(), t_update_tr_.end());
          ++iter_;
          st_ = St::IterMul1;
          return detail::Need::None;
        }
        trunc2_ = std::make_unique<TruncTask>(bundle_.trunc_fb,
                                              std::span<const uint64_t>(t_update_.data(), t_update_.size()),
                                              std::span<uint64_t>(t_update_tr_.data(), t_update_tr_.size()));
        st_ = St::Trunc2;
        return detail::Need::None;
      }
      case St::Trunc2: {
        auto need = trunc2_->step(R);
        if (!trunc2_->done()) return need;
#ifdef SUF_HAVE_CUDA
        if (R.device_pipeline) {
          uint64_t* steal_ptr = nullptr;
          size_t steal_elems = 0;
          if (trunc2_->device_out()) {
            // Take ownership of the device buffer so we can optionally
            // expose it downstream without double-free.
            steal_ptr = trunc2_->take_device_out(&steal_elems);
            size_t n = std::min(t_update_tr_.size(), steal_elems);
            if (n > 0) cudaMemcpy(t_update_tr_.data(), steal_ptr, n * sizeof(uint64_t), cudaMemcpyDeviceToHost);
            // If this is the final iteration, keep the device buffer for downstream consumers.
            if (iter_ + 1 >= bundle_.nr_iters) {
              d_out_device_ = steal_ptr;
              d_out_elems_ = steal_elems;
              steal_ptr = nullptr;
            }
            if (steal_ptr) cudaFree(steal_ptr);
          }
        }
#endif
        if (dbg_rec_ && !t_update_tr_.empty()) {
          std::cerr << "[RecipTask p" << R.party << "] iter" << iter_ << " y_new[0]=" << t_update_tr_[0] << "\n";
        }
        y_.assign(t_update_tr_.begin(), t_update_tr_.end());
        ++iter_;
        st_ = St::IterMul1;
        return detail::Need::None;
      }
      case St::Done:
        return detail::Need::None;
    }
    return detail::Need::None;
  }

 private:
  enum class St {
    OpenXhat,
    WaitXhatOpen,
    EnqueueCoeff,
    WaitCoeff,
    InitMul,
    InitTrunc,
    IterMul1,
    Mul1,
    Trunc1,
    Mul2,
    Trunc2,
    Done
  } st_ = St::OpenXhat;

  RecipTaskBundle bundle_;
  const gates::CompositePartyKey* key_ = nullptr;
  std::span<const uint64_t> x_;
  std::span<uint64_t> out_;

  std::vector<uint64_t> masked_;
  std::vector<uint64_t> opened_;
  OpenHandle h_open_{};
  PfssHandle coeff_handle_{};

  std::span<const proto::BeaverTriple64Share> triple_span_;
  size_t triple_cursor_ = 0;
  bool dbg_rec_ = false;

  // coeff storage
  std::vector<uint64_t> coeff_buf_;
  std::vector<uint64_t> soa_buf_;
  std::span<const uint64_t> c0_;
  std::span<const uint64_t> c1_;

  std::vector<uint64_t> y_;
  std::vector<uint64_t> init_y_debug_;
  int iter_ = 0;

  std::vector<uint64_t> init_mul_out_;
  std::vector<uint64_t> init_trunc_out_;
  std::unique_ptr<MulTask> init_mul_;
  std::unique_ptr<TruncTask> init_trunc_;

  std::vector<uint64_t> t_xy_;
  std::vector<uint64_t> t_xy_tr_;
  std::vector<uint64_t> two_minus_;
  std::vector<uint64_t> t_update_;
  std::vector<uint64_t> t_update_tr_;

  std::unique_ptr<MulTask> mul1_;
  std::unique_ptr<MulTask> mul2_;
  std::unique_ptr<TruncTask> trunc1_;
  std::unique_ptr<TruncTask> trunc2_;
  bool ref_trunc_ = false;
  bool force_ref_full_ = false;
  int party_ = -1;

#ifdef SUF_HAVE_CUDA
  uint64_t* d_out_device_ = nullptr;
  size_t d_out_elems_ = 0;
#endif

  void plain_trunc(const std::vector<uint64_t>& in, std::vector<uint64_t>& out) {
    int shift_bits = bundle_.frac_bits;
    if (bundle_.trunc_fb && !bundle_.trunc_fb->keys.k0.compiled.extra_u64.empty()) {
      shift_bits = static_cast<int>(bundle_.trunc_fb->keys.k0.compiled.extra_u64[0] & 0xFFFFu);
    }
    out.resize(in.size());
    for (size_t i = 0; i < in.size(); ++i) {
      int64_t v = to_signed(in[i]);
      int64_t shifted = (shift_bits >= 64) ? 0ll : (v >> shift_bits);
      out[i] = (party_ == 0) ? static_cast<uint64_t>(shifted) : 0ull;
    }
  }

  std::span<const proto::BeaverTriple64Share> next_triples(size_t n) {
    if (triple_cursor_ + n > triple_span_.size()) {
      throw std::runtime_error("RecipTask: not enough triples");
    }
    auto s = triple_span_.subspan(triple_cursor_, n);
    triple_cursor_ += n;
    return s;
  }
};

}  // namespace runtime
