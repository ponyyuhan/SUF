#include <chrono>
#include <condition_variable>
#include <cctype>
#include <cstdio>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <map>
#include <csignal>
#include <execinfo.h>
#include <cstdlib>
#include <atomic>
#include <mutex>
#include <queue>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <algorithm>
#include <vector>
#include <cstring>
#include <sys/resource.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#if __has_include(<span>)
#include <span>
#elif __has_include(<experimental/span>)
#include <experimental/span>
namespace std { using std::experimental::span; }
#endif

#include "compiler/truncation_lowering.hpp"
#include "gates/nexp_composite.hpp"
#include "gates/reciprocal_composite.hpp"
#include "nn/model_specs.hpp"
#include "nn/transformer_layer.hpp"
#include "nn/kv_cache.hpp"
#include "nn/tensor_view.hpp"
#include "nn/softmax_block_task.hpp"
#include "proto/backend_clear.hpp"
#include "proto/backend_gpu.hpp"
#include "proto/beaver.hpp"
#include "proto/common.hpp"
#include "proto/channel.hpp"
#include "proto/sigma_fast_backend_ext.hpp"
#include "runtime/phase_executor.hpp"
#include "runtime/phase_tasks.hpp"
#include "runtime/pfss_phase_planner.hpp"
#include "runtime/pfss_superbatch.hpp"
#include "mpc/net.hpp"
#include "gates/composite_fss.hpp"
#include "runtime/bench_accounting.hpp"
#include "runtime/bench_key_cost.hpp"

namespace {

struct Args {
  std::string model;
  std::string backend = "cpu";
  int seq_len = 128;
  int batch_size = 1;
  int n_iters = 1;
  int n_layers = 0;  // 0 => use model spec
  std::string log_json;
  int per_element_masks = -1;   // -1 => use env/default
  int open_pack_effbits = -1;   // -1 => use env/default
  int omp_threads = 0;          // 0 => don't override
  int gelu_const = -1;          // -1 => use env/default
  int gelu_const_segments = 0;  // 0 => leave env/default
};

Args parse(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    std::string s(argv[i]);
    auto take_val = [&](const std::string& prefix) -> std::string {
      return s.substr(prefix.size());
    };
    auto next_val = [&](int& idx) -> std::string {
      if (idx + 1 >= argc) throw std::runtime_error("flag requires value: " + s);
      return std::string(argv[++idx]);
    };
    if (s == "--model") a.model = next_val(i);
    else if (s == "--backend") a.backend = next_val(i);
    else if (s == "--seq-len") a.seq_len = std::stoi(next_val(i));
    else if (s == "--batch-size") a.batch_size = std::stoi(next_val(i));
    else if (s == "--n-iters") a.n_iters = std::stoi(next_val(i));
    else if (s == "--n-layers") a.n_layers = std::stoi(next_val(i));
    else if (s == "--log-json") a.log_json = next_val(i);
    else if (s == "--per-element-masks") a.per_element_masks = std::stoi(next_val(i));
    else if (s == "--open-pack") a.open_pack_effbits = std::stoi(next_val(i));
    else if (s == "--omp-threads") a.omp_threads = std::stoi(next_val(i));
    else if (s == "--gelu-const") a.gelu_const = std::stoi(next_val(i));
    else if (s == "--gelu-const-segments") a.gelu_const_segments = std::stoi(next_val(i));
    else if (s.rfind("--model=", 0) == 0) a.model = take_val("--model=");
    else if (s.rfind("--backend=", 0) == 0) a.backend = take_val("--backend=");
    else if (s.rfind("--seq-len=", 0) == 0) a.seq_len = std::stoi(take_val("--seq-len="));
    else if (s.rfind("--batch-size=", 0) == 0) a.batch_size = std::stoi(take_val("--batch-size="));
    else if (s.rfind("--n-iters=", 0) == 0) a.n_iters = std::stoi(take_val("--n-iters="));
    else if (s.rfind("--n-layers=", 0) == 0) a.n_layers = std::stoi(take_val("--n-layers="));
    else if (s.rfind("--log-json=", 0) == 0) a.log_json = take_val("--log-json=");
    else if (s.rfind("--per-element-masks=", 0) == 0) a.per_element_masks = std::stoi(take_val("--per-element-masks="));
    else if (s.rfind("--open-pack=", 0) == 0) a.open_pack_effbits = std::stoi(take_val("--open-pack="));
    else if (s.rfind("--omp-threads=", 0) == 0) a.omp_threads = std::stoi(take_val("--omp-threads="));
    else if (s.rfind("--gelu-const=", 0) == 0) a.gelu_const = std::stoi(take_val("--gelu-const="));
    else if (s.rfind("--gelu-const-segments=", 0) == 0) a.gelu_const_segments = std::stoi(take_val("--gelu-const-segments="));
    else if (s == "--help" || s == "-h") {
      std::cerr << "Usage: bench_suf_transformer --model NAME [--backend cpu|gpu] "
                << "[--seq-len L] [--batch-size B] [--n-iters N] [--n-layers LAYERS]\n"
                << "  [--per-element-masks 0|1] [--open-pack 0|1] [--omp-threads T]\n"
                << "  [--gelu-const 0|1] [--gelu-const-segments K] [--log-json PATH]\n";
      std::exit(0);
    }
  }
  if (a.model.empty()) throw std::runtime_error("missing --model");
  if (a.log_json.empty()) throw std::runtime_error("missing --log-json");
  return a;
}

struct CountingChan : proto::IChannel {
  struct Ring {
    explicit Ring(size_t cap_pow2 = (1ull << 24)) : buf(cap_pow2, 0), mask(cap_pow2 - 1) {
      if ((cap_pow2 & (cap_pow2 - 1)) != 0) {
        throw std::runtime_error("CountingChan::Ring: capacity must be power of two");
      }
    }
    std::vector<uint8_t> buf;
    size_t mask = 0;
    std::atomic<uint64_t> head{0};
    std::atomic<uint64_t> tail{0};

    void reset() {
      head.store(0, std::memory_order_relaxed);
      tail.store(0, std::memory_order_relaxed);
    }

    static inline void spin_pause() {
#if defined(__x86_64__) || defined(_M_X64)
      for (int i = 0; i < 64; ++i) __builtin_ia32_pause();
#else
      std::this_thread::yield();
#endif
    }

    void push_many(const uint8_t* src, size_t n) {
      if (n == 0) return;
      const uint64_t cap = static_cast<uint64_t>(buf.size());
      size_t off = 0;
      uint64_t h = head.load(std::memory_order_relaxed);
      while (off < n) {
        size_t chunk = std::min(n - off, static_cast<size_t>(cap));
        for (;;) {
          uint64_t t = tail.load(std::memory_order_acquire);
          if (h + static_cast<uint64_t>(chunk) - t <= cap) break;
          spin_pause();
        }
        size_t idx = static_cast<size_t>(h) & mask;
        size_t first = std::min(chunk, buf.size() - idx);
        std::memcpy(&buf[idx], src + off, first);
        if (first < chunk) {
          std::memcpy(&buf[0], src + off + first, chunk - first);
        }
        h += static_cast<uint64_t>(chunk);
        head.store(h, std::memory_order_release);
        off += chunk;
      }
    }

    void pop_many(uint8_t* dst, size_t n) {
      if (n == 0) return;
      const uint64_t cap = static_cast<uint64_t>(buf.size());
      size_t off = 0;
      uint64_t t = tail.load(std::memory_order_relaxed);
      while (off < n) {
        size_t chunk = std::min(n - off, static_cast<size_t>(cap));
        for (;;) {
          uint64_t h = head.load(std::memory_order_acquire);
          if (static_cast<uint64_t>(chunk) <= (h - t)) break;
          spin_pause();
        }
        size_t idx = static_cast<size_t>(t) & mask;
        size_t first = std::min(chunk, buf.size() - idx);
        std::memcpy(dst + off, &buf[idx], first);
        if (first < chunk) {
          std::memcpy(dst + off + first, &buf[0], chunk - first);
        }
        t += static_cast<uint64_t>(chunk);
        tail.store(t, std::memory_order_release);
        off += chunk;
      }
    }
  };

  struct Shared {
    Ring q0to1;
    Ring q1to0;
    std::atomic<uint64_t> sent0{0};
    std::atomic<uint64_t> sent1{0};
    std::atomic<uint64_t> calls0{0};
    std::atomic<uint64_t> calls1{0};
    std::atomic<uint64_t> bytes1{0};
    std::atomic<uint64_t> bytes8{0};
    std::atomic<uint64_t> bytes_other{0};
  };

  Shared* sh = nullptr;
  bool is0 = false;
  CountingChan(Shared* s, bool party0) : sh(s), is0(party0) {}
  void send_bytes(const void* data, size_t n) override {
    if (n == 0) return;
    const auto* src = reinterpret_cast<const uint8_t*>(data);
    if (is0) {
      sh->q0to1.push_many(src, n);
      sh->sent0.fetch_add(n, std::memory_order_relaxed);
      sh->calls0.fetch_add(1, std::memory_order_relaxed);
    } else {
      sh->q1to0.push_many(src, n);
      sh->sent1.fetch_add(n, std::memory_order_relaxed);
      sh->calls1.fetch_add(1, std::memory_order_relaxed);
    }
    if (n == 1) sh->bytes1.fetch_add(1, std::memory_order_relaxed);
    else if (n == 8) sh->bytes8.fetch_add(8, std::memory_order_relaxed);
    else sh->bytes_other.fetch_add(n, std::memory_order_relaxed);
  }
  void recv_bytes(void* data, size_t n) override {
    if (n == 0) return;
    auto* dst = reinterpret_cast<uint8_t*>(data);
    if (is0) sh->q1to0.pop_many(dst, n);
    else sh->q0to1.pop_many(dst, n);
  }
};

struct CountingNetChan : net::Chan {
  struct Ring {
    // Default large enough to hold a full OpenCollector flush (order ~ millions of words)
    // without forcing the sender to spin on backpressure.
    explicit Ring(size_t cap_pow2 = (1ull << 22)) : buf(cap_pow2, 0), mask(cap_pow2 - 1) {
      if ((cap_pow2 & (cap_pow2 - 1)) != 0) throw std::runtime_error("Ring: capacity must be power of two");
    }
    std::vector<uint64_t> buf;
    size_t mask = 0;
    std::atomic<uint64_t> head{0};
    std::atomic<uint64_t> tail{0};

    void reset() {
      head.store(0, std::memory_order_relaxed);
      tail.store(0, std::memory_order_relaxed);
    }

    static inline void spin_pause() {
#if defined(__x86_64__) || defined(_M_X64)
      for (int i = 0; i < 64; ++i) __builtin_ia32_pause();
#else
      std::this_thread::yield();
#endif
    }

    void push(uint64_t v) {
      uint64_t h = head.load(std::memory_order_relaxed);
      for (;;) {
        uint64_t t = tail.load(std::memory_order_acquire);
        if (h - t < buf.size()) break;
        spin_pause();
      }
      buf[static_cast<size_t>(h) & mask] = v;
      head.store(h + 1, std::memory_order_release);
    }

    void push_many(const uint64_t* src, size_t n) {
      if (n == 0) return;
      const uint64_t cap = static_cast<uint64_t>(buf.size());
      size_t off = 0;
      uint64_t h = head.load(std::memory_order_relaxed);
      while (off < n) {
        size_t chunk = std::min(n - off, static_cast<size_t>(cap));
        for (;;) {
          uint64_t t = tail.load(std::memory_order_acquire);
          if (h + static_cast<uint64_t>(chunk) - t <= cap) break;
          spin_pause();
        }
        size_t idx = static_cast<size_t>(h) & mask;
        size_t first = std::min(chunk, buf.size() - idx);
        std::memcpy(&buf[idx], src + off, first * sizeof(uint64_t));
        if (first < chunk) {
          std::memcpy(&buf[0], src + off + first, (chunk - first) * sizeof(uint64_t));
        }
        h += static_cast<uint64_t>(chunk);
        head.store(h, std::memory_order_release);
        off += chunk;
      }
    }

    uint64_t pop() {
      uint64_t t = tail.load(std::memory_order_relaxed);
      for (;;) {
        uint64_t h = head.load(std::memory_order_acquire);
        if (t < h) break;
        spin_pause();
      }
      uint64_t v = buf[static_cast<size_t>(t) & mask];
      tail.store(t + 1, std::memory_order_release);
      return v;
    }

    void pop_many(uint64_t* dst, size_t n) {
      if (n == 0) return;
      const uint64_t cap = static_cast<uint64_t>(buf.size());
      size_t off = 0;
      uint64_t t = tail.load(std::memory_order_relaxed);
      while (off < n) {
        size_t chunk = std::min(n - off, static_cast<size_t>(cap));
        for (;;) {
          uint64_t h = head.load(std::memory_order_acquire);
          if (static_cast<uint64_t>(chunk) <= (h - t)) break;
          spin_pause();
        }
        size_t idx = static_cast<size_t>(t) & mask;
        size_t first = std::min(chunk, buf.size() - idx);
        std::memcpy(dst + off, &buf[idx], first * sizeof(uint64_t));
        if (first < chunk) {
          std::memcpy(dst + off + first, &buf[0], (chunk - first) * sizeof(uint64_t));
        }
        t += static_cast<uint64_t>(chunk);
        tail.store(t, std::memory_order_release);
        off += chunk;
      }
    }
  };

  struct Shared {
    Ring q0to1;
    Ring q1to0;
    std::atomic<uint64_t> sent0{0};
    std::atomic<uint64_t> sent1{0};
  };
  Shared* sh = nullptr;
  bool is0 = false;
  CountingNetChan(Shared* s, bool p0) : sh(s), is0(p0) {}
  void send_u64(uint64_t v) override {
    if (is0) {
      sh->q0to1.push(v);
      sh->sent0.fetch_add(sizeof(uint64_t), std::memory_order_relaxed);
    } else {
      sh->q1to0.push(v);
      sh->sent1.fetch_add(sizeof(uint64_t), std::memory_order_relaxed);
    }
  }
  uint64_t recv_u64() override {
    return is0 ? sh->q1to0.pop() : sh->q0to1.pop();
  }

  void send_u64s(const uint64_t* data, size_t n) override {
    if (n == 0) return;
    if (is0) {
      sh->q0to1.push_many(data, n);
      sh->sent0.fetch_add(n * sizeof(uint64_t), std::memory_order_relaxed);
    } else {
      sh->q1to0.push_many(data, n);
      sh->sent1.fetch_add(n * sizeof(uint64_t), std::memory_order_relaxed);
    }
  }

  void recv_u64s(uint64_t* data, size_t n) override {
    if (n == 0) return;
    if (is0) sh->q1to0.pop_many(data, n);
    else sh->q0to1.pop_many(data, n);
  }
};

thread_local int tl_party = -1;

static void composite_keygen_hook(const gates::CompositeKeyPair& kp) {
  // Count composite PFSS key material during initial warmup/keygen.
  // Note: task materials are cached and may be generated by either party thread.
  if (!runtime::bench::offline_counting_enabled()) return;
  runtime::bench::charge_offline_bytes(runtime::bench::composite_keypair_cost(kp));
}

void write_json(const Args& a,
                const nn::ModelSpec& spec,
                int n_layers_run,
                double keygen_s,
                double elapsed_s_mean,
                double elapsed_s_max,
                uint64_t pfss_bytes_mean,
                uint64_t net_bytes_mean,
                double cpu_user_s,
                double cpu_sys_s,
                double cpu_util_avg,
                long max_rss_kb,
                double wall_time_s,
                int n_iters,
                const runtime::PhaseExecutor::Stats& stats0,
                const runtime::PhaseExecutor::Stats& stats1) {
  auto env_flag = [](const char* name) -> bool {
    const char* v = std::getenv(name);
    if (!v) return false;
    std::string s(v);
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return !(s == "0" || s == "false" || s == "off" || s == "no");
  };
  auto env_int = [](const char* name) -> int64_t {
    const char* v = std::getenv(name);
    if (!v) return 0;
    try {
      return std::stoll(std::string(v));
    } catch (...) {
      return 0;
    }
  };
  const uint64_t coeff_jobs = stats0.pfss_coeff_jobs + stats1.pfss_coeff_jobs;
  const uint64_t trunc_jobs = stats0.pfss_trunc_jobs + stats1.pfss_trunc_jobs;
  const uint64_t coeff_hatx = stats0.pfss_coeff_hatx_words + stats1.pfss_coeff_hatx_words;
  const uint64_t trunc_hatx = stats0.pfss_trunc_hatx_words + stats1.pfss_trunc_hatx_words;
  const uint64_t coeff_flushes = stats0.pfss_coeff_flushes + stats1.pfss_coeff_flushes;
  const uint64_t trunc_flushes = stats0.pfss_trunc_flushes + stats1.pfss_trunc_flushes;
  const uint64_t open_flushes = stats0.open_flushes + stats1.open_flushes;
  const uint64_t opened_words = stats0.opened_words + stats1.opened_words;
  const uint64_t opened_words_beaver = stats0.opened_words_beaver + stats1.opened_words_beaver;
  const uint64_t opened_words_mask = stats0.opened_words_mask + stats1.opened_words_mask;
  const uint64_t opened_words_other = stats0.opened_words_other + stats1.opened_words_other;
  const uint64_t open_bytes = opened_words * sizeof(uint64_t);
  auto packed_open_bytes = [&](uint64_t words) -> uint64_t {
    const uint64_t n_bits = static_cast<uint64_t>(spec.n_bits);
    if (n_bits == 0 || n_bits > 64) return words * sizeof(uint64_t);
    __uint128_t bits = static_cast<__uint128_t>(words) * static_cast<__uint128_t>(n_bits);
    bits += 7;
    return static_cast<uint64_t>(bits / 8);
  };
  const uint64_t open_packed_bytes = packed_open_bytes(opened_words);
  const uint64_t open_bytes_beaver = opened_words_beaver * sizeof(uint64_t);
  const uint64_t open_bytes_mask = opened_words_mask * sizeof(uint64_t);
  const uint64_t open_bytes_other = opened_words_other * sizeof(uint64_t);
  const uint64_t open_packed_bytes_beaver = packed_open_bytes(opened_words_beaver);
  const uint64_t open_packed_bytes_mask = packed_open_bytes(opened_words_mask);
  const uint64_t open_packed_bytes_other = packed_open_bytes(opened_words_other);
  const uint64_t pfss_related_bytes = pfss_bytes_mean + open_bytes_mask;
  const uint64_t pfss_related_packed_bytes = pfss_bytes_mean + open_packed_bytes_mask;
  const uint64_t beaver_related_bytes = open_bytes_beaver;
  const uint64_t beaver_related_packed_bytes = open_packed_bytes_beaver;

  std::filesystem::path out_path(a.log_json);
  if (out_path.has_parent_path()) {
    std::error_code ec;
    std::filesystem::create_directories(out_path.parent_path(), ec);
  }
  std::ofstream f(a.log_json);
  if (!f) throw std::runtime_error("failed to open log json: " + a.log_json);
  f << "{\n";
  f << "  \"system\": \"suf\",\n";
  f << "  \"backend\": \"" << a.backend << "\",\n";
  f << "  \"model\": \"" << a.model << "\",\n";
  f << "  \"seq_len\": " << a.seq_len << ",\n";
  f << "  \"batch_size\": " << a.batch_size << ",\n";
  f << "  \"n_layers\": " << spec.n_layers << ",\n";
  f << "  \"n_layers_run\": " << n_layers_run << ",\n";
  f << "  \"n_iters\": " << n_iters << ",\n";
  f << "  \"n_heads\": " << spec.n_heads << ",\n";
  f << "  \"d_model\": " << spec.d_model << ",\n";
  f << "  \"n_bits\": " << spec.n_bits << ",\n";
  f << "  \"frac_bits\": " << spec.frac_bits << ",\n";
  f << "  \"timing\": { "
    << "\"keygen_time_s\": " << keygen_s
    << ", \"online_time_s\": " << elapsed_s_mean
    << ", \"online_time_s_mean\": " << elapsed_s_mean
    << ", \"online_time_s_max\": " << elapsed_s_max
    << ", \"wall_time_s\": " << wall_time_s
    << " },\n";
  f << "  \"resources\": { "
    << "\"cpu_user_s\": " << cpu_user_s
    << ", \"cpu_sys_s\": " << cpu_sys_s
    << ", \"cpu_util_avg\": " << cpu_util_avg
    << ", \"max_rss_kb\": " << max_rss_kb
    << " },\n";
  const uint64_t key_bytes_total = runtime::bench::offline_bytes_total();
  const uint64_t key_bytes_composite = runtime::bench::offline_bytes(runtime::bench::OfflineBytesKind::CompositeTape);
  const uint64_t key_bytes_matmul = runtime::bench::offline_bytes(runtime::bench::OfflineBytesKind::MatmulTriple);
  const uint64_t key_bytes_beaver = runtime::bench::offline_bytes(runtime::bench::OfflineBytesKind::BeaverTriple);
  const uint64_t key_bytes_row = runtime::bench::offline_bytes(runtime::bench::OfflineBytesKind::RowBroadcastTriple);
  const uint64_t key_bytes_other = runtime::bench::offline_bytes(runtime::bench::OfflineBytesKind::Other);
  f << "  \"preprocessing\": { \"key_bytes\": " << key_bytes_total
    << ", \"key_bytes_composite\": " << key_bytes_composite
    << ", \"key_bytes_matmul_triples\": " << key_bytes_matmul
    << ", \"key_bytes_beaver_triples\": " << key_bytes_beaver
    << ", \"key_bytes_row_triples\": " << key_bytes_row
    << ", \"key_bytes_other\": " << key_bytes_other
    << ", \"key_bytes_scope\": \"dealer_total\""
    << ", \"cache_material\": " << (env_flag("SUF_BENCH_CACHE_MATERIAL") ? "true" : "false")
    << ", \"per_element_masks\": " << (env_flag("SUF_PER_ELEMENT_MASKS") ? "true" : "false")
    << ", \"open_pack_effbits\": " << (env_flag("SUF_OPEN_PACK_EFFBITS") ? "true" : "false")
    << ", \"gelu_const\": " << (env_flag("SUF_GELU_CONST") ? "true" : "false")
    << ", \"gelu_const_segments\": " << env_int("SUF_GELU_CONST_SEGMENTS")
    << " },\n";
  f << "  \"communication\": { \"pfss_bytes\": " << pfss_bytes_mean
    << ", \"net_bytes\": " << net_bytes_mean
    << ", \"open_bytes\": " << open_bytes
    << ", \"open_bytes_beaver\": " << open_bytes_beaver
    << ", \"open_bytes_mask\": " << open_bytes_mask
    << ", \"open_bytes_other\": " << open_bytes_other
    << ", \"open_packed_bytes\": " << open_packed_bytes
    << ", \"open_packed_bytes_beaver\": " << open_packed_bytes_beaver
    << ", \"open_packed_bytes_mask\": " << open_packed_bytes_mask
    << ", \"open_packed_bytes_other\": " << open_packed_bytes_other
    << ", \"pfss_related_bytes\": " << pfss_related_bytes
    << ", \"pfss_related_packed_bytes\": " << pfss_related_packed_bytes
    << ", \"beaver_related_bytes\": " << beaver_related_bytes
    << ", \"beaver_related_packed_bytes\": " << beaver_related_packed_bytes
    << ", \"online_bytes\": " << (pfss_bytes_mean + net_bytes_mean)
    << ", \"online_packed_bytes\": " << (pfss_bytes_mean +
                                         ((net_bytes_mean > open_bytes) ? (net_bytes_mean - open_bytes) : 0ull) +
                                         open_packed_bytes)
    << " },\n";
  f << "  \"pfss\": {\n";
  f << "    \"num_jobs\": " << (coeff_jobs + trunc_jobs) << ",\n";
  f << "    \"num_flushes\": " << (coeff_flushes + trunc_flushes) << ",\n";
  f << "    \"total_hatx_words\": " << (coeff_hatx + trunc_hatx) << ",\n";
  f << "    \"coeff_jobs\": " << coeff_jobs << ",\n";
  f << "    \"trunc_jobs\": " << trunc_jobs << ",\n";
  f << "    \"coeff_hatx_words\": " << coeff_hatx << ",\n";
  f << "    \"trunc_hatx_words\": " << trunc_hatx << ",\n";
  f << "    \"open_flushes\": " << open_flushes << ",\n";
  f << "    \"opened_words\": " << opened_words << ",\n";
  f << "    \"opened_words_beaver\": " << opened_words_beaver << ",\n";
  f << "    \"opened_words_mask\": " << opened_words_mask << ",\n";
  f << "    \"opened_words_other\": " << opened_words_other << "\n";
  f << "  },\n";
  f << "  \"notes\": \"transformer forward (PFSS-backed)\"\n";
  f << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
  rusage ru_start{};
  rusage ru_end{};
  auto dump_bt = [](int sig) {
    void* addrs[64];
    int n = backtrace(addrs, 64);
    std::fprintf(stderr, "\n[bench_suf] caught signal %d, backtrace:\n", sig);
    backtrace_symbols_fd(addrs, n, STDERR_FILENO);
    std::_Exit(128 + sig);
  };
  std::signal(SIGABRT, dump_bt);
  std::signal(SIGSEGV, dump_bt);
  std::signal(SIGILL, dump_bt);
	  try {
	    auto args = parse(argc, argv);
#ifdef _OPENMP
	    if (args.omp_threads > 0) {
	      // Apply to all threads (bench runs two party threads).
	      const std::string omp_threads = std::to_string(args.omp_threads);
	      ::setenv("OMP_NUM_THREADS", omp_threads.c_str(), /*overwrite=*/1);
	      omp_set_dynamic(0);
	      omp_set_num_threads(args.omp_threads);
	    }
#endif
	    // Cache expensive dealer-generated task materials (activation/softmax) so
	    // we can separate one-time keygen from steady-state online time.
	    ::setenv("SUF_BENCH_CACHE_MATERIAL", "1", /*overwrite=*/0);
    if (args.per_element_masks == 0 || args.per_element_masks == 1) {
      ::setenv("SUF_PER_ELEMENT_MASKS", args.per_element_masks ? "1" : "0", /*overwrite=*/1);
    } else {
      // Per-element trunc/ARS masks prevent batching and massively increase both
      // key size and runtime overhead for large vectors. Disable by default for
      // end-to-end transformer benchmarking (can be re-enabled by exporting it).
      ::setenv("SUF_PER_ELEMENT_MASKS", "0", /*overwrite=*/0);
    }
    // Prefer GapARS when a conservative (but not explicitly proof-tagged) bound
    // is available; this matches the benchmark setting where n_bits guidance is
    // treated as a hard cap and keeps truncation communication competitive.
    ::setenv("SUF_GAPARS_ALLOW_HINT", "1", /*overwrite=*/0);
    if (args.open_pack_effbits == 0 || args.open_pack_effbits == 1) {
      ::setenv("SUF_OPEN_PACK_EFFBITS", args.open_pack_effbits ? "1" : "0", /*overwrite=*/1);
    } else {
      // Open packing can reduce wire bytes but adds local pack/unpack overhead.
      // Default OFF for stable end-to-end timing; enable explicitly for WAN/LAN benches.
      ::setenv("SUF_OPEN_PACK_EFFBITS", "0", /*overwrite=*/0);
    }
    if (args.gelu_const == 0) {
      ::unsetenv("SUF_GELU_CONST");
    } else if (args.gelu_const == 1) {
      ::setenv("SUF_GELU_CONST", "1", /*overwrite=*/1);
    }
    if (args.gelu_const_segments > 0) {
      const std::string seg = std::to_string(args.gelu_const_segments);
      ::setenv("SUF_GELU_CONST_SEGMENTS", seg.c_str(), /*overwrite=*/1);
    }
	    // For benchmarking, we force the PFSS pipeline by default on GPU runs.
	    // CPU runs use the deterministic reference fast-path unless `SUF_FORCE_PFSS`
	    // is explicitly exported by the user.
	    if (args.backend == "gpu") {
	      ::setenv("SUF_FORCE_PFSS", "1", /*overwrite=*/0);
	    }
	    const auto& spec = nn::get_model_spec(args.model);
	    proto::set_ring_bits(static_cast<int>(spec.n_bits));
	    // BERT-Tiny throughput-focused default: approximate GeLU with a degree-0
	    // piecewise constant SUF so online Beaver/trunc opens don't dominate.
	    if (spec.name == "bert-tiny" && args.gelu_const == -1) {
	      ::setenv("SUF_GELU_CONST", "1", /*overwrite=*/0);
	      ::setenv("SUF_GELU_CONST_SEGMENTS", "256", /*overwrite=*/0);
	    }
	    const int n_layers_run = (args.n_layers > 0) ? args.n_layers : static_cast<int>(spec.n_layers);

    const int B = args.batch_size;
    const int rows = args.seq_len;
    const int cols = static_cast<int>(spec.d_model);
    const int fb = static_cast<int>(spec.frac_bits);
    const int H = static_cast<int>(spec.n_heads);
    const int Dh = cols / H;
    const int Dff = static_cast<int>(spec.d_ff);
    std::mt19937_64 rng(1234);
    std::vector<int64_t> Wqkv(static_cast<size_t>(cols * cols * 3));
    std::vector<int64_t> Wout(static_cast<size_t>(cols * cols));
    std::vector<int64_t> W1(static_cast<size_t>(cols * Dff));
    std::vector<int64_t> W2(static_cast<size_t>(Dff * cols));
    auto fill_w = [&](std::vector<int64_t>& w) {
      for (auto& v : w) v = static_cast<int64_t>(rng() % 64);
    };
    fill_w(Wqkv); fill_w(Wout); fill_w(W1); fill_w(W2);

    std::vector<uint64_t> X_plain(static_cast<size_t>(B * rows * cols));
    for (auto& v : X_plain) v = rng() & ((uint64_t(1) << fb) - 1ull);
    std::vector<uint64_t> X0(X_plain.size()), X1(X_plain.size());
    for (size_t i = 0; i < X_plain.size(); ++i) {
      uint64_t s0 = proto::norm_mod(rng());
      X0[i] = s0;
      X1[i] = proto::sub_mod(X_plain[i], s0);
    }

    // Shared channels
    CountingChan::Shared pfss_sh;
    CountingChan ch0(&pfss_sh, true), ch1(&pfss_sh, false);
    CountingNetChan::Shared net_sh;
    CountingNetChan nch0(&net_sh, true), nch1(&net_sh, false);

	    // Backends
	    std::unique_ptr<proto::PfssBackendBatch> be0, be1;
#ifdef SUF_HAVE_CUDA
	    if (args.backend == "gpu") {
	      be0 = proto::make_real_gpu_backend();
	      be1 = proto::make_real_gpu_backend();
	    }
#endif
	    if (!be0 || !be1) {
	      be0 = std::make_unique<proto::ReferenceBackend>();
	      be1 = std::make_unique<proto::ReferenceBackend>();
	    }

    compiler::TruncationPassContext trunc_ctx0(*be0, 0x77726c3064756c6cull);
    compiler::TruncationPassContext trunc_ctx1(*be1, 0x77726c3164756c6cull);

    gates::set_composite_keygen_hook(&composite_keygen_hook);
    runtime::bench::reset_offline_bytes();
    runtime::bench::set_offline_counting_enabled(true);

    auto run_party = [&](int party,
                         proto::PfssBackendBatch& be,
                         CountingChan& pfss_ch,
                         CountingNetChan& net_ch,
                         compiler::TruncationPassContext& trunc_ctx,
                         runtime::PhaseExecutor::Stats& stats_out,
                         std::vector<uint64_t>& Xshare) {
      tl_party = party;
#ifdef _OPENMP
      if (args.omp_threads > 0) {
        omp_set_dynamic(0);
        omp_set_num_threads(args.omp_threads);
      }
#endif
      if (std::getenv("SUF_BENCH_TRACE")) {
        std::fprintf(stderr, "[bench_suf] party %d start backend=%s rows=%d cols=%d\n",
                     party, args.backend.c_str(), rows, cols);
      }
      runtime::PhaseExecutor pe;
      if (args.backend == "gpu") {
        // Default benchmark path materializes PFSS outputs to host immediately.
        // Keeping PFSS outputs on device is only beneficial when downstream tasks
        // can consume device pointers; otherwise it adds extra staging overhead.
        if (std::getenv("SUF_BENCH_DEVICE_PIPELINE")) {
          pe.set_device_pipeline(true);
          pe.set_device_pipeline_materialize(false);
          pe.pfss_coeff_batch().set_device_outputs(true);
          pe.pfss_trunc_batch().set_device_outputs(true);
        }
      }

      nn::LayerContext ctx;
      ctx.trunc_ctx = &trunc_ctx;
      ctx.pfss_backend_override = &be;
      ctx.pfss_chan = &pfss_ch;
      ctx.frac_bits = fb;
      // Benchmarking default: rely on stall-driven flushing (no explicit phase barriers)
      // to maximize batching and reduce synchronization overhead. Set
      // `SUF_BENCH_KEEP_BARRIERS=1` to restore explicit barriers.
      if (!std::getenv("SUF_BENCH_KEEP_BARRIERS")) {
        ctx.disable_inner_barriers = true;
      }
      runtime::PfssLayerPlanner bench_layer_planner;
      {
        runtime::PfssLayerPlanner::Limits lim;
        // Be generous: end-to-end runs can easily exceed the conservative defaults.
        lim.max_phases = 1ull << 20;
        lim.max_coeff_jobs = 1ull << 22;
        lim.max_trunc_jobs = 1ull << 22;
        lim.max_coeff_hatx_words = 1ull << 26;
        lim.max_trunc_hatx_words = 1ull << 26;
        lim.max_coeff_hatx_bytes = lim.max_coeff_hatx_words * sizeof(uint64_t);
        lim.max_trunc_hatx_bytes = lim.max_trunc_hatx_words * sizeof(uint64_t);
        lim.max_coeff_flushes = 1ull << 16;
        lim.max_trunc_flushes = 1ull << 16;
        lim.max_coeff_active_elems = 1ull << 26;
        lim.max_trunc_active_elems = 1ull << 26;
        lim.max_coeff_cost_effbits = 1ull << 62;
        lim.max_trunc_cost_effbits = 1ull << 62;
        bench_layer_planner.set_limits(lim);
      }
      ctx.pfss_layer_planner = &bench_layer_planner;
      if (args.backend == "gpu" && std::getenv("SUF_BENCH_EAGER_PFSS")) {
        // Optional: eager PFSS flushing for GPU stress benches.
        ctx.force_eager_pfss = true;
      }
#ifdef SUF_HAVE_CUDA
      std::unique_ptr<runtime::CudaPfssStager> cuda_stager;
      if (ctx.uses_gpu_backend()) {
        void* stream = nullptr;
        if (auto* gpu_eval = dynamic_cast<proto::PfssGpuStagedEval*>(&be)) {
          stream = gpu_eval->device_stream();
        }
        cuda_stager = std::make_unique<runtime::CudaPfssStager>(stream);
        ctx.pfss_gpu_stager = cuda_stager.get();
      }
#endif

      runtime::PhaseResources R{};
      R.party = party;
      R.pfss_backend = &be;
      R.pfss_chan = &pfss_ch;
      R.net_chan = &net_ch;
      R.pfss_coeff = &pe.pfss_coeff_batch();
      R.pfss_trunc = &pe.pfss_trunc_batch();
      R.opens = &pe.open_collector();

      nn::TransformerConfig cfg;
      cfg.frac_bits = fb;
      cfg.attn.D = cols;
      cfg.attn.H = H;
      cfg.attn.Dh = Dh;
      cfg.attn.S_max = rows;
      cfg.attn.frac_bits = fb;
      cfg.attn.causal = spec.causal;
      cfg.mlp.D = cols;
      cfg.mlp.frac_bits = fb;
      cfg.mlp.Hidden = Dff;
      cfg.mlp.activation = (spec.mlp_activation == "gelu")
          ? nn::MLPConfig::Activation::GeLU
          : nn::MLPConfig::Activation::SiLU;

      std::vector<uint64_t> Y(static_cast<size_t>(B * rows * cols), 0ull);
      std::vector<nn::KVCache> caches;
      caches.reserve(static_cast<size_t>(n_layers_run));
      for (int l = 0; l < n_layers_run; ++l) {
        caches.emplace_back(/*B=*/B, /*H=*/H, /*S_max=*/rows, /*Dh=*/Dh);
      }

      if (std::getenv("SUF_BENCH_TRACE")) {
        std::fprintf(stderr, "[bench_suf] party %d entering transformer_layer_forward\n", party);
      }
      for (int l = 0; l < n_layers_run; ++l) {
        std::fill(Y.begin(), Y.end(), 0ull);
        nn::transformer_layer_forward(
            cfg,
            party,
            net_ch,
            nn::view3(Xshare.data(), B, rows, cols),
            nn::view2(Wqkv.data(), cols, cols * 3),
            nn::view2(Wout.data(), cols, cols),
            nn::view2(W1.data(), cols, Dff),
            nn::view2(W2.data(), Dff, cols),
            caches[static_cast<size_t>(l)],
            nn::view3(Y.data(), B, rows, cols),
            &ctx,
            &pe);
        Xshare.swap(Y);
      }
      if (std::getenv("SUF_BENCH_TRACE")) {
        std::fprintf(stderr, "[bench_suf] party %d finished transformer_layer_forward\n", party);
      }

      stats_out = pe.stats();
    };

	    runtime::PhaseExecutor::Stats st0, st1;
	    std::vector<double> it_s;
	    it_s.reserve(static_cast<size_t>(std::max(1, args.n_iters)));
      std::vector<uint64_t> it_pfss_bytes;
      std::vector<uint64_t> it_net_bytes;
      it_pfss_bytes.reserve(static_cast<size_t>(std::max(1, args.n_iters)));
      it_net_bytes.reserve(static_cast<size_t>(std::max(1, args.n_iters)));
	    double wall_start_s = 0.0;
	    double wall_end_s = 0.0;
	    auto wall_start = std::chrono::steady_clock::now();
      ::getrusage(RUSAGE_SELF, &ru_start);
	    for (int it = 0; it < std::max(1, args.n_iters); ++it) {
	      // Reset comm counters between iterations.
	      {
        pfss_sh.sent0.store(0, std::memory_order_relaxed);
        pfss_sh.sent1.store(0, std::memory_order_relaxed);
        pfss_sh.calls0.store(0, std::memory_order_relaxed);
        pfss_sh.calls1.store(0, std::memory_order_relaxed);
        pfss_sh.bytes1.store(0, std::memory_order_relaxed);
        pfss_sh.bytes8.store(0, std::memory_order_relaxed);
        pfss_sh.bytes_other.store(0, std::memory_order_relaxed);
        pfss_sh.q0to1.reset();
        pfss_sh.q1to0.reset();
      }
      {
        net_sh.sent0.store(0, std::memory_order_relaxed);
        net_sh.sent1.store(0, std::memory_order_relaxed);
        net_sh.q0to1.reset();
        net_sh.q1to0.reset();
      }

      auto start = std::chrono::steady_clock::now();
      std::exception_ptr exc1;
      std::exception_ptr exc0;
      std::thread t([&] {
        try {
          run_party(1, *be1, ch1, nch1, trunc_ctx1, st1, X1);
        } catch (...) { exc1 = std::current_exception(); }
      });
      try {
        run_party(0, *be0, ch0, nch0, trunc_ctx0, st0, X0);
      } catch (...) {
        exc0 = std::current_exception();
      }
	      if (t.joinable()) t.join();
	      if (exc1) std::rethrow_exception(exc1);
	      if (exc0) std::rethrow_exception(exc0);
	      if (it == 0) {
	        // Stop counting offline key material after the first end-to-end run.
	        runtime::bench::set_offline_counting_enabled(false);
	      }
	      auto end = std::chrono::steady_clock::now();
      double elapsed_s = std::chrono::duration<double>(end - start).count();
      it_s.push_back(elapsed_s);

      uint64_t pfss_bytes = pfss_sh.sent0.load(std::memory_order_relaxed) +
                            pfss_sh.sent1.load(std::memory_order_relaxed);
      if (std::getenv("SUF_BENCH_CHAN_HIST")) {
        uint64_t calls = pfss_sh.calls0.load(std::memory_order_relaxed) +
                         pfss_sh.calls1.load(std::memory_order_relaxed);
        uint64_t b1 = pfss_sh.bytes1.load(std::memory_order_relaxed);
        uint64_t b8 = pfss_sh.bytes8.load(std::memory_order_relaxed);
        uint64_t bo = pfss_sh.bytes_other.load(std::memory_order_relaxed);
        std::fprintf(stderr,
                     "[bench_suf] pfss_ch hist: bytes=%llu calls=%llu bytes1=%llu bytes8=%llu bytes_other=%llu\n",
                     (unsigned long long)pfss_bytes,
                     (unsigned long long)calls,
                     (unsigned long long)b1,
                     (unsigned long long)b8,
                     (unsigned long long)bo);
      }
      uint64_t net_bytes = net_sh.sent0 + net_sh.sent1;
      it_pfss_bytes.push_back(pfss_bytes);
      it_net_bytes.push_back(net_bytes);
    }
    auto wall_end = std::chrono::steady_clock::now();
    ::getrusage(RUSAGE_SELF, &ru_end);
    wall_start_s = 0.0;
    wall_end_s = std::chrono::duration<double>(wall_end - wall_start).count();

    const int n_meas = std::max(1, args.n_iters);
    double max_s = 0.0;
    for (double v : it_s) max_s = std::max(max_s, v);
    double mean_all = 0.0;
    for (double v : it_s) mean_all += v;
    mean_all /= static_cast<double>(n_meas);

    // Treat the first iteration as (keygen+online); steady-state online is mean over iters[1:].
    double online_mean = mean_all;
    double keygen_s = 0.0;
    if (it_s.size() >= 2) {
      online_mean = 0.0;
      for (size_t i = 1; i < it_s.size(); ++i) online_mean += it_s[i];
      online_mean /= static_cast<double>(it_s.size() - 1);
      keygen_s = std::max(0.0, it_s[0] - online_mean);
    }

    auto mean_u64 = [&](const std::vector<uint64_t>& v, size_t start) -> uint64_t {
      if (v.empty() || start >= v.size()) return 0;
      __uint128_t sum = 0;
      for (size_t i = start; i < v.size(); ++i) sum += v[i];
      uint64_t denom = static_cast<uint64_t>(v.size() - start);
      return denom ? static_cast<uint64_t>(sum / denom) : 0;
    };
    const size_t online_start = (it_s.size() >= 2) ? 1 : 0;
    uint64_t mean_pfss_bytes = mean_u64(it_pfss_bytes, online_start);
    uint64_t mean_net_bytes = mean_u64(it_net_bytes, online_start);

    auto tv_to_s = [](const timeval& tv) -> double {
      return static_cast<double>(tv.tv_sec) + static_cast<double>(tv.tv_usec) / 1e6;
    };
    const double cpu_user_s = tv_to_s(ru_end.ru_utime) - tv_to_s(ru_start.ru_utime);
    const double cpu_sys_s = tv_to_s(ru_end.ru_stime) - tv_to_s(ru_start.ru_stime);
    const double cpu_util_avg = (wall_end_s > 0.0) ? ((cpu_user_s + cpu_sys_s) / wall_end_s) : 0.0;
    const long max_rss_kb = ru_end.ru_maxrss;

    gates::set_composite_keygen_hook(nullptr);

    write_json(args,
               spec,
               n_layers_run,
               keygen_s,
               online_mean,
               max_s,
               mean_pfss_bytes,
               mean_net_bytes,
               cpu_user_s,
               cpu_sys_s,
               cpu_util_avg,
               max_rss_kb,
               wall_end_s,
               n_meas,
               st0,
               st1);
    std::cout << "Wrote bench log to " << args.log_json << "\n";
  } catch (const std::exception& e) {
    std::cerr << "bench_suf_transformer error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
