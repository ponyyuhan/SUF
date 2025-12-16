#include <chrono>
#include <condition_variable>
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
#include "proto/channel.hpp"
#include "proto/sigma_fast_backend_ext.hpp"
#include "runtime/phase_executor.hpp"
#include "runtime/phase_tasks.hpp"
#include "runtime/pfss_phase_planner.hpp"
#include "runtime/pfss_superbatch.hpp"
#include "mpc/net.hpp"
#include "gates/composite_fss.hpp"

namespace {

struct Args {
  std::string model;
  std::string backend = "cpu";
  int seq_len = 128;
  int batch_size = 1;
  int n_iters = 1;
  int n_layers = 0;  // 0 => use model spec
  std::string log_json;
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
    else if (s.rfind("--model=", 0) == 0) a.model = take_val("--model=");
    else if (s.rfind("--backend=", 0) == 0) a.backend = take_val("--backend=");
    else if (s.rfind("--seq-len=", 0) == 0) a.seq_len = std::stoi(take_val("--seq-len="));
    else if (s.rfind("--batch-size=", 0) == 0) a.batch_size = std::stoi(take_val("--batch-size="));
    else if (s.rfind("--n-iters=", 0) == 0) a.n_iters = std::stoi(take_val("--n-iters="));
    else if (s.rfind("--n-layers=", 0) == 0) a.n_layers = std::stoi(take_val("--n-layers="));
    else if (s.rfind("--log-json=", 0) == 0) a.log_json = take_val("--log-json=");
    else if (s == "--help" || s == "-h") {
      std::cerr << "Usage: bench_suf_transformer --model NAME [--backend cpu|gpu] "
                << "[--seq-len L] [--batch-size B] [--n-iters N] [--n-layers LAYERS] [--log-json PATH]\n";
      std::exit(0);
    }
  }
  if (a.model.empty()) throw std::runtime_error("missing --model");
  if (a.log_json.empty()) throw std::runtime_error("missing --log-json");
  return a;
}

struct CountingChan : proto::IChannel {
  struct Shared {
    std::mutex mu;
    std::condition_variable cv;
    std::queue<std::vector<uint8_t>> q0to1;
    std::queue<std::vector<uint8_t>> q1to0;
    uint64_t sent0 = 0;
    uint64_t sent1 = 0;
  };
  Shared* sh = nullptr;
  bool is0 = false;
  CountingChan(Shared* s, bool party0) : sh(s), is0(party0) {}
  void send_bytes(const void* data, size_t n) override {
    std::vector<uint8_t> buf(n);
    std::memcpy(buf.data(), data, n);
    {
      std::lock_guard<std::mutex> lk(sh->mu);
      auto& q = is0 ? sh->q0to1 : sh->q1to0;
      q.push(std::move(buf));
      if (is0) sh->sent0 += n; else sh->sent1 += n;
    }
    sh->cv.notify_all();
  }
  void recv_bytes(void* data, size_t n) override {
    std::unique_lock<std::mutex> lk(sh->mu);
    auto& q = is0 ? sh->q1to0 : sh->q0to1;
    sh->cv.wait(lk, [&]{ return !q.empty(); });
    auto buf = std::move(q.front());
    q.pop();
    if (buf.size() != n) throw std::runtime_error("CountingChan: size mismatch");
    std::memcpy(data, buf.data(), n);
  }
};

struct CountingNetChan : net::Chan {
  struct Ring {
    explicit Ring(size_t cap_pow2 = (1ull << 20)) : buf(cap_pow2, 0), mask(cap_pow2 - 1) {
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
thread_local int tl_iter = -1;
static std::atomic<uint64_t> g_key_bytes{0};

static void composite_keygen_hook(const gates::CompositeKeyPair& kp) {
  // Our unit-test/bench harness generates full key pairs inside each party thread.
  // Count keys once (party 0) and only on the first iteration.
  if (tl_party != 0 || tl_iter != 0) return;
  auto tapes = gates::composite_write_tapes(kp);
  g_key_bytes.fetch_add(tapes.t0.bytes.size() + tapes.t1.bytes.size(), std::memory_order_relaxed);
}

void write_json(const Args& a,
                const nn::ModelSpec& spec,
                int n_layers_run,
                double keygen_s,
                double elapsed_s_mean,
                double elapsed_s_max,
                uint64_t pfss_bytes_mean,
                uint64_t net_bytes_mean,
                uint64_t key_bytes,
                double wall_time_s,
                int n_iters,
                const runtime::PhaseExecutor::Stats& stats0,
                const runtime::PhaseExecutor::Stats& stats1) {
  const uint64_t coeff_jobs = stats0.pfss_coeff_jobs + stats1.pfss_coeff_jobs;
  const uint64_t trunc_jobs = stats0.pfss_trunc_jobs + stats1.pfss_trunc_jobs;
  const uint64_t coeff_hatx = stats0.pfss_coeff_hatx_words + stats1.pfss_coeff_hatx_words;
  const uint64_t trunc_hatx = stats0.pfss_trunc_hatx_words + stats1.pfss_trunc_hatx_words;
  const uint64_t coeff_flushes = stats0.pfss_coeff_flushes + stats1.pfss_coeff_flushes;
  const uint64_t trunc_flushes = stats0.pfss_trunc_flushes + stats1.pfss_trunc_flushes;
  const uint64_t open_flushes = stats0.open_flushes + stats1.open_flushes;
  const uint64_t opened_words = stats0.opened_words + stats1.opened_words;

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
  f << "  \"preprocessing\": { \"key_bytes\": " << key_bytes << " },\n";
  f << "  \"communication\": { \"pfss_bytes\": " << pfss_bytes_mean
    << ", \"net_bytes\": " << net_bytes_mean
    << ", \"online_bytes\": " << (pfss_bytes_mean + net_bytes_mean) << " },\n";
  f << "  \"pfss\": {\n";
  f << "    \"num_jobs\": " << (coeff_jobs + trunc_jobs) << ",\n";
  f << "    \"num_flushes\": " << (coeff_flushes + trunc_flushes) << ",\n";
  f << "    \"total_hatx_words\": " << (coeff_hatx + trunc_hatx) << ",\n";
  f << "    \"coeff_jobs\": " << coeff_jobs << ",\n";
  f << "    \"trunc_jobs\": " << trunc_jobs << ",\n";
  f << "    \"coeff_hatx_words\": " << coeff_hatx << ",\n";
  f << "    \"trunc_hatx_words\": " << trunc_hatx << ",\n";
  f << "    \"open_flushes\": " << open_flushes << ",\n";
  f << "    \"opened_words\": " << opened_words << "\n";
  f << "  },\n";
  f << "  \"notes\": \"transformer forward (PFSS-backed)\"\n";
  f << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
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
    // For performance benchmarking, drive the full PFSS pipeline even when the
    // underlying backend is a clear/reference implementation.
    ::setenv("SUF_FORCE_PFSS", "1", /*overwrite=*/0);
    // Cache expensive dealer-generated task materials (activation/softmax) so
    // we can separate one-time keygen from steady-state online time.
    ::setenv("SUF_BENCH_CACHE_MATERIAL", "1", /*overwrite=*/0);
    // Per-element trunc/ARS masks prevent batching and massively increase both
    // key size and runtime overhead for large vectors. Disable by default for
    // end-to-end transformer benchmarking (can be re-enabled by exporting it).
    ::setenv("SUF_PER_ELEMENT_MASKS", "0", /*overwrite=*/0);
    auto args = parse(argc, argv);
    const auto& spec = nn::get_model_spec(args.model);
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
      uint64_t s0 = rng();
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
    g_key_bytes.store(0, std::memory_order_relaxed);

    auto run_party = [&](int party,
                         proto::PfssBackendBatch& be,
                         CountingChan& pfss_ch,
                         CountingNetChan& net_ch,
                         compiler::TruncationPassContext& trunc_ctx,
                         runtime::PhaseExecutor::Stats& stats_out,
                         std::vector<uint64_t>& Xshare) {
      tl_party = party;
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
      ctx.frac_bits = fb;
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
    double wall_start_s = 0.0;
    double wall_end_s = 0.0;
    uint64_t sum_pfss_bytes = 0;
    uint64_t sum_net_bytes = 0;
    auto wall_start = std::chrono::steady_clock::now();
    for (int it = 0; it < std::max(1, args.n_iters); ++it) {
      tl_iter = it;
      // Reset comm counters between iterations.
      {
        std::lock_guard<std::mutex> lk(pfss_sh.mu);
        pfss_sh.sent0 = pfss_sh.sent1 = 0;
        std::queue<std::vector<uint8_t>>().swap(pfss_sh.q0to1);
        std::queue<std::vector<uint8_t>>().swap(pfss_sh.q1to0);
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
      auto end = std::chrono::steady_clock::now();
      double elapsed_s = std::chrono::duration<double>(end - start).count();
      it_s.push_back(elapsed_s);

      uint64_t pfss_bytes = pfss_sh.sent0 + pfss_sh.sent1;
      uint64_t net_bytes = net_sh.sent0 + net_sh.sent1;
      sum_pfss_bytes += pfss_bytes;
      sum_net_bytes += net_bytes;
    }
    auto wall_end = std::chrono::steady_clock::now();
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

    uint64_t mean_pfss_bytes = sum_pfss_bytes / static_cast<uint64_t>(n_meas);
    uint64_t mean_net_bytes = sum_net_bytes / static_cast<uint64_t>(n_meas);

    uint64_t key_bytes = g_key_bytes.load(std::memory_order_relaxed);
    gates::set_composite_keygen_hook(nullptr);

    write_json(args,
               spec,
               n_layers_run,
               keygen_s,
               online_mean,
               max_s,
               mean_pfss_bytes,
               mean_net_bytes,
               key_bytes,
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
