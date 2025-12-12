#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <csignal>
#include <execinfo.h>
#include <mutex>
#include <queue>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
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

namespace {

struct Args {
  std::string model;
  std::string backend = "cpu";
  int seq_len = 128;
  int batch_size = 1;
  int n_iters = 1;
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
    else if (s == "--log-json") a.log_json = next_val(i);
    else if (s.rfind("--model=", 0) == 0) a.model = take_val("--model=");
    else if (s.rfind("--backend=", 0) == 0) a.backend = take_val("--backend=");
    else if (s.rfind("--seq-len=", 0) == 0) a.seq_len = std::stoi(take_val("--seq-len="));
    else if (s.rfind("--batch-size=", 0) == 0) a.batch_size = std::stoi(take_val("--batch-size="));
    else if (s.rfind("--n-iters=", 0) == 0) a.n_iters = std::stoi(take_val("--n-iters="));
    else if (s.rfind("--log-json=", 0) == 0) a.log_json = take_val("--log-json=");
    else if (s == "--help" || s == "-h") {
      std::cerr << "Usage: bench_suf_transformer --model NAME [--backend cpu|gpu] "
                << "[--seq-len L] [--batch-size B] [--n-iters N] [--log-json PATH]\n";
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
  struct Shared {
    std::mutex mu;
    std::condition_variable cv;
    std::queue<uint64_t> q0to1;
    std::queue<uint64_t> q1to0;
    uint64_t sent0 = 0;
    uint64_t sent1 = 0;
  };
  Shared* sh = nullptr;
  bool is0 = false;
  CountingNetChan(Shared* s, bool p0) : sh(s), is0(p0) {}
  void send_u64(uint64_t v) override {
    {
      std::lock_guard<std::mutex> lk(sh->mu);
      auto& q = is0 ? sh->q0to1 : sh->q1to0;
      q.push(v);
      if (is0) sh->sent0 += sizeof(uint64_t); else sh->sent1 += sizeof(uint64_t);
    }
    sh->cv.notify_all();
  }
  uint64_t recv_u64() override {
    std::unique_lock<std::mutex> lk(sh->mu);
    auto& q = is0 ? sh->q1to0 : sh->q0to1;
    sh->cv.wait(lk, [&]{ return !q.empty(); });
    uint64_t v = q.front();
    q.pop();
    return v;
  }
};

void write_json(const Args& a,
                const nn::ModelSpec& spec,
                double elapsed_s,
                uint64_t pfss_bytes,
                uint64_t net_bytes,
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

  std::ofstream f(a.log_json);
  if (!f) throw std::runtime_error("failed to open log json: " + a.log_json);
  f << "{\n";
  f << "  \"system\": \"suf\",\n";
  f << "  \"backend\": \"" << a.backend << "\",\n";
  f << "  \"model\": \"" << a.model << "\",\n";
  f << "  \"seq_len\": " << a.seq_len << ",\n";
  f << "  \"batch_size\": " << a.batch_size << ",\n";
  f << "  \"n_layers\": " << spec.n_layers << ",\n";
  f << "  \"n_heads\": " << spec.n_heads << ",\n";
  f << "  \"d_model\": " << spec.d_model << ",\n";
  f << "  \"n_bits\": " << spec.n_bits << ",\n";
  f << "  \"frac_bits\": " << spec.frac_bits << ",\n";
  f << "  \"timing\": { \"online_time_s_mean\": " << elapsed_s << " },\n";
  f << "  \"communication\": { \"pfss_bytes\": " << pfss_bytes
    << ", \"net_bytes\": " << net_bytes
    << ", \"online_bytes\": " << (pfss_bytes + net_bytes) << " },\n";
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
  f << "  \"notes\": \"single-layer transformer forward (PFSS-backed)\"\n";
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
    auto args = parse(argc, argv);
    const auto& spec = nn::get_model_spec(args.model);

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

    auto run_party = [&](int party,
                         proto::PfssBackendBatch& be,
                         CountingChan& pfss_ch,
                         CountingNetChan& net_ch,
                         compiler::TruncationPassContext& trunc_ctx,
                         runtime::PhaseExecutor::Stats& stats_out,
                         std::vector<uint64_t>& Xshare) {
      if (std::getenv("SUF_BENCH_TRACE")) {
        std::fprintf(stderr, "[bench_suf] party %d start backend=%s rows=%d cols=%d\n",
                     party, args.backend.c_str(), rows, cols);
      }
      runtime::PhaseExecutor pe;
      if (args.backend == "gpu") {
        pe.set_device_pipeline(true);
        pe.set_device_pipeline_materialize(false);
        pe.pfss_coeff_batch().set_device_outputs(true);
        pe.pfss_trunc_batch().set_device_outputs(true);
      }

      nn::LayerContext ctx;
      ctx.trunc_ctx = &trunc_ctx;
      ctx.pfss_backend_override = &be;
      ctx.frac_bits = fb;
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
      cfg.mlp.D = cols;
      cfg.mlp.frac_bits = fb;
      cfg.mlp.Hidden = Dff;

      std::vector<uint64_t> Y(static_cast<size_t>(rows * cols), 0ull);
      nn::KVCache cache(/*B=*/B, /*H=*/H, /*S_max=*/rows, /*Dh=*/Dh);

      if (std::getenv("SUF_BENCH_TRACE")) {
        std::fprintf(stderr, "[bench_suf] party %d entering transformer_layer_forward\n", party);
      }
      nn::transformer_layer_forward(
          cfg,
          party,
          net_ch,
          nn::view3(Xshare.data(), B, rows, cols),
          nn::view2(Wqkv.data(), cols, cols * 3),
          nn::view2(Wout.data(), cols, cols),
          nn::view2(W1.data(), cols, Dff),
          nn::view2(W2.data(), Dff, cols),
          cache,
          nn::view3(Y.data(), B, rows, cols),
          &ctx,
          &pe);
      if (std::getenv("SUF_BENCH_TRACE")) {
        std::fprintf(stderr, "[bench_suf] party %d finished transformer_layer_forward\n", party);
      }

      stats_out = pe.stats();
    };

    runtime::PhaseExecutor::Stats st0, st1;
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

    uint64_t pfss_bytes = pfss_sh.sent0 + pfss_sh.sent1;
    uint64_t net_bytes = net_sh.sent0 + net_sh.sent1;
    write_json(args, spec, elapsed_s, pfss_bytes, net_bytes, st0, st1);
    std::cout << "Wrote bench log to " << args.log_json << "\n";
  } catch (const std::exception& e) {
    std::cerr << "bench_suf_transformer error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
