#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <mutex>
#include <queue>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "compiler/truncation_pass_runner.hpp"
#include "nn/kv_cache.hpp"
#include "nn/model_specs.hpp"
#include "nn/tensor_view.hpp"
#include "nn/transformer_layer.hpp"
#include "proto/backend_factory.hpp"
#include "proto/pfss_backend_batch.hpp"
#include "runtime/phase_executor.hpp"
#include "runtime/pfss_phase_planner.hpp"
#include "runtime/pfss_gpu_staging.hpp"

namespace {

struct Args {
  std::string model;
  std::string backend = "cpu";  // cpu|gpu|auto (respects SUF_PFSS_BACKEND if auto)
  int seq_len = 128;
  int batch_size = 1;
  int n_iters = 5;
};

Args parse(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    std::string s(argv[i]);
    auto next = [&](int& idx) -> std::string {
      if (idx + 1 >= argc) throw std::runtime_error("flag requires value: " + s);
      return std::string(argv[++idx]);
    };
    auto take = [&](const std::string& pfx) -> std::string { return s.substr(pfx.size()); };
    if (s == "--model") a.model = next(i);
    else if (s == "--backend") a.backend = next(i);
    else if (s == "--seq-len") a.seq_len = std::stoi(next(i));
    else if (s == "--batch-size") a.batch_size = std::stoi(next(i));
    else if (s == "--n-iters") a.n_iters = std::stoi(next(i));
    else if (s.rfind("--model=", 0) == 0) a.model = take("--model=");
    else if (s.rfind("--backend=", 0) == 0) a.backend = take("--backend=");
    else if (s.rfind("--seq-len=", 0) == 0) a.seq_len = std::stoi(take("--seq-len="));
    else if (s.rfind("--batch-size=", 0) == 0) a.batch_size = std::stoi(take("--batch-size="));
    else if (s.rfind("--n-iters=", 0) == 0) a.n_iters = std::stoi(take("--n-iters="));
    else if (s == "--help" || s == "-h") {
      std::cerr << "Usage: bench_layer_breakdown --model NAME [--backend cpu|gpu|auto]"
                << " [--seq-len L] [--batch-size B] [--n-iters N]\n";
      std::exit(0);
    }
  }
  if (a.model.empty()) throw std::runtime_error("missing --model");
  if (a.n_iters <= 0) throw std::runtime_error("--n-iters must be > 0");
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
      if (is0) sh->sent0 += n;
      else sh->sent1 += n;
    }
    sh->cv.notify_all();
  }
  void recv_bytes(void* data, size_t n) override {
    std::unique_lock<std::mutex> lk(sh->mu);
    auto& q = is0 ? sh->q1to0 : sh->q0to1;
    sh->cv.wait(lk, [&] { return !q.empty(); });
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
  CountingNetChan(Shared* s, bool party0) : sh(s), is0(party0) {}
  void send_u64(uint64_t v) override {
    {
      std::lock_guard<std::mutex> lk(sh->mu);
      auto& q = is0 ? sh->q0to1 : sh->q1to0;
      q.push(v);
      if (is0) sh->sent0 += sizeof(uint64_t);
      else sh->sent1 += sizeof(uint64_t);
    }
    sh->cv.notify_all();
  }
  uint64_t recv_u64() override {
    std::unique_lock<std::mutex> lk(sh->mu);
    auto& q = is0 ? sh->q1to0 : sh->q0to1;
    sh->cv.wait(lk, [&] { return !q.empty(); });
    uint64_t v = q.front();
    q.pop();
    return v;
  }
};

struct PartyBreakdown {
  runtime::PfssLayerPlanner::Totals pfss{};
  runtime::OpenCollector::Stats opens{};
};

static std::unique_ptr<proto::PfssBackendBatch> make_backend(const std::string& name) {
  proto::PfssBackendOptions opts;
  if (name == "cpu") opts.kind = proto::PfssBackendKind::Cpu;
  else if (name == "gpu") opts.kind = proto::PfssBackendKind::Gpu;
  else opts.kind = proto::PfssBackendKind::Auto;
  return proto::make_pfss_backend(opts);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    Args args = parse(argc, argv);
    const auto& spec = nn::get_model_spec(args.model);

    const int B = args.batch_size;
    const int T = args.seq_len;
    const int D = static_cast<int>(spec.d_model);
    const int fb = static_cast<int>(spec.frac_bits);
    const int H = static_cast<int>(spec.n_heads);
    const int Dh = D / H;
    const int Dff = static_cast<int>(spec.d_ff);

    std::mt19937_64 rng(1234);
    std::vector<int64_t> Wqkv(static_cast<size_t>(D * D * 3));
    std::vector<int64_t> Wout(static_cast<size_t>(D * D));
    std::vector<int64_t> W1(static_cast<size_t>(D * Dff));
    std::vector<int64_t> W2(static_cast<size_t>(Dff * D));
    auto fill_w = [&](std::vector<int64_t>& w) {
      for (auto& v : w) v = static_cast<int64_t>(rng() % 64);
    };
    fill_w(Wqkv);
    fill_w(Wout);
    fill_w(W1);
    fill_w(W2);

    std::vector<uint64_t> X_plain(static_cast<size_t>(B * T * D));
    for (auto& v : X_plain) v = rng() & ((uint64_t(1) << fb) - 1ull);
    std::vector<uint64_t> X0(X_plain.size()), X1(X_plain.size());
    for (size_t i = 0; i < X_plain.size(); ++i) {
      uint64_t s0 = rng();
      X0[i] = s0;
      X1[i] = proto::sub_mod(X_plain[i], s0);
    }

    // Backends and compilation contexts (reused across iterations).
    auto be0 = make_backend(args.backend);
    auto be1 = make_backend(args.backend);
    compiler::TruncationPassContext trunc_ctx0(*be0, 0x626c306275666675ull /*"bl0buffu"*/);
    compiler::TruncationPassContext trunc_ctx1(*be1, 0x626c316275666675ull /*"bl1buffu"*/);

    double sum_ms = 0.0;
    uint64_t sum_pfss_bytes = 0;
    uint64_t sum_net_bytes = 0;
    uint64_t sum_coeff_flush_ns = 0;
    uint64_t sum_trunc_flush_ns = 0;
    uint64_t sum_open_flush_ns = 0;
    uint64_t sum_opened_words = 0;
    uint64_t sum_open_flushes = 0;
    uint64_t sum_pfss_jobs = 0;
    uint64_t sum_pfss_flushes = 0;
    uint64_t sum_pfss_coeff_hatx_bytes = 0;
    uint64_t sum_pfss_trunc_hatx_bytes = 0;
    uint64_t sum_pfss_coeff_active_elems = 0;
    uint64_t sum_pfss_trunc_active_elems = 0;
    uint64_t sum_pfss_coeff_cost_effbits = 0;
    uint64_t sum_pfss_trunc_cost_effbits = 0;

    for (int it = 0; it < args.n_iters; ++it) {
      CountingChan::Shared pfss_sh;
      CountingChan ch0(&pfss_sh, true), ch1(&pfss_sh, false);
      CountingNetChan::Shared net_sh;
      CountingNetChan nch0(&net_sh, true), nch1(&net_sh, false);

      PartyBreakdown b0, b1;
      auto run_party = [&](int party,
                           proto::PfssBackendBatch& be,
                           CountingChan& pfss_ch,
                           CountingNetChan& net_ch,
	                           compiler::TruncationPassContext& trunc_ctx,
	                           std::vector<uint64_t>& Xshare,
	                           PartyBreakdown& bout) {
#ifdef SUF_HAVE_CUDA
	        // Must outlive `PhaseExecutor`: PFSS tasks may retain staged device buffers
	        // whose deleters call back into the stager during executor teardown.
	        std::unique_ptr<runtime::CudaPfssStager> cuda_stager;
#endif
	        runtime::PhaseExecutor pe;

        nn::LayerContext ctx;
        ctx.trunc_ctx = &trunc_ctx;
        ctx.pfss_backend_override = &be;
        // Use the dedicated PFSS byte channel for Beaver/PFSS traffic so we can
        // measure PFSS-side bytes separately from net opens.
        ctx.pfss_chan = &pfss_ch;
        ctx.frac_bits = fb;
        runtime::PfssLayerPlanner layer_planner;
        ctx.pfss_layer_planner = &layer_planner;
#ifdef SUF_HAVE_CUDA
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
        cfg.attn.D = D;
        cfg.attn.H = H;
        cfg.attn.Dh = Dh;
        cfg.attn.S_max = T;
        cfg.attn.frac_bits = fb;
        cfg.mlp.D = D;
        cfg.mlp.frac_bits = fb;
        cfg.mlp.Hidden = Dff;

        std::vector<uint64_t> Y(static_cast<size_t>(B * T * D), 0ull);
        nn::KVCache cache(/*B=*/B, /*H=*/H, /*S_max=*/T, /*Dh=*/Dh);
        nn::transformer_layer_forward(cfg,
                                      party,
                                      net_ch,
                                      nn::view3(Xshare.data(), B, T, D),
                                      nn::view2(Wqkv.data(), D, D * 3),
                                      nn::view2(Wout.data(), D, D),
                                      nn::view2(W1.data(), D, Dff),
                                      nn::view2(W2.data(), Dff, D),
                                      cache,
                                      nn::view3(Y.data(), B, T, D),
                                      &ctx,
                                      &pe);

        bout.pfss = layer_planner.totals();
        bout.opens = pe.open_collector().stats();
      };

      auto t0 = std::chrono::steady_clock::now();
      std::exception_ptr e0, e1;
      std::thread th([&] {
        try {
          run_party(1, *be1, ch1, nch1, trunc_ctx1, X1, b1);
        } catch (...) {
          e1 = std::current_exception();
        }
      });
      try {
        run_party(0, *be0, ch0, nch0, trunc_ctx0, X0, b0);
      } catch (...) {
        e0 = std::current_exception();
      }
      if (th.joinable()) th.join();
      if (e1) std::rethrow_exception(e1);
      if (e0) std::rethrow_exception(e0);
      auto t1 = std::chrono::steady_clock::now();

      double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
      uint64_t pfss_bytes = pfss_sh.sent0 + pfss_sh.sent1;
      uint64_t net_bytes = net_sh.sent0 + net_sh.sent1;

      // Sum both parties (counts are generally symmetric; sum reflects total work).
      uint64_t coeff_flush_ns = b0.pfss.coeff_flush_ns + b1.pfss.coeff_flush_ns;
      uint64_t trunc_flush_ns = b0.pfss.trunc_flush_ns + b1.pfss.trunc_flush_ns;
      uint64_t open_flush_ns = b0.opens.flush_ns + b1.opens.flush_ns;
      uint64_t opened_words = b0.opens.opened_words + b1.opens.opened_words;
      uint64_t open_flushes = b0.opens.flushes + b1.opens.flushes;
      uint64_t pfss_jobs = b0.pfss.coeff_jobs + b1.pfss.coeff_jobs +
                           b0.pfss.trunc_jobs + b1.pfss.trunc_jobs;
      uint64_t pfss_flushes = b0.pfss.coeff_flushes + b1.pfss.coeff_flushes +
                              b0.pfss.trunc_flushes + b1.pfss.trunc_flushes;
      uint64_t coeff_hatx_bytes = b0.pfss.coeff_hatx_bytes + b1.pfss.coeff_hatx_bytes;
      uint64_t trunc_hatx_bytes = b0.pfss.trunc_hatx_bytes + b1.pfss.trunc_hatx_bytes;
      uint64_t coeff_active_elems = b0.pfss.coeff_active_elems + b1.pfss.coeff_active_elems;
      uint64_t trunc_active_elems = b0.pfss.trunc_active_elems + b1.pfss.trunc_active_elems;
      uint64_t coeff_cost_effbits = b0.pfss.coeff_cost_effbits + b1.pfss.coeff_cost_effbits;
      uint64_t trunc_cost_effbits = b0.pfss.trunc_cost_effbits + b1.pfss.trunc_cost_effbits;

      sum_ms += ms;
      sum_pfss_bytes += pfss_bytes;
      sum_net_bytes += net_bytes;
      sum_coeff_flush_ns += coeff_flush_ns;
      sum_trunc_flush_ns += trunc_flush_ns;
      sum_open_flush_ns += open_flush_ns;
      sum_opened_words += opened_words;
      sum_open_flushes += open_flushes;
      sum_pfss_jobs += pfss_jobs;
      sum_pfss_flushes += pfss_flushes;
      sum_pfss_coeff_hatx_bytes += coeff_hatx_bytes;
      sum_pfss_trunc_hatx_bytes += trunc_hatx_bytes;
      sum_pfss_coeff_active_elems += coeff_active_elems;
      sum_pfss_trunc_active_elems += trunc_active_elems;
      sum_pfss_coeff_cost_effbits += coeff_cost_effbits;
      sum_pfss_trunc_cost_effbits += trunc_cost_effbits;
    }

    double mean_ms = sum_ms / static_cast<double>(args.n_iters);
    std::cout << "bench_layer_breakdown model=" << args.model
              << " backend=" << args.backend
              << " B=" << B << " T=" << T << " D=" << D << " H=" << H
              << " fb=" << fb
              << " iters=" << args.n_iters << "\n";
    std::cout << "  time_ms_mean=" << mean_ms << "\n";
    std::cout << "  comm_pfss_bytes_mean=" << (sum_pfss_bytes / static_cast<uint64_t>(args.n_iters)) << "\n";
    std::cout << "  comm_open_bytes_mean=" << (sum_net_bytes / static_cast<uint64_t>(args.n_iters)) << "\n";
    std::cout << "  pfss_jobs_sum_mean=" << (sum_pfss_jobs / static_cast<uint64_t>(args.n_iters)) << "\n";
    std::cout << "  pfss_flushes_sum_mean=" << (sum_pfss_flushes / static_cast<uint64_t>(args.n_iters)) << "\n";
    std::cout << "  pfss_hatx_bytes_sum_mean="
              << ((sum_pfss_coeff_hatx_bytes + sum_pfss_trunc_hatx_bytes) /
                  static_cast<uint64_t>(args.n_iters))
              << "\n";
    std::cout << "  pfss_active_elems_sum_mean="
              << ((sum_pfss_coeff_active_elems + sum_pfss_trunc_active_elems) /
                  static_cast<uint64_t>(args.n_iters))
              << "\n";
    std::cout << "  pfss_cost_effbits_sum_mean="
              << ((sum_pfss_coeff_cost_effbits + sum_pfss_trunc_cost_effbits) /
                  static_cast<uint64_t>(args.n_iters))
              << "\n";
    std::cout << "  pfss_coeff_flush_ms_sum_mean=" << (static_cast<double>(sum_coeff_flush_ns) / 1e6 / args.n_iters) << "\n";
    std::cout << "  pfss_trunc_flush_ms_sum_mean=" << (static_cast<double>(sum_trunc_flush_ns) / 1e6 / args.n_iters) << "\n";
    std::cout << "  open_flushes_sum_mean=" << (sum_open_flushes / static_cast<uint64_t>(args.n_iters)) << "\n";
    std::cout << "  opened_words_sum_mean=" << (sum_opened_words / static_cast<uint64_t>(args.n_iters)) << "\n";
    std::cout << "  open_flush_ms_sum_mean=" << (static_cast<double>(sum_open_flush_ns) / 1e6 / args.n_iters) << "\n";
  } catch (const std::exception& e) {
    std::cerr << "bench_layer_breakdown error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
