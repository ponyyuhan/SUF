#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <random>
#include <thread>
#include <vector>
#include <limits>
#include <algorithm>

#include "compiler/truncation_lowering.hpp"
#include "gates/rsqrt_gate.hpp"
#include "gates/tables/rsqrt_piecewise_affine_init.hpp"
#include "proto/pfss_backend.hpp"
#include "proto/pfss_backend_batch.hpp"
#include "proto/reference_backend.hpp"
#include "runtime/phase_tasks.hpp"
#include "suf/suf_silu_builders.hpp"

namespace {

struct LocalChan : net::Chan {
  struct Shared {
    std::mutex m;
    std::condition_variable cv;
    std::queue<uint64_t> q0to1, q1to0;
  };
  Shared* sh = nullptr;
  bool is0 = false;
  LocalChan(Shared* s, bool p) : sh(s), is0(p) {}
  void send_u64(uint64_t v) override {
    std::unique_lock<std::mutex> lk(sh->m);
    auto& q = is0 ? sh->q0to1 : sh->q1to0;
    q.push(v);
    sh->cv.notify_all();
  }
  uint64_t recv_u64() override {
    std::unique_lock<std::mutex> lk(sh->m);
    auto& q = is0 ? sh->q1to0 : sh->q0to1;
    sh->cv.wait(lk, [&] { return !q.empty(); });
    uint64_t v = q.front();
    q.pop();
    return v;
  }
};

struct RowBroadcastTripleMaterial {
  int rows = 0;
  int cols = 0;
  std::vector<uint64_t> A0, A1;
  std::vector<uint64_t> B0, B1;
  std::vector<uint64_t> C0, C1;
};

RowBroadcastTripleMaterial make_row_broadcast_triples(int rows, int cols, std::mt19937_64& rng) {
  RowBroadcastTripleMaterial mat;
  mat.rows = rows;
  mat.cols = cols;
  size_t count = static_cast<size_t>(rows) * static_cast<size_t>(cols);
  mat.A0.resize(count);
  mat.A1.resize(count);
  mat.B0.resize(static_cast<size_t>(rows));
  mat.B1.resize(static_cast<size_t>(rows));
  mat.C0.resize(count);
  mat.C1.resize(count);

  std::vector<uint64_t> B(rows);
  for (int r = 0; r < rows; ++r) {
    uint64_t b = rng();
    uint64_t b0 = rng();
    uint64_t b1 = b - b0;
    B[static_cast<size_t>(r)] = b;
    mat.B0[static_cast<size_t>(r)] = b0;
    mat.B1[static_cast<size_t>(r)] = b1;
  }
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      size_t idx = static_cast<size_t>(r * cols + c);
      uint64_t a = rng();
      uint64_t a0 = rng();
      uint64_t a1 = a - a0;
      uint64_t c_val = proto::mul_mod(a, B[static_cast<size_t>(r)]);
      uint64_t c0 = rng();
      uint64_t c1 = c_val - c0;
      mat.A0[idx] = a0;
      mat.A1[idx] = a1;
      mat.C0[idx] = c0;
      mat.C1[idx] = c1;
    }
  }
  return mat;
}

class RowBroadcastTripleProviderImpl : public runtime::RowBroadcastTripleProvider {
 public:
  RowBroadcastTripleProviderImpl(const RowBroadcastTripleMaterial& mat, int party)
      : mat_(mat), party_(party) {}

  runtime::RowBroadcastTriple reserve_mul(int rows, int cols) override {
    if (rows != mat_.rows || cols != mat_.cols) {
      throw std::runtime_error("RowBroadcastTripleProviderImpl: shape mismatch");
    }
    size_t count = static_cast<size_t>(rows) * static_cast<size_t>(cols);
    const auto& A = (party_ == 0) ? mat_.A0 : mat_.A1;
    const auto& B = (party_ == 0) ? mat_.B0 : mat_.B1;
    const auto& C = (party_ == 0) ? mat_.C0 : mat_.C1;
    return {std::span<const uint64_t>(A.data(), count),
            std::span<const uint64_t>(B.data(), static_cast<size_t>(rows)),
            std::span<const uint64_t>(C.data(), count)};
  }

 private:
 const RowBroadcastTripleMaterial& mat_;
  int party_ = 0;
};

std::pair<std::vector<proto::BeaverTriple64Share>, std::vector<proto::BeaverTriple64Share>>
make_triples(size_t need, std::mt19937_64& rng) {
  std::vector<proto::BeaverTriple64Share> t0(need), t1(need);
  for (size_t i = 0; i < need; ++i) {
    uint64_t a = rng();
    uint64_t b = rng();
    uint64_t c = proto::mul_mod(a, b);
    uint64_t a0 = rng();
    uint64_t b0 = rng();
    uint64_t c0 = rng();
    t0[i] = {a0, b0, c0};
    t1[i] = {a - a0, b - b0, c - c0};
  }
  return {std::move(t0), std::move(t1)};
}

// Build a SUF that emits affine rsqrt init coefficients adjusted for fixed-point
// evaluation: out0 = a0 - (a1 * offset >> fb), out1 = a1.
inline suf::SUF<uint64_t> build_rsqrt_affine_eval_suf(const gates::PiecewisePolySpec& spec) {
  suf::SUF<uint64_t> F;
  F.n_bits = 64;
  F.r_out = 2;
  F.l_out = 0;
  F.degree = 0;

  std::vector<gates::PiecewiseInterval> intervals = spec.intervals;
  std::sort(intervals.begin(), intervals.end(),
            [](const gates::PiecewiseInterval& a, const gates::PiecewiseInterval& b) {
              return a.start < b.start;
            });
  if (intervals.empty()) return F;
  F.alpha.clear();
  F.alpha.reserve(intervals.size() + 1);
  F.alpha.push_back(intervals.front().start);

  if (F.alpha.front() != 0) {
    suf::SufPiece<uint64_t> zero_piece;
    zero_piece.polys.resize(2);
    zero_piece.polys[0].coeffs = {0};
    zero_piece.polys[1].coeffs = {0};
    F.pieces.push_back(std::move(zero_piece));
    F.alpha.insert(F.alpha.begin(), 0ull);
  }

  auto clamp_to_ring = []( __int128 v) -> uint64_t {
    if (v > static_cast<__int128>(std::numeric_limits<int64_t>::max())) {
      v = static_cast<__int128>(std::numeric_limits<int64_t>::max());
    }
    if (v < static_cast<__int128>(std::numeric_limits<int64_t>::min())) {
      v = static_cast<__int128>(std::numeric_limits<int64_t>::min());
    }
    return static_cast<uint64_t>(static_cast<int64_t>(v));
  };

  for (const auto& iv : intervals) {
    int64_t a0 = (!iv.pack.coeffs.empty()) ? iv.pack.coeffs[0] : 0;
    int64_t a1 = (iv.pack.coeffs.size() > 1) ? iv.pack.coeffs[1] : 0;
    int64_t offset = iv.pack.offset;
    __int128 prod = static_cast<__int128>(a1) * static_cast<__int128>(offset);
    int64_t offset_term = static_cast<int64_t>(prod >> spec.frac_bits_in);
    int64_t adj = a0 - offset_term;

    suf::SufPiece<uint64_t> piece;
    piece.polys.resize(2);
    piece.polys[0].coeffs = {clamp_to_ring(adj)};
    piece.polys[1].coeffs = {clamp_to_ring(a1)};
    F.pieces.push_back(std::move(piece));
    F.alpha.push_back(iv.end);
  }
  return F;
}

struct RsqrtMaterial {
  suf::SUF<uint64_t> suf;
  gates::CompositeKeyPair keys;
  compiler::TruncationLoweringResult trunc_f;
  compiler::TruncationLoweringResult trunc_2f;
  gates::PiecewisePolySpec init_spec;
  int frac_bits = 0;
  int nr_iters = 1;
  runtime::RsqrtTaskBundle bundle() {
    runtime::RsqrtTaskBundle b{};
    b.suf = &suf;
    b.key0 = &keys.k0;
    b.key1 = &keys.k1;
    b.trunc_f = &trunc_f;
    b.trunc_2f = &trunc_2f;
    b.init_spec = &init_spec;
    b.frac_bits = frac_bits;
    b.nr_iters = nr_iters;
    return b;
  }
};

RsqrtMaterial make_rsqrt_material(int frac_bits,
                                  int nr_iters,
                                  double eps,
                                  double vmax,
                                  proto::PfssBackendBatch& backend,
                                  std::mt19937_64& rng,
                                  int rows) {
  RsqrtMaterial mat;
  auto spec = gates::make_rsqrt_affine_init_spec(frac_bits, eps, vmax);
  mat.init_spec = spec;
  mat.suf = build_rsqrt_affine_eval_suf(spec);
  std::vector<uint64_t> r_out(static_cast<size_t>(mat.suf.r_out), 0ull);
  mat.keys = gates::composite_gen_backend_with_masks(mat.suf,
                                                     backend,
                                                     rng,
                                                     rng(),
                                                     r_out,
                                                     static_cast<size_t>(rows));
  mat.keys.k0.compiled.gate_kind = compiler::GateKind::Rsqrt;
  mat.keys.k1.compiled.gate_kind = compiler::GateKind::Rsqrt;

  compiler::GateParams p;
  p.kind = compiler::GateKind::FaithfulARS;
  p.frac_bits = frac_bits;
  p.per_element_masks = true;
  mat.trunc_f = compiler::lower_truncation_gate(backend, rng, p, static_cast<size_t>(rows));
  p.frac_bits = 2 * frac_bits;
  p.per_element_masks = true;
  mat.trunc_2f = compiler::lower_truncation_gate(backend, rng, p, static_cast<size_t>(rows));
  std::fill(mat.trunc_f.keys.k0.r_out_share.begin(), mat.trunc_f.keys.k0.r_out_share.end(), 0ull);
  std::fill(mat.trunc_f.keys.k1.r_out_share.begin(), mat.trunc_f.keys.k1.r_out_share.end(), 0ull);
  std::fill(mat.trunc_2f.keys.k0.r_out_share.begin(), mat.trunc_2f.keys.k0.r_out_share.end(), 0ull);
  std::fill(mat.trunc_2f.keys.k1.r_out_share.begin(), mat.trunc_2f.keys.k1.r_out_share.end(), 0ull);
  mat.frac_bits = frac_bits;
  mat.nr_iters = nr_iters;

  // Zero r_out so reconstruction is direct.
  std::fill(mat.keys.k0.r_out_share.begin(), mat.keys.k0.r_out_share.end(), 0ull);
  std::fill(mat.keys.k1.r_out_share.begin(), mat.keys.k1.r_out_share.end(), 0ull);

  size_t need_triples = 3 * static_cast<size_t>(nr_iters);
  auto fill = [&](std::vector<proto::BeaverTriple64Share>& dst, int party_id) {
    while (dst.size() < need_triples) {
      uint64_t a = rng(), b = rng(), c = proto::mul_mod(a, b);
      uint64_t a0 = rng(), b0 = rng(), c0 = rng();
      dst.push_back((party_id == 0) ? proto::BeaverTriple64Share{a0, b0, c0}
                                    : proto::BeaverTriple64Share{a - a0, b - b0, c - c0});
    }
  };
  fill(mat.keys.k0.triples, 0);
  fill(mat.keys.k1.triples, 1);

  mat.keys.k0.r_in_share_vec.assign(static_cast<size_t>(rows), mat.keys.k0.r_in_share);
  mat.keys.k1.r_in_share_vec.assign(static_cast<size_t>(rows), mat.keys.k1.r_in_share);
  return mat;
}

// Fixed-point reference LayerNorm (gamma/beta omitted).
std::vector<int64_t> ref_layernorm(const std::vector<int64_t>& x,
                                   int rows,
                                   int cols,
                                   int fb,
                                   const gates::PiecewisePolySpec& rsqrt_spec,
                                   double eps,
                                   int nr_iters) {
  std::vector<int64_t> out(x.size(), 0);
  int64_t one = int64_t(1) << fb;
  for (int r = 0; r < rows; ++r) {
    int64_t sum = 0;
    for (int c = 0; c < cols; ++c) sum += x[static_cast<size_t>(r * cols + c)];
    int64_t mu = sum / cols;
    int64_t var_acc = 0;
    for (int c = 0; c < cols; ++c) {
      int64_t d = x[static_cast<size_t>(r * cols + c)] - mu;
      __int128 sq = static_cast<__int128>(d) * static_cast<__int128>(d);
      var_acc += static_cast<int64_t>(sq >> fb);
    }
    int64_t var = var_acc / cols;
    int64_t var_eps = var + static_cast<int64_t>(std::llround(eps * std::ldexp(1.0, fb)));
    int64_t rcp_sqrt = gates::ref_rsqrt_fixed(rsqrt_spec, var_eps, fb, nr_iters);
    for (int c = 0; c < cols; ++c) {
      int64_t d = x[static_cast<size_t>(r * cols + c)] - mu;
      __int128 prod = static_cast<__int128>(d) * static_cast<__int128>(rcp_sqrt);
      int64_t y = static_cast<int64_t>(prod >> fb);
      out[static_cast<size_t>(r * cols + c)] = y;
    }
  }
  return out;
}

}  // namespace

int main() {
  const int fb = 12;
  const int rows = 2;
  const int cols = 4;
  const double eps = 1.0 / 1024.0;
  const int rsqrt_iters = 2;

  // Simple deterministic inputs (party0 holds full value, party1 zeros).
  std::vector<uint64_t> x_share(rows * cols, 0);
  for (int i = 0; i < rows * cols; ++i) {
    x_share[static_cast<size_t>(i)] = static_cast<uint64_t>((i + 1) << fb);
  }

  proto::ReferenceBackend backend;
  std::mt19937_64 rng(42);

  // Trunc bundles.
  auto make_trunc = [&](compiler::GateKind kind, size_t batch, int frac_bits_param) {
    compiler::GateParams p;
    p.kind = kind;
    p.frac_bits = frac_bits_param;
    auto bundle = compiler::lower_truncation_gate(backend, rng, p, batch);
    std::fill(bundle.keys.k0.r_out_share.begin(), bundle.keys.k0.r_out_share.end(), 0ull);
    std::fill(bundle.keys.k1.r_out_share.begin(), bundle.keys.k1.r_out_share.end(), 0ull);
    return bundle;
  };
  auto mean_trunc = make_trunc(compiler::GateKind::FaithfulARS, rows, fb);
  auto var_trunc = make_trunc(compiler::GateKind::FaithfulARS, rows, 2 * fb);
  auto norm_trunc = make_trunc(compiler::GateKind::FaithfulARS, rows * cols, fb);

  // Row-broadcast triples and mul triples.
  RowBroadcastTripleMaterial rb_mat = make_row_broadcast_triples(rows, cols, rng);
  RowBroadcastTripleProviderImpl rb_p0(rb_mat, 0);
  RowBroadcastTripleProviderImpl rb_p1(rb_mat, 1);
  auto [mul_triples0, mul_triples1] = make_triples(static_cast<size_t>(rows * cols), rng);
  uint64_t inv_len_qf = static_cast<uint64_t>(std::llround((1.0 / cols) * std::ldexp(1.0, fb)));

  auto ensure_pair_triples = [&](std::vector<proto::BeaverTriple64Share>& t0,
                                 std::vector<proto::BeaverTriple64Share>& t1,
                                 size_t need) {
    while (t0.size() < need || t1.size() < need) {
      uint64_t a = rng();
      uint64_t b = rng();
      uint64_t c = proto::mul_mod(a, b);
      uint64_t a0 = rng();
      uint64_t b0 = rng();
      uint64_t c0 = rng();
      t0.push_back({a0, b0, c0});
      t1.push_back({a - a0, b - b0, c - c0});
    }
  };
  size_t ample = 1024;
  ensure_pair_triples(mean_trunc.keys.k0.triples, mean_trunc.keys.k1.triples, ample);
  ensure_pair_triples(var_trunc.keys.k0.triples, var_trunc.keys.k1.triples, ample);
  ensure_pair_triples(norm_trunc.keys.k0.triples, norm_trunc.keys.k1.triples, ample);

  // Rsqrt bundle (shared key pair; party is chosen at runtime).
  std::mt19937_64 rng_rsqrt(rng());
  auto rsqrt_mat = make_rsqrt_material(fb, rsqrt_iters, eps, 16.0, backend, rng_rsqrt, rows);
  ensure_pair_triples(rsqrt_mat.keys.k0.triples, rsqrt_mat.keys.k1.triples, ample);
  ensure_pair_triples(rsqrt_mat.trunc_f.keys.k0.triples, rsqrt_mat.trunc_f.keys.k1.triples, ample);
  ensure_pair_triples(rsqrt_mat.trunc_2f.keys.k0.triples, rsqrt_mat.trunc_2f.keys.k1.triples, ample);
  auto rsqrt_bundle = rsqrt_mat.bundle();

  // Shared channel and run two parties.
  LocalChan::Shared sh;
  LocalChan c0(&sh, true), c1(&sh, false);

struct PartyResult {
  std::vector<uint64_t> out;
  std::vector<uint64_t> mu_q2f;
  std::vector<uint64_t> mu_qf;
  std::vector<uint64_t> var_q3f;
  std::vector<uint64_t> var_qf;
  std::vector<uint64_t> rsqrt_qf;
  std::vector<uint64_t> rsqrt_init_qf;
  std::vector<uint64_t> rsqrt_xy2_f;
  std::vector<uint64_t> rsqrt_c0;
  std::vector<uint64_t> rsqrt_c1;
  std::vector<uint64_t> rsqrt_x_plain;
  int rsqrt_init_r = 0;
  std::vector<uint64_t> norm_qf;
};

auto run_party = [&](int party, const runtime::RsqrtTaskBundle& rsqrt_bundle,
                     std::span<const proto::BeaverTriple64Share> mul_triples,
                     runtime::RowBroadcastTripleProvider* rb,
                     std::span<const uint64_t> x_share_local) {
  runtime::LayerNormTaskBundle b{};
  b.mean_trunc = {nullptr, &mean_trunc, fb, true};
  b.var_trunc = {nullptr, &var_trunc, 2 * fb, true};
  b.norm_trunc = {nullptr, &norm_trunc, fb, true};
  b.rsqrt = rsqrt_bundle;
  b.inv_len_qf = inv_len_qf;
  b.eps_qf = (party == 0)
                 ? static_cast<uint64_t>(std::llround(eps * std::ldexp(1.0, fb)))
                 : 0ull;
  b.frac_bits = fb;
  b.mul_triples = mul_triples;
  b.row_triples = rb;

  runtime::PhaseExecutor pe;
  runtime::PhaseResources R{};
  R.party = party;
  R.pfss_backend = &backend;
  runtime::ProtoChanFromNet pch(party == 0 ? c0 : c1);
  R.pfss_chan = &pch;
    R.net_chan = (party == 0) ? static_cast<net::Chan*>(&c0) : static_cast<net::Chan*>(&c1);
    R.pfss_coeff = &pe.pfss_coeff_batch();
    R.pfss_trunc = &pe.pfss_trunc_batch();
    R.opens = &pe.open_collector();

  std::vector<uint64_t> out(x_share_local.size(), 0);
  auto task = std::make_unique<runtime::LayerNormTask>(
      b,
      x_share_local,
      std::span<uint64_t>(out.data(), out.size()),
      rows,
      cols);
  auto* task_ptr = task.get();
  pe.add_task(std::move(task));
  pe.run(R);
  PartyResult pr;
  pr.out = std::move(out);
  pr.mu_q2f.assign(task_ptr->mu_q2f_debug().begin(), task_ptr->mu_q2f_debug().end());
  pr.mu_qf.assign(task_ptr->mu_qf_debug().begin(), task_ptr->mu_qf_debug().end());
  pr.var_q3f.assign(task_ptr->var_q3f_debug().begin(), task_ptr->var_q3f_debug().end());
  pr.var_qf.assign(task_ptr->var_qf_debug().begin(), task_ptr->var_qf_debug().end());
  pr.rsqrt_qf.assign(task_ptr->rsqrt_qf_debug().begin(), task_ptr->rsqrt_qf_debug().end());
  pr.rsqrt_init_qf.assign(task_ptr->rsqrt_init_debug().begin(), task_ptr->rsqrt_init_debug().end());
  pr.rsqrt_xy2_f.assign(task_ptr->rsqrt_xy2_f_debug().begin(), task_ptr->rsqrt_xy2_f_debug().end());
  pr.rsqrt_c0.assign(task_ptr->rsqrt_c0_debug().begin(), task_ptr->rsqrt_c0_debug().end());
  pr.rsqrt_c1.assign(task_ptr->rsqrt_c1_debug().begin(), task_ptr->rsqrt_c1_debug().end());
  pr.rsqrt_init_r = task_ptr->rsqrt_init_r_debug();
  pr.rsqrt_x_plain.assign(task_ptr->rsqrt_x_plain_debug().begin(),
                          task_ptr->rsqrt_x_plain_debug().end());
  pr.norm_qf.assign(task_ptr->norm_qf_debug().begin(), task_ptr->norm_qf_debug().end());
  return pr;
};

  std::vector<uint64_t> x_share_p0 = x_share;
  std::vector<uint64_t> x_share_p1(x_share.size(), 0ull);
  bool fail = false;
  std::string err;
  PartyResult y0, y1;
  std::thread t0([&] {
    try {
      y0 = run_party(0,
                     rsqrt_bundle,
                     std::span<const proto::BeaverTriple64Share>(mul_triples0),
                     &rb_p0,
                     std::span<const uint64_t>(x_share_p0.data(), x_share_p0.size()));
    } catch (const std::exception& e) {
      fail = true;
      err = e.what();
    }
  });
  std::thread t1([&] {
    try {
      y1 = run_party(1,
                     rsqrt_bundle,
                     std::span<const proto::BeaverTriple64Share>(mul_triples1),
                     &rb_p1,
                     std::span<const uint64_t>(x_share_p1.data(), x_share_p1.size()));
    } catch (const std::exception& e) {
      fail = true;
      err = e.what();
    }
  });
  t0.join();
  t1.join();
  if (fail) {
    std::cerr << "LayerNorm task failed: " << err << std::endl;
    return 1;
  }

  std::vector<int64_t> plain(x_share.size(), 0);
  for (size_t i = 0; i < plain.size(); ++i) {
    plain[i] = static_cast<int64_t>(y0.out[i] + y1.out[i]);
  }
  auto ref = ref_layernorm(
      [&] {
        std::vector<int64_t> tmp(x_share.size(), 0);
        for (size_t i = 0; i < x_share.size(); ++i) tmp[i] = static_cast<int64_t>(x_share[i]);
        return tmp;
      }(),
      rows,
      cols,
      fb,
      rsqrt_mat.init_spec,
      eps,
      rsqrt_iters);

  for (size_t i = 0; i < plain.size(); ++i) {
    if (std::llabs(static_cast<long long>(plain[i] - ref[i])) > 2) {
      auto reconstruct = [](const std::vector<uint64_t>& a, const std::vector<uint64_t>& b) {
        std::vector<int64_t> out(a.size(), 0);
        for (size_t i = 0; i < a.size(); ++i) out[i] = static_cast<int64_t>(a[i] + b[i]);
        return out;
      };
      auto mu_q2f_plain = reconstruct(y0.mu_q2f, y1.mu_q2f);
      auto mu_plain = reconstruct(y0.mu_qf, y1.mu_qf);
      auto var_q3f_plain = reconstruct(y0.var_q3f, y1.var_q3f);
      auto var_plain = reconstruct(y0.var_qf, y1.var_qf);
      auto rsqrt_plain = reconstruct(y0.rsqrt_qf, y1.rsqrt_qf);
      auto rsqrt_init_plain = reconstruct(y0.rsqrt_init_qf, y1.rsqrt_init_qf);
      auto xy2_plain = reconstruct(y0.rsqrt_xy2_f, y1.rsqrt_xy2_f);
      auto c0_plain = reconstruct(y0.rsqrt_c0, y1.rsqrt_c0);
      auto c1_plain = reconstruct(y0.rsqrt_c1, y1.rsqrt_c1);
      auto x_plain_dbg = reconstruct(y0.rsqrt_x_plain, y1.rsqrt_x_plain);
      auto norm_plain = reconstruct(y0.norm_qf, y1.norm_qf);
      std::cerr << "Mismatch at " << i << ": got " << plain[i] << " expected " << ref[i] << std::endl;
      std::cerr << "inv_len_qf=" << inv_len_qf << "\n";
      std::cerr << "mu_q2f: ";
      for (auto v : mu_q2f_plain) std::cerr << v << " ";
      std::cerr << "\n";
      std::cerr << "mu: ";
      for (auto v : mu_plain) std::cerr << v << " ";
      std::cerr << "\nvar_qf: ";
      for (auto v : var_plain) std::cerr << v << " ";
      std::cerr << "\nvar_q3f: ";
      for (auto v : var_q3f_plain) std::cerr << v << " ";
      std::cerr << "\nrsqrt_qf: ";
      for (auto v : rsqrt_plain) std::cerr << v << " ";
      std::cerr << "\nrsqrt init y: ";
      for (auto v : rsqrt_init_plain) std::cerr << v << " ";
      std::cerr << "\nxy2_f: ";
      for (auto v : xy2_plain) std::cerr << v << " ";
      std::cerr << "\nc0: ";
      for (auto v : c0_plain) std::cerr << v << " ";
      std::cerr << "\nc1: ";
      for (auto v : c1_plain) std::cerr << v << " ";
      std::cerr << "\nx_plain_dbg: ";
      for (auto v : x_plain_dbg) std::cerr << v << " ";
      std::cerr << "\ninit_r=" << y0.rsqrt_init_r << " " << y1.rsqrt_init_r;
      std::cerr << "\nnorm_qf (first row): ";
      for (int c = 0; c < cols && c < static_cast<int>(norm_plain.size()); ++c) {
        std::cerr << norm_plain[static_cast<size_t>(c)] << " ";
      }
      std::cerr << std::endl;
      return 1;
    }
  }
  std::cout << "LayerNormTask test passed\n";
  return 0;
}
