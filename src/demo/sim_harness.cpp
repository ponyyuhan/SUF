#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <cstdlib>
#include <mutex>
#include <queue>
#include <random>
#include <thread>
#include <vector>
#include <span>
#include <string>
#include <array>

#include "proto/backend_clear.hpp"
#include "proto/beaver.hpp"
#include "proto/beaver_mul64.hpp"
#include "proto/bit_ring_ops.hpp"
#include "proto/channel.hpp"
#include "proto/gelu_online_step_dcf.hpp"
#include "proto/gelu_spline_dealer.hpp"
#include "proto/pack_utils.hpp"
#include "proto/tape.hpp"
#include "proto/pfss_utils.hpp"
#include "proto/reluars_dealer.hpp"
#include "proto/reluars_online_complete.hpp"

using namespace proto;

// In-memory duplex channel for two-party simulation
struct LocalChan : IChannel {
  struct Shared {
    std::mutex m;
    std::condition_variable cv;
    std::queue<std::vector<uint8_t>> q0to1, q1to0;
  };
  Shared* shared = nullptr;
  bool is0 = false;
  LocalChan() = default;
  LocalChan(Shared* s, bool is0_) : shared(s), is0(is0_) {}

  void send_bytes(const void* data, size_t n) override {
    std::vector<uint8_t> buf(n);
    std::memcpy(buf.data(), data, n);
    {
      std::lock_guard<std::mutex> lk(shared->m);
      auto& q = is0 ? shared->q0to1 : shared->q1to0;
      q.push(std::move(buf));
    }
    shared->cv.notify_all();
  }
  void recv_bytes(void* data, size_t n) override {
    std::unique_lock<std::mutex> lk(shared->m);
    auto& q = is0 ? shared->q1to0 : shared->q0to1;
    shared->cv.wait(lk, [&]{ return !q.empty(); });
    auto buf = std::move(q.front());
    q.pop();
    if (buf.size() != n) throw std::runtime_error("recv_bytes size mismatch");
    std::memcpy(data, buf.data(), n);
  }
};

// Convenience: split into additive shares
static std::pair<u64,u64> split_add(u64 v, std::mt19937_64& rng) {
  u64 s0 = rng();
  u64 s1 = sub_mod(v, s0);
  return {s0, s1};
}

// Batch beaver self-test
static bool beaver_batch_selftest() {
  std::mt19937_64 rng(2024);
  const size_t N = 1024;
  std::vector<BeaverTriple64Share> t0, t1;
  t0.reserve(N); t1.reserve(N);
  for (size_t i = 0; i < N; i++) {
    u64 a = rng(), b = rng(), c = mul_mod(a, b);
    auto [a0, a1] = split_add(a, rng);
    auto [b0, b1] = split_add(b, rng);
    auto [c0, c1] = split_add(c, rng);
    t0.push_back({a0, b0, c0});
    t1.push_back({a1, b1, c1});
  }
  std::vector<u64> x(N), y(N);
  for (size_t i = 0; i < N; i++) {
    x[i] = rng();
    y[i] = rng();
  }
  // channels
  LocalChan::Shared sh;
  LocalChan c0(&sh, true), c1(&sh, false);
  std::vector<u64> x0(N), x1(N), y0(N), y1(N);
  for (size_t i = 0; i < N; i++) {
    auto [xs0, xs1] = split_add(x[i], rng);
    auto [ys0, ys1] = split_add(y[i], rng);
    x0[i] = xs0; x1[i] = xs1; y0[i] = ys0; y1[i] = ys1;
  }
  std::vector<u64> z0(N), z1(N);
  BeaverMul64Batch bm0(0, c0, std::span<const BeaverTriple64Share>(t0.data(), t0.size()));
  BeaverMul64Batch bm1(1, c1, std::span<const BeaverTriple64Share>(t1.data(), t1.size()));
  std::thread tA([&]{ bm0.mul(std::span<const u64>(x0), std::span<const u64>(y0), std::span<u64>(z0)); });
  std::thread tB([&]{ bm1.mul(std::span<const u64>(x1), std::span<const u64>(y1), std::span<u64>(z1)); });
  tA.join(); tB.join();
  for (size_t i = 0; i < N; i++) {
    u64 z = add_mod(z0[i], z1[i]);
    if (z != mul_mod(x[i], y[i])) return false;
  }
  return true;
}

static bool beaver_zero_share_selftest() {
  std::mt19937_64 rng(404);
  BeaverTriple64Share t0, t1;
  u64 a = rng(), b = rng(), c = mul_mod(a, b);
  auto [a0, a1] = split_add(a, rng);
  auto [b0, b1] = split_add(b, rng);
  auto [c0, c1] = split_add(c, rng);
  t0 = {a0, b0, c0};
  t1 = {a1, b1, c1};
  LocalChan::Shared sh;
  LocalChan c0ch(&sh, true), c1ch(&sh, false);
  BeaverMul64 m0{0, c0ch, std::vector<BeaverTriple64Share>{t0}}, m1{1, c1ch, std::vector<BeaverTriple64Share>{t1}};
  u64 x0 = 12345;  // arbitrary
  u64 x1 = sub_mod(0ull, x0);
  u64 y0 = 1, y1 = 0;
  u64 z0 = 0, z1 = 0;
  std::thread tA([&]{ z0 = m0.mul(x0, y0); });
  std::thread tB([&]{ z1 = m1.mul(x1, y1); });
  tA.join(); tB.join();
  return add_mod(z0, z1) == 0;
}

static int get_iters(const char* env_name, int def_val) {
  if (const char* v = std::getenv(env_name)) {
    try {
      int parsed = std::stoi(v);
      if (parsed > 0) return parsed;
    } catch (...) {
    }
  }
  return def_val;
}

static bool tape_roundtrip_selftest() {
  TapeWriter tw;
  std::vector<u8> bytes = {1,2,3,4,5};
  std::vector<u64> vec = {10, 20, 30};
  BeaverTriple64Share triple{7, 11, 13};
  std::vector<BeaverTriple64Share> triples = {triple, {100,200,300}};
  tw.write_u64(42);
  tw.write_bytes(bytes);
  tw.write_u64_vec(vec);
  tw.write_triple64(triple);
  tw.write_triple64_vec<BeaverTriple64Share>(std::span<const BeaverTriple64Share>(triples));

  TapeReader tr(tw.data());
  if (tr.read_u64() != 42) return false;
  if (tr.read_bytes() != bytes) return false;
  if (tr.read_u64_vec() != vec) return false;
  auto tr_single = tr.read_triple64<BeaverTriple64Share>();
  if (tr_single.a != triple.a || tr_single.b != triple.b || tr_single.c != triple.c) return false;
  auto tr_vec = tr.read_triple64_vec<BeaverTriple64Share>();
  if (tr_vec.size() != triples.size()) return false;
  for (size_t i = 0; i < tr_vec.size(); i++) {
    if (tr_vec[i].a != triples[i].a || tr_vec[i].b != triples[i].b || tr_vec[i].c != triples[i].c) return false;
  }
  return tr.eof();
}

static bool bitops_lut_selftest() {
  std::mt19937_64 rng(99);
  const size_t Trials = 64;
  const size_t mul_per_trial = 11; // AND, OR, XOR, SEL, LUT8(7 muls) = 11 total
  const size_t triple_need = Trials * mul_per_trial;
  std::vector<BeaverTriple64Share> t0, t1;
  t0.reserve(triple_need); t1.reserve(triple_need);
  for (size_t i = 0; i < triple_need; i++) {
    u64 a = rng() & 1ull;
    u64 b = rng() & 1ull;
    u64 c = mul_mod(a, b);
    auto [a0, a1] = split_add(a, rng);
    auto [b0, b1] = split_add(b, rng);
    auto [c0, c1] = split_add(c, rng);
    t0.push_back({a0, b0, c0});
    t1.push_back({a1, b1, c1});
  }

  LocalChan::Shared sh;
  LocalChan c0(&sh, true), c1(&sh, false);

  std::vector<u64> a_plain(Trials), b_plain(Trials), c_plain(Trials);
  std::vector<u64> a0(Trials), a1(Trials), b0v(Trials), b1v(Trials), c0v(Trials), c1v(Trials);
  for (size_t i = 0; i < Trials; i++) {
    a_plain[i] = rng() & 1ull;
    b_plain[i] = rng() & 1ull;
    c_plain[i] = rng() & 1ull;
    auto [sa0, sa1] = split_add(a_plain[i], rng);
    auto [sb0, sb1] = split_add(b_plain[i], rng);
    auto [sc0, sc1] = split_add(c_plain[i], rng);
    a0[i] = sa0; a1[i] = sa1;
    b0v[i] = sb0; b1v[i] = sb1;
    c0v[i] = sc0; c1v[i] = sc1;
  }

  std::vector<u64> and0(Trials), and1v(Trials), or0(Trials), or1v(Trials), xor0(Trials), xor1v(Trials);
  std::vector<u64> sel0(Trials), sel1v(Trials), lut0(Trials), lut1v(Trials);
  std::array<u64,8> table{};
  for (size_t i = 0; i < table.size(); i++) table[i] = rng();

  std::thread tA([&]{
    BeaverMul64 mul{0, c0, t0, 0};
    BitRingOps B{0, mul};
    for (size_t i = 0; i < Trials; i++) {
      and0[i] = B.AND(a0[i], b0v[i]);
      or0[i] = B.OR(a0[i], b0v[i]);
      xor0[i] = B.XOR(a0[i], b0v[i]);
      sel0[i] = B.SEL(a0[i], b0v[i], c0v[i]);
      lut0[i] = lut8_select(B, a0[i], b0v[i], c0v[i], table);
    }
  });

  std::thread tB([&]{
    BeaverMul64 mul{1, c1, t1, 0};
    BitRingOps B{1, mul};
    for (size_t i = 0; i < Trials; i++) {
      and1v[i] = B.AND(a1[i], b1v[i]);
      or1v[i] = B.OR(a1[i], b1v[i]);
      xor1v[i] = B.XOR(a1[i], b1v[i]);
      sel1v[i] = B.SEL(a1[i], b1v[i], c1v[i]);
      lut1v[i] = lut8_select(B, a1[i], b1v[i], c1v[i], table);
    }
  });
  tA.join(); tB.join();

  for (size_t i = 0; i < Trials; i++) {
    auto recon_and = add_mod(and0[i], and1v[i]) & 1ull;
    auto recon_or  = add_mod(or0[i], or1v[i]) & 1ull;
    auto recon_xor = add_mod(xor0[i], xor1v[i]) & 1ull;
    auto recon_sel = add_mod(sel0[i], sel1v[i]);
    auto recon_lut = add_mod(lut0[i], lut1v[i]);
    if (recon_and != (a_plain[i] & b_plain[i])) return false;
    if (recon_or  != ((a_plain[i] | b_plain[i]) & 1ull)) return false;
    if (recon_xor != ((a_plain[i] ^ b_plain[i]) & 1ull)) return false;
    if (recon_sel != (a_plain[i] ? c_plain[i] : b_plain[i])) return false;
    const size_t idx = (static_cast<size_t>(a_plain[i]) << 2) |
                       (static_cast<size_t>(b_plain[i]) << 1) |
                       static_cast<size_t>(c_plain[i]);
    if (recon_lut != table[idx]) return false;
  }
  return true;
}

static bool pack_evalmany_selftest() {
  ClearBackend backend;
  std::mt19937_64 rng(321);
  const size_t N = 16;
  std::vector<FssKey> keys;
  std::vector<u64> xs;
  keys.reserve(N);
  xs.reserve(N);
  for (size_t i = 0; i < N; i++) {
    u64 alpha = rng();
    u64 payload = rng();
    auto bits = backend.u64_to_bits_msb(alpha, 64);
    auto kp = backend.gen_dcf(64, bits, pack_u64_le(payload));
    keys.push_back(kp.k0);
    xs.push_back(rng());
  }

  auto flat = pack_keys_flat(keys);
  std::vector<uint8_t> outs_many(N * 8);
  backend.eval_dcf_many_u64(64, keys.front().bytes.size(), flat.data(), xs, 8, outs_many.data());
  for (size_t i = 0; i < N; i++) {
    FssKey kb = keys[i];
    auto single = backend.eval_dcf(64, kb, backend.u64_to_bits_msb(xs[i], 64));
    u64 single_val = unpack_u64_le(single.data());
    u64 many_val = unpack_u64_le(outs_many.data() + 8 * i);
    if (single_val != many_val) return false;
  }
  return true;
}
// Protocol-faithful reference for ReluARS with delta=0.
static u64 reluars_plain(u64 x, u64 r, int f) {
  u64 off = (f == 0) ? 0ull : (1ull << (f - 1));
  u64 hatx = add_mod(x, r);
  u64 z_hat = add_mod(hatx, off);
  u64 r_low = (f == 64) ? r : (r & ((u64(1) << f) - 1));
  u64 z_low = (f == 64) ? z_hat : (z_hat & ((u64(1) << f) - 1));
  u64 t = (z_low < r_low) ? 1ull : 0ull;
  u64 q = sub_mod(sub_mod((f == 64) ? 0ull : (z_hat >> f), (f == 64) ? 0ull : (r >> f)), t);
  u64 w = (static_cast<int64_t>(x) >= 0) ? 1ull : 0ull;
  return w ? q : 0ull;
}

// Plain reference for toy GeLU spline in dealer below.
static u64 gelu_plain_piecewise(u64 x_twos,
                                const std::vector<u64>& cut_bias,
                                const std::vector<std::vector<u64>>& coeffs,
                                int d,
                                u64 r_mask) {
  // Rotate comparison cuts into masked domain to mirror the online evaluator.
  u64 hatx = add_mod(x_twos, r_mask);
  u64 hatx_bias = add_mod(hatx, (u64(1) << 63));

  std::vector<u64> coeff = coeffs.front();  // base interval 0
  for (size_t i = 0; i < cut_bias.size(); i++) {
    u64 rotated = add_mod(cut_bias[i], r_mask);
    bool ge_cut = !(hatx_bias < rotated);   // 1[x >= cut]
    if (ge_cut) {
      for (int k = 0; k <= d; k++) {
        u64 delta = sub_mod(coeffs[i + 1][static_cast<size_t>(k)], coeffs[i][static_cast<size_t>(k)]);
        coeff[static_cast<size_t>(k)] = add_mod(coeff[static_cast<size_t>(k)], delta);
      }
    }
  }

  u64 acc = coeff[static_cast<size_t>(d)];
  for (int k = d - 1; k >= 0; k--) acc = add_mod(mul_mod(acc, x_twos), coeff[static_cast<size_t>(k)]);
  u64 x_plus = (static_cast<int64_t>(x_twos) < 0) ? 0ull : x_twos;
  return add_mod(x_plus, acc);
}

// ---------------- Dealer helpers ----------------
struct ReluARSSimKeys {
  ReluARSPartyKeyOnline party0;
  ReluARSPartyKeyOnline party1;
  u64 r_full;
  u64 r_out_full;
};

static ReluARSSimKeys dealer_make_reluars_sim(ClearBackend& fss, int f, std::mt19937_64& rng) {
  ReluARSSimKeys out;
  ReluARSParamsOnline params;
  params.f = f;
  params.delta.fill(0);
  out.party0.f = out.party1.f = f;

  auto [r0, r1] = split_add(rng(), rng);
  auto [rhi0, rhi1] = split_add((f == 64) ? 0ull : (add_mod(r0, r1) >> f), rng);
  auto [ro0, ro1] = split_add(rng(), rng);
  out.party0.r_in_share = r0; out.party1.r_in_share = r1;
  out.party0.r_hi_share = rhi0; out.party1.r_hi_share = rhi1;
  out.party0.r_out_share = ro0; out.party1.r_out_share = ro1;
  out.party0.params = params; out.party1.params = params;
  out.r_full = add_mod(r0, r1);
  out.r_out_full = add_mod(ro0, ro1);

  u64 r = out.r_full;
  u64 r2 = r + (u64(1) << 63);
  bool wrap = (r2 < r);
  u64 wrap0 = rng() & 1ull;
  if (!wrap) wrap0 = 0;
  u64 wrap1 = wrap ? (1ull - wrap0) : 0ull;
  out.party0.wrap_sign_share = wrap0;
  out.party1.wrap_sign_share = wrap1;
  auto payload_bit1 = []() {
    return pack_u64_le(1ull);
  };
  auto one = payload_bit1();
  auto k1 = fss.gen_dcf(64, fss.u64_to_bits_msb(r, 64), one);
  auto k2 = fss.gen_dcf(64, fss.u64_to_bits_msb(r2, 64), one);
  out.party0.dcf_hat_lt_r = k1.k0; out.party1.dcf_hat_lt_r = k1.k1;
  out.party0.dcf_hat_lt_r_plus_2p63 = k2.k0; out.party1.dcf_hat_lt_r_plus_2p63 = k2.k1;

  u64 r_low = (f == 64) ? r : (r & ((u64(1) << f) - 1));
  u64 r_low1 = (f == 64) ? (r_low + 1) : ((r_low + 1) & ((u64(1) << f) - 1));
  auto kt = fss.gen_dcf(f, fss.u64_to_bits_msb(r_low, f), one);
  auto kd = fss.gen_dcf(f, fss.u64_to_bits_msb(r_low1, f), one);
  out.party0.dcf_low_lt_r_low = kt.k0; out.party1.dcf_low_lt_r_low = kt.k1;
  out.party0.dcf_low_lt_r_low_plus1 = kd.k0; out.party1.dcf_low_lt_r_low_plus1 = kd.k1;

  // triples: fixed count for evaluator
  auto gen_beaver = [&](size_t n) {
    std::vector<BeaverTriple64Share> a(n), b(n);
    for (size_t i = 0; i < n; i++) {
      u64 aa = rng(), bb = rng(), cc = mul_mod(aa, bb);
      auto [a0, a1] = split_add(aa, rng);
      auto [b0, b1] = split_add(bb, rng);
      auto [c0, c1] = split_add(cc, rng);
      a[i] = {a0, b0, c0};
      b[i] = {a1, b1, c1};
    }
    return std::make_pair(a, b);
  };
  auto t = gen_beaver(reluars_triples64_needed());
  out.party0.triples64 = t.first;
  out.party1.triples64 = t.second;
  return out;
}

struct GeluSimKeys {
  GeluStepDCFPartyKey party0;
  GeluStepDCFPartyKey party1;
  u64 r_full;
  u64 r_out_full;
  std::vector<u64> cut_bias;
  std::vector<std::vector<u64>> coeffs;
};

static GeluSimKeys dealer_make_gelu_sim(ClearBackend& fss, int d, std::mt19937_64& rng) {
  GeluSimKeys out;
  out.party0.d = out.party1.d = d;
  auto [r0, r1] = split_add(rng(), rng);
  auto [ro0, ro1] = split_add(rng(), rng);
  out.r_full = add_mod(r0, r1);
  out.r_out_full = add_mod(ro0, ro1);
  out.party0.r_in_share = r0; out.party1.r_in_share = r1;
  out.party0.r_out_share = ro0; out.party1.r_out_share = ro1;

  u64 r = out.r_full;
  u64 r2 = r + (u64(1) << 63);
  bool wrap = (r2 < r);
  u64 wrap0 = rng() & 1ull;
  if (!wrap) wrap0 = 0;
  u64 wrap1 = wrap ? (1ull - wrap0) : 0ull;
  out.party0.wrap_sign_share = wrap0;
  out.party1.wrap_sign_share = wrap1;
  auto payload_bit1 = []() {
    return pack_u64_le(1ull);
  };
  auto one = payload_bit1();
  auto k1 = fss.gen_dcf(64, fss.u64_to_bits_msb(r, 64), one);
  auto k2 = fss.gen_dcf(64, fss.u64_to_bits_msb(r2, 64), one);
  out.party0.dcf_hat_lt_r = k1.k0; out.party1.dcf_hat_lt_r = k1.k1;
  out.party0.dcf_hat_lt_r_plus_2p63 = k2.k0; out.party1.dcf_hat_lt_r_plus_2p63 = k2.k1;

  // Toy spline: 4 intervals determined by 3 cutpoints in biased domain
  u64 c0 = (u64(1) << 63) - (u64(1) << 62);
  u64 c1 = (u64(1) << 63);
  u64 c2 = (u64(1) << 63) + (u64(1) << 62);
  out.cut_bias = {c0, c1, c2};
  out.coeffs = {
      {0, 0, 0, 0},       // interval 0: delta=0
      {0, 1, 0, 0},       // interval 1: delta = x
      {1, 2, 0, 0},       // interval 2: delta = 1 + 2x
      {0, 1, 0, 0},       // interval 3: delta = x
  };

  // Base coeff = first interval; payloads = v_j - v_{j+1}
  out.party0.base_coeff = out.coeffs.front();
  out.party1.base_coeff.assign(out.party0.base_coeff.size(), 0);
  auto flatten_payload = [&](const std::vector<u64>& v) {
    return pack_u64_vec_le(v);
  };

  for (size_t j = 0; j < out.cut_bias.size(); j++) {
    std::vector<u64> delta_vec(out.party0.base_coeff.size());
    for (size_t k = 0; k < delta_vec.size(); k++) {
      delta_vec[k] = sub_mod(out.coeffs[j + 1][k], out.coeffs[j][k]);
    }
    auto bytes = flatten_payload(delta_vec);
    u64 rotated_cut = add_mod(out.cut_bias[j], out.r_full);
    auto kp = fss.gen_dcf(64, fss.u64_to_bits_msb(rotated_cut, 64), bytes);
    StepCutVec sc;
    sc.dcf_key = kp.k0;
    sc.delta_vec = delta_vec;
    out.party0.cuts.push_back(sc);
    sc.dcf_key = kp.k1;
    out.party1.cuts.push_back(sc);
  }

  auto gen_beaver = [&](size_t n) {
    std::vector<BeaverTriple64Share> a(n), b(n);
    for (size_t i = 0; i < n; i++) {
      u64 aa = rng(), bb = rng(), cc = mul_mod(aa, bb);
      auto [a0, a1] = split_add(aa, rng);
      auto [b0, b1] = split_add(bb, rng);
      auto [c0, c1] = split_add(cc, rng);
      a[i] = {a0, b0, c0};
      b[i] = {a1, b1, c1};
    }
    return std::make_pair(a, b);
  };
  auto t = gen_beaver(static_cast<size_t>(d + 6));
  out.party0.triples64 = t.first;
  out.party1.triples64 = t.second;
  return out;
}

// ---------------- Simulation harness ----------------
int main() {
  try {
  if (!beaver_batch_selftest()) {
    std::cerr << "Beaver batch self-test failed\n";
    return 1;
  }
  if (!beaver_zero_share_selftest()) {
    std::cerr << "Beaver zero-share self-test failed\n";
    return 1;
  }
  if (!tape_roundtrip_selftest()) {
    std::cerr << "Tape roundtrip self-test failed\n";
    return 1;
  }
  if (!bitops_lut_selftest()) {
    std::cerr << "Bit/LUT self-test failed\n";
    return 1;
  }
  if (!pack_evalmany_selftest()) {
    std::cerr << "pack_keys_flat/eval_many self-test failed\n";
    return 1;
  }
  ClearBackend backend;
  std::mt19937_64 rng(12345);

  // ReluARS test
  {
    const int N = get_iters("RELU_ITERS", 2000);
    int pass = 0;
    for (int i = 0; i < N; i++) {
      auto relu_keys = dealer_make_reluars_sim(backend, 4, rng);
      u64 x = rng();
      u64 hatx = add_mod(x, relu_keys.r_full);
      LocalChan::Shared sh;
      ReluARSOut o0{}, o1{};
      LocalChan c0(&sh, true), c1(&sh, false);
      std::thread t0([&] { o0 = eval_reluars_one(0, backend, c0, relu_keys.party0, hatx); });
      std::thread t1([&] { o1 = eval_reluars_one(1, backend, c1, relu_keys.party1, hatx); });
      t0.join();
      t1.join();
      u64 y = add_mod(o0.y_share, o1.y_share);
      u64 yref = reluars_plain(x, relu_keys.r_full, relu_keys.party0.params.f);

      // Tape path
      TapeWriter tw0, tw1;
      tw0.write_u64(relu_keys.party0.wrap_sign_share);
      tw1.write_u64(relu_keys.party1.wrap_sign_share);
      tw0.write_u64(relu_keys.party0.r_in_share);
      tw1.write_u64(relu_keys.party1.r_in_share);
      tw0.write_u64(relu_keys.party0.r_hi_share);
      tw1.write_u64(relu_keys.party1.r_hi_share);
      tw0.write_u64(relu_keys.party0.r_out_share);
      tw1.write_u64(relu_keys.party1.r_out_share);
      tw0.write_bytes(relu_keys.party0.dcf_hat_lt_r.bytes);
      tw1.write_bytes(relu_keys.party1.dcf_hat_lt_r.bytes);
      tw0.write_bytes(relu_keys.party0.dcf_hat_lt_r_plus_2p63.bytes);
      tw1.write_bytes(relu_keys.party1.dcf_hat_lt_r_plus_2p63.bytes);
      tw0.write_bytes(relu_keys.party0.dcf_low_lt_r_low.bytes);
      tw1.write_bytes(relu_keys.party1.dcf_low_lt_r_low.bytes);
      tw0.write_bytes(relu_keys.party0.dcf_low_lt_r_low_plus1.bytes);
      tw1.write_bytes(relu_keys.party1.dcf_low_lt_r_low_plus1.bytes);
      tw0.write_triple64_vec<BeaverTriple64Share>(std::span<const BeaverTriple64Share>(relu_keys.party0.triples64));
      tw1.write_triple64_vec<BeaverTriple64Share>(std::span<const BeaverTriple64Share>(relu_keys.party1.triples64));
      TapeReader tr0(tw0.data()), tr1(tw1.data());
      LocalChan::Shared sh2;
      ReluARSOut ot0{}, ot1{};
      LocalChan tc0(&sh2, true), tc1(&sh2, false);
      std::thread tt0([&] { ot0 = eval_reluars_from_tape(0, backend, tc0, tr0, relu_keys.party0.params, hatx); });
      std::thread tt1([&] { ot1 = eval_reluars_from_tape(1, backend, tc1, tr1, relu_keys.party1.params, hatx); });
      tt0.join();
      tt1.join();
      u64 y_tape = add_mod(ot0.y_share, ot1.y_share);

      if (y == y_tape && y == yref) {
        pass++;
      } else if (std::getenv("DEBUG_REL_FAIL")) {
        u64 r = relu_keys.r_full;
        u64 r2 = add_mod(r, (u64(1) << 63));
        bool wrap_plain = r2 < r;
        bool a_plain = hatx < r;
        bool b_plain = hatx < r2;
        bool w_plain = wrap_plain ? ((!a_plain) || b_plain) : ((!a_plain) && b_plain);
        auto recon_dcf = [&](const FssKey& k0, const FssKey& k1, int bits) {
          auto s0 = backend.eval_dcf(bits, k0, backend.u64_to_bits_msb(hatx, bits));
          auto s1 = backend.eval_dcf(bits, k1, backend.u64_to_bits_msb(hatx, bits));
          u64 v0 = unpack_u64_le(s0.data());
          u64 v1 = unpack_u64_le(s1.data());
          return add_mod(v0, v1);
        };
        u64 a_rec = recon_dcf(relu_keys.party0.dcf_hat_lt_r, relu_keys.party1.dcf_hat_lt_r, 64);
        u64 b_rec = recon_dcf(relu_keys.party0.dcf_hat_lt_r_plus_2p63, relu_keys.party1.dcf_hat_lt_r_plus_2p63, 64);
        std::cerr << "Relu mismatch: x=" << x
                  << " hatx=" << hatx
                  << " y=" << y
                  << " y_tape=" << y_tape
                  << " yref=" << yref
                  << " w=" << add_mod(o0.w, o1.w)
                  << " t=" << add_mod(o0.t, o1.t)
                  << " d=" << add_mod(o0.d, o1.d)
                  << " wrap=" << add_mod(relu_keys.party0.wrap_sign_share, relu_keys.party1.wrap_sign_share)
                  << " wrap_plain=" << wrap_plain
                  << " w_plain=" << w_plain << " a=" << a_plain << " b=" << b_plain
                  << " a_rec=" << a_rec << " b_rec=" << b_rec
                  << "\n";
      }
    }
    if (pass != N) {
      std::cerr << "ReluARS test failed: " << pass << "/" << N << " passed\n";
      return 1;
    } else {
      std::cout << "ReluARS test passed: " << pass << "/" << N << "\n";
    }
  }

  // GeLU test (toy)
  {
    const int N = get_iters("GELU_ITERS", 2000);
    int pass = 0;
    for (int i = 0; i < N; i++) {
      auto gelu_keys = dealer_make_gelu_sim(backend, 1, rng);
      u64 x = rng();
      u64 hatx = add_mod(x, gelu_keys.r_full);
      LocalChan::Shared sh;
      GeluOut g0{}, g1{};
      LocalChan c0(&sh, true), c1(&sh, false);
      std::thread t0([&] { g0 = eval_gelu_step_dcf_one(0, backend, c0, gelu_keys.party0, hatx); });
      std::thread t1([&] { g1 = eval_gelu_step_dcf_one(1, backend, c1, gelu_keys.party1, hatx); });
      t0.join();
      t1.join();
      u64 y = add_mod(g0.y_share, g1.y_share);
      u64 yref = gelu_plain_piecewise(x, gelu_keys.cut_bias, gelu_keys.coeffs, gelu_keys.party0.d, gelu_keys.r_full);

      // Tape path
      TapeWriter tw0, tw1;
      tw0.write_u64(gelu_keys.party0.wrap_sign_share);
      tw1.write_u64(gelu_keys.party1.wrap_sign_share);
      tw0.write_u64(gelu_keys.party0.r_in_share);
      tw1.write_u64(gelu_keys.party1.r_in_share);
      tw0.write_u64(gelu_keys.party0.r_out_share);
      tw1.write_u64(gelu_keys.party1.r_out_share);
      tw0.write_bytes(gelu_keys.party0.dcf_hat_lt_r.bytes);
      tw1.write_bytes(gelu_keys.party1.dcf_hat_lt_r.bytes);
      tw0.write_bytes(gelu_keys.party0.dcf_hat_lt_r_plus_2p63.bytes);
      tw1.write_bytes(gelu_keys.party1.dcf_hat_lt_r_plus_2p63.bytes);
      tw0.write_u64_vec(gelu_keys.party0.base_coeff);
      tw1.write_u64_vec(gelu_keys.party1.base_coeff);
      tw0.write_u64(static_cast<uint64_t>(gelu_keys.party0.cuts.size()));
      tw1.write_u64(static_cast<uint64_t>(gelu_keys.party1.cuts.size()));
      for (size_t ci = 0; ci < gelu_keys.party0.cuts.size(); ci++) {
        tw0.write_bytes(gelu_keys.party0.cuts[ci].dcf_key.bytes);
        tw0.write_u64_vec(gelu_keys.party0.cuts[ci].delta_vec);
        tw1.write_bytes(gelu_keys.party1.cuts[ci].dcf_key.bytes);
        tw1.write_u64_vec(gelu_keys.party1.cuts[ci].delta_vec);
      }
      tw0.write_triple64_vec<BeaverTriple64Share>(std::span<const BeaverTriple64Share>(gelu_keys.party0.triples64));
      tw1.write_triple64_vec<BeaverTriple64Share>(std::span<const BeaverTriple64Share>(gelu_keys.party1.triples64));
      TapeReader tr0(tw0.data()), tr1(tw1.data());
      LocalChan::Shared sh2;
      GeluOut gt0{}, gt1{};
      LocalChan tc0(&sh2, true), tc1(&sh2, false);
      std::thread tt0([&] { gt0 = eval_gelu_step_dcf_from_tape(0, backend, tc0, tr0, hatx); });
      std::thread tt1([&] { gt1 = eval_gelu_step_dcf_from_tape(1, backend, tc1, tr1, hatx); });
      tt0.join();
      tt1.join();
      u64 y_tape = add_mod(gt0.y_share, gt1.y_share);

      if (y == y_tape && y == yref) {
        pass++;
      } else if (std::getenv("DEBUG_GELU_FAIL")) {
        u64 delta_calc = sub_mod(y, x);
        std::cerr << "Gelu mismatch: x=" << x
                  << " hatx=" << hatx
                  << " y=" << y
                  << " y_tape=" << y_tape
                  << " yref=" << yref
                  << " w=" << add_mod(g0.w, g1.w)
                  << " delta=" << delta_calc
                  << "\n";
      }
    }
    if (pass != N) {
      std::cerr << "GeLU (toy) test failed: " << pass << "/" << N << " passed\n";
      return 1;
    } else {
      std::cout << "GeLU (toy) test passed: " << pass << "/" << N << "\n";
    }
  }

  // Demonstrate pack_keys_flat utility
  // Demonstrate packing on the last generated keys
  auto demo_keys = dealer_make_reluars_sim(backend, 4, rng);
  std::vector<FssKey> sign_keys = {demo_keys.party0.dcf_hat_lt_r, demo_keys.party0.dcf_hat_lt_r_plus_2p63};
  auto flat = pack_keys_flat(sign_keys);
  std::cout << "Packed " << flat.size() << " bytes for 2 sign keys (party0)\n";

  return 0;
  } catch (const std::exception& e) {
    std::cerr << "sim_harness exception: " << e.what() << "\n";
    return 1;
  }
}
