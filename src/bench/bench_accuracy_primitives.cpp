#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "gates/nexp_gate.hpp"
#include "gates/reciprocal_gate.hpp"
#include "gates/rsqrt_gate.hpp"
#include "gates/silu_spline_gate.hpp"
#include "gates/tables/gelu_spline_table.hpp"

namespace {

struct Args {
  int frac_bits = 12;
  int samples = 200000;
  uint64_t seed = 1234;
  std::string out_json;
  std::string out_md;
  int gelu_segments = 16;
  int silu_segments = 16;
  int nexp_segments = 16;
  int recip_iters = 1;
  int rsqrt_iters = 1;
};

Args parse(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    std::string s(argv[i]);
    auto next_val = [&](int& idx) -> std::string {
      if (idx + 1 >= argc) throw std::runtime_error("flag requires value: " + s);
      return std::string(argv[++idx]);
    };
    if (s == "--frac-bits") a.frac_bits = std::stoi(next_val(i));
    else if (s == "--samples") a.samples = std::stoi(next_val(i));
    else if (s == "--seed") a.seed = static_cast<uint64_t>(std::stoull(next_val(i)));
    else if (s == "--out-json") a.out_json = next_val(i);
    else if (s == "--out-md") a.out_md = next_val(i);
    else if (s == "--gelu-segments") a.gelu_segments = std::stoi(next_val(i));
    else if (s == "--silu-segments") a.silu_segments = std::stoi(next_val(i));
    else if (s == "--nexp-segments") a.nexp_segments = std::stoi(next_val(i));
    else if (s == "--recip-iters") a.recip_iters = std::stoi(next_val(i));
    else if (s == "--rsqrt-iters") a.rsqrt_iters = std::stoi(next_val(i));
    else if (s == "--help" || s == "-h") {
      std::cerr
          << "Usage: bench_accuracy_primitives [--frac-bits F] [--samples N] [--seed S]\n"
             "  [--out-json PATH] [--out-md PATH]\n"
             "  [--gelu-segments K] [--silu-segments K] [--nexp-segments K]\n"
             "  [--recip-iters I] [--rsqrt-iters I]\n";
      std::exit(0);
    }
  }
  return a;
}

static inline int64_t clamp_i64(__int128 v) {
  if (v > static_cast<__int128>(std::numeric_limits<int64_t>::max())) {
    v = static_cast<__int128>(std::numeric_limits<int64_t>::max());
  }
  if (v < static_cast<__int128>(std::numeric_limits<int64_t>::min())) {
    v = static_cast<__int128>(std::numeric_limits<int64_t>::min());
  }
  return static_cast<int64_t>(v);
}

static inline int64_t to_fixed(double x, int frac_bits) {
  const double s = std::ldexp(1.0, frac_bits);
  return static_cast<int64_t>(std::llround(x * s));
}

static inline double from_fixed(int64_t x, int frac_bits) {
  return static_cast<double>(x) / std::ldexp(1.0, frac_bits);
}

struct ErrStats {
  double mean_abs = 0.0;
  double max_abs = 0.0;
  double p99_abs = 0.0;
  double mean_rel = 0.0;
  double p99_rel = 0.0;
  double max_rel = 0.0;
};

ErrStats summarize_errors(const std::vector<double>& abs_errs,
                          const std::vector<double>& rel_errs) {
  ErrStats st;
  if (!abs_errs.empty()) {
    double sum = 0.0;
    double mx = 0.0;
    for (double v : abs_errs) {
      sum += v;
      mx = std::max(mx, v);
    }
    st.mean_abs = sum / static_cast<double>(abs_errs.size());
    st.max_abs = mx;
    auto tmp = abs_errs;
    std::nth_element(tmp.begin(), tmp.begin() + (tmp.size() * 99) / 100, tmp.end());
    st.p99_abs = tmp[(tmp.size() * 99) / 100];
  }
  if (!rel_errs.empty()) {
    double sum = 0.0;
    double mx = 0.0;
    for (double v : rel_errs) {
      sum += v;
      mx = std::max(mx, v);
    }
    st.mean_rel = sum / static_cast<double>(rel_errs.size());
    st.max_rel = mx;
    auto tmp = rel_errs;
    std::nth_element(tmp.begin(), tmp.begin() + (tmp.size() * 99) / 100, tmp.end());
    st.p99_rel = tmp[(tmp.size() * 99) / 100];
  }
  return st;
}

template <typename TrueFn, typename ApproxFn>
ErrStats bench_one(const std::string& name,
                   int frac_bits,
                   int samples,
                   std::mt19937_64& rng,
                   double x_lo,
                   double x_hi,
                   TrueFn true_fn,
                   ApproxFn approx_fn) {
  std::uniform_real_distribution<double> dist(x_lo, x_hi);
  std::vector<double> abs_errs;
  abs_errs.reserve(static_cast<size_t>(samples));
  std::vector<double> rel_errs;
  rel_errs.reserve(static_cast<size_t>(samples));

  for (int i = 0; i < samples; ++i) {
    double x = dist(rng);
    int64_t x_fixed = to_fixed(x, frac_bits);
    int64_t y_true_fixed = to_fixed(true_fn(x), frac_bits);
    int64_t y_approx_fixed = approx_fn(x_fixed);
    int64_t diff = y_approx_fixed - y_true_fixed;
    double abs_e = std::abs(from_fixed(diff, frac_bits));
    abs_errs.push_back(abs_e);
    double denom = std::abs(from_fixed(y_true_fixed, frac_bits));
    if (denom > 0) {
      rel_errs.push_back(abs_e / denom);
    }
  }
  (void)name;
  return summarize_errors(abs_errs, rel_errs);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    Args args = parse(argc, argv);
    if (args.samples <= 0) throw std::runtime_error("--samples must be > 0");
    if (args.frac_bits < 1 || args.frac_bits > 30) throw std::runtime_error("--frac-bits must be in [1,30]");

    std::mt19937_64 rng(args.seed);
    const int fb = args.frac_bits;

    // GeLU spline approximation (same PiecewisePolySpec family used by the composite gate material).
    gates::PiecewisePolySpec gelu_spec = gates::make_gelu_spline_spec(fb, args.gelu_segments);
    auto gelu_stats = bench_one(
        "gelu",
        fb,
        args.samples,
        rng,
        /*x_lo=*/-8.0,
        /*x_hi=*/8.0,
        [](double x) { return gates::gelu_fn(x); },
        [&](int64_t x_fixed) -> int64_t { return gates::eval_piecewise_poly_ref(gelu_spec, x_fixed); });

    // SiLU spline approximation.
    gates::SiLUGateParams silu_params{.frac_bits = fb, .segments = args.silu_segments};
    gates::PiecewisePolySpec silu_spec = gates::make_silu_spec(silu_params);
    auto silu_true = [](double x) {
      // x / (1 + exp(-x))
      double ex = std::exp(-x);
      return x / (1.0 + ex);
    };
    auto silu_stats = bench_one(
        "silu",
        fb,
        args.samples,
        rng,
        /*x_lo=*/-8.0,
        /*x_hi=*/8.0,
        silu_true,
        [&](int64_t x_fixed) -> int64_t { return gates::ref_silu_fixed(silu_spec, x_fixed); });

    // nExp approximation used in softmax: exp(-x) where x is clamped to [0,16].
    gates::NExpGateParams nexp_params{.frac_bits = fb, .segments = args.nexp_segments};
    gates::PiecewisePolySpec nexp_spec = gates::make_nexp_spec(nexp_params);
    auto nexp_true = [](double x) { return std::exp(-x); };
    auto nexp_stats = bench_one(
        "nexp",
        fb,
        args.samples,
        rng,
        /*x_lo=*/0.0,
        /*x_hi=*/16.0,
        nexp_true,
        [&](int64_t x_fixed) -> int64_t { return gates::ref_nexp_fixed(nexp_spec, x_fixed); });

    // Reciprocal init+NR approximation for x in [1, 1024].
    gates::ReciprocalParams recip_params{.frac_bits = fb, .nr_iters = args.recip_iters, .nmax = 1024.0};
    gates::PiecewisePolySpec recip_init = gates::make_recip_affine_init_spec(fb, recip_params.nmax);
    auto recip_true = [](double x) { return 1.0 / x; };
    auto recip_stats = bench_one(
        "recip",
        fb,
        args.samples,
        rng,
        /*x_lo=*/1.0,
        /*x_hi=*/recip_params.nmax,
        recip_true,
        [&](int64_t x_fixed) -> int64_t {
          // Clamp to >= 1.0 (matches init spec).
          if (x_fixed < (int64_t(1) << fb)) x_fixed = (int64_t(1) << fb);
          return gates::ref_reciprocal_fixed(recip_init, x_fixed, fb, recip_params.nr_iters);
        });

    // Rsqrt init+NR approximation for x in [eps, vmax].
    gates::RsqrtParams rsqrt_params{.frac_bits = fb, .nr_iters = args.rsqrt_iters, .eps = 1.0 / 1024.0, .vmax = 16.0};
    gates::PiecewisePolySpec rsqrt_init = gates::make_rsqrt_affine_init_spec(fb, rsqrt_params.eps, rsqrt_params.vmax);
    auto rsqrt_true = [](double x) { return 1.0 / std::sqrt(x); };
    auto rsqrt_stats = bench_one(
        "rsqrt",
        fb,
        args.samples,
        rng,
        /*x_lo=*/rsqrt_params.eps,
        /*x_hi=*/rsqrt_params.vmax,
        rsqrt_true,
        [&](int64_t x_fixed) -> int64_t {
          // Clamp to >= eps (matches init spec).
          int64_t eps_q = to_fixed(rsqrt_params.eps, fb);
          if (eps_q <= 0) eps_q = 1;
          if (x_fixed < eps_q) x_fixed = eps_q;
          return gates::ref_rsqrt_fixed(rsqrt_init, x_fixed, fb, rsqrt_params.nr_iters);
        });

    auto write_json = [&](std::ostream& o) {
      auto dump = [&](const char* nm, const ErrStats& s) {
        o << "    \"" << nm << "\": {"
          << "\"mean_abs\": " << std::setprecision(10) << s.mean_abs
          << ", \"p99_abs\": " << std::setprecision(10) << s.p99_abs
          << ", \"max_abs\": " << std::setprecision(10) << s.max_abs
          << ", \"mean_rel\": " << std::setprecision(10) << s.mean_rel
          << ", \"p99_rel\": " << std::setprecision(10) << s.p99_rel
          << ", \"max_rel\": " << std::setprecision(10) << s.max_rel
          << "}";
      };

      o << "{\n";
      o << "  \"frac_bits\": " << fb << ",\n";
      o << "  \"samples\": " << args.samples << ",\n";
      o << "  \"seed\": " << args.seed << ",\n";
      o << "  \"configs\": {\n";
      o << "    \"gelu_segments\": " << args.gelu_segments << ",\n";
      o << "    \"silu_segments\": " << args.silu_segments << ",\n";
      o << "    \"nexp_segments\": " << args.nexp_segments << ",\n";
      o << "    \"recip_iters\": " << args.recip_iters << ",\n";
      o << "    \"rsqrt_iters\": " << args.rsqrt_iters << "\n";
      o << "  },\n";
      o << "  \"metrics\": {\n";
      dump("gelu", gelu_stats); o << ",\n";
      dump("silu", silu_stats); o << ",\n";
      dump("nexp", nexp_stats); o << ",\n";
      dump("recip", recip_stats); o << ",\n";
      dump("rsqrt", rsqrt_stats); o << "\n";
      o << "  }\n";
      o << "}\n";
    };

    auto write_md = [&](std::ostream& o) {
      auto row = [&](const char* nm, const ErrStats& s) {
        o << "| " << nm
          << " | " << std::scientific << std::setprecision(3) << s.mean_abs
          << " | " << std::scientific << std::setprecision(3) << s.p99_abs
          << " | " << std::scientific << std::setprecision(3) << s.max_abs
          << " | " << std::scientific << std::setprecision(3) << s.mean_rel
          << " | " << std::scientific << std::setprecision(3) << s.p99_rel
          << " | " << std::scientific << std::setprecision(3) << s.max_rel
          << " |\n";
      };
      o << "# Primitive approximation accuracy (fixed-point)\n\n";
      o << "- frac_bits: `" << fb << "`\n";
      o << "- samples: `" << args.samples << "`\n";
      o << "- seed: `" << args.seed << "`\n\n";
      o << "| primitive | mean_abs | p99_abs | max_abs | mean_rel | p99_rel | max_rel |\n";
      o << "|---|---:|---:|---:|---:|---:|---:|\n";
      row("gelu", gelu_stats);
      row("silu", silu_stats);
      row("nexp(exp(-x))", nexp_stats);
      row("recip(1/x)", recip_stats);
      row("rsqrt(1/sqrt(x))", rsqrt_stats);
      o << "\n";
      o << "Notes:\n";
      o << "- Errors are measured in real units after dividing fixed-point deltas by `2^frac_bits`.\n";
      o << "- `rel` uses `abs_err / abs(true)` when `true != 0`.\n";
      o << "- Domains: GeLU/SiLU x∈[-8,8], nExp x∈[0,16], recip x∈[1,1024], rsqrt x∈[1/1024,16].\n";
    };

    if (!args.out_json.empty()) {
      std::ofstream f(args.out_json);
      if (!f) throw std::runtime_error("failed to open --out-json: " + args.out_json);
      write_json(f);
    }
    if (!args.out_md.empty()) {
      std::ofstream f(args.out_md);
      if (!f) throw std::runtime_error("failed to open --out-md: " + args.out_md);
      write_md(f);
    }

    write_json(std::cout);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "bench_accuracy_primitives error: " << e.what() << "\n";
    return 1;
  }
}

