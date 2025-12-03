#include <cassert>
#include <iostream>
#include <random>
#include "suf/mask_rewrite.hpp"
#include "suf/mask_rewrite_eval.hpp"

using namespace suf;

static void test_lt_u64() {
  std::mt19937_64 rng(42);
  for (int i = 0; i < 5000; i++) {
    uint64_t r = rng();
    uint64_t beta = rng();
    auto rec = rewrite_lt_u64(r, beta);
    uint64_t x = rng();
    uint64_t hatx = x + r;
    bool ref = x < beta;
    bool got = eval_rot_recipe(rec, hatx);
    assert(ref == got);
  }
  std::cout << "mask_rewrite lt_u64 ok\n";
}

static void test_ltlow() {
  std::mt19937_64 rng(99);
  for (int i = 0; i < 3000; i++) {
    uint64_t r = rng();
    int f = 1 + (rng() % 64);
    uint64_t mask = (f >= 64) ? ~uint64_t(0) : ((uint64_t(1) << f) - 1);
    uint64_t gamma = rng() & mask;
    auto rec = rewrite_ltlow(r, f, gamma);
    uint64_t x = rng();
    uint64_t hatx = x + r;
    uint64_t x_low = (f >= 64) ? x : (x & mask);
    bool ref = x_low < gamma;
    bool got = eval_rot_low_recipe(rec, hatx);
    assert(ref == got);
  }
  std::cout << "mask_rewrite ltlow ok\n";
}

static void test_msb_add() {
  std::mt19937_64 rng(1234);
  for (int i = 0; i < 5000; i++) {
    uint64_t r = rng();
    uint64_t c = rng();
    auto rec = rewrite_msb_add(r, c);
    uint64_t x = rng();
    uint64_t hatx = x + r;
    bool ref = ((x + c) >> 63) & 1u;
    bool got = eval_msb_add_recipe(rec, hatx);
    assert(ref == got);
  }
  std::cout << "mask_rewrite msb_add ok\n";
}

int main() {
  test_lt_u64();
  test_ltlow();
  test_msb_add();
  return 0;
}
