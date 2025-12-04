#pragma once

#include <vector>
#include "suf/suf_ir.hpp"
#include "proto/beaver_mul64.hpp"
#include "proto/bit_ring_ops.hpp"

namespace gates {

// Evaluate BoolExpr in XOR domain for predicate bits; wrap bits are additive shares.
// Returns XOR share (0/1) of the boolean. Uses bit-level triples via BeaverMul64.
inline uint64_t eval_bool_xor(const suf::BoolExpr& e,
                              const std::vector<uint64_t>& pred_bits_xor,
                              const std::vector<uint64_t>& wrap_bits_add,
                              proto::BeaverMul64& mul,
                              const proto::BeaverTripleBitShare* bit_triples = nullptr,
                              size_t* bit_idx = nullptr) {
  return std::visit([&](auto&& n) -> uint64_t {
    using T = std::decay_t<decltype(n)>;
    if constexpr (std::is_same_v<T, suf::BConst>) return n.v ? 1ull : 0ull;
    else if constexpr (std::is_same_v<T, suf::BVar>) {
      int idx = n.pred_idx;
      if (idx >= 0) {
        size_t u = static_cast<size_t>(idx);
        if (u < pred_bits_xor.size()) return pred_bits_xor[u] & 1ull;
        return 0;
      } else {
        size_t w = static_cast<size_t>(-1 - idx);
        if (w < wrap_bits_add.size()) return wrap_bits_add[w] & 1ull;
        return 0;
      }
    } else if constexpr (std::is_same_v<T, suf::BNot>) {
      return 1ull ^ eval_bool_xor(*n.a, pred_bits_xor, wrap_bits_add, mul, bit_triples, bit_idx);
    } else if constexpr (std::is_same_v<T, suf::BXor>) {
      return eval_bool_xor(*n.a, pred_bits_xor, wrap_bits_add, mul, bit_triples, bit_idx) ^ eval_bool_xor(*n.b, pred_bits_xor, wrap_bits_add, mul, bit_triples, bit_idx);
    } else if constexpr (std::is_same_v<T, suf::BAnd>) {
      uint64_t ax = eval_bool_xor(*n.a, pred_bits_xor, wrap_bits_add, mul, bit_triples, bit_idx) & 1ull;
      uint64_t ay = eval_bool_xor(*n.b, pred_bits_xor, wrap_bits_add, mul, bit_triples, bit_idx) & 1ull;
      if (bit_triples && bit_idx) {
        size_t idx_local = (*bit_idx)++;
        const auto& t = bit_triples[idx_local];
        uint8_t e_share = static_cast<uint8_t>(ax) ^ t.a;
        uint8_t f_share = static_cast<uint8_t>(ay) ^ t.b;
        uint8_t other_e = 0, other_f = 0;
        mul.ch.send_bytes(&e_share, 1);
        mul.ch.send_bytes(&f_share, 1);
        mul.ch.recv_bytes(&other_e, 1);
        mul.ch.recv_bytes(&other_f, 1);
        uint8_t e = static_cast<uint8_t>(e_share ^ other_e);
        uint8_t f = static_cast<uint8_t>(f_share ^ other_f);
        uint8_t z = static_cast<uint8_t>(t.c ^ (e & t.b) ^ (f & t.a));
        if (mul.party == 0) z ^= static_cast<uint8_t>(e & f);
        return static_cast<uint64_t>(z & 1u);
      } else {
        // fallback to additive Beaver
        uint64_t a_share = (mul.party == 0) ? ax : 0ull;
        uint64_t b_share = (mul.party == 1) ? ax : 0ull;
        uint64_t c_share = (mul.party == 0) ? ay : 0ull;
        uint64_t d_share = (mul.party == 1) ? ay : 0ull;
        uint64_t prod = mul.mul(a_share, c_share);
        uint64_t two_prod = proto::add_mod(prod, prod);
        uint64_t add_a = proto::sub_mod(proto::add_mod(a_share, b_share), two_prod);
        uint64_t add_b = proto::sub_mod(proto::add_mod(c_share, d_share), two_prod);
        uint64_t and_add = mul.mul(add_a, add_b);
        return and_add & 1ull;
      }
    } else { // BOr
      uint64_t ax = eval_bool_xor(*n.a, pred_bits_xor, wrap_bits_add, mul, bit_triples, bit_idx) & 1ull;
      uint64_t ay = eval_bool_xor(*n.b, pred_bits_xor, wrap_bits_add, mul, bit_triples, bit_idx) & 1ull;
      uint64_t xorv = ax ^ ay;
      uint64_t andv = ax & ay;
      return xorv ^ andv;
    }
  }, e.node);
}

// Convert XOR bit to additive using 1 Beaver multiply: b0,b1 XOR shares -> additive share
inline uint64_t b2a_bit(uint64_t bx, int party, proto::BeaverMul64& mul) {
  uint64_t a_share = (party == 0) ? bx : 0ull;
  uint64_t b_share = (party == 1) ? bx : 0ull;
  uint64_t prod = mul.mul(a_share, b_share);
  uint64_t two_prod = proto::add_mod(prod, prod);
  return proto::sub_mod(proto::add_mod(a_share, b_share), two_prod);
}

}  // namespace gates
