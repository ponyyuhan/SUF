#pragma once

#include <vector>
#include "suf/suf_ir.hpp"
#include "proto/beaver_mul64.hpp"
#include "proto/bit_ring_ops.hpp"
#include "gates/pred_view.hpp"

namespace gates {

// Generic view-based evaluator: get_pred(idx) must return XOR bit (0/1) for predicate idx.
template<typename PredAccessor>
inline uint64_t eval_bool_xor_view(const suf::BoolExpr& e,
                                   PredAccessor&& get_pred,
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
        return get_pred(static_cast<size_t>(idx)) & 1ull;
      } else {
        size_t w = static_cast<size_t>(-1 - idx);
        if (w < wrap_bits_add.size()) return wrap_bits_add[w] & 1ull;
        return 0;
      }
    } else if constexpr (std::is_same_v<T, suf::BNot>) {
      return 1ull ^ eval_bool_xor_view(*n.a, get_pred, wrap_bits_add, mul, bit_triples, bit_idx);
    } else if constexpr (std::is_same_v<T, suf::BXor>) {
      return eval_bool_xor_view(*n.a, get_pred, wrap_bits_add, mul, bit_triples, bit_idx) ^
             eval_bool_xor_view(*n.b, get_pred, wrap_bits_add, mul, bit_triples, bit_idx);
    } else if constexpr (std::is_same_v<T, suf::BAnd>) {
      uint64_t ax = eval_bool_xor_view(*n.a, get_pred, wrap_bits_add, mul, bit_triples, bit_idx) & 1ull;
      uint64_t ay = eval_bool_xor_view(*n.b, get_pred, wrap_bits_add, mul, bit_triples, bit_idx) & 1ull;
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
      uint64_t ax = eval_bool_xor_view(*n.a, get_pred, wrap_bits_add, mul, bit_triples, bit_idx) & 1ull;
      uint64_t ay = eval_bool_xor_view(*n.b, get_pred, wrap_bits_add, mul, bit_triples, bit_idx) & 1ull;
      uint64_t xorv = ax ^ ay;
      uint64_t andv = ax & ay;
      return xorv ^ andv;
    }
  }, e.node);
}

// Legacy convenience wrapper using flat bit vector.
inline uint64_t eval_bool_xor(const suf::BoolExpr& e,
                              const std::vector<uint64_t>& pred_bits_xor,
                              const std::vector<uint64_t>& wrap_bits_add,
                              proto::BeaverMul64& mul,
                              const proto::BeaverTripleBitShare* bit_triples = nullptr,
                              size_t* bit_idx = nullptr) {
  auto get_pred = [&](size_t idx) -> uint64_t {
    if (idx < pred_bits_xor.size()) return pred_bits_xor[idx] & 1ull;
    return 0;
  };
  return eval_bool_xor_view(e, get_pred, wrap_bits_add, mul, bit_triples, bit_idx);
}

// Convert XOR bit to additive using 1 Beaver multiply: b0,b1 XOR shares -> additive share
inline uint64_t b2a_bit(uint64_t bx, int party, proto::BeaverMul64& mul) {
  uint64_t a_share = (party == 0) ? bx : 0ull;
  uint64_t b_share = (party == 1) ? bx : 0ull;
  uint64_t prod = mul.mul(a_share, b_share);
  uint64_t two_prod = proto::add_mod(prod, prod);
  return proto::sub_mod(proto::add_mod(a_share, b_share), two_prod);
}

// Packed evaluator over blocks: pred_masks packed, wrap additive; outputs additive bits.
inline void eval_bool_xor_packed_block(const std::vector<suf::BoolExpr>& exprs,
                                       const PredViewPacked& pred_masks,
                                       const std::vector<uint64_t>& wrap_bits_add,
                                       int party,
                                       proto::BeaverMul64& mul,
                                       const proto::BeaverTripleBitShare* bit_triples,
                                       size_t bit_base,
                                       std::vector<uint64_t>& out_add) {
  out_add.resize(exprs.size());
  size_t bit_idx = bit_base;
  auto get_pred = [&](size_t idx)->uint64_t { return pred_masks.get(idx); };
  for (size_t i = 0; i < exprs.size(); i++) {
    uint64_t bx = eval_bool_xor_view(exprs[i], get_pred, wrap_bits_add, mul, bit_triples, &bit_idx) & 1ull;
    out_add[i] = b2a_bit(bx, party, mul);
  }
}

// Packed evaluator over a block of elements (SoA layout). masks_base points to [block][words_per_elem].
inline void eval_bool_xor_packed_block_soa(const std::vector<suf::BoolExpr>& exprs,
                                           const uint64_t* masks_base,
                                           size_t words_per_elem,
                                           size_t block_size,
                                           const std::vector<uint64_t>& wrap_bits_add,
                                           int party,
                                           proto::BeaverMul64& mul,
                                           const proto::BeaverTripleBitShare* bit_triples,
                                           size_t bit_base,
                                           std::vector<uint64_t>& out_add_flat) {
  out_add_flat.assign(exprs.size() * block_size, 0);
  for (size_t i = 0; i < block_size; i++) {
    PredViewPacked pv{masks_base + i * words_per_elem, words_per_elem};
    size_t bit_idx = bit_base;
    auto get_pred = [&](size_t idx) -> uint64_t { return pv.get(idx); };
    for (size_t e = 0; e < exprs.size(); e++) {
      uint64_t bx = eval_bool_xor_view(exprs[e], get_pred, wrap_bits_add, mul, bit_triples, &bit_idx) & 1ull;
      out_add_flat[e * block_size + i] = b2a_bit(bx, party, mul);
    }
  }
}

}  // namespace gates
