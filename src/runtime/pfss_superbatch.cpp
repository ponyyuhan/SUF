#include "runtime/pfss_superbatch.hpp"

#include <stdexcept>
#include <tuple>
#include <unordered_map>

namespace runtime {

void PfssSuperBatch::flush_and_finalize(int party,
                                        proto::PfssBackendBatch& backend,
                                        proto::IChannel& ch) {
  using Key = std::tuple<const suf::SUF<uint64_t>*, const gates::CompositePartyKey*, const gates::PostProcHook*>;
  struct KeyHash {
    size_t operator()(const Key& k) const {
      return std::hash<const void*>{}(std::get<0>(k)) ^
             (std::hash<const void*>{}(std::get<1>(k)) << 1) ^
             (std::hash<const void*>{}(std::get<2>(k)) << 2);
    }
  };
  struct Slice {
    size_t job_idx;
    size_t start;
    size_t len;
  };
  std::unordered_map<Key, size_t, KeyHash> group_index;
  std::vector<std::vector<uint64_t>> grouped_hatx;
  std::vector<std::vector<Slice>> grouped_slices;

  for (size_t idx = 0; idx < jobs_.size(); ++idx) {
    auto& job = jobs_[idx];
    if (!job.suf || !job.key || !job.hook) {
      throw std::runtime_error("PfssSuperBatch: incomplete composite job");
    }
    Key k{job.suf, job.key, job.hook};
    size_t g;
    auto it = group_index.find(k);
    if (it == group_index.end()) {
      g = grouped_hatx.size();
      group_index.emplace(k, g);
      grouped_hatx.emplace_back();
      grouped_slices.emplace_back();
    } else {
      g = it->second;
    }
    size_t start = grouped_hatx[g].size();
    grouped_hatx[g].insert(grouped_hatx[g].end(), job.hatx_public.begin(), job.hatx_public.end());
    grouped_slices[g].push_back(Slice{idx, start, job.hatx_public.size()});
  }

  for (auto& kv : group_index) {
    const auto& ktuple = kv.first;
    size_t g = kv.second;
    const auto* suf = std::get<0>(ktuple);
    const auto* key = std::get<1>(ktuple);
    auto* hook = const_cast<gates::PostProcHook*>(std::get<2>(ktuple));
    gates::CompositeBatchInput in{grouped_hatx[g].data(), grouped_hatx[g].size()};
    auto out = gates::composite_eval_batch_with_postproc(
        party, backend, ch, *key, *suf, in, *hook);
    size_t r = static_cast<size_t>(key->compiled.r);
    uint64_t r_out_share = key->r_out_share.empty() ? 0ull : key->r_out_share[0];
    for (const auto& sl : grouped_slices[g]) {
      auto& job = jobs_[sl.job_idx];
      for (size_t i = 0; i < sl.len; ++i) {
        size_t global_idx = sl.start + i;
        for (size_t rr = 0; rr < r; ++rr) {
          size_t out_idx = global_idx * r + rr;
          size_t dst = i * r + rr;
          if (dst >= job.out.numel()) continue;
          job.out.data[dst] = proto::sub_mod(out.haty_share[out_idx], r_out_share);
        }
      }
    }
  }
  jobs_.clear();
}

void run_truncation_now(int party,
                        proto::PfssBackendBatch& backend,
                        proto::IChannel& ch,
                        const compiler::TruncationLoweringResult& bundle,
                        const std::vector<uint64_t>& x_share,
                        std::vector<uint64_t>& y_share) {
  const auto& key = (party == 0) ? bundle.keys.k0 : bundle.keys.k1;
  gates::PostProcHook* hook = (party == 0) ? bundle.hook0.get() : bundle.hook1.get();
  if (!hook) throw std::runtime_error("run_truncation_now: missing postproc hook");

  size_t n = x_share.size();
  std::vector<uint64_t> hatx_share(n);
  for (size_t i = 0; i < n; ++i) {
    hatx_share[i] = proto::add_mod(x_share[i], key.r_in_share);
  }

  // Open hatx (masked) to both parties.
  std::vector<uint64_t> other(n, 0);
  size_t byte_len = n * sizeof(uint64_t);
  if (party == 0) {
    ch.send_bytes(hatx_share.data(), byte_len);
    ch.recv_bytes(other.data(), byte_len);
  } else {
    ch.recv_bytes(other.data(), byte_len);
    ch.send_bytes(hatx_share.data(), byte_len);
  }
  std::vector<uint64_t> hatx_public(n);
  for (size_t i = 0; i < n; ++i) {
    hatx_public[i] = proto::add_mod(hatx_share[i], other[i]);
  }

  gates::CompositeBatchInput in{hatx_public.data(), n};
  auto out = gates::composite_eval_batch_with_postproc(
      party, backend, ch, key, bundle.suf, in, *hook);
  uint64_t r_out_share = key.r_out_share.empty() ? 0ull : key.r_out_share[0];
  y_share.resize(n);
  for (size_t i = 0; i < n; ++i) {
    y_share[i] = proto::sub_mod(out.haty_share[i], r_out_share);
  }
}

}  // namespace runtime
